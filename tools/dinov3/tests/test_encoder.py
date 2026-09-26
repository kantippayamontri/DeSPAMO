import traceback
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from requests import Response
from requests.exceptions import Timeout

from tools.dinov3.encoder import extract, load_model
from tools.dinov3.frames import preprocess
from tools.dinov3.identity import MODEL


class Fake:
    def __init__(self):
        self.calls = []

    def __call__(self, pixel_values):
        assert pixel_values.shape[0:2] == (1, 3)
        assert pixel_values.shape[-2] == pixel_values.shape[-1]
        assert pixel_values.dtype == torch.float32
        assert torch.is_inference_mode_enabled()
        self.calls.append(pixel_values.shape[-1])
        side = pixel_values.shape[-1]
        cls = torch.full((1, 1024), float(side), device=pixel_values.device)
        register = torch.full((4, 1024), -1.0, device=pixel_values.device)
        patch = torch.full((1, 1024), -2.0, device=pixel_values.device)
        return SimpleNamespace(last_hidden_state=torch.cat((cls, register, patch))[None])


def test_two_pass_cls_order_width_and_determinism(tmp_path):
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    Image.new("RGB", (210, 260), "white").save(first_image)
    Image.new("RGB", (120, 90), "black").save(second_image)
    model = Fake()
    first, resolutions = extract([first_image, second_image], model, "cpu")
    second, _ = extract([first_image, second_image], model, "cpu")

    assert resolutions == [[210, 260], [120, 90]]
    assert model.calls == [224, 448, 224, 448] * 2
    assert first.shape == (2, 2048) and first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, :1024], np.full((2, 1024), 224))
    np.testing.assert_array_equal(first[:, 1024:], np.full((2, 1024), 448))


def test_batched_two_scale_cls_preserves_frame_order(tmp_path):
    paths = []
    for index in range(10):
        path = tmp_path / f"{index:02}.png"
        Image.new("RGB", (210, 260), (index * 20, 0, 0)).save(path)
        paths.append(path)

    class BatchModel:
        def __init__(self):
            self.calls = []
            self.tf32_flags = []
            self.cudnn_enabled = []

        def __call__(self, pixel_values):
            size = pixel_values.shape[-1]
            batch = pixel_values.shape[0]
            self.calls.append((size, batch))
            self.tf32_flags.append(torch.backends.cudnn.allow_tf32)
            self.cudnn_enabled.append(torch.backends.cudnn.enabled)
            cls = pixel_values[:, :1, 0, 0].expand(batch, 1024) + size
            registers = torch.full((batch, 4, 1024), -1.0)
            return SimpleNamespace(last_hidden_state=torch.cat((cls[:, None], registers), 1))

    model = BatchModel()
    prior_tf32 = torch.backends.cudnn.allow_tf32
    result, resolutions = extract(paths, model, "cpu", batch_size=4)
    assert result.shape == (10, 2048) and result.dtype == np.float32
    assert resolutions == [[210, 260]] * 10
    assert model.calls == [(224, 4), (448, 4), (224, 4), (448, 4), (224, 2), (448, 2)]
    assert model.tf32_flags == [False] * 6
    assert model.cudnn_enabled == [torch.backends.cudnn.enabled] * 6
    assert torch.backends.cudnn.allow_tf32 == prior_tf32
    markers = np.array([preprocess(path, 224)[0, 0, 0].item() for path in paths], dtype=np.float32)
    np.testing.assert_allclose(result[:, 0], markers + 224)
    np.testing.assert_allclose(result[:, 1024], markers + 448)
    assert np.all(result[:, 1:1024] == result[:, :1])


def test_batch_size_must_be_positive(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (210, 260)).save(path)
    with pytest.raises(ValueError, match="batch_size"):
        extract([path], Fake(), "cpu", batch_size=0)


def test_extract_bad_png_crc_is_path_qualified_value_error(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (3, 2), "white").save(image)
    raw = bytearray(image.read_bytes())
    offset = raw.index(b"IDAT")
    length = int.from_bytes(raw[offset - 4 : offset], "big")
    raw[offset + 4 + length] ^= 1
    image.write_bytes(raw)

    model = Fake()
    with pytest.raises(ValueError, match=r"invalid PNG .*frame\.png") as caught:
        extract([image], model, "cpu")
    assert isinstance(caught.value.__cause__, SyntaxError)
    assert model.calls == []


@pytest.mark.parametrize("shape", [(1024,), (1, 1024), (2, 5, 1024), (1, 0, 1024), (1, 5, 768)])
def test_invalid_hidden_state_shape_fails_without_fallback(tmp_path, shape):
    image = tmp_path / "frame.png"
    Image.new("RGB", (210, 260)).save(image)

    class Bad:
        def __call__(self, pixel_values):
            return SimpleNamespace(last_hidden_state=torch.ones(shape))

    with pytest.raises(RuntimeError, match="CLS"):
        extract([image], Bad(), "cpu")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_cls_fails_without_fallback(tmp_path, value):
    image = tmp_path / "frame.png"
    Image.new("RGB", (210, 260)).save(image)

    class Bad:
        def __call__(self, pixel_values):
            states = torch.ones((1, 5, 1024))
            states[0, 0, 0] = value
            return SimpleNamespace(last_hidden_state=states)

    with pytest.raises(RuntimeError, match="non-finite CLS"):
        extract([image], Bad(), "cpu")


def test_extract_sanitizes_model_runtime_error(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (3, 2)).save(image)

    class Broken:
        def __call__(self, pixel_values):
            raise RuntimeError("encoder incompatible")

    with pytest.raises(RuntimeError, match="DINOv3 model forward failed"):
        extract([image], Broken(), "cpu")


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, TypeError, AttributeError])
def test_model_forward_errors_are_fatal_and_hide_upstream_details(tmp_path, error_type):
    image = tmp_path / "frame.png"
    Image.new("RGB", (3, 2)).save(image)
    token = "synthetic-private-token"
    url = f"https://invalid.example/signed?token={token}"

    class Broken:
        def __call__(self, pixel_values):
            raise error_type(f"failed at {url}")

    with pytest.raises(RuntimeError, match="DINOv3 model forward failed") as caught:
        extract([image], Broken(), "cpu")
    rendered = "".join(traceback.format_exception(caught.value))
    assert token not in rendered and url not in rendered
    assert len(str(caught.value)) < 100


def test_device_transfer_error_is_fatal_and_hides_details(tmp_path, monkeypatch):
    image = tmp_path / "frame.png"
    Image.new("RGB", (3, 2)).save(image)
    token = "synthetic-private-token"
    url = f"https://invalid.example/signed?token={token}"

    def failed_transfer(self, *args, **kwargs):
        raise ValueError(f"failed at {url}")

    monkeypatch.setattr(torch.Tensor, "to", failed_transfer)
    with pytest.raises(RuntimeError, match="DINOv3 device transfer failed") as caught:
        extract([image], Fake(), "cpu")
    rendered = "".join(traceback.format_exception(caught.value))
    assert token not in rendered and url not in rendered


@pytest.mark.parametrize("states", [torch.ones(1, 2, 768), torch.full((1, 2, 1024), float("nan"))])
def test_bad_cls_category_omits_source_filename(tmp_path, states):
    token = "synthetic-private-token"
    image = tmp_path / f"frame-{token}.png"
    Image.new("RGB", (3, 2)).save(image)

    class Broken:
        def __call__(self, pixel_values):
            return SimpleNamespace(last_hidden_state=states)

    with pytest.raises(RuntimeError, match="CLS") as caught:
        extract([image], Broken(), "cpu")
    assert token not in "".join(traceback.format_exception(caught.value))


def test_load_model_pins_official_weights_and_freezes_parameters(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(hidden_size=1024, patch_size=16, num_register_tokens=4)
    calls = []

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return model

    monkeypatch.setattr("tools.dinov3.encoder.AutoModel.from_pretrained", fake_from_pretrained)
    assert load_model("a" * 40, "fake-token", tmp_path, "cpu") is model
    assert calls == [
        (
            (MODEL,),
            dict(
                revision="a" * 40,
                token="fake-token",
                cache_dir=tmp_path,
                trust_remote_code=False,
                use_safetensors=True,
            ),
        )
    ]
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert next(model.parameters()).device.type == "cpu"


def test_load_model_rejects_mutable_revision_before_loading(tmp_path, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid SHA must not load weights")

    monkeypatch.setattr("tools.dinov3.encoder.AutoModel.from_pretrained", unexpected_load)
    with pytest.raises(ValueError, match="SHA"):
        load_model("main", "fake-token", tmp_path, "cpu")


@pytest.mark.parametrize(
    ("field", "value"),
    [("hidden_size", 768), ("patch_size", 14), ("num_register_tokens", 0)],
)
def test_load_model_rejects_wrong_config(tmp_path, monkeypatch, field, value):
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(hidden_size=1024, patch_size=16, num_register_tokens=4)
    setattr(model.config, field, value)
    monkeypatch.setattr("tools.dinov3.encoder.AutoModel.from_pretrained", lambda *a, **kw: model)
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_model("a" * 40, "fake-token", tmp_path, "cpu")


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "access denied"),
        (403, "access denied"),
        (400, "Hub request rejected"),
        (404, "model revision"),
        (503, "Hub unavailable"),
    ],
)
def test_load_model_classifies_hub_status_without_leaking_details(
    tmp_path, monkeypatch, status, message
):
    token = "fake-" + "token"
    url = f"https://invalid.example/models?token={token}"
    response = Response()
    response.status_code = status
    response.url = url
    error = HfHubHTTPError(f"upstream failure at {url}", response=response)

    def fake_from_pretrained(*args, **kwargs):
        raise error

    monkeypatch.setattr("tools.dinov3.encoder.AutoModel.from_pretrained", fake_from_pretrained)
    with pytest.raises(RuntimeError, match=message) as caught:
        load_model("a" * 40, token, tmp_path, "cpu")
    rendered = "".join(traceback.format_exception(caught.value))
    assert (status in (401, 403)) == ("access denied" in str(caught.value))
    assert token not in rendered and url not in rendered
    assert "HfHubHTTPError" not in rendered


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (Timeout, "Hub request failed"),
        (torch.cuda.OutOfMemoryError, "CUDA out of memory"),
        (ValueError, "incompatible"),
    ],
)
def test_load_model_classifies_network_cuda_and_incompatible_without_leaks(
    tmp_path, monkeypatch, error_type, message
):
    token = "fake-" + "token"
    url = f"https://invalid.example/models?token={token}"

    def fake_from_pretrained(*args, **kwargs):
        raise error_type(f"failed at {url}")

    monkeypatch.setattr("tools.dinov3.encoder.AutoModel.from_pretrained", fake_from_pretrained)
    with pytest.raises(RuntimeError, match=message) as caught:
        load_model("a" * 40, token, tmp_path, "cpu")
    rendered = "".join(traceback.format_exception(caught.value))
    assert token not in rendered and url not in rendered
    assert error_type.__name__ not in rendered
    assert "gated" not in str(caught.value)
