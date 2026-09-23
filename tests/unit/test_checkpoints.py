import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from despamo import checkpoints
from despamo.checkpoints import convert_spamo_state_dict


def test_checkpoint_key_mapping() -> None:
    source = {
        "logit_scale": torch.tensor(1.0),
        "spatio_proj.weight": torch.tensor([1, 2], dtype=torch.int64),
        "spatiotemp_proj.weight": torch.tensor([3, 4], dtype=torch.float64),
        "fusion_proj.0.weight": torch.tensor([5, 6], dtype=torch.int32),
        "temporal_encoder.temporal_conv.0.weight": torch.tensor([7, 8], dtype=torch.float32),
        "t5_model.shared.weight": torch.tensor([9, 10], dtype=torch.float16),
    }
    converted = convert_spamo_state_dict(source)
    expected = {
        "logit_scale": "vt_align.logit_scale",
        "spatio_proj.weight": "visual_adapter.spatial_projector.weight",
        "spatiotemp_proj.weight": "visual_adapter.motion_projector.weight",
        "fusion_proj.0.weight": "visual_adapter.multimodal_projector.0.weight",
        "temporal_encoder.temporal_conv.0.weight": (
            "visual_adapter.temporal_encoder.temporal_conv.0.weight"
        ),
        "t5_model.shared.weight": "language_model.model.shared.weight",
    }
    assert set(converted) == set(expected.values())
    for original, target in expected.items():
        assert converted[target] is source[original]
        assert converted[target].dtype == source[original].dtype
        torch.testing.assert_close(converted[target], source[original])


def test_wrapped_lora_source_converts_to_wrapped_target_with_strict_load() -> None:
    class TinyPeft(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_model = torch.nn.Module()
            self.base_model.model = torch.nn.Module()
            self.base_model.model.shared = torch.nn.Linear(2, 2, bias=False)
            self.base_model.model.encoder = torch.nn.Module()
            attention = torch.nn.Module()
            attention.base_layer = torch.nn.Linear(2, 2, bias=False)
            attention.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False)})
            attention.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})
            self.base_model.model.encoder.q = attention

    source = torch.nn.Module()
    source.t5_model = TinyPeft()
    target = torch.nn.Module()
    target.language_model = torch.nn.Module()
    target.language_model.model = TinyPeft()
    with torch.no_grad():
        for index, tensor in enumerate(source.state_dict().values(), start=1):
            tensor.fill_(index)
        for tensor in target.state_dict().values():
            tensor.zero_()

    converted = convert_spamo_state_dict(source.state_dict())

    assert set(converted) == set(target.state_dict())
    assert any(".base_layer." in key for key in converted)
    assert any(".lora_A.default." in key for key in converted)
    assert any(".lora_B.default." in key for key in converted)
    target.load_state_dict(converted, strict=True)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, converted[key])


def test_checkpoint_mapping_rejects_unknown_key() -> None:
    with pytest.raises(KeyError, match="unmapped SpaMo key"):
        convert_spamo_state_dict({"unknown.weight": torch.zeros(1)})


def test_checkpoint_mapping_rejects_colliding_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        checkpoints,
        "PREFIXES",
        (("first.", "visual_adapter.same."), ("second.", "visual_adapter.same.")),
    )
    with pytest.raises(KeyError, match="checkpoint mapping collision"):
        convert_spamo_state_dict({"first.weight": torch.zeros(1), "second.weight": torch.ones(1)})


def _run_converter(
    input_path: Path, output_path: Path, *options: str
) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "convert_spamo_checkpoint.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            *options,
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )


def _write_schema(path: Path, entries: dict[str, dict[str, object]]) -> Path:
    path.write_text(json.dumps({"state_dict": entries}), encoding="utf-8")
    return path


def test_converter_cli_writes_state_dict_and_source_hash(tmp_path: Path) -> None:
    source_path = tmp_path / "source.ckpt"
    output_path = tmp_path / "nested" / "converted.pt"
    source = {"spatio_proj.weight": torch.tensor([[3.0]]), "logit_scale": torch.tensor(1.0)}
    torch.save({"state_dict": source}, source_path)
    schema_path = _write_schema(
        tmp_path / "target-schema.json",
        {
            "visual_adapter.spatial_projector.weight": {
                "shape": [1, 1],
                "dtype": "torch.float32",
            },
            "vt_align.logit_scale": {"shape": [], "dtype": "torch.float32"},
        },
    )

    result = _run_converter(source_path, output_path, "--target-schema", str(schema_path))

    assert result.returncode == 0, result.stderr
    assert "converted 2 tensors" in result.stdout
    converted = torch.load(output_path, map_location="cpu", weights_only=True)
    assert set(converted["state_dict"]) == {
        "visual_adapter.spatial_projector.weight",
        "vt_align.logit_scale",
    }
    torch.testing.assert_close(
        converted["state_dict"]["visual_adapter.spatial_projector.weight"],
        source["spatio_proj.weight"],
    )
    assert converted["metadata"] == {
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "target_schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        "source_tensor_count": 2,
        "target_tensor_count": 2,
    }


def test_converter_requires_explicit_trust_for_unsafe_pickle(tmp_path: Path) -> None:
    source_path = tmp_path / "legacy.ckpt"
    output_path = tmp_path / "converted.pt"
    torch.save(
        {"state_dict": {"logit_scale": torch.tensor(1.0)}, "extra": Path("metadata")},
        source_path,
    )
    schema_path = _write_schema(
        tmp_path / "schema.json", {"vt_align.logit_scale": {"shape": [], "dtype": "torch.float32"}}
    )

    result = _run_converter(source_path, output_path, "--target-schema", str(schema_path))

    assert result.returncode != 0
    assert "--trusted-source" in result.stderr
    assert not output_path.exists()

    trusted_result = _run_converter(
        source_path, output_path, "--target-schema", str(schema_path), "--trusted-source"
    )
    assert trusted_result.returncode == 0, trusted_result.stderr
    assert set(torch.load(output_path, weights_only=True)["state_dict"]) == {"vt_align.logit_scale"}


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hard_link"])
def test_converter_rejects_output_alias_without_touching_source(tmp_path: Path, alias: str) -> None:
    source_path = tmp_path / "source.ckpt"
    torch.save({"state_dict": {"logit_scale": torch.tensor(1.0)}}, source_path)
    schema_path = _write_schema(
        tmp_path / "schema.json", {"vt_align.logit_scale": {"shape": [], "dtype": "torch.float32"}}
    )
    output_path = tmp_path / "output.pt"
    if alias == "same_path":
        output_path = source_path
    elif alias == "symlink":
        try:
            output_path.symlink_to(source_path)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
    else:
        try:
            os.link(source_path, output_path)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"hard links unavailable: {exc}")
    original = source_path.read_bytes()

    result = _run_converter(source_path, output_path, "--target-schema", str(schema_path))

    assert result.returncode != 0
    assert "input and output must be different files" in result.stderr
    assert source_path.read_bytes() == original
    assert output_path.read_bytes() == original


def test_converter_requires_target_schema(tmp_path: Path) -> None:
    source_path = tmp_path / "source.ckpt"
    output_path = tmp_path / "converted.pt"
    torch.save({"state_dict": {"logit_scale": torch.tensor(1.0)}}, source_path)

    result = _run_converter(source_path, output_path)

    assert result.returncode != 0
    assert "--target-schema" in result.stderr
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("case", "entries", "diagnostic"),
    [
        (
            "missing",
            {
                "vt_align.logit_scale": {"shape": [], "dtype": "torch.float32"},
                "visual_adapter.spatial_projector.weight": {
                    "shape": [1, 1],
                    "dtype": "torch.float32",
                },
                "unexpected.weight": {"shape": [1], "dtype": "torch.float32"},
            },
            "missing keys",
        ),
        (
            "extra",
            {"vt_align.logit_scale": {"shape": [], "dtype": "torch.float32"}},
            "extra keys",
        ),
        (
            "shape",
            {
                "vt_align.logit_scale": {"shape": [], "dtype": "torch.float32"},
                "visual_adapter.spatial_projector.weight": {
                    "shape": [2, 1],
                    "dtype": "torch.float32",
                },
            },
            "shape mismatches",
        ),
        (
            "dtype",
            {
                "vt_align.logit_scale": {"shape": [], "dtype": "torch.float32"},
                "visual_adapter.spatial_projector.weight": {
                    "shape": [1, 1],
                    "dtype": "torch.float16",
                },
            },
            "dtype mismatches",
        ),
    ],
)
def test_converter_rejects_schema_mismatch_without_output(
    tmp_path: Path, case: str, entries: dict[str, dict[str, object]], diagnostic: str
) -> None:
    source_path = tmp_path / "source.ckpt"
    output_path = tmp_path / "nested" / "converted.pt"
    torch.save(
        {"state_dict": {"logit_scale": torch.tensor(1.0), "spatio_proj.weight": torch.ones(1, 1)}},
        source_path,
    )
    schema_path = _write_schema(tmp_path / f"{case}.json", entries)
    original = source_path.read_bytes()

    result = _run_converter(source_path, output_path, "--target-schema", str(schema_path))

    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert not output_path.exists()
    assert not output_path.parent.exists()
    assert source_path.read_bytes() == original


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hard_link"])
def test_converter_rejects_target_schema_output_alias_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], alias: str
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    converter = importlib.import_module("convert_spamo_checkpoint")
    source_path = tmp_path / "source.ckpt"
    torch.save({"state_dict": {"logit_scale": torch.tensor(1.0)}}, source_path)
    schema_path = _write_schema(
        tmp_path / "schema.json", {"vt_align.logit_scale": {"shape": [], "dtype": "torch.float32"}}
    )
    output_path = tmp_path / "output.pt"
    if alias == "same_path":
        output_path = schema_path
    elif alias == "symlink":
        try:
            output_path.symlink_to(schema_path)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
    else:
        try:
            os.link(schema_path, output_path)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"hard links unavailable: {exc}")
    source_bytes = source_path.read_bytes()
    schema_bytes = schema_path.read_bytes()

    def unexpected_load(*args: object, **kwargs: object) -> None:
        pytest.fail("checkpoint loaded before schema/output alias rejection")

    monkeypatch.setattr(converter.torch, "load", unexpected_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_spamo_checkpoint.py",
            "--input",
            str(source_path),
            "--target-schema",
            str(schema_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        converter.main()

    assert exc.value.code == 2
    assert "target schema and output must be different files" in capsys.readouterr().err
    assert source_path.read_bytes() == source_bytes
    assert schema_path.read_bytes() == schema_bytes
    assert output_path.read_bytes() == schema_bytes
