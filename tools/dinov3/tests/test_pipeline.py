import hashlib
import json
import shutil
import traceback
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tools.dinov3.pipeline import run


class Fake:
    def __call__(self, pixel_values):
        cls = torch.ones(1, 1024) * pixel_values.shape[-1]
        return SimpleNamespace(last_hidden_state=cls[:, None, :])


def fixture(tmp_path):
    frames, ann = tmp_path / "frames", tmp_path / "ann"
    ann.mkdir()
    records = []
    for split in ("train", "dev", "test"):
        folder = frames / split / split
        folder.mkdir(parents=True)
        for i in range(5):
            Image.new("RGB", (210, 260), "white").save(folder / f"{i:06}.png")
        np.save(ann / f"{split}_info_ml.npy", {0: {"fileid": split, "num_frames": 5}})
        records.append(
            dict(
                clip_id=split,
                split=split,
                path=f"{split}/{split}.npy",
                length=5,
                width=2048,
                dtype="float32",
            )
        )
    clip = tmp_path / "clip.json"
    clip.write_text(
        json.dumps(dict(schema_version=1, encoder="clip", expected_dim=2048, records=records))
    )
    return frames, ann, clip


def test_bounded_receipt_and_failure_never_publish_complete(tmp_path):
    frames, ann, clip = fixture(tmp_path)
    output = tmp_path / "out"
    metadata = {"model_sha": "a" * 40}
    stats = run(frames, ann, clip, output, "key", metadata, Fake(), "cpu", only=("train", "train"))
    assert stats == {"completed": 1, "complete": False}
    assert (output / "receipts/train/train.json").is_file()
    assert not (output / "complete").exists()
    (frames / "dev/dev/000000.png").write_bytes(b"broken")
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", metadata, Fake(), "cpu")
    failures = json.loads((output / "failures.json").read_text())
    assert failures["dev/dev"]["stage"] == "source"
    assert not (output / "complete").exists()
    assert (output / "train/train.npy").is_file()


def test_pipeline_uses_versioned_batch_size(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    from tools.dinov3 import pipeline

    observed = []

    def batched(paths, model, device, *, batch_size):
        observed.append(batch_size)
        return np.ones((len(paths), 2048), dtype=np.float32), [[210, 260]] * len(paths)

    monkeypatch.setattr(pipeline, "extract", batched)
    result = run(
        frames,
        ann,
        clip,
        tmp_path / "out",
        "fixture-key",
        {"model_sha": "a" * 40, "inference_batch_size": 4},
        Fake(),
        "cpu",
        only=("train", "train"),
    )
    assert result == {"completed": 1, "complete": False}
    assert observed == [4]


def test_complete_map_matches_factor_contract_with_small_fixture(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", dict(train=1, dev=1, test=1))
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED_PNG", dict(train=5, dev=5, test=5))
    output = tmp_path / "out"
    assert run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu") == {
        "completed": 3,
        "complete": True,
        "counts": {"train": 1, "dev": 1, "test": 1},
    }
    manifest = json.loads((output / "complete/manifest.json").read_text())
    mapping = json.loads((output / "complete/frame_rows.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["expected_dim"] == 2048
    assert manifest["encoder"] == "dinov3:key"
    assert {
        (r["split"], r["clip_id"], r["path"], r["length"], r["width"], r["dtype"])
        for r in manifest["records"]
    } == {(s, s, f"{s}/{s}.npy", 5, 2048, "float32") for s in ("train", "dev", "test")}
    assert mapping["encoder_key"] == "key"
    assert (
        mapping["spatial_manifest_hash"]
        == hashlib.sha256((output / "complete/manifest.json").read_bytes()).hexdigest()
    )
    for split in ("train", "dev", "test"):
        row = mapping["clips"][split]
        assert row["source_indices"] == list(range(5))
        assert set(row["sampled_images"]) == set(map(str, range(5)))
        for index, image_hash in row["sampled_images"].items():
            assert (
                image_hash
                == hashlib.sha256(
                    (frames / split / split / f"{int(index):06}.png").read_bytes()
                ).hexdigest()
            )
        assert (
            row["feature_hash"]
            == hashlib.sha256((output / split / f"{split}.npy").read_bytes()).hexdigest()
        )
        assert row["frame_count"] == 5
        assert row["paths"] == [f"{split}/{split}/{i:06}.png" for i in range(5)]
        assert row["resolutions"] == [[210, 260]] * 5
    assert json.loads((output / "failures.json").read_text()) == {}


def test_completed_rerun_read_only_and_rejects_source_drift(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", dict(train=1, dev=1, test=1))
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED_PNG", dict(train=5, dev=5, test=5))
    output = tmp_path / "out"
    metadata = {"model_sha": "a" * 40}
    run(frames, ann, clip, output, "key", metadata, Fake(), "cpu")

    def snapshot():
        return {
            path.relative_to(output).as_posix(): (
                path.read_bytes() if path.is_file() else None,
                path.stat().st_mtime_ns,
            )
            for path in (output, *output.rglob("*"))
        }

    before = snapshot()
    assert run(frames, ann, clip, output, "key", metadata, Fake(), "cpu")["complete"]
    assert snapshot() == before
    Image.new("RGB", (210, 260), "black").save(frames / "train/train/000000.png")
    with pytest.raises(ValueError, match="train/train"):
        run(frames, ann, clip, output, "key", metadata, Fake(), "cpu")
    assert snapshot() == before


def test_joint_publication_failure_never_exposes_half_manifest(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", dict(train=1, dev=1, test=1))
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED_PNG", dict(train=5, dev=5, test=5))
    from tools.dinov3 import pipeline

    write = pipeline.atomic_json

    def crash_on_row_map(path, payload):
        if path.name == "frame_rows.json":
            raise OSError("injected staging failure")
        return write(path, payload)

    monkeypatch.setattr(pipeline, "atomic_json", crash_on_row_map)
    with pytest.raises(OSError, match="injected"):
        run(frames, ann, clip, tmp_path / "out", "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    assert not (tmp_path / "out/complete").exists()
    assert not list((tmp_path / "out").glob(".complete-staging-*"))


def test_model_runtime_error_journaled_then_aborts_before_third_clip(tmp_path):
    frames, ann, clip = fixture(tmp_path)

    class Broken(Fake):
        calls = 0

        def __call__(self, pixel_values):
            self.calls += 1
            if self.calls == 11:
                raise RuntimeError("encoder incompatible")
            return super().__call__(pixel_values)

    model = Broken()
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="dev/dev.*model forward failed"):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, model, "cpu")
    assert model.calls == 11
    assert (output / "train/train.npy").is_file()
    assert not (output / "test/test.npy").exists()
    assert json.loads((output / "failures.json").read_text())["dev/dev"]["stage"] == "encode"
    assert not (output / "complete").exists()


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, TypeError, AttributeError])
def test_model_forward_failure_aborts_and_never_records_upstream_details(tmp_path, error_type):
    frames, ann, clip = fixture(tmp_path)
    token = "synthetic-private-token"
    url = f"https://invalid.example/signed?token={token}"

    class Broken(Fake):
        calls = 0

        def __call__(self, pixel_values):
            self.calls += 1
            if self.calls == 11:
                raise error_type(f"failed at {url}")
            return super().__call__(pixel_values)

    model = Broken()
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="dev/dev.*model forward failed") as caught:
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, model, "cpu")
    rendered = "".join(traceback.format_exception(caught.value))
    journal = (output / "failures.json").read_text()
    assert token not in journal + rendered and url not in journal + rendered
    assert len(json.loads(journal)["dev/dev"]["reason"]) < 100
    assert json.loads(journal)["dev/dev"]["stage"] == "encode"
    assert model.calls == 11
    assert (output / "train/train.npy").is_file()
    assert not (output / "test/test.npy").exists()
    assert not (output / "complete").exists()


def test_unexpected_runtime_error_in_pipeline_is_journaled_with_static_reason(
    tmp_path, monkeypatch
):
    frames, ann, clip = fixture(tmp_path)
    token = "synthetic-private-token"
    url = f"https://invalid.example/signed?token={token}"

    def broken_extract(paths, model, device):
        raise RuntimeError(f"unexpected failure at {url}")

    monkeypatch.setattr("tools.dinov3.pipeline.extract", broken_extract)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="train/train.*extraction runtime failure") as caught:
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    journal = (output / "failures.json").read_text()
    rendered = "".join(traceback.format_exception(caught.value))
    assert json.loads(journal)["train/train"] == {
        "stage": "encode",
        "reason": "DINOv3 extraction runtime failure",
    }
    assert token not in journal + rendered and url not in journal + rendered
    assert not (output / "dev/dev.npy").exists()


def test_png_corrupted_after_scan_is_journaled_and_next_clip_runs(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    from tools.dinov3 import pipeline

    actual_extract = pipeline.extract

    def corrupt_before_decode(paths, model, device):
        if paths[0].parent.name == "dev":
            paths[0].write_bytes(b"broken")
        return actual_extract(paths, model, device)

    monkeypatch.setattr(pipeline, "extract", corrupt_before_decode)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    failure = json.loads((output / "failures.json").read_text())["dev/dev"]
    assert failure["stage"] == "encode"
    assert "invalid PNG" in failure["reason"]
    assert "dev/dev/000000.png" in failure["reason"]
    assert (output / "test/test.npy").is_file()
    assert not (output / "complete").exists()


@pytest.mark.parametrize(
    ("states", "category"),
    [
        (torch.ones(1, 2, 768), "CLS width/shape mismatch"),
        (torch.full((1, 2, 1024), float("nan")), "non-finite CLS"),
    ],
)
def test_bad_cls_aborts_with_static_category(tmp_path, states, category):
    frames, ann, clip = fixture(tmp_path)

    class Bad(Fake):
        calls = 0

        def __call__(self, pixel_values):
            self.calls += 1
            if self.calls == 11:
                return SimpleNamespace(last_hidden_state=states)
            return super().__call__(pixel_values)

    model = Bad()
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="dev/dev.*" + category):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, model, "cpu")
    assert json.loads((output / "failures.json").read_text())["dev/dev"] == {
        "stage": "encode",
        "reason": category,
    }
    assert model.calls == 11
    assert not (output / "test/test.npy").exists()


def test_malformed_resume_receipt_is_journaled_with_clip_id(tmp_path):
    frames, ann, clip = fixture(tmp_path)
    output = tmp_path / "out"
    metadata = {"model_sha": "a" * 40}
    run(frames, ann, clip, output, "key", metadata, Fake(), "cpu", only=("train", "train"))
    (output / "receipts/train/train.json").write_text("{")
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", metadata, Fake(), "cpu", only=("train", "train"))
    assert (
        "train/train: malformed receipt"
        in (json.loads((output / "failures.json").read_text())["train/train"]["reason"])
    )


def test_full_gate_rejects_partial_split_counts(tmp_path):
    frames, ann, clip = fixture(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, tmp_path / "out", "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    assert not (tmp_path / "out/complete").exists()


def test_source_change_between_scan_and_encode_cannot_publish(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    from tools.dinov3 import pipeline

    real_extract = pipeline.extract

    def change_input(paths, model, device):
        result = real_extract(paths, model, device)
        Image.new("RGB", (210, 260), "black").save(paths[0])
        return result

    monkeypatch.setattr(pipeline, "extract", change_input)
    with pytest.raises(ValueError, match="incomplete"):
        run(
            frames,
            ann,
            clip,
            tmp_path / "out",
            "key",
            {"model_sha": "a" * 40},
            Fake(),
            "cpu",
            only=("train", "train"),
        )
    assert "source PNG changed" in (tmp_path / "out/failures.json").read_text()
    assert not (tmp_path / "out/train/train.npy").exists()


def test_resumed_source_change_before_final_promotion_fails(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    output = tmp_path / "out"
    metadata = {"model_sha": "a" * 40}
    run(frames, ann, clip, output, "key", metadata, Fake(), "cpu", only=("train", "train"))
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", dict(train=1, dev=1, test=1))
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED_PNG", dict(train=5, dev=5, test=5))
    from tools.dinov3 import pipeline

    original = pipeline.check_receipt

    def change_after_resume(root, split, clip_id, source, key):
        receipt = original(root, split, clip_id, source, key)
        if receipt is not None and split == "train":
            Image.new("RGB", (210, 260), "black").save(frames / "train/train/000000.png")
        return receipt

    monkeypatch.setattr(pipeline, "check_receipt", change_after_resume)
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", metadata, Fake(), "cpu")
    assert "train/train" in json.loads((output / "failures.json").read_text())
    assert not (output / "complete").exists()


def test_fresh_source_change_during_staging_is_caught_before_promotion(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    output = tmp_path / "out"
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", dict(train=1, dev=1, test=1))
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED_PNG", dict(train=5, dev=5, test=5))
    from tools.dinov3 import pipeline

    write = pipeline.atomic_json

    def mutate_after_manifest(path, payload):
        write(path, payload)
        if path.name == "manifest.json":
            Image.new("RGB", (210, 260), "black").save(frames / "dev/dev/000001.png")

    monkeypatch.setattr(pipeline, "atomic_json", mutate_after_manifest)
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    assert "dev/dev" in json.loads((output / "failures.json").read_text())
    assert not (output / "complete").exists()


def test_missing_source_directory_is_journaled_per_clip(tmp_path):
    frames, ann, clip = fixture(tmp_path)
    shutil.rmtree(frames / "dev/dev")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    assert json.loads((output / "failures.json").read_text())["dev/dev"]["stage"] == "source"
    assert not (output / "complete").exists()


def test_full_gate_checks_source_png_totals_after_split_parity(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", dict(train=1, dev=1, test=1))
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED_PNG", dict(train=6, dev=5, test=5))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="incomplete.*PNG totals"):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    assert not (output / "complete").exists()


def test_failed_clip_can_resume_after_input_repair_without_reencoding_good_clip(
    tmp_path, monkeypatch
):
    frames, ann, clip = fixture(tmp_path)
    output = tmp_path / "out"
    metadata = {"model_sha": "a" * 40}
    (frames / "dev/dev/000000.png").write_bytes(b"broken")
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", metadata, Fake(), "cpu")
    original = (output / "train/train.npy").read_bytes()
    Image.new("RGB", (210, 260), "white").save(frames / "dev/dev/000000.png")
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", dict(train=1, dev=1, test=1))
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED_PNG", dict(train=5, dev=5, test=5))

    class NeverEncodeGood(Fake):
        def __call__(self, pixel_values):
            assert (output / "train/train.npy").read_bytes() == original
            return super().__call__(pixel_values)

    assert run(frames, ann, clip, output, "key", metadata, NeverEncodeGood(), "cpu")["complete"]
    assert (output / "train/train.npy").read_bytes() == original
    assert json.loads((output / "failures.json").read_text()) == {}


def test_resumed_source_drift_journals_failure_and_preserves_receipt(tmp_path):
    frames, ann, clip = fixture(tmp_path)
    output = tmp_path / "out"
    metadata = {"model_sha": "a" * 40}
    run(frames, ann, clip, output, "key", metadata, Fake(), "cpu", only=("train", "train"))
    feature = (output / "train/train.npy").read_bytes()
    receipt = (output / "receipts/train/train.json").read_bytes()
    Image.new("RGB", (210, 260), "black").save(frames / "train/train/000000.png")
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", metadata, Fake(), "cpu", only=("train", "train"))
    assert (
        "incompatible receipt"
        in (json.loads((output / "failures.json").read_text())["train/train"]["reason"])
    )
    assert (output / "train/train.npy").read_bytes() == feature
    assert (output / "receipts/train/train.json").read_bytes() == receipt
    assert not (output / "complete").exists()
