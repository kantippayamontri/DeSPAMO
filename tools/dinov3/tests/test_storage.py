import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools.dinov3 import storage
from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import (
    atomic_array,
    atomic_json,
    check_receipt,
    publish_clip,
    verify_location,
    writer,
)


def source():
    return dict(
        paths=[f"train/c/{i}.png" for i in range(5)],
        source_indices=list(range(5)),
        source_hash="input",
        sampled_images={str(i): f"{i + 1:064x}" for i in range(5)},
        resolutions=[[210, 260]] * 5,
    )


def features():
    return np.ones((5, 2048), dtype=np.float32)


def test_writer_rejects_checkout_before_creating_output(tmp_path):
    checkout = storage.Path(__file__).resolve().parents[3]
    output = checkout / "task4-never-create"
    with pytest.raises(ValueError, match="inside Git checkout"):
        with writer(output):
            pass
    assert not output.exists()


def test_writer_contention_is_nonblocking_and_lock_releases(tmp_path):
    root = tmp_path / "output"
    with writer(root):
        assert (root / ".writer.lock").is_file()
        with pytest.raises(RuntimeError, match="writer"):
            with writer(root):
                pass
    with writer(root):
        assert (root / ".writer.lock").is_file()


def test_completed_root_without_lock_fails_closed(tmp_path):
    root = tmp_path / "output"
    (root / "complete").mkdir(parents=True)
    with pytest.raises(ValueError, match="missing writer lock"):
        with writer(root):
            pass
    assert not (root / ".writer.lock").exists()


def test_checkout_guard_is_read_only_and_resolves_symlinks(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    with pytest.raises(ValueError, match="inside Git checkout"):
        verify_location(checkout / "cache", checkout)
    alias = tmp_path / "alias"
    alias.symlink_to(checkout, target_is_directory=True)
    with pytest.raises(ValueError, match="inside Git checkout"):
        verify_location(alias / "outputs", checkout)
    assert not (checkout / "cache").exists()
    assert not (checkout / "outputs").exists()
    verify_location(tmp_path / "external", checkout)
    assert not (tmp_path / "external").exists()


def test_mount_guard_uses_nearest_existing_ancestor_without_writing(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    target = tmp_path / "external" / "child"
    calls = []

    def findmnt(args, **kwargs):
        calls.append((args, kwargs))
        return "drvfs\n"

    monkeypatch.setattr(storage.subprocess, "check_output", findmnt)
    with pytest.raises(ValueError, match="Linux-native filesystem"):
        verify_location(target, checkout)
    assert calls[0][0] == ["findmnt", "-no", "FSTYPE", "-T", str(tmp_path)]
    assert "HF_TOKEN" not in calls[0][1]["env"]
    assert not target.parent.exists()


def test_atomic_array_rejects_invalid_shapes_dtype_and_nonfinite_before_writing(tmp_path):
    path = tmp_path / "nested" / "c.npy"
    for array in (
        np.ones((4, 2048), dtype=np.float32),
        np.ones((5, 1024), dtype=np.float32),
        np.ones((5, 2048), dtype=np.float64),
        np.full((5, 2048), np.nan, dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="invalid feature array"):
            atomic_array(path, array)
    assert not path.parent.exists()


def test_atomic_writes_round_trip_and_leave_no_partial_files(tmp_path):
    data = features()
    array_path = tmp_path / "train" / "c.npy"
    receipt_path = tmp_path / "receipts" / "c.json"
    atomic_array(array_path, data)
    atomic_json(receipt_path, {"key": ["one", "two"]})
    np.testing.assert_array_equal(np.load(array_path, allow_pickle=False), data)
    assert json.loads(receipt_path.read_text()) == {"key": ["one", "two"]}
    assert not list(tmp_path.rglob(".partial-*"))


@pytest.mark.parametrize("payload", [{"values": (1, 2)}, {1: "value"}])
def test_atomic_json_rejects_type_loss_before_replacing_or_creating_dirs(tmp_path, payload):
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"original")
    for path in (existing, tmp_path / "new" / "invalid.json"):
        with pytest.raises(ValueError, match="JSON round-trip mismatch"):
            atomic_json(path, payload)
    assert existing.read_bytes() == b"original"
    assert not (tmp_path / "new").exists()
    assert not list(tmp_path.rglob(".partial-*"))


@pytest.mark.parametrize(
    "function,suffix,payload", [(atomic_array, ".npy", None), (atomic_json, ".json", {"new": 1})]
)
def test_atomic_replace_failure_keeps_previous_file_and_cleans_temp(
    tmp_path, monkeypatch, function, suffix, payload
):
    path = tmp_path / "existing" / f"c{suffix}"
    path.parent.mkdir()
    path.write_bytes(b"original")

    def fail_replace(*args):
        raise OSError("replace interrupted")

    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace interrupted"):
        function(path, features() if payload is None else payload)
    assert path.read_bytes() == b"original"
    assert not list(path.parent.glob(".partial-*"))


def test_file_fsync_failure_preserves_destination_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    path.write_text("original")

    def fail_fsync(fd):
        raise OSError("disk sync failed")

    monkeypatch.setattr(storage.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="disk sync failed"):
        atomic_json(path, {"key": "new"})
    assert path.read_text() == "original"
    assert not list(tmp_path.glob(".partial-*"))


def test_creating_atomic_parent_syncs_its_parent_directory(tmp_path, monkeypatch):
    synced = []
    real_fsync = os.fsync

    def record_fsync(fd):
        synced.append(os.fstat(fd).st_ino)
        return real_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", record_fsync)
    atomic_json(tmp_path / "new" / "c.json", {"key": "value"})
    assert tmp_path.stat().st_ino in synced


def test_resume_integrity_rejects_source_feature_and_key_drift(tmp_path):
    original = source()
    with writer(tmp_path):
        receipt = publish_clip(tmp_path, "train", "c", original, "key", features())
        assert receipt == check_receipt(tmp_path, "train", "c", original, "key")
        assert check_receipt(tmp_path, "train", "missing", original, "key") is None
        for changed, key in (
            ({**original, "source_hash": "changed"}, "key"),
            ({**original, "resolutions": [[999, 999]] * 5}, "key"),
            (original, "other"),
        ):
            with pytest.raises(ValueError, match="incompatible receipt"):
                check_receipt(tmp_path, "train", "c", changed, key)
        # scan() supplies resolutions=[] on resume; receipt holds encoded resolutions.
        assert (
            check_receipt(tmp_path, "train", "c", {**original, "resolutions": []}, "key") == receipt
        )
        before = (tmp_path / "train/c.npy").read_bytes()
        with pytest.raises(ValueError, match="already exists"):
            publish_clip(tmp_path, "train", "c", original, "key", features() * 2)
        assert (tmp_path / "train/c.npy").read_bytes() == before
        (tmp_path / "train/c.npy").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="feature hash"):
            check_receipt(tmp_path, "train", "c", original, "key")
        with pytest.raises(ValueError, match="feature hash"):
            publish_clip(tmp_path, "train", "c", original, "key", features())
        assert (tmp_path / "train/c.npy").read_bytes() == b"corrupt"


def test_malformed_receipt_has_clip_context_and_cannot_be_overwritten(tmp_path):
    path = tmp_path / "receipts/train/c.json"
    path.parent.mkdir(parents=True)
    for content in ('{"feature_hash":', '{"encoder_key": "key"}', "[]", "null"):
        path.write_text(content)
        with pytest.raises(ValueError, match="train/c: malformed receipt"):
            check_receipt(tmp_path, "train", "c", source(), "key")
        with pytest.raises(ValueError, match="train/c: malformed receipt"):
            publish_clip(tmp_path, "train", "c", source(), "key", features())
        assert path.read_text() == content
        assert not (tmp_path / "train/c.npy").exists()


def test_receipt_hash_matching_invalid_feature_still_fails_closed(tmp_path):
    original = source()
    publish_clip(tmp_path, "train", "c", original, "key", features())
    feature = tmp_path / "train/c.npy"
    np.save(feature, np.ones((5, 2048), dtype=np.float64))
    path = tmp_path / "receipts/train/c.json"
    receipt = json.loads(path.read_text())
    receipt["feature_hash"] = sha256_file(feature)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="feature shape/dtype/finiteness"):
        check_receipt(tmp_path, "train", "c", original, "key")


def test_orphan_array_can_be_replaced_but_invalid_rows_cannot(tmp_path):
    feature = tmp_path / "train/c.npy"
    feature.parent.mkdir()
    feature.write_bytes(b"orphan")
    original = source()
    with pytest.raises(ValueError, match="source/feature row mismatch"):
        publish_clip(tmp_path, "train", "c", original, "key", features()[:4])
    assert feature.read_bytes() == b"orphan"
    receipt = publish_clip(tmp_path, "train", "c", original, "key", features())
    assert receipt["feature_hash"] == sha256_file(feature)
    assert check_receipt(tmp_path, "train", "c", original, "key") == receipt


@pytest.mark.parametrize("dangling", [False, True])
def test_writer_rejects_lock_symlink_without_following_it(tmp_path, dangling):
    root = tmp_path / "out"
    root.mkdir()
    target = tmp_path / "missing" if dangling else tmp_path / "target"
    if not dangling:
        target.write_bytes(b"keep")
    (root / ".writer.lock").symlink_to(target)
    with pytest.raises(ValueError, match="writer lock"):
        with writer(root):
            pass
    assert (root / ".writer.lock").is_symlink()
    assert not target.exists() if dangling else target.read_bytes() == b"keep"


def test_writer_opens_regular_lock_without_following_symlinks(tmp_path, monkeypatch):
    opened = []
    real_open = os.open

    def track_open(path, flags, *args, **kwargs):
        if str(path).endswith(".writer.lock"):
            opened.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", track_open)
    with writer(tmp_path / "out"):
        pass
    assert len(opened) == 1 and opened[0] & os.O_NOFOLLOW


@pytest.mark.parametrize("component", ["train", "receipts/train"])
def test_nested_checkout_symlink_cannot_publish_or_read(tmp_path, component):
    root = tmp_path / "out"
    root.mkdir()
    checkout = Path(__file__).resolve().parents[3]
    alias = root / component
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(checkout, target_is_directory=True)
    with writer(root):
        with pytest.raises(ValueError, match="unsafe output path"):
            check_receipt(root, "train", "c", source(), "key")
        with pytest.raises(ValueError, match="unsafe output path"):
            publish_clip(root, "train", "c", source(), "key", features())
    assert alias.is_symlink()
    assert not (checkout / "c.npy").exists()
    assert not (checkout / "c.json").exists()


def test_symlinked_feature_leaf_cannot_be_replaced(tmp_path):
    root = tmp_path / "out"
    (root / "train").mkdir(parents=True)
    outside = tmp_path / "original"
    outside.write_bytes(b"keep")
    (root / "train/c.npy").symlink_to(outside)
    with writer(root):
        with pytest.raises(ValueError, match="unsafe output path"):
            publish_clip(root, "train", "c", source(), "key", features())
    assert outside.read_bytes() == b"keep"
    assert not (root / "receipts").exists()


def test_derived_device_change_rejected_without_findmnt_per_clip(tmp_path, monkeypatch):
    root = tmp_path / "out"
    (root / "train").mkdir(parents=True)
    calls = []
    real_findmnt = storage.subprocess.check_output
    real_stat = os.stat

    def findmnt(args, **kwargs):
        calls.append(args)
        return real_findmnt(args, **kwargs)

    def foreign_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if Path(path) == root / "train":
            values = list(result)
            values[2] += 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(storage.subprocess, "check_output", findmnt)
    with writer(root):
        monkeypatch.setattr(storage.os, "stat", foreign_stat)
        with pytest.raises(ValueError, match="output filesystem"):
            publish_clip(root, "train", "c", source(), "key", features())
    assert len(calls) == 1
    assert not (root / "train/c.npy").exists()


@pytest.mark.parametrize(
    "split,clip",
    [("../escape", "c"), ("train", "../escape"), ("dev", ""), ("test", ".."), ("train", "a\\b")],
)
def test_invalid_clip_identity_rejected_before_io(tmp_path, split, clip):
    root = tmp_path / "out"
    for operation in (
        lambda: check_receipt(root, split, clip, source(), "key"),
        lambda: publish_clip(root, split, clip, source(), "key", features()),
    ):
        with pytest.raises(ValueError, match="invalid clip"):
            operation()
    assert not root.exists()


@pytest.mark.parametrize(
    "images",
    [
        {str(i): f"{i + 1:064x}" for i in range(4)},
        {**{str(i): f"{i + 1:064x}" for i in range(4)}, "5": f"{5:064x}"},
        {**{str(i): f"{i + 1:064x}" for i in range(4)}, "-1": f"{5:064x}"},
        {**{str(i): f"{i + 1:064x}" for i in range(4)}, "4": "A" * 64},
        {**{str(i): f"{i + 1:064x}" for i in range(4)}, "4": "sha"},
    ],
)
def test_invalid_source_samples_cannot_publish(tmp_path, images):
    original = {**source(), "sampled_images": images}
    with pytest.raises(ValueError, match="sampled_images"):
        publish_clip(tmp_path, "train", "c", original, "key", features())
    assert not (tmp_path / "train").exists()


def test_identical_png_hashes_at_distinct_source_ordinals_can_resume(tmp_path):
    original = {**source(), "sampled_images": {str(i): "a" * 64 for i in range(5)}}
    receipt = publish_clip(tmp_path, "train", "c", original, "key", features())
    assert receipt["sampled_images"] == original["sampled_images"]
    assert check_receipt(tmp_path, "train", "c", original, "key") == receipt


def test_missing_source_ordinals_rejected_before_publish(tmp_path):
    original = {**source(), "source_indices": [0, 1, 2, 3, 4, 5]}
    with pytest.raises(ValueError, match="sampled_images"):
        publish_clip(tmp_path, "train", "c", original, "key", features())
    assert not (tmp_path / "train").exists()


@pytest.mark.parametrize(
    "changes,key",
    [
        ({"resolutions": [[0, 0]] * 5}, "key"),
        ({"resolutions": [[210, -1]] * 5}, "key"),
        ({"resolutions": [(210, 260)] * 5}, "key"),
        ({"source_hash": 7}, "key"),
        ({"paths": ["train/c/0.png", "train/c/1.png", 2, "train/c/3.png", "train/c/4.png"]}, "key"),
        ({}, 3),
    ],
)
def test_invalid_source_metadata_cannot_publish_files(tmp_path, changes, key):
    root = tmp_path / "output"
    with pytest.raises(ValueError, match="invalid"):
        publish_clip(root, "train", "c", {**source(), **changes}, key, features())
    assert not root.exists()


def test_malformed_sampled_images_in_receipt_cannot_resume_or_be_replaced(tmp_path):
    original = source()
    publish_clip(tmp_path, "train", "c", original, "key", features())
    receipt_path = tmp_path / "receipts/train/c.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["sampled_images"]["4"] = "sha"
    receipt_path.write_text(json.dumps(receipt))
    before = (tmp_path / "train/c.npy").read_bytes()
    with pytest.raises(ValueError, match="train/c: malformed receipt"):
        check_receipt(tmp_path, "train", "c", original, "key")
    with pytest.raises(ValueError, match="train/c: malformed receipt"):
        publish_clip(tmp_path, "train", "c", original, "key", features())
    assert (tmp_path / "train/c.npy").read_bytes() == before
