import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from despamo.appearance.provenance import digest, file_hash
from tests.unit.appearance.merge_fixtures import REVISION, inputs, refreshed


class Encoder:
    revision = REVISION
    width = 768

    def encode(self, texts):
        return np.stack([np.full(768, len(t), np.float32) for t in texts])


def module():
    return importlib.import_module("despamo.appearance.merged_text_features")


def prepared(tmp_path, invalid_old=False, masked=False, old_width=768):
    from despamo.appearance.merged_targets import merge_targets

    value = inputs()
    old_root = tmp_path / "old"
    old_root.mkdir()
    if masked:
        value[1]["records"][0]["targets"][4].update(text=None, mask_reason="not_detected")
        for target in value[0]["records"][1]["targets"]:
            target["text"] = None
    value = refreshed(value)
    for record in value[0]["records"]:
        mask = np.array([t["text"] is not None for t in record["targets"]])
        vectors = np.repeat(np.arange(1, 20, dtype=np.float32)[:, None], old_width, axis=1)
        vectors[~mask] = 0
        path = old_root / record["path"]
        np.savez_compressed(path, vectors=vectors, valid=mask)
        record["file_hash"] = file_hash(path)
        if invalid_old:
            record["file_hash"] = "f"*64
    parents = {}
    for name, content in zip(("old_manifest", "crop_export", "source"), value[:3], strict=True):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(content))
        parents[name] = dict(path=str(p), sha256=file_hash(p))
    document = merge_targets(*value, parents=parents)
    return document, old_root


def test_stable_rows_unchanged_and_new_rows_encoded(tmp_path):
    doc, old = prepared(tmp_path, masked=True)
    manifest = module().encode_merged_targets(doc, old, Encoder(), tmp_path / "out")
    result = json.loads(manifest.read_text())
    assert result["metadata"]["revision"] == REVISION
    assert result["human_review_status"] == "not_assessed"
    for row in result["records"]:
        with (np.load(manifest.parent / row["path"]) as new,
              np.load(old / f"{row['clip_id']}.npz") as v1):
            assert new["vectors"].shape == (19, 768)
            assert new["vectors"].dtype == np.float32 and new["valid"].dtype == np.bool_
            assert new["vectors"][:4].tobytes() == v1["vectors"][:4].tobytes()
            assert np.array_equal(new["valid"][:4], v1["valid"][:4])
            assert not new["vectors"][~new["valid"]].any()
            if row["clip_id"] == "a":
                assert not new["valid"][4]
                assert new["vectors"][5, 0] == len("new right_hand 5")
            else:
                assert new["valid"][4:].all()
    assert (manifest.parent / "complete.json").is_file()


def test_changed_old_npz_rejected(tmp_path):
    doc, old = prepared(tmp_path, invalid_old=True)
    with pytest.raises(ValueError, match="hash"):
        module().encode_merged_targets(doc, old, Encoder(), tmp_path / "out")
    assert not list((tmp_path / "out").glob("*/complete.json"))


def test_idempotent_cache_verifies_without_reencoding(tmp_path):
    doc, old = prepared(tmp_path)
    path = module().encode_merged_targets(doc, old, Encoder(), tmp_path / "out")

    class Fails(Encoder):
        def encode(self, texts):
            raise AssertionError("completed cache should not reencode")

    assert module().encode_merged_targets(doc, old, Fails(), tmp_path / "out") == path
    data = json.loads(path.read_text())
    (path.parent / data["records"][0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError):
        module().encode_merged_targets(doc, old, Fails(), tmp_path / "out")


def test_sealed_inputs_survive_workspace_edit_but_detect_snapshot_change(tmp_path):
    doc, old = prepared(tmp_path)
    path = module().encode_merged_targets(doc, old, Encoder(), tmp_path / "out")
    original = doc["parents"]["crop_export"]["path"]
    Path(original).write_text("changed later")
    assert module().verify_merged_manifest(path)["records"]
    (path.parent / "inputs/crop_export.json").write_text("tampered")
    with pytest.raises(ValueError):
        module().verify_merged_manifest(path)


@pytest.mark.parametrize("failure", ["encoder", "snapshot", "revision", "width"])
def test_failed_publication_has_no_completion(tmp_path, failure):
    doc, old = prepared(tmp_path)

    class Bad(Encoder):
        def encode(self, texts):
            raise ValueError("token overflow")

    encoder = Bad() if failure == "encoder" else Encoder()
    if failure == "snapshot":
        doc["parents"]["source"]["sha256"] = "f"*64
    if failure == "revision":
        encoder.revision = "wrong"
    if failure == "width":
        encoder.width = 512
    with pytest.raises(ValueError):
        module().encode_merged_targets(doc, old, encoder, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*/complete.json"))


def test_new_source_with_same_encoder_has_distinct_cache(tmp_path):
    doc, old = prepared(tmp_path)
    first = module().encode_merged_targets(doc, old, Encoder(), tmp_path / "out")
    changed = deepcopy(doc)
    changed["records"][0]["targets"][4]["text"] = "changed articulator"
    second = module().encode_merged_targets(changed, old, Encoder(), tmp_path / "out")
    assert first.parent != second.parent
    assert digest(doc) != digest(changed)


def test_offline_pinned_loader_is_cpu_frozen(monkeypatch):
    import huggingface_hub
    import torch

    cache = Path("/home/kan/datasets/spamo/hf_cache")
    if not (cache / f"models--openai--clip-vit-large-patch14/snapshots/{REVISION}").is_dir():
        pytest.skip("pinned offline snapshot unavailable on this host")

    def no_remote(*args, **kwargs):
        raise AssertionError("remote revision resolution prohibited")

    monkeypatch.setattr(huggingface_hub, "model_info", no_remote)
    encoder = module().load_pinned_encoder(cache)
    assert encoder.revision == REVISION and encoder.width == 768
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad
               for p in encoder.model.parameters())


def test_non_cpu_encoder_cannot_publish_cpu_identity(tmp_path):
    import torch

    doc, old = prepared(tmp_path)
    encoder = Encoder()
    encoder.model = torch.nn.Linear(1, 1, device="meta")
    with pytest.raises(ValueError, match="CPU"):
        module().encode_merged_targets(doc, old, encoder, tmp_path / "out")


def test_producer_source_is_sealed_with_cache(tmp_path):
    doc, old = prepared(tmp_path)
    path = module().encode_merged_targets(doc, old, Encoder(), tmp_path / "out")
    source = path.parent / "producer/merged_targets.py"
    assert source.is_file()
    source.write_text("changed implementation")
    with pytest.raises(ValueError):
        module().verify_merged_manifest(path)


def test_cached_version_verifies_after_consumer_producer_changes(tmp_path, monkeypatch):
    doc, old = prepared(tmp_path)
    path = module().encode_merged_targets(doc, old, Encoder(), tmp_path / "out")
    updated = deepcopy(module().ENCODING["producer_sha256"])
    updated["merged_targets.py"] = "f"*64
    monkeypatch.setitem(module().ENCODING, "producer_sha256", updated)
    assert module().verify_merged_manifest(path)["records"]


def test_fabricated_human_review_rejected_even_with_rehashed_receipt(tmp_path):
    doc, old = prepared(tmp_path)
    path = module().encode_merged_targets(doc, old, Encoder(), tmp_path / "out")
    data = json.loads(path.read_text())
    data["human_review_status"] = "assessed"
    path.write_text(json.dumps(data))
    complete = path.parent / "complete.json"
    receipt = json.loads(complete.read_text())
    receipt["manifest_sha256"] = file_hash(path)
    complete.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        module().verify_merged_manifest(path)


def test_parent_mutation_during_snapshot_has_no_completion(tmp_path, monkeypatch):
    doc, old = prepared(tmp_path)
    atomic = module()._atomic

    def mutate_after_copy(path, write):
        atomic(path, write)
        if path.name == "crop_export.json":
            Path(doc["parents"]["crop_export"]["path"]).write_text("changed during copy")

    monkeypatch.setattr(module(), "_atomic", mutate_after_copy)
    with pytest.raises(ValueError, match="during snapshot"):
        module().encode_merged_targets(doc, old, Encoder(), tmp_path / "out")
    assert not list((tmp_path / "out").glob("*/complete.json"))


def test_valid_hash_wrong_old_array_shape_rejected(tmp_path):
    doc, old = prepared(tmp_path, old_width=512)
    with pytest.raises(ValueError, match="shape"):
        module().encode_merged_targets(doc, old, Encoder(), tmp_path / "out")


def test_real_text_encoder_overflow_has_no_completion(tmp_path):
    from types import SimpleNamespace

    import torch

    from despamo.appearance.text_features import TextEncoder

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.ones(len(texts), 78, dtype=torch.long)}

    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(projection_dim=768, max_position_embeddings=77)
    encoder = TextEncoder(Tokenizer(), model, REVISION)
    doc, old = prepared(tmp_path)
    with pytest.raises(ValueError, match="token overflow"):
        module().encode_merged_targets(doc, old, encoder, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*/complete.json"))
