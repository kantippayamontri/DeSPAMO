import json

import pytest

from tools.checkpoint_storage.tests.test_publish import bundle


def test_register_requires_complete_publication_and_rehash_on_restart(tmp_path):
    from tools.checkpoint_storage.publish import publish_bundle
    from tools.checkpoint_storage.registry import read_committed, register_verified

    b, _ = bundle(tmp_path)
    control = tmp_path / "control"
    published = publish_bundle(b, control)
    entry = register_verified(control, b, published)
    assert read_committed(control, b.artifact_id) == entry
    b.members[0].destination_path.unlink()
    with pytest.raises(ValueError, match="missing|hash|bytes"):
        read_committed(control, b.artifact_id)


def test_false_marker_and_wrong_artifact_cannot_register(tmp_path):
    from tools.checkpoint_storage.registry import register_verified

    b, _ = bundle(tmp_path)
    with pytest.raises(ValueError):
        register_verified(tmp_path / "control", b, {"status": "verified", "artifact_id": "wrong"})


def test_capacity_is_volume_specific(tmp_path, monkeypatch):
    from tools.checkpoint_storage import registry as r

    reserve = 50 * 1024**3
    monkeypatch.setattr(r, "backing_free", lambda drive: reserve + (100 if drive == "C:" else 1000))
    assert r.require_capacity(tmp_path, tmp_path / "B", 99, 999)["C_free_bytes"] == reserve + 100
    with pytest.raises(ValueError, match="C:"):
        r.require_capacity(tmp_path, tmp_path / "B", 101, 1)
    with pytest.raises(ValueError, match="B:"):
        r.require_capacity(tmp_path, tmp_path / "B", 1, 1001)


@pytest.mark.parametrize("fault", ["missing_head", "head_hash", "head_sequence", "orphan_intent"])
def test_registry_authority_fault_blocks_checkpoint_admission(tmp_path, fault):
    from tools.checkpoint_storage.publish import publish_bundle
    from tools.checkpoint_storage.registry import read_committed, register_verified
    from tools.dinov3.storage import atomic_json

    b, _ = bundle(tmp_path)
    control = tmp_path / "control"
    register_verified(control, b, publish_bundle(b, control))
    head = control / "head.json"
    if fault == "missing_head":
        head.unlink()
    elif fault == "orphan_intent":
        atomic_json(control / "journal/00000002.json", {"interrupted": True})
    else:
        value = json.loads(head.read_text())
        value["entry_sha256" if fault == "head_hash" else "sequence"] = (
            "0" * 64 if fault == "head_hash" else 999
        )
        atomic_json(head, value)
    with pytest.raises(ValueError):
        read_committed(control, b.artifact_id)
    assert all(member.source_path.exists() for member in b.members)


def test_commit_marker_without_published_head_requires_review(tmp_path, monkeypatch):
    from tools.checkpoint_storage import registry
    from tools.checkpoint_storage.publish import publish_bundle

    b, _ = bundle(tmp_path)
    control = tmp_path / "control"
    publication = publish_bundle(b, control)
    real_write = registry.atomic_json

    def interrupted_head(path, value):
        if path.name == "head.json":
            raise OSError("injected head publication failure")
        return real_write(path, value)

    with monkeypatch.context() as injection:
        injection.setattr(registry, "atomic_json", interrupted_head)
        with pytest.raises(OSError, match="injected"):
            registry.register_verified(control, b, publication)
    with pytest.raises(ValueError):
        registry.read_committed(control, b.artifact_id)
    with pytest.raises(ValueError):
        registry.register_verified(control, b, publication)
    assert all(member.source_path.exists() for member in b.members)


def test_valid_latest_head_keeps_older_committed_bundle_admissible(tmp_path):
    from tools.checkpoint_storage.publish import publish_bundle
    from tools.checkpoint_storage.registry import read_committed, register_verified

    first, _ = bundle(tmp_path / "first")
    second, _ = bundle(tmp_path / "second")
    control = tmp_path / "control"
    one = register_verified(control, first, publish_bundle(first, control))
    two = register_verified(control, second, publish_bundle(second, control))
    assert one["sequence"] == 1 and two["sequence"] == 2
    assert read_committed(control, first.artifact_id) == one
    assert read_committed(control, second.artifact_id) == two
