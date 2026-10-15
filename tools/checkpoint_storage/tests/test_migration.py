import pytest

from tools.checkpoint_storage.inventory import file_hash
from tools.checkpoint_storage.tests.test_publish import bundle


def test_registered_bundle_moves_and_resolves_original_paths(tmp_path, monkeypatch):
    from tools.checkpoint_storage import migration as m
    from tools.checkpoint_storage.resolver import resolve_checkpoint

    b, _ = bundle(tmp_path)
    monkeypatch.setattr(m, "require_capacity", lambda *a: {})
    original = {x.logical_path: x.source_path.read_bytes() for x in b.members}
    result = m.migrate_inactive(b, tmp_path / "control", authorize_remove=True)
    assert result["status"] == "native_removed"
    for logical, data in original.items():
        assert not logical.exists()
        assert resolve_checkpoint(logical, tmp_path / "control").read_bytes() == data


def test_copy_only_and_changed_source_never_deleted(tmp_path, monkeypatch):
    from tools.checkpoint_storage import migration as m

    b, _ = bundle(tmp_path)
    monkeypatch.setattr(m, "require_capacity", lambda *a: {})
    assert (
        m.migrate_inactive(b, tmp_path / "control", authorize_remove=False)["status"]
        == "registered"
    )
    assert all(x.source_path.exists() for x in b.members)
    b.members[0].source_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        m.migrate_inactive(b, tmp_path / "control", authorize_remove=True)
    assert b.members[0].source_path.exists()


def test_live_owner_and_marker_prevent_removal(tmp_path, monkeypatch):
    from tools.checkpoint_storage import migration as m

    b, _ = bundle(tmp_path)
    monkeypatch.setattr(m, "require_capacity", lambda *a: {})
    monkeypatch.setattr(m, "owner_active", lambda owner: True)
    (b.logical_path.parent / "budget.json").write_text('{"owner":{"pid":123},"status":"running"}')
    with pytest.raises(RuntimeError, match="active"):
        m.migrate_inactive(b, tmp_path / "control", authorize_remove=True)
    assert all(x.source_path.exists() for x in b.members)


def test_partial_cleanup_replay_and_byte_exact_restore(tmp_path, monkeypatch):
    from tools.checkpoint_storage import migration as m
    from tools.checkpoint_storage.resolver import resolve_checkpoint

    b, _ = bundle(tmp_path)
    monkeypatch.setattr(m, "require_capacity", lambda *a: {})
    m.migrate_inactive(b, tmp_path / "control", authorize_remove=False)
    b.members[0].source_path.unlink()
    m.migrate_inactive(b, tmp_path / "control", authorize_remove=True)
    assert resolve_checkpoint(b.logical_path, tmp_path / "control").exists()
    m.restore_bundle(b.artifact_id, tmp_path / "control", authorize_restore=True)
    assert all(file_hash(x.source_path) == x.sha256 for x in b.members)


def test_resolver_rejects_corrupt_receipt_and_unregistered_path(tmp_path, monkeypatch):
    from tools.checkpoint_storage import migration as m
    from tools.checkpoint_storage.resolver import resolve_checkpoint

    b, _ = bundle(tmp_path)
    monkeypatch.setattr(m, "require_capacity", lambda *a: {})
    m.migrate_inactive(b, tmp_path / "control", authorize_remove=True)
    b.members[1].destination_path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        resolve_checkpoint(b.logical_path, tmp_path / "control")


def test_existing_native_writer_lock_blocks_migration(tmp_path, monkeypatch):
    import fcntl

    from tools.checkpoint_storage import migration as m

    b, _ = bundle(tmp_path)
    monkeypatch.setattr(m, "require_capacity", lambda *a: {})
    lock = b.logical_path.parent / ".writer.lock"
    with lock.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="writer|lock"):
            m.migrate_inactive(b, tmp_path / "control", authorize_remove=True)
        assert b.logical_path.exists()


def test_restore_rejects_held_native_writer_before_recreating_payload(tmp_path, monkeypatch):
    import fcntl

    from tools.checkpoint_storage import migration as m

    b, _ = bundle(tmp_path)
    control = tmp_path / "control"
    monkeypatch.setattr(m, "require_capacity", lambda *a: {})
    m.migrate_inactive(b, control, authorize_remove=True)
    with (b.logical_path.parent / ".writer.lock").open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="writer|lock"):
            m.restore_bundle(b.artifact_id, control, authorize_restore=True)
    assert not b.logical_path.exists()


def test_restore_rechecks_owner_after_capacity_admission(tmp_path, monkeypatch):
    import os

    from tools.checkpoint_storage import migration as m
    from tools.dinov3.official_runtime import process_owner
    from tools.dinov3.storage import atomic_json

    b, _ = bundle(tmp_path)
    control = tmp_path / "control"
    monkeypatch.setattr(m, "require_capacity", lambda *a: {})
    m.migrate_inactive(b, control, authorize_remove=True)

    def owner_appeared(*args):
        atomic_json(b.logical_path.parent / "budget.json", {"owner": process_owner(os.getpid())})
        return {}

    monkeypatch.setattr(m, "require_capacity", owner_appeared)
    with pytest.raises(RuntimeError, match="active"):
        m.restore_bundle(b.artifact_id, control, authorize_restore=True)
    assert not b.logical_path.exists()


def test_source_mutation_during_final_B_rehash_is_retained(tmp_path, monkeypatch):
    from tools.checkpoint_storage import migration as m

    b, _ = bundle(tmp_path)
    control = tmp_path / "control"
    monkeypatch.setattr(m, "require_capacity", lambda *args: {})
    m.migrate_inactive(b, control, authorize_remove=False)
    payload = b.members[0]
    original_hash = m.file_hash

    def source_changed_during_destination_read(path):
        if path == payload.destination_path:
            payload.source_path.write_bytes(b"new source bytes after initial checks")
        return original_hash(path)

    monkeypatch.setattr(m, "file_hash", source_changed_during_destination_read)
    with pytest.raises(ValueError, match="Source changed"):
        m.migrate_inactive(b, control, authorize_remove=True)
    assert payload.source_path.read_bytes() == b"new source bytes after initial checks"
