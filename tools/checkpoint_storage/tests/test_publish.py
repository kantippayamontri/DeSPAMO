import pytest

from tools.checkpoint_storage.inventory import discover_bundles, file_hash
from tools.checkpoint_storage.tests.test_inventory import checkpoint


def bundle(tmp_path):
    source, target = tmp_path / "source", tmp_path / "B"
    checkpoint(source / "signer-pilot/run", "resume-step-4.pt", sidecars=True)
    return discover_bundles(source, target)[0], target


def test_publication_preserves_source_and_verifies_every_member(tmp_path):
    from tools.checkpoint_storage.publish import publish_bundle, verify_publication

    b, _ = bundle(tmp_path)
    result = publish_bundle(b, tmp_path / "control")
    verify_publication(result)
    assert result["status"] == "verified"
    for member in b.members:
        assert member.source_path.exists()
        assert file_hash(member.destination_path) == member.sha256


def test_matching_archive_reused_foreign_bytes_rejected(tmp_path):
    from tools.checkpoint_storage.publish import publish_bundle

    b, _ = bundle(tmp_path)
    first = publish_bundle(b, tmp_path / "control")
    assert publish_bundle(b, tmp_path / "control") == first
    b.members[0].destination_path.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="hash|bytes|differ"):
        publish_bundle(b, tmp_path / "control")
    assert all(m.source_path.exists() for m in b.members)


@pytest.mark.parametrize("point", ["copy", "flush", "rename", "manifest"])
def test_publication_failure_never_removes_source(tmp_path, monkeypatch, point):
    from tools.checkpoint_storage import publish as p

    b, _ = bundle(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(
        p,
        {
            "copy": "copy_file",
            "flush": "flush_file",
            "rename": "replace_file",
            "manifest": "publish_manifest",
        }[point],
        fail,
    )
    with pytest.raises(OSError, match="injected"):
        p.publish_bundle(b, tmp_path / "control")
    assert all(m.source_path.exists() for m in b.members)


def test_unsupported_directory_flush_recorded_unexpected_error_rejected(tmp_path, monkeypatch):
    from tools.checkpoint_storage import publish as p

    b, _ = bundle(tmp_path)
    monkeypatch.setattr(p, "directory_fsync", lambda path: False)
    assert p.publish_bundle(b, tmp_path / "control")["directory_fsync_supported"] is False
    assert all(m.source_path.exists() for m in b.members)


def test_publication_rejects_receipt_payload_hash_drift(tmp_path):
    import json

    from tools.checkpoint_storage.inventory import discover_bundles
    from tools.checkpoint_storage.publish import publish_bundle

    b, target = bundle(tmp_path)
    receipt = b.logical_path.with_suffix(".json")
    value = json.loads(receipt.read_text())
    value["sha256"] = "0" * 64
    receipt.write_text(json.dumps(value))
    b = discover_bundles(tmp_path / "source", target)[0]
    with pytest.raises(ValueError, match="receipt|payload"):
        publish_bundle(b, tmp_path / "control")
    assert b.logical_path.exists()
