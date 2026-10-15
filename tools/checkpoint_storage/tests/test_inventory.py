import hashlib
import json

import pytest


def checkpoint(root, name, *, receipt=True, sidecars=False):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint-payload")
    if receipt:
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                    "identity": {"seed": 0, "arm": "reference"},
                    "step": 4,
                }
            )
        )
    if sidecars:
        path.with_suffix(".runtime.json").write_text('{"binding_hash":"old"}')
        path.with_suffix(".guard.json").write_text('{"last_update":4,"ce_values":[]}')
    return path


def test_inventory_classifies_compact_encoder_and_legacy(tmp_path):
    from tools.checkpoint_storage.inventory import discover_bundles

    source, target = tmp_path / "source", tmp_path / "B"
    checkpoint(source / "signer-pilot/run", "resume-step-4.pt", sidecars=True)
    checkpoint(source / "official-phoenix14t-dino-adaptation-v1/exports", "epoch-40.pt")
    checkpoint(source / "comparison-2026-09-26", "legacy.ckpt", receipt=False)
    rows = discover_bundles(source, target)
    assert len(rows) == 3
    by_name = {row.logical_path.name: row for row in rows}
    assert len(by_name["resume-step-4.pt"].members) == 4
    assert by_name["resume-step-4.pt"].runnable
    assert not by_name["legacy.ckpt"].runnable
    assert by_name["legacy.ckpt"].layout == "legacy_unreceipted"


def test_inventory_excludes_features_caches_links_and_marks_active(tmp_path):
    from tools.checkpoint_storage.inventory import discover_bundles

    source, target = tmp_path / "source", tmp_path / "B"
    checkpoint(source / "signer-pilot/features", "features.pt")
    checkpoint(source / "signer-pilot/hf_cache", "model.safetensors")
    p = checkpoint(source / "signer-pilot/run", "selected.pt")
    (p.parent / "budget.json").write_text(
        json.dumps(
            {
                "status": "running",
                "owner": {"pid": 12, "boot_id": "current", "start_ticks": "9"},
            }
        )
    )
    checkpoint(source / "outside", "checkpoint.pt")
    rows = discover_bundles(source, target, owner_check=lambda owner: owner["pid"] == 12)
    assert len(rows) == 1 and rows[0].active
    assert rows[0].logical_path == p


@pytest.mark.parametrize("name", ["CON.pt", "foo:bar.pt", "bad?.pt"])
def test_destination_rejects_windows_reserved_names(tmp_path, name):
    from tools.checkpoint_storage.inventory import discover_bundles

    checkpoint(tmp_path / "source/signer-pilot", name)
    with pytest.raises(ValueError, match="Windows|destination"):
        discover_bundles(tmp_path / "source", tmp_path / "B")


def test_case_collision_and_destination_symlink_rejected(tmp_path):
    from tools.checkpoint_storage.inventory import discover_bundles

    source = tmp_path / "source"
    checkpoint(source / "signer-pilot", "Foo.pt")
    checkpoint(source / "signer-pilot", "foo.pt")
    with pytest.raises(ValueError, match="collision"):
        discover_bundles(source, tmp_path / "B")
    (source / "signer-pilot/foo.pt").unlink()
    target = tmp_path / "B"
    target.mkdir()
    (target / "signer-pilot").symlink_to(source / "signer-pilot", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        discover_bundles(source, target)
