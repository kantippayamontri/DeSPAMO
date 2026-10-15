import json
from dataclasses import replace

import pytest

from tools.checkpoint_storage import cli, migration
from tools.checkpoint_storage.tests.test_publish import bundle


def test_cli_inventory_is_read_only(tmp_path):
    b, target = bundle(tmp_path)
    original = {m.source_path: m.source_path.read_bytes() for m in b.members}
    inventory = tmp_path / "inventory.json"
    cli.main(
        [
            "inventory",
            "--source-base",
            str(tmp_path / "source"),
            "--payload-root",
            str(target),
            "--control-root",
            str(tmp_path / "control"),
            "--inventory",
            str(inventory),
        ]
    )
    assert len(json.loads(inventory.read_text())["bundles"]) == 1
    assert not target.exists()
    assert all(path.read_bytes() == data for path, data in original.items())


def test_cli_remove_without_authorization_rejects_before_copy(tmp_path):
    b, target = bundle(tmp_path)
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "remove-native",
                "--inventory",
                str(tmp_path / "missing-inventory.json"),
                "--source-base",
                str(tmp_path / "source"),
                "--payload-root",
                str(target),
                "--control-root",
                str(tmp_path / "control"),
            ]
        )
    assert caught.value.code == 2
    assert not target.exists()
    assert all(m.source_path.exists() for m in b.members)


@pytest.mark.parametrize("foreign_root", ["payload", "source"])
def test_cleanup_cli_enforces_explicit_source_and_payload_roots(
    tmp_path, monkeypatch, foreign_root
):
    b, target = bundle(tmp_path)
    source = tmp_path / "source"
    original_paths = [m.source_path for m in b.members]
    if foreign_root == "payload":
        b = replace(
            b,
            members=(
                replace(b.members[0], destination_path=tmp_path / "outside/payload.pt"),
                *b.members[1:],
            ),
        )
    else:
        source = tmp_path / "different-source-base"
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"bundles": [b.to_dict()]}))
    monkeypatch.setattr(migration, "require_capacity", lambda *args: {})
    with pytest.raises(ValueError, match="root|Source"):
        cli.main(
            [
                "remove-native",
                "--authorize-remove-native",
                "--inventory",
                str(inventory),
                "--source-base",
                str(source),
                "--payload-root",
                str(target),
                "--control-root",
                str(tmp_path / "control"),
            ]
        )
    assert all(path.exists() for path in original_paths)
