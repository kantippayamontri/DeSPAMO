"""Explicit inventory/copy/verify/native-cleanup/restore commands."""

import argparse
import json
from pathlib import Path

from tools.checkpoint_storage.handoff import request_handoff, seal_handoff
from tools.checkpoint_storage.inventory import (
    ArtifactBundle,
    discover_bundles,
    no_symlinks,
    validate_destination,
)
from tools.checkpoint_storage.migration import migrate_inactive, restore_bundle
from tools.checkpoint_storage.registry import read_committed
from tools.dinov3.storage import atomic_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "inventory",
            "copy",
            "verify",
            "remove-native",
            "restore",
            "request-handoff",
            "seal-handoff",
        ],
    )
    parser.add_argument("--source-base", type=Path, default=Path("/home/kan/datasets/despamo"))
    parser.add_argument("--payload-root", type=Path, default=Path("/mnt/b/DeSpaMo/checkpoints-v1"))
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("/home/kan/datasets/despamo/checkpoint-storage-v1"),
    )
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--artifact-id")
    parser.add_argument("--authorize-remove-native", action="store_true")
    parser.add_argument("--authorize-restore", action="store_true")
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--authorize-handoff", action="store_true")
    args = parser.parse_args(argv)
    if args.mode in {"request-handoff", "seal-handoff"}:
        if not args.source_config:
            parser.error("Source runtime configuration required")
        result = (
            request_handoff(
                args.source_config, args.control_root, authorize_handoff=args.authorize_handoff
            )
            if args.mode == "request-handoff"
            else seal_handoff(args.source_config, args.control_root)
        )
        print(json.dumps(result))
        return
    if args.mode == "inventory":
        rows = discover_bundles(args.source_base, args.payload_root)
        data = {
            "schema_version": 1,
            "source_base": str(args.source_base),
            "payload_root": str(args.payload_root),
            "bundles": [r.to_dict() for r in rows],
        }
        if args.inventory:
            atomic_json(args.inventory, data)
        print(
            json.dumps(
                {
                    "bundles": len(rows),
                    "active": sum(r.active for r in rows),
                    "bytes": sum(m.bytes for r in rows for m in r.members),
                }
            )
        )
        return
    if args.mode == "restore":
        print(
            json.dumps(
                restore_bundle(
                    args.artifact_id, args.control_root, authorize_restore=args.authorize_restore
                )
            )
        )
        return
    if args.mode == "verify":
        print(json.dumps(read_committed(args.control_root, args.artifact_id)))
        return
    if not args.inventory:
        parser.error("Approved inventory required")
    if args.mode == "remove-native" and not args.authorize_remove_native:
        parser.error("Explicit native cleanup authorization required")
    rows = [ArtifactBundle.from_dict(r) for r in json.loads(args.inventory.read_text())["bundles"]]
    no_symlinks(args.source_base)
    for row in rows:
        validate_destination(row, args.payload_root)
        for member in row.members:
            if (
                args.source_base not in member.source_path.parents
                or args.source_base not in member.logical_path.parents
                or ".." in member.source_path.parts
                or ".." in member.logical_path.parts
                or args.mode == "remove-native"
                and member.source_path != member.logical_path
            ):
                raise ValueError("Source/logical path outside admitted source root")
    for row in rows:
        if args.artifact_id and row.artifact_id != args.artifact_id:
            continue
        if row.active:
            print(json.dumps({"status": "blocked_active", "artifact_id": row.artifact_id}))
            continue
        print(
            json.dumps(
                migrate_inactive(
                    row, args.control_root, authorize_remove=args.mode == "remove-native"
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
