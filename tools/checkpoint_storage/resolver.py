"""Logical path relocation without rewriting original checkpoint/report bytes."""

import hashlib
import json
from pathlib import Path

from tools.checkpoint_storage.registry import read_committed


def resolve_checkpoint(logical_path: Path, control_root: Path) -> Path:
    logical = str(logical_path.absolute())
    aid = hashlib.sha256(logical.encode()).hexdigest()
    candidates = [control_root / "entries" / f"{aid}.json"]
    if not candidates[0].exists():
        candidates = sorted((control_root / "entries").glob("*.json"))
    for path in candidates:
        if path.name.endswith(".commit.json") or not path.exists():
            continue
        raw = json.loads(path.read_text())
        members = raw["publication"]["bundle"]["members"]
        if any(member["logical_path"] == logical for member in members):
            entry = read_committed(control_root, raw["artifact_id"])
            found = next(
                m for m in entry["publication"]["bundle"]["members"] if m["logical_path"] == logical
            )
            return Path(found["destination_path"])
    raise ValueError(f"No committed relocation for {logical_path}")
