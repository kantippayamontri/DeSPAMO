"""Native single-writer registry; B: payloads are revalidated on each admission."""

import json
import subprocess
from pathlib import Path

from tools.checkpoint_storage.inventory import ArtifactBundle, file_hash, no_symlinks
from tools.checkpoint_storage.publish import verify_publication
from tools.dinov3.storage import atomic_json, writer

RESERVE = 50 * 1024**3


def backing_free(drive: str) -> int:
    if drive not in {"C:", "B:"}:
        raise ValueError("Unknown backing drive")
    command = f"(Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='{drive}'\").FreeSpace"
    result = subprocess.run(
        [
            "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def require_capacity(control_root: Path, payload_root: Path, c_demand: int, b_demand: int) -> dict:
    for demand in (c_demand, b_demand):
        if type(demand) is not int or demand < 0:
            raise ValueError("Nonnegative byte projection required")
    no_symlinks(control_root)
    no_symlinks(payload_root)
    c_free, b_free = backing_free("C:"), backing_free("B:")
    if c_free < RESERVE + c_demand:
        raise ValueError("C: below metadata demand plus50GiB reserve")
    if b_free < RESERVE + b_demand:
        raise ValueError("B: below payload/staging demand plus50GiB reserve")
    return {
        "C_free_bytes": c_free,
        "B_free_bytes": b_free,
        "C_projected_bytes": c_demand,
        "B_projected_bytes": b_demand,
        "reserve_bytes": RESERVE,
    }


def register_verified(control_root: Path, bundle: ArtifactBundle, publication: dict) -> dict:
    if publication.get("artifact_id") != bundle.artifact_id:
        raise ValueError("Wrong artifact publication")
    verify_publication(publication)
    with writer(control_root / "authority"):
        previous, lineage = _read_head(control_root)
        entry_path = control_root / "entries" / f"{bundle.artifact_id}.json"
        if entry_path.exists():
            existing = _read_committed(control_root, bundle.artifact_id, lineage)
            if existing["publication"] != publication:
                raise ValueError("Registered publication differs")
            return existing
        head = control_root / "head.json"
        sequence = previous["sequence"] + 1 if previous else 1
        entry = {
            "schema_version": 1,
            "artifact_id": bundle.artifact_id,
            "logical_path": str(bundle.logical_path),
            "publication": publication,
            "sequence": sequence,
            "previous_head": previous,
        }
        intent = control_root / "journal" / f"{sequence:08d}.json"
        if intent.exists():
            raise ValueError("Orphan registry intention requires reviewed reconciliation")
        atomic_json(intent, entry)
        atomic_json(entry_path, entry)
        commit = {
            "entry_sha256": file_hash(entry_path),
            "intent_sha256": file_hash(intent),
            "sequence": sequence,
            "artifact_id": bundle.artifact_id,
        }
        atomic_json(entry_path.with_suffix(".commit.json"), commit)
        atomic_json(head, commit)
        return entry


def _registry_row(control_root: Path, artifact_id: str) -> tuple[dict, dict]:
    if len(artifact_id) != 64 or any(c not in "0123456789abcdef" for c in artifact_id):
        raise ValueError("Canonical artifact digest required")
    path = control_root / "entries" / f"{artifact_id}.json"
    no_symlinks(path)
    marker = path.with_suffix(".commit.json")
    no_symlinks(marker)
    if not path.is_file() or not marker.is_file():
        raise ValueError("Committed registry entry required")
    commit = json.loads(marker.read_text())
    if file_hash(path) != commit["entry_sha256"]:
        raise ValueError("Registry entry hash drift")
    value = json.loads(path.read_text())
    if (
        value.get("artifact_id") != artifact_id
        or commit.get("artifact_id") != artifact_id
        or type(value.get("sequence")) is not int
        or value["sequence"] < 1
        or commit.get("sequence") != value["sequence"]
    ):
        raise ValueError("Registry artifact/sequence drift")
    intent = control_root / "journal" / f"{value['sequence']:08d}.json"
    no_symlinks(intent)
    if file_hash(intent) != commit["intent_sha256"] or json.loads(intent.read_text()) != value:
        raise ValueError("Registry intention drift")
    return value, commit


def _read_head(control_root: Path) -> tuple[dict | None, dict[str, dict]]:
    """Validate native commit lineage while holding the registry writer lease."""
    head_path = control_root / "head.json"
    no_symlinks(head_path)
    if not head_path.exists():
        if any((control_root / "entries").glob("*.json")) or any(
            (control_root / "journal").glob("*.json")
        ):
            raise ValueError("Orphan registry without committed head requires review")
        return None, {}
    head = json.loads(head_path.read_text())
    if type(head.get("sequence")) is not int or head["sequence"] < 1:
        raise ValueError("Invalid registry head sequence")
    if (control_root / "journal" / f"{head['sequence'] + 1:08d}.json").exists():
        raise ValueError("Orphan registry intention requires reviewed reconciliation")
    lineage = {}
    current = head
    while current:
        row, commit = _registry_row(control_root, current["artifact_id"])
        if current != commit:
            raise ValueError("Registry head/commit proof drift")
        lineage[row["artifact_id"]] = commit
        previous = row.get("previous_head")
        if row["sequence"] == 1:
            if previous is not None:
                raise ValueError("First registry entry has foreign previous head")
        elif (
            not isinstance(previous, dict)
            or type(previous.get("sequence")) is not int
            or previous["sequence"] != row["sequence"] - 1
        ):
            raise ValueError("Registry previous-head sequence drift")
        current = previous
    return head, lineage


def _read_committed(control_root: Path, artifact_id: str, lineage: dict[str, dict]) -> dict:
    value, commit = _registry_row(control_root, artifact_id)
    if lineage.get(artifact_id) != commit:
        raise ValueError("Registry entry outside committed head lineage")
    verify_publication(value["publication"])
    return value


def read_committed(control_root: Path, artifact_id: str) -> dict:
    with writer(control_root / "authority"):
        _, lineage = _read_head(control_root)
        return _read_committed(control_root, artifact_id, lineage)
