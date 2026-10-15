"""Explicitly approved native cleanup only after revalidated B: registration."""

import fcntl
from contextlib import ExitStack, contextmanager
from pathlib import Path

from tools.checkpoint_storage.inventory import (
    ArtifactBundle,
    activity,
    file_hash,
    no_symlinks,
    owner_active,
    stat_signature,
)
from tools.checkpoint_storage.publish import copy_file, publish_bundle
from tools.checkpoint_storage.registry import read_committed, register_verified, require_capacity
from tools.dinov3.storage import atomic_json, writer


def _family_root(bundle: ArtifactBundle) -> Path:
    if bundle.family == "continuation":
        for parent in bundle.logical_path.parents:
            if parent.name in {"checkpoints", "selected"}:
                return parent.parent
        raise ValueError("Generated bundle has no admitted runtime namespace")
    return next(p for p in bundle.logical_path.parents if p.name == bundle.family)


def _inactive(bundle: ArtifactBundle) -> None:
    if activity(bundle.logical_path, _family_root(bundle), owner_active):
        raise RuntimeError("Checkpoint family has active owner")
    for parent in (bundle.logical_path.parent, *bundle.logical_path.parent.parents):
        if parent == _family_root(bundle).parent:
            break
        if any(parent.glob("*.partial")) or any(parent.glob(".partial-*")):
            raise RuntimeError("Checkpoint publication temporary file present")


def _source_unchanged(bundle: ArtifactBundle) -> None:
    for member in bundle.members:
        if member.source_path.exists():
            no_symlinks(member.source_path)
            if (
                stat_signature(member.source_path) != member.source_stat
                or file_hash(member.source_path) != member.sha256
            ):
                raise ValueError("Source changed since approved inventory")


@contextmanager
def native_leases(bundle):
    paths = set()
    family = _family_root(bundle)
    for parent in (bundle.logical_path.parent, *bundle.logical_path.parent.parents):
        if parent == family.parent:
            break
        paths.update(parent.glob("*.lock"))
    with ExitStack() as stack:
        for path in sorted(paths):
            no_symlinks(path)
            handle = stack.enter_context(path.open("rb"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Existing native writer lock held: {path}") from error
        yield


def migrate_inactive(bundle: ArtifactBundle, control_root: Path, *, authorize_remove: bool) -> dict:
    with native_leases(bundle):
        return _migrate_inactive(bundle, control_root, authorize_remove=authorize_remove)


def _migrate_inactive(
    bundle: ArtifactBundle, control_root: Path, *, authorize_remove: bool
) -> dict:
    _inactive(bundle)
    _source_unchanged(bundle)
    destination = bundle.members[0].destination_path
    require_capacity(
        control_root,
        destination.parent,
        1024**2,
        sum(m.bytes for m in bundle.members) + max(m.bytes for m in bundle.members),
    )
    entry_path = control_root / "entries" / f"{bundle.artifact_id}.json"
    if entry_path.exists():
        entry = read_committed(control_root, bundle.artifact_id)
        if entry["publication"]["bundle"] != bundle.to_dict():
            raise ValueError("Registered inventory differs")
    else:
        publication = publish_bundle(bundle, control_root)
        entry = register_verified(control_root, bundle, publication)
    result = {"artifact_id": bundle.artifact_id, "status": "registered", "removed": []}
    if not authorize_remove:
        return result
    with writer(control_root / "leases" / bundle.artifact_id):
        read_committed(control_root, bundle.artifact_id)
        _inactive(bundle)
        _source_unchanged(bundle)
        operation = control_root / "operations" / f"{bundle.artifact_id}.json"
        atomic_json(
            operation, {**result, "status": "removing", "entry_sequence": entry["sequence"]}
        )
        for member in bundle.members:
            if member.source_path.exists():
                if file_hash(member.destination_path) != member.sha256:
                    raise ValueError("B: bytes changed before native removal")
                no_symlinks(member.source_path)
                if (
                    stat_signature(member.source_path) != member.source_stat
                    or file_hash(member.source_path) != member.sha256
                ):
                    raise ValueError("Source changed during final B: verification")
                member.source_path.unlink()
            result["removed"].append(str(member.source_path))
            atomic_json(operation, {**result, "status": "removing"})
        result["status"] = "native_removed"
        atomic_json(operation, result)
    return result


def restore_bundle(artifact_id: str, control_root: Path, *, authorize_restore: bool) -> dict:
    if not authorize_restore:
        raise RuntimeError("Explicit restore authorization required")
    entry = read_committed(control_root, artifact_id)
    bundle = ArtifactBundle.from_dict(entry["publication"]["bundle"])
    with native_leases(bundle), writer(control_root / "leases" / artifact_id):
        _inactive(bundle)
        require_capacity(
            control_root,
            bundle.members[0].destination_path.parent,
            sum(m.bytes for m in bundle.members) + max(m.bytes for m in bundle.members),
            1024**2,
        )
        read_committed(control_root, artifact_id)
        _inactive(bundle)
        for member in bundle.members:
            copy_file(member.destination_path, member.logical_path, member.sha256, member.bytes)
        result = {
            "status": "restored_verified",
            "artifact_id": artifact_id,
            "restored": [str(m.logical_path) for m in bundle.members],
        }
        atomic_json(control_root / "restores" / f"{artifact_id}.json", result)
        return result
