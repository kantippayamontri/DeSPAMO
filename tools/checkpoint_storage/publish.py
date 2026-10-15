"""B: file-flush/rehashed publication with truthful directory durability."""

import errno
import json
import os
import shutil
import tempfile
from pathlib import Path

from tools.checkpoint_storage.inventory import (
    STREAM_BYTES,
    ArtifactBundle,
    file_hash,
    no_symlinks,
    stat_signature,
)
from tools.dinov3.storage import writer


def directory_fsync(path: Path) -> bool:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            os.fsync(fd)
            return True
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EACCES}:
                raise
            return False
    finally:
        os.close(fd)


def flush_file(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def replace_file(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def copy_file(source: Path, destination: Path, expected_sha: str, expected_bytes: int) -> bool:
    no_symlinks(source)
    no_symlinks(destination)
    if destination.exists():
        if (
            not destination.is_file()
            or destination.stat().st_size != expected_bytes
            or file_hash(destination) != expected_sha
        ):
            raise ValueError("Existing destination bytes/hash differ")
        return directory_fsync(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".partial-", delete=False
    ) as out:
        temp = Path(out.name)
        with source.open("rb", buffering=0) as src:
            shutil.copyfileobj(src, out, STREAM_BYTES)
        flush_file(out)
    if temp.stat().st_size != expected_bytes or file_hash(temp) != expected_sha:
        raise ValueError(f"Temporary copy hash/bytes mismatch; retained {temp}")
    # Native operation lease serializes publishers; never replace foreign final bytes.
    if destination.exists():
        raise ValueError("Destination appeared during publication")
    replace_file(temp, destination)
    supported = directory_fsync(destination.parent)
    if file_hash(destination) != expected_sha:
        raise ValueError("Final destination hash mismatch")
    return supported


def publish_manifest(path: Path, value: dict) -> bool:
    no_symlinks(path)
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("Existing bundle manifest differs")
        return directory_fsync(path.parent)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".partial-manifest-", delete=False
    ) as out:
        temp = Path(out.name)
        out.write(encoded)
        flush_file(out)
    replace_file(temp, path)
    return directory_fsync(path.parent)


def verify_publication(value: dict) -> None:
    if value.get("status") != "verified" or not isinstance(value.get("bundle"), dict):
        raise ValueError("Complete bundle publication required")
    manifest = Path(value["manifest"])
    no_symlinks(manifest)
    if not manifest.is_file() or file_hash(manifest) != value["manifest_sha256"]:
        raise ValueError("Bundle marker missing/hash drift")
    data = json.loads(manifest.read_text())
    if data["artifact_id"] != value["artifact_id"] or data["bundle"] != value["bundle"]:
        raise ValueError("Bundle marker identity drift")
    for member in value["bundle"]["members"]:
        path = Path(member["destination_path"])
        no_symlinks(path)
        if (
            not path.is_file()
            or path.stat().st_size != member["bytes"]
            or file_hash(path) != member["sha256"]
        ):
            raise ValueError("Published member missing/hash/bytes drift")
    payload = next(m for m in value["bundle"]["members"] if m["role"] in {"payload", ".pt"})
    receipt = next(
        (m for m in value["bundle"]["members"] if m["role"] in {".json", ".appended.json"}), None
    )
    if receipt:
        original = json.loads(Path(receipt["destination_path"]).read_text())
        if "sha256" in original and original["sha256"] != payload["sha256"]:
            raise ValueError("Original receipt payload hash mismatch")
        if "bytes" in original and original["bytes"] != payload["bytes"]:
            raise ValueError("Original receipt payload bytes mismatch")


def publish_bundle(bundle: ArtifactBundle, control_root: Path) -> dict:
    with writer(control_root / "leases" / bundle.artifact_id):
        manifest = bundle.members[0].destination_path.parent / f".bundle-{bundle.artifact_id}.json"
        if manifest.exists():
            data = json.loads(manifest.read_text())
            result = {**data, "manifest": str(manifest), "manifest_sha256": file_hash(manifest)}
            if data["bundle"] != bundle.to_dict():
                raise ValueError("Existing bundle differs from source inventory")
            verify_publication(result)
            return result
        supported = True
        for member in bundle.members:
            if (
                stat_signature(member.source_path) != member.source_stat
                or file_hash(member.source_path) != member.sha256
            ):
                raise ValueError("Source changed before copy")
            supported = (
                copy_file(member.source_path, member.destination_path, member.sha256, member.bytes)
                and supported
            )
        if any(stat_signature(m.source_path) != m.source_stat for m in bundle.members):
            raise ValueError("Source changed during bundle copy")
        data = {
            "schema_version": 1,
            "status": "verified",
            "artifact_id": bundle.artifact_id,
            "bundle": bundle.to_dict(),
            "directory_fsync_supported": supported,
            "durability": "file-fsync-rehashed-atomic-rename",
        }
        publish_manifest(manifest, data)
        result = {**data, "manifest": str(manifest), "manifest_sha256": file_hash(manifest)}
        verify_publication(result)
        return result
