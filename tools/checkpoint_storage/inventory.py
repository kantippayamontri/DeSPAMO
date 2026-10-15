"""Read-only checkpoint bundle discovery; original paths remain logical identities."""

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

FAMILIES = (
    "comparison-2026-09-26",
    "comparison-smokes-2026-09-26",
    "official-phoenix14t-dino-adaptation-v1",
    "official-phoenix14t-translation-clip-s2-spamo200-v1",
    "official-phoenix14t-translation-clip-s2-spamo200-v2",
    "official-phoenix14t-translation-clip-s2-spamo200-v3",
    "official-phoenix14t-translation-spamo200-v1",
    "official-phoenix14t-translation-v1",
    "signer-pilot",
    "unreviewed-factor-100step",
    "unreviewed-factor-smoke",
)
SKIP = {
    "hf_cache",
    "internvl3_hf_cache",
    "model_cache",
    "model-cache",
    "frames",
    "features",
    ".venv",
    ".git",
    "__pycache__",
    "node_modules",
}
EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors"}
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)}
# Keep 9p request allocations small during checkpoint streaming and rehashing.
STREAM_BYTES = 16 * 1024


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for chunk in iter(lambda: handle.read(STREAM_BYTES), b""):
            result.update(chunk)
    return result.hexdigest()


def stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_nlink


def no_symlinks(path: Path) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Unsafe symlink path: {path}")


@dataclass(frozen=True)
class ArtifactMember:
    role: str
    logical_path: Path
    source_path: Path
    destination_path: Path
    bytes: int
    sha256: str
    source_stat: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ArtifactBundle:
    artifact_id: str
    family: str
    logical_path: Path
    layout: str
    original_identity_digest: str | None
    members: tuple[ArtifactMember, ...]
    runnable: bool
    active: bool

    def to_dict(self) -> dict:
        result = asdict(self)
        result["logical_path"] = str(self.logical_path)
        for member in result["members"]:
            for key in ("logical_path", "source_path", "destination_path"):
                member[key] = str(member[key])
        return json.loads(json.dumps(result))

    @classmethod
    def from_dict(cls, value: dict) -> "ArtifactBundle":
        v = dict(value)
        v["logical_path"] = Path(v["logical_path"])
        members = []
        for item in v["members"]:
            d = dict(item)
            for key in ("logical_path", "source_path", "destination_path"):
                d[key] = Path(d[key])
            d["source_stat"] = tuple(d["source_stat"])
            members.append(ArtifactMember(**d))
        v["members"] = tuple(members)
        return cls(**v)


def owner_active(owner: dict) -> bool:
    if type(owner.get("pid")) is not int or owner["pid"] <= 0:
        return False
    proc = Path("/proc") / str(owner["pid"])
    if not proc.exists():
        return False
    if not owner.get("boot_id") or not owner.get("start_ticks"):
        return True
    if owner.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        return False
    try:
        text = (proc / "stat").read_text()
        return text[text.rfind(")") + 2 :].split()[19] == str(owner.get("start_ticks"))
    except OSError:
        return False


def activity(path: Path, family_root: Path, check: Callable[[dict], bool]) -> bool:
    for directory in (path.parent, *path.parent.parents):
        if directory != family_root and family_root not in directory.parents:
            break
        for name in ("budget.json", "state.json"):
            candidate = directory / name
            if candidate.is_file() and not candidate.is_symlink():
                value = json.loads(candidate.read_text())
                if check(value.get("owner") or {}):
                    return True
    return False


def validate_destination(bundle: ArtifactBundle, payload_root: Path) -> None:
    no_symlinks(payload_root)
    for member in bundle.members:
        dest = member.destination_path
        if not dest.is_absolute() or payload_root not in dest.parents or ".." in dest.parts:
            raise ValueError("Destination escapes configured root")
        no_symlinks(dest)
        for part in dest.relative_to(payload_root).parts:
            if (
                part.endswith((".", " "))
                or re.search(r'[<>:"/\\|?*\x00-\x1f]', part)
                or part.split(".")[0].upper() in RESERVED
            ):
                raise ValueError(f"Invalid Windows destination component: {part}")


def discover_bundles(
    source_base: Path, payload_root: Path, *, owner_check: Callable[[dict], bool] = owner_active
) -> list[ArtifactBundle]:
    source_base, payload_root = source_base.absolute(), payload_root.absolute()
    no_symlinks(source_base)
    results = []
    occupied = set()
    for family in FAMILIES:
        root = source_base / family
        if not root.is_dir() or root.is_symlink():
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in SKIP
                and not d.startswith(("spatial-", "motion-"))
                and not (Path(current) / d).is_symlink()
            )
            for name in sorted(files):
                path = Path(current) / name
                if path.is_symlink() or (
                    path.suffix not in EXTENSIONS and name not in {"model.bin", "pytorch_model.bin"}
                ):
                    continue
                if re.search(r"(^|[-_])(feature|embedding|target|caption)([-_.]|$)", name, re.I):
                    continue
                parts = [("payload", path)]
                for suffix in (".json", ".runtime.json", ".guard.json"):
                    side = path.with_suffix(suffix)
                    if side.is_file():
                        parts.append((suffix, side))
                appended = Path(str(path) + ".json")
                if appended.is_file() and appended not in [p for _, p in parts]:
                    parts.append((".appended.json", appended))
                members = []
                for role, item in parts:
                    no_symlinks(item)
                    before = stat_signature(item)
                    digest = file_hash(item)
                    if stat_signature(item) != before:
                        raise ValueError("Source changed during inventory")
                    members.append(
                        ArtifactMember(
                            role,
                            item,
                            item,
                            payload_root / item.relative_to(source_base),
                            before[2],
                            digest,
                            before,
                        )
                    )
                receipt = next(
                    (p for role, p in parts if role in {".json", ".appended.json"}), None
                )
                value = json.loads(receipt.read_text()) if receipt else {}
                identity = value.get("identity")
                identity_digest = (
                    hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                    if identity
                    else None
                )
                layout = (
                    "compact"
                    if identity and "step" in value
                    else "receipted"
                    if receipt
                    else "legacy_unreceipted"
                )
                aid = hashlib.sha256(str(path).encode()).hexdigest()
                bundle = ArtifactBundle(
                    aid,
                    family,
                    path,
                    layout,
                    identity_digest,
                    tuple(members),
                    bool(identity),
                    activity(path, root, owner_check),
                )
                validate_destination(bundle, payload_root)
                for member in members:
                    folded = str(member.destination_path).casefold()
                    if folded in occupied:
                        raise ValueError("Windows case-fold destination collision")
                    occupied.add(folded)
                results.append(bundle)
    return sorted(results, key=lambda b: str(b.logical_path))
