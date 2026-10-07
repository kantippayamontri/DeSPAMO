"""PHOENIX14T frame source: the release tar.gz, indexed once, then selectively
extracted with a single GNU-tar member-list read (no full unpack).

The 2026-10-01 design targets PHOENIX 210x260 frames; only the release tarball is
present on this host, so sampling indexes the tar (one streaming pass), verifies
the annotation frame counts, and extracts exactly the five sampled frames per
clip into ``root/frames``.
"""

import re
import subprocess
import tarfile
from pathlib import Path

import cv2
import numpy as np

FRAME_RE = re.compile(r"^(.*)/(images\d{4})\.png$")
EXCLUDED_SIGNERS = ("Signer03", "Signer07")
_SAFE_NAME = re.compile(r"^[\w./+-]+$", re.UNICODE)


def load_clips(annotation_path, excluded=EXCLUDED_SIGNERS) -> list[dict]:
    raw = np.load(annotation_path, allow_pickle=True).item()
    rows = []
    for value in raw.values():
        if not isinstance(value, dict) or "fileid" not in value:
            continue
        if excluded and value.get("signer") in excluded:
            continue
        count = int(value["num_frames"])
        if count < 5:
            raise ValueError(f"{value['fileid']}: fewer than five frames")
        rows.append(dict(clip=value["fileid"], stratum=value["signer"], num_frames=count))
    clips = [row["clip"] for row in rows]
    if len(clips) != len(set(clips)):
        raise ValueError("duplicate clip ids in annotations")
    return rows


def _clip_of(member_name: str):
    match = FRAME_RE.search(member_name)
    if not match or "/fullFrame-210x260px/" not in member_name:
        return None
    return match.group(1).rsplit("/", 1)[-1]


def index_frames(tar_path, wanted: set[str]) -> dict[str, dict[str, str]]:
    """One streaming pass (compression auto-detected): clip -> {serial: member}."""
    frames: dict[str, dict[str, str]] = {}
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive:
            clip = _clip_of(member.name) if member.isfile() else None
            if clip in wanted:
                frames.setdefault(clip, {})[FRAME_RE.match(member.name).group(2)] = member.name
    missing = sorted(wanted - set(frames))
    if missing:
        raise ValueError(f"clips absent from tar: {missing[:5]}")
    return frames


def verify_counts(frames: dict[str, dict[str, str]], counts: dict[str, int]) -> None:
    for clip, expected in counts.items():
        got = len(frames[clip])
        if got != expected:
            raise ValueError(f"{clip}: tar holds {got} frames, annotation says {expected}")


def extract_members(tar_path, member_names: list[str], root) -> None:
    """GNU tar reads the archive once and writes root/{clip}/{serial}.png."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in member_names:
        if not _SAFE_NAME.match(name):
            raise ValueError(f"unsafe member name: {name!r}")
    strip = min(name.count("/") for name in member_names) - 1
    if strip < 0:
        raise ValueError("member names must end with {clip}/{serial}.png")
    listing = root / ".extract-list.txt"
    listing.write_text("\n".join(sorted(member_names)) + "\n", encoding="utf-8")
    tar = subprocess.run(
        [
            "tar",
            "-xf",
            str(tar_path),
            "-T",
            str(listing),
            "--force-local",
            f"--strip-components={strip}",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    listing.unlink(missing_ok=True)
    if tar.returncode != 0:
        raise ValueError(f"tar extraction failed: {tar.stderr[:400]}")


def planned_indices(
    frames: dict[str, dict[str, str]], clip_indices: dict[str, list[int]]
) -> dict[str, list[str]]:
    """Resolve 0-based positions to serials per clip (extract targets)."""
    plan = {}
    for clip, indices in clip_indices.items():
        serials = sorted(frames[clip])
        for index in indices:
            if not 0 <= index < len(serials):
                raise ValueError(f"{clip}: frame {index} outside {len(serials)} tar frames")
        plan[clip] = sorted({serials[index] for index in indices})
    return plan


def extract_frames(
    tar_path, frames: dict[str, dict[str, str]], plan: dict[str, list[str]], root
) -> None:
    members = [frames[clip][serial] for clip, serials in plan.items() for serial in serials]
    extract_members(tar_path, members, root)
    for clip, serials in plan.items():
        present = sorted(path.stem for path in (Path(root) / clip).glob("*.png"))
        if present != serials:
            raise ValueError(f"{clip}: extracted {present}, planned {serials}")


def read_frame(frames_dir, clip: str, position: int):
    serials = sorted(path.stem for path in (Path(frames_dir) / clip).glob("*.png"))
    if not 0 <= position < len(serials):
        raise ValueError(
            f"{clip}: frame position {position} outside {len(serials)} extracted frames"
        )
    image = cv2.imread(str(Path(frames_dir) / clip / f"{serials[position]}.png"))
    if image is None:
        raise ValueError(f"{clip}: cannot read extracted frame {position}")
    return image


class FrameIndex:
    """Sorted serial names for one clip with safe absolute-path resolution."""

    def __init__(self, clip: str, directory, serials):
        self.clip = clip
        self.directory = directory
        self.serials = list(serials)

    def __getitem__(self, item):
        return self.serials[item]

    def __len__(self):
        return len(self.serials)

    def abs_path(self, serial: str):
        if serial not in self.serials:
            raise ValueError(f"{self.clip}: unknown frame serial {serial}")
        return self.directory / f"{serial}.png"


def _valid_clip(clip: str) -> bool:
    return bool(re.fullmatch(r"[\w.-]+", clip)) and not clip.startswith(".") and "/" not in clip


def load_dir_clips(annotation_path, frames_root, split="train", excluded=EXCLUDED_SIGNERS):
    """Annotation rows verified against on-disk frame directories."""
    frames_root = Path(frames_root)
    rows = load_clips(annotation_path, excluded=excluded)
    checked = []
    for row in rows:
        clip = row["clip"]
        if not _valid_clip(clip):
            raise ValueError(f"invalid clip id from annotations: {clip!r}")
        directory = frames_root / split / clip
        present = len(list(directory.glob("*.png"))) if directory.is_dir() else -1
        if present < 0:
            raise ValueError(f"{clip}: missing frame directory under {directory}")
        if present != row["num_frames"]:
            raise ValueError(
                f"{clip}: dir holds {present} frames, annotation says {row['num_frames']}"
            )
        checked.append(row)
    return checked


def index_frames_dir(rows, frames_root, split="train"):
    """clip -> FrameIndex (serials in extraction order)."""
    frames_root = Path(frames_root)
    index = {}
    for row in rows:
        clip = row["clip"]
        if not _valid_clip(clip):
            raise ValueError(f"invalid clip id: {clip!r}")
        directory = frames_root / split / clip
        serials = sorted(path.stem for path in directory.glob("*.png"))
        if len(serials) != row["num_frames"]:
            raise ValueError(f"{clip}: dir count {len(serials)} != {row['num_frames']}")
        index[clip] = FrameIndex(clip, directory, serials)
    return index
