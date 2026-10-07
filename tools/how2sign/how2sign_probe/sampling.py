"""Deterministic video-stratified clip sampling for the How2Sign crop probe.

The upstream probe design stratifies by signer, but How2Sign signer IDs are not
recoverable from the realigned sentence CSV on this host (the ``-{N}-`` suffix is
a camera/view id). Stratification therefore uses ``VIDEO_ID-camera`` strata, one
signer per source video, and the deviation is recorded in probe provenance.
"""

import csv
import hashlib
import re
from pathlib import Path

REQUIRED_COLUMNS = ("VIDEO_ID", "SENTENCE_NAME")
VIEW_SUFFIX = re.compile(r"-(\d+)-rgb_front$")


def stratum_of(row: dict) -> str:
    video = row["VIDEO_ID"]
    match = VIEW_SUFFIX.search(row["SENTENCE_NAME"])
    return f"{video}-{match.group(1)}" if match else video


def clip_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not set(REQUIRED_COLUMNS) <= set(reader.fieldnames):
            raise ValueError(f"missing required columns: {REQUIRED_COLUMNS}")
        rows = []
        for raw in reader:
            if not raw.get("VIDEO_ID") or not raw.get("SENTENCE_NAME"):
                raise ValueError("VIDEO_ID and SENTENCE_NAME must be non-empty")
            rows.append(dict(raw, clip=raw["SENTENCE_NAME"], stratum=stratum_of(raw)))
    clips = [row["clip"] for row in rows]
    if len(clips) != len(set(clips)):
        raise ValueError("duplicate clip ids in manifest")
    return rows


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    if total < 0 or any(count < 0 for count in counts.values()):
        raise ValueError("negative counts")
    if not counts or total == 0:
        return {}
    population = sum(counts.values())
    total = min(total, population)
    got = {key: min(total * counts[key] // population, counts[key]) for key in counts}
    starved = {key: total * counts[key] < population for key in counts}
    remaining = total - sum(got.values())
    while remaining > 0:
        order = sorted(
            (key for key in counts if got[key] < counts[key]),
            key=lambda key: (
                0 if starved[key] else 1,
                -(total * counts[key] - population * got[key]),
                key,
            ),
        )
        if not order:
            raise ValueError("allocation failed")
        for key in order:
            if remaining == 0:
                break
            got[key] += 1
            remaining -= 1
    if sum(got.values()) != total:
        raise ValueError("allocation failed")
    return got


def _rng_key(seed: int, clip: str) -> bytes:
    return hashlib.sha256(f"{seed}\n{clip}".encode()).digest()


def sample_clips(rows: list[dict], total: int, seed: int) -> list[dict]:
    strata: dict[str, list[dict]] = {}
    for row in rows:
        strata.setdefault(row["stratum"], []).append(row)
    quotas = allocate({key: len(items) for key, items in strata.items()}, total)
    picked = []
    for key, members in sorted(strata.items()):
        order = sorted(members, key=lambda row: (_rng_key(seed, row["clip"]), row["clip"]))
        picked.extend(order[: quotas.get(key, 0)])
    return sorted(picked, key=lambda row: row["clip"])
