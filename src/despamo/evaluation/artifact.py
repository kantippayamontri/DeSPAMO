import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch

from despamo.utils.hashing import sha256_file


def collect_runtime_metadata(
    seed: int,
    package_lock: Path,
    spatial_manifest: Path,
    motion_manifest: Path,
) -> dict[str, Any]:
    lock_directory = package_lock.resolve().parent
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=lock_directory,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=lock_directory,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "seed": seed,
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else "NO_COMMIT",
        "git_dirty": bool(status.stdout.strip()),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "package_lock_sha256": sha256_file(package_lock),
        "spatial_manifest_sha256": sha256_file(spatial_manifest),
        "motion_manifest_sha256": sha256_file(motion_manifest),
    }


def write_result_artifact(
    path: Path,
    clip_ids: list[str],
    predictions: list[str],
    references: list[str],
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> None:
    if not clip_ids or not (len(clip_ids) == len(predictions) == len(references)):
        raise ValueError("clip_ids, predictions, and references must have equal non-zero length")
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("duplicate clip_id")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("metrics must be finite")
    payload = {
        "metadata": metadata,
        "metrics": metrics,
        "items": [
            {"clip_id": clip_id, "prediction": prediction, "reference": reference}
            for clip_id, prediction, reference in zip(
                clip_ids, predictions, references, strict=True
            )
        ],
    }
    output = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output)
