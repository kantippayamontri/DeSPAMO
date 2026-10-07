"""Provenance for probe runs: code hash, data hashes, model digest, deviations."""

import hashlib
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

DATASET_DEVIATIONS = {
    "stratification": (
        "The design stratifies by signer; How2Sign signer ids are not recoverable on "
        "this host (the -{N}-rgb_front suffix is a camera id), so sampling stratifies by "
        "VIDEO_ID-camera, one signer per source video."
    ),
    "resolution": (
        "The design's 3.9x/5.5x magnification rationale was written for PHOENIX 210x260; "
        "How2Sign frames are 1280x720, so crops are usually downscaled to 224 instead of magnified."
    ),
    "frozen_indices": (
        "There is no PHOENIX-style frozen source.json for How2Sign; the probe defines the "
        "5 sampled frame indices itself using the same 0.1-0.9 position recipe."
    ),
    "review_status": "labels are LLM-generated and unreviewed; human_review_status=not_assessed",
}


def package_code_sha256() -> str:
    hasher = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        hasher.update(path.name.encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ollama_model_digest(model: str, port: int = 11434) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=30) as stream:
        tags = json.loads(stream.read())
    for entry in tags.get("models", []):
        if entry.get("name") == model:
            return entry["digest"]
    raise ValueError(f"model {model} not installed in Ollama")


def build_provenance(csv_path: Path, model: str, seed: int, total: int) -> dict:
    versions = {}
    for name in ("cv2", "mediapipe", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except ImportError:
            versions[name] = None
    return {
        "created_unix": time.time(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "code_sha256": package_code_sha256(),
        "csv_path": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "model": model,
        "model_digest": ollama_model_digest(model),
        "seed": seed,
        "requested_clips": total,
        "versions": versions,
        "deviations": dict(DATASET_DEVIATIONS),
    }
