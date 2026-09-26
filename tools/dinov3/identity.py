import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException

MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
SHA = re.compile(r"[0-9a-f]{40}\Z")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def read_token(path: Path) -> str:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[7:].strip()
        name, separator, value = text.partition("=")
        if not separator or name.strip() != "HF_TOKEN":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        values.append(value)
    if len(values) != 1 or not values[0] or "\n" in values[0]:
        raise ValueError("exactly one nonempty HF_TOKEN required in env file")
    return values[0]


def resolve_sha(token: str, revision: str) -> str:
    try:
        sha = HfApi().model_info(MODEL, revision=revision, token=token).sha
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else None
        if status in (401, 403):
            raise RuntimeError("DINOv3 model access denied; check license and HF_TOKEN") from None
        if status == 404:
            raise RuntimeError("DINOv3 Hub request rejected; check model revision") from None
        if status is not None and 500 <= status < 600:
            raise RuntimeError("DINOv3 Hub unavailable; retry later") from None
        raise RuntimeError("DINOv3 Hub request rejected") from None
    except RequestException:
        raise RuntimeError("DINOv3 Hub request failed; check connectivity and retry") from None
    except Exception:
        raise RuntimeError("DINOv3 model revision lookup failed") from None
    if not isinstance(sha, str) or not SHA.fullmatch(sha):
        raise ValueError("Hub did not resolve model to immutable SHA")
    return sha


def identity(sha: str, lock: Path) -> tuple[str, dict]:
    if not SHA.fullmatch(sha):
        raise ValueError("model revision must be a 40-hex SHA")
    metadata = dict(
        schema_version=1,
        feature_format="npy-float32-Tx2048",
        model=MODEL,
        model_sha=sha,
        lock_sha256=sha256_file(lock),
        source_resolution_policy="record-per-frame",
        scales=[224, 448],
        inference_batch_size=8,
        cudnn_tf32=False,
        color="RGB",
        resize="Pillow.BICUBIC square",
        range="float32/255",
        mean=MEAN,
        std=STD,
        tokens="last_hidden_state[:,0,:]",
        augmentation="none",
        libraries={
            name: version(package)
            for name, package in (
                ("torch", "torch"),
                ("transformers", "transformers"),
                ("pillow", "Pillow"),
                ("numpy", "numpy"),
            )
        },
    )
    return digest(metadata), metadata
