# DINOv3 Offline Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce complete, versioned PHOENIX14T DINOv3 S2 features and an exact source-frame-to-feature-row manifest, without changing or importing the accepted training runtime.

**Architecture:** A standalone Python 3.11 extractor under `tools/dinov3/` reads trusted annotations, raw frames, and the existing CLIP manifest. It resolves a gated model to an immutable SHA, derives an encoder key from a separate lock and preprocessing policy, atomically publishes verified per-clip arrays/receipts, then renames a staged directory containing both schema-1 manifest and factor-compatible row map only after a zero-failure, three-split gate. Completed versions are validated read-only on rerun. The baseline Python environment reads the resulting manifest in a separate integration check.

**Tech Stack:** Python 3.11; independent uv project/lock with Transformers 4.56.2, PyTorch CUDA build validated on RTX 4080 SUPER, Pillow, NumPy, huggingface-hub, pytest; baseline remains Torch 2.0.1 / Transformers 4.32.0.

## Global Constraints

- Approved source: `docs/superpowers/specs/2026-09-24-dinov3-stage-design.md`, sections 1–4, 6–7. Controlled training comparison belongs to a **different** plan.
- Model `facebook/dinov3-vitl16-pretrain-lvd1689m`; require gated access and exact 40-hex commit SHA; never use random weights or persist mutable `main`.
- Frame root: `/mnt/e/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px`; pass as `PHOENIX14T_FRAME_ROOT`. Annotation split filenames: `train_info_ml.npy`, `dev_info_ml.npy`, `test_info_ml.npy`.
- Expected splits `train=7096`, `dev=519`, `test=642`; PNG totals `827354`, `55775`, `64627`, total `947756`. Every source clip has >=5 PNGs; count equals annotation `num_frames` and CLIP feature length. CLIP manifest is baseline `FeatureManifest` schema 1.
- Sorted source PNGs; no flip, random selection, temporal crop, or subsampling. Square RGB bicubic 224 and 448, float32 `[0,1]`, ImageNet mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)`; two complete frozen passes, `last_hidden_state[:,0,:]` each width 1024, concatenate 224 then 448 to finite float32 `[T,2048]`.
- Encoder key includes model ID/SHA, extractor `uv.lock` SHA-256, ordered preprocessing, schema/format version. Keep `pyproject.toml`, root `uv.lock`, baseline feature files, and VideoMAE untouched. All model cache and output data external to Git on Linux-native filesystem.
- Reject model cache and output paths inside the Git checkout (including symlink-resolved descendants) before creating directories, reading token, or making Hub requests. Validate nearest existing ancestor filesystem before writing; refuse Windows-mounted output/cache.
- Row-map JSON **exact consumer keys**: `spatial_manifest_hash`, `encoder_key`, `clips[clip_id].feature_hash`, `.source_indices`, `.sampled_images` (string source ordinal -> source PNG SHA-256); extra provenance fields may be added without changing those names. Five distinct indices computed by `int((T-1)*position+0.5)` for positions `(0.1,0.3,0.5,0.7,0.9)`.
- `--env-file /home/kan/Research/DeSpaMo/.env` parses **only** `HF_TOKEN` as data, never shell-sources it; never print, persist, or log token. Worktree has no `.env`. No full GPU extraction until separate explicit authorization; do not commit in this plan or its execution unless requested.
- Official API reference: <https://huggingface.co/docs/transformers/v4.56.2/en/model_doc/dinov3> uses `AutoModel.from_pretrained` and `outputs.last_hidden_state[:,0,:]`, with four register tokens after CLS. Explicit manual preprocessing below overrides processor defaults.

## File Map And Contracts

| File | Single responsibility |
|---|---|
| `tools/dinov3/pyproject.toml`, `tools/dinov3/uv.lock` | Separate locked extractor runtime; lock frozen **after** CUDA smoke; root lock untouched |
| `tools/dinov3/identity.py` | Safe token read, immutable revision, canonical version metadata/key |
| `tools/dinov3/frames.py` | Annotation/CLIP validation, sorted source frames/hashes, five-frame indices, image preprocessing |
| `tools/dinov3/encoder.py` | Official model load and two-pass CLS extraction; injectable fake encoder |
| `tools/dinov3/storage.py` | Checkout/filesystem preflight, single-writer lock, durable atomic writes, clip-qualified receipt validation |
| `tools/dinov3/pipeline.py` | Resume, per-clip failures, complete split gate, schema-1 manifest and exact row-map publication |
| `tools/dinov3/cli.py` | Access check, bounded clip, and separately authorized full extraction |
| `tools/dinov3/tests/test_identity.py`, `test_frames.py`, `test_encoder.py`, `test_storage.py`, `test_pipeline.py`, `test_cli.py` | Offline independent-environment TDD; no HF downloads; root pytest never collects these |
| `tools/dinov3/tests/test_live.py` | Opt-in real gated encoder and one real clip in extractor env; outside root `tests/` |
| `tests/integration/test_dinov3_baseline.py` | Opt-in `FeatureManifest.load`/`build_data` test in baseline env |

External layout: `${DINO_OUTPUT}/<encoder_key>/version.json`, `train|dev|test/<clip_id>.npy`, `receipts/<split>/<clip_id>.json`, `failures.json`, `complete/manifest.json`, `complete/frame_rows.json`, `.writer.lock`. No `complete/` exists during incomplete runs; never delete or replace an existing `complete/`. `FeatureRecord.path` stays relative to the encoder-key root, **not** `complete/`. A same-root staging directory contains both JSON files; `os.replace(staging_dir, root/complete)` publishes them together only after hash/count/source verification. All writes use same-directory temp files, `fsync`, `os.replace`, and read-back; each clip receipt records frame paths, source content/size hash, sampled image hashes, feature file hash, resolution list, and encoder key. No extractor module imports `despamo`; JSON schema is duplicated deliberately at output boundary. Run extractor tests explicitly from the repository root with `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests`; root `pytest tests/` must never import extractor libraries. File-labelled Python blocks are complete file contents unless explicitly labelled as insertions into those files. Review each task's files after passing tests; no automatic commits.

---

### Task 1: Locked Runtime And Immutable Identity

**Files:** Create `tools/dinov3/pyproject.toml`, `tools/dinov3/identity.py`, `tools/dinov3/tests/test_identity.py`; generate `tools/dinov3/uv.lock` with uv after local CUDA compatibility check.

**Interfaces:** `read_token(path: Path) -> str`; `resolve_sha(token: str, revision: str) -> str`; `identity(sha: str, lock: Path) -> tuple[str, dict]`; `digest(value: object) -> str`; `sha256_file(path: Path) -> str`. Later modules use these exact names. No baseline imports.

- [ ] **Step 1: Write failing tests** (`tools/dinov3/tests/test_identity.py`):

```python
import json

import pytest

from tools.dinov3.identity import digest, identity, read_token, resolve_sha


def test_token_file_is_data_and_never_executed(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OTHER=ignored\nHF_TOKEN='hf_test; touch /tmp/not-executed'\n")
    assert read_token(env) == "hf_test; touch /tmp/not-executed"
    env.write_text("HF_TOKEN=one\nHF_TOKEN=two\n")
    with pytest.raises(ValueError, match="HF_TOKEN"):
        read_token(env)


def test_revision_pin_and_canonical_version(tmp_path, monkeypatch):
    class Info:
        sha = "a" * 40

    class Api:
        def model_info(self, repo_id, revision, token):
            assert revision == "main" and token == "secret"
            return Info()

    monkeypatch.setattr("tools.dinov3.identity.HfApi", lambda: Api())
    assert resolve_sha("secret", "main") == "a" * 40
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"fixed-lock")
    key, meta = identity("a" * 40, lock)
    assert key == digest(meta) and meta["model_sha"] == "a" * 40
    assert meta["scales"] == [224, 448]
    assert meta["source_resolution_policy"] == "record-per-frame"
    assert "source_resolution" not in meta
    assert "secret" not in json.dumps(meta)
    with pytest.raises(ValueError, match="SHA"):
        identity("main", lock)
```

- [ ] **Step 2: Declare isolated exact inputs** (`tools/dinov3/pyproject.toml`):

```toml
[project]
name = "despamo-dinov3-extractor"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "transformers==4.56.2",
  "torch==2.5.1+cu121",
  "pillow==11.3.0",
  "numpy==1.26.4",
  "huggingface-hub==0.35.3",
  "pytest==8.4.2",
]

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
```

- [ ] **Step 3: Check hardware/runtime, resolve and smoke before accepting lock.** Run `nvidia-smi` (expect RTX 4080 SUPER, driver reporting CUDA >=12.1); `uv lock --project tools/dinov3` (expect resolver success and **only** `tools/dinov3/uv.lock` created); `uv sync --project tools/dinov3 --locked` (expect pinned interpreter 3.11, CUDA wheel), then `uv run --project tools/dinov3 --locked python -c 'import torch,transformers,PIL,numpy; print(torch.__version__,transformers.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0)); assert torch.cuda.is_available(); assert transformers.__version__ == "4.56.2"'` (expect `2.5.1+cu121 4.56.2 True NVIDIA GeForce RTX 4080 SUPER`). If wheel/driver incompatible, stop, choose an actually supported **exact** Torch/CUDA pin, rerun these commands and record resulting exact pin/versions in this project file before freezing lock; never drift root dependencies. `uv lock` is part of later **implementation**, not authorization to install now.
- [ ] **Step 4: Observe red test.** Run `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_identity.py -q`; expect collection `ModuleNotFoundError: tools.dinov3.identity` (before module exists). Do not use root `uv run` for extractor tests.
- [ ] **Step 5: Implement** `tools/dinov3/identity.py`:

```python
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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


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
        raise RuntimeError("DINOv3 Hub request rejected; check model revision") from None
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
    metadata = dict(schema_version=1, feature_format="npy-float32-Tx2048",
                    model=MODEL, model_sha=sha, lock_sha256=sha256_file(lock),
                    source_resolution_policy="record-per-frame",
                    scales=[224, 448], color="RGB", resize="Pillow.BICUBIC square",
                    range="float32/255", mean=MEAN, std=STD,
                    tokens="last_hidden_state[:,0,:]", augmentation="none",
                    libraries={name: version(package) for name, package in (
                        ("torch", "torch"), ("transformers", "transformers"),
                        ("pillow", "Pillow"), ("numpy", "numpy"))})
    return digest(metadata), metadata
```

- [ ] **Step 6: Run** `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_identity.py -q` (expect 2 passed). Review `git status --short` and `git diff -- tools/dinov3`; new untracked files also need direct inspection. Lock present and root lock unchanged.

### Task 2: Frame Discovery, Source Integrity, Preprocessing

**Files:** Create `tools/dinov3/frames.py`, `tools/dinov3/tests/test_frames.py`.

**Interfaces:** `annotations(root: Path, clip_manifest: Path) -> dict[str,dict[str,tuple[int,int]]]` maps split -> clip ID -> `(annotation_count, CLIP_rows)`; `scan(root: Path, split: str, clip_id: str, expected: tuple[int,int]) -> dict` returns `paths`, `source_hash`, `sampled_images`, `source_indices`, `resolutions` (resolutions filled during encode); `preprocess(path: Path, size: int) -> torch.Tensor[3,size,size]`; `sample_indices(count: int) -> list[int]`.

- [ ] **Step 1: Write failing tests** (`tools/dinov3/tests/test_frames.py`):

```python
import json

import numpy as np
import pytest
import torch
from PIL import Image

from tools.dinov3.frames import annotations, preprocess, sample_indices, scan


def test_sorted_rows_hash_and_five_distinct_samples(tmp_path):
    folder = tmp_path / "train" / "c"
    folder.mkdir(parents=True)
    for i in (4, 0, 2, 1, 3):
        Image.new("RGB", (210, 260), (i, 0, 0)).save(folder / f"c-{i:06}.png")
    source = scan(tmp_path, "train", "c", (5, 5))
    assert source["source_indices"] == list(range(5))
    assert source["paths"] == [f"train/c/c-{i:06}.png" for i in range(5)]
    assert list(source["sampled_images"]) == ["0", "1", "2", "3", "4"]
    before = source["source_hash"]
    Image.new("RGB", (210, 260), (255, 0, 0)).save(folder / "c-000000.png")
    assert scan(tmp_path, "train", "c", (5, 5))["source_hash"] != before
    with pytest.raises(ValueError, match="train/c.*count"):
        scan(tmp_path, "train", "c", (6, 6))


def test_preprocess_square_rgb_and_imagenet(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGB", (210, 260), "white").save(path)
    for size in (224, 448):
        pixels = preprocess(path, size)
        assert pixels.shape == (3, size, size)
        assert pixels.dtype == torch.float32
        np.testing.assert_array_equal(pixels[:, 0, 0],
            (np.ones(3, dtype=np.float32) - np.array((.485, .456, .406),
             dtype=np.float32)) / np.array((.229, .224, .225), dtype=np.float32))
    with pytest.raises(ValueError, match="five"):
        sample_indices(4)


def test_nonuniform_bicubic_matches_pillow_at_both_scales(tmp_path):
    path = tmp_path / "gradient.png"
    source = np.array([[[0, 80, 255], [40, 160, 200], [255, 20, 10]],
                       [[255, 240, 0], [100, 5, 100], [0, 255, 160]]], dtype=np.uint8)
    Image.fromarray(source, "RGB").save(path)
    for size in (224, 448):
        bicubic = np.asarray(Image.open(path).resize((size, size), Image.Resampling.BICUBIC),
                             dtype=np.float32) / np.float32(255)
        reference = (bicubic - np.array((.485, .456, .406), dtype=np.float32)) / (
            np.array((.229, .224, .225), dtype=np.float32))
        actual = preprocess(path, size)
        assert actual.dtype == torch.float32
        np.testing.assert_array_equal(actual.numpy(), reference.transpose(2, 0, 1))
        bilinear = np.asarray(Image.open(path).resize((size, size), Image.Resampling.BILINEAR))
        assert not np.array_equal(bilinear, np.asarray(Image.open(path).resize(
            (size, size), Image.Resampling.BICUBIC)))


def test_annotation_and_clip_parity(tmp_path):
    ann = tmp_path / "ann"
    ann.mkdir()
    for split, file in (("train", "train"), ("dev", "dev"), ("test", "test")):
        np.save(ann / f"{file}_info_ml.npy", {0: {"fileid": split, "num_frames": 5}})
    manifest = tmp_path / "clip.json"
    manifest.write_text(json.dumps({"schema_version": 1, "encoder": "clip",
        "expected_dim": 2048, "records": [dict(clip_id=s, split=s, path=f"{s}/{s}.npy",
             length=5, width=2048, dtype="float32") for s in ("train", "dev", "test")]}))
    assert annotations(ann, manifest)["dev"]["dev"] == (5, 5)
    data = json.loads(manifest.read_text())
    data["records"][1]["length"] = 6
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="dev/dev.*CLIP"):
        annotations(ann, manifest)
```

- [ ] **Step 2: Run** `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_frames.py -q` (expect import failure).
- [ ] **Step 3: Implement** `tools/dinov3/frames.py`:

```python
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.dinov3.identity import MEAN, STD, digest, sha256_file

SPLITS = {"train": "train_info_ml.npy", "dev": "dev_info_ml.npy",
          "test": "test_info_ml.npy"}
POSITIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


def annotations(root: Path, clip_manifest: Path) -> dict[str, dict[str, tuple[int, int]]]:
    manifest = json.loads(clip_manifest.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("expected_dim") != 2048:
        raise ValueError("CLIP schema/width mismatch")
    records = {}
    for record in manifest["records"]:
        split, clip = record["split"], record["clip_id"]
        if split not in SPLITS or not clip or "/" in clip or "\\" in clip:
            raise ValueError("invalid CLIP split/clip")
        if (set(record) != {"clip_id", "split", "path", "length", "width", "dtype"}
                or record["path"] != f"{split}/{clip}.npy" or record["width"] != 2048
                or record["dtype"] not in {"float16", "float32", "float64"}
                or type(record["length"]) is not int or record["length"] < 5
                or (split, clip) in records):
            raise ValueError(f"invalid/duplicate CLIP record: {split}/{clip}")
        records[split, clip] = record["length"]
    result = {}
    for split, filename in SPLITS.items():
        raw = np.load(root / filename, allow_pickle=True).item()  # trusted local annotation
        entries = {}
        for key in sorted(k for k in raw if type(k) is int):
            item = raw[key]
            clip, count = item["fileid"], item["num_frames"]
            if not isinstance(clip, str) or not clip or "/" in clip or "\\" in clip:
                raise ValueError(f"invalid annotation clip in {split}")
            if type(count) is not int or count < 5 or clip in entries:
                raise ValueError(f"invalid/duplicate annotation count: {split}/{clip}")
            if records.get((split, clip)) != count:
                raise ValueError(f"{split}/{clip}: annotation/CLIP row count mismatch")
            entries[clip] = (count, records[split, clip])
        result[split] = entries
    if set(records) != {(split, clip) for split, clips in result.items() for clip in clips}:
        raise ValueError("CLIP manifest extra/missing clips")
    if len({clip for clips in result.values() for clip in clips}) != len(records):
        raise ValueError("duplicate clip ID across splits; row-map keys would collide")
    return result


def sample_indices(count: int) -> list[int]:
    if type(count) is not int or count < 5:
        raise ValueError("at least five frames required")
    indices = [int((count - 1) * position + 0.5) for position in POSITIONS]
    if len(set(indices)) != 5:
        raise ValueError("five sampled source indices must be distinct")
    return indices


def scan(root: Path, split: str, clip_id: str, expected: tuple[int, int]) -> dict:
    if split not in SPLITS or not clip_id or "/" in clip_id or "\\" in clip_id:
        raise ValueError("invalid source clip")
    directory = root / split / clip_id
    if not directory.is_dir():
        raise ValueError(f"{split}/{clip_id}: missing source directory")
    frames = sorted(directory.glob("*.png"), key=lambda p: p.name)
    if len(frames) < 5 or len(frames) != expected[0] or len(frames) != expected[1]:
        raise ValueError(f"{split}/{clip_id}: PNG count/annotation/CLIP mismatch")
    serials = [re.search(r"(\d+)\.png\Z", p.name) for p in frames]
    if any(match is None for match in serials) or len({int(m.group(1)) for m in serials}) != len(frames):
        raise ValueError(f"{split}/{clip_id}: duplicate/invalid frame ordinal")
    items = [dict(path=p.relative_to(root).as_posix(), size=p.stat().st_size,
                  sha256=sha256_file(p)) for p in frames]
    sampled = {str(i): items[i]["sha256"] for i in sample_indices(len(frames))}
    return dict(paths=[item["path"] for item in items], source_hash=digest(items),
                source_indices=list(range(len(items))), sampled_images=sampled,
                resolutions=[])


def preprocess(path: Path, size: int) -> torch.Tensor:
    if size not in (224, 448):
        raise ValueError("unsupported DINOv3 scale")
    with Image.open(path) as source:
        image = source.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        values = np.asarray(image, dtype=np.float32).copy() / 255.0
    values = (values - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return torch.from_numpy(values.transpose(2, 0, 1).copy())
```

- [ ] **Step 4: Run** `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_frames.py -q` (expect 4 passed). Review sorted-path and source hash contracts.

### Task 3: Official Two-Pass Frozen Encoder

**Files:** Create `tools/dinov3/encoder.py`, `tools/dinov3/tests/test_encoder.py`.

**Interfaces:** `extract(paths: list[Path], model: object, device: str) -> tuple[np.ndarray, list[list[int]]]`; `load_model(sha: str, token: str, cache: Path, device: str) -> object`. Output resolutions list is `[width,height]` for each source image. Fake `model(pixel_values=...)` returns `.last_hidden_state`.

- [ ] **Step 1: Write failing tests** (`tools/dinov3/tests/test_encoder.py`):

```python
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tools.dinov3.encoder import extract


class Fake:
    def __call__(self, pixel_values):
        side = pixel_values.shape[-1]
        cls = torch.full((1, 1024), float(side))
        register = torch.full((1, 1024), -1.0)
        return SimpleNamespace(last_hidden_state=torch.stack((cls[0], register[0]))[None])


def test_two_pass_cls_order_width_and_determinism(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (210, 260), "white").save(image)
    first, resolutions = extract([image, image], Fake(), "cpu")
    second, _ = extract([image, image], Fake(), "cpu")
    assert resolutions == [[210, 260]] * 2
    assert first.shape == (2, 2048) and first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0, :1024], np.full(1024, 224))
    np.testing.assert_array_equal(first[0, 1024:], np.full(1024, 448))


def test_invalid_cls_fails_without_fallback(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (210, 260)).save(image)
    class Bad:
        def __call__(self, pixel_values):
            return SimpleNamespace(last_hidden_state=torch.full((1, 5, 1024), float("nan")))
    with pytest.raises(RuntimeError, match="CLS"):
        extract([image], Bad(), "cpu")
```

- [ ] **Step 2: Run** `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_encoder.py -q` (expect missing module).
- [ ] **Step 3: Implement** `tools/dinov3/encoder.py`:

```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from tools.dinov3.frames import preprocess
from tools.dinov3.identity import MODEL, SHA


def load_model(sha: str, token: str, cache: Path, device: str) -> object:
    if not SHA.fullmatch(sha):
        raise ValueError("immutable model SHA required")
    try:
        model = AutoModel.from_pretrained(MODEL, revision=sha, token=token,
                                          cache_dir=cache, trust_remote_code=False,
                                          use_safetensors=True).eval().to(device)
    except Exception:
        raise RuntimeError("DINOv3 gated weights inaccessible/incompatible; check access and CUDA") from None
    if model.config.hidden_size != 1024 or model.config.patch_size != 16 or (
            model.config.num_register_tokens != 4):
        raise ValueError("DINOv3 model configuration mismatch")
    return model


def extract(paths: list[Path], model: object, device: str) -> tuple[np.ndarray, list[list[int]]]:
    output, resolutions = [], []
    for path in paths:
        with Image.open(path) as image:
            image.verify()  # fail on broken PNG even if metadata exists
        with Image.open(path) as image:
            resolutions.append([image.width, image.height])
        scales = []
        for size in (224, 448):
            pixels = preprocess(path, size).unsqueeze(0).to(device)
            with torch.inference_mode():
                states = model(pixel_values=pixels).last_hidden_state
            if states.ndim != 3 or states.shape[0] != 1 or states.shape[2] != 1024:
                raise RuntimeError(f"CLS width/shape mismatch: {path.name}")
            cls = states[:, 0, :].detach().float().cpu().numpy()[0]
            if not np.isfinite(cls).all():
                raise RuntimeError(f"non-finite CLS: {path.name}")
            scales.append(cls)
        output.append(np.concatenate(scales).astype(np.float32, copy=False))
    result = np.stack(output)
    if result.shape != (len(paths), 2048) or result.dtype != np.float32:
        raise RuntimeError("invalid extracted feature shape/dtype")
    return result, resolutions
```

- [ ] **Step 4: Run** `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_encoder.py -q` (expect 2 passed). Confirm no `despamo` import or feature-width fallback.

### Task 4: Atomic Storage, Single Writer, Safe Resume

**Files:** Create `tools/dinov3/storage.py`, `tools/dinov3/tests/test_storage.py`.

**Interfaces:** `verify_location(path: Path, checkout: Path) -> None` checks resolved checkout exclusion and Linux-native mount **without creating anything**; `writer(root: Path)` obtains nonblocking `flock` after preflight; `atomic_json(path: Path, payload: object) -> None`; `atomic_array(path: Path, array: np.ndarray) -> None`; `check_receipt(root: Path, split: str, clip_id: str, source: dict, key: str) -> dict | None`; `publish_clip(root: Path, split: str, clip_id: str, source: dict, key: str, array: np.ndarray) -> dict`. Existing receipt/source conflict fails closed; corrupt feature or malformed receipt requires a fresh external output base/key (no destructive repair). Only an orphan array without receipt may be replaced.

- [ ] **Step 1: Write failing tests** (`tools/dinov3/tests/test_storage.py`):

```python
import numpy as np
import pytest

from tools.dinov3.storage import check_receipt, publish_clip, verify_location, writer


def test_resume_integrity_rejects_source_feature_and_key_drift(tmp_path):
    source = dict(paths=[f"train/c/{i}.png" for i in range(5)], source_indices=list(range(5)),
                  source_hash="input", sampled_images={str(i): "sha" for i in range(5)},
                  resolutions=[[210, 260]] * 5)
    with writer(tmp_path):
        with pytest.raises(RuntimeError, match="writer"):
            with writer(tmp_path):
                pass
        receipt = publish_clip(tmp_path, "train", "c", source, "key", np.ones((5, 2048), np.float32))
        assert receipt == check_receipt(tmp_path, "train", "c", source, "key")
        for altered, key in (({**source, "source_hash": "changed"}, "key"), (source, "other")):
            with pytest.raises(ValueError, match="incompatible receipt"):
                check_receipt(tmp_path, "train", "c", altered, key)
        feature = tmp_path / "train/c.npy"
        feature.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="feature hash"):
            check_receipt(tmp_path, "train", "c", source, "key")


def test_malformed_receipt_has_clip_context(tmp_path):
    path = tmp_path / "receipts/train/c.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"feature_hash":')
    with pytest.raises(ValueError, match="train/c: malformed receipt"):
        check_receipt(tmp_path, "train", "c", {}, "key")
    path.write_text('{"encoder_key": "key"}')
    with pytest.raises(ValueError, match="train/c: malformed receipt"):
        check_receipt(tmp_path, "train", "c", {}, "key")


def test_checkout_guard_is_read_only_and_resolves_symlinks(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    with pytest.raises(ValueError, match="inside Git checkout"):
        verify_location(checkout / "cache", checkout)
    link = tmp_path / "alias"
    link.symlink_to(checkout, target_is_directory=True)
    with pytest.raises(ValueError, match="inside Git checkout"):
        verify_location(link / "outputs", checkout)
    assert not (checkout / "cache").exists()
    assert not (checkout / "outputs").exists()
    verify_location(tmp_path / "external", checkout)
```

- [ ] **Step 2: Run** `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_storage.py -q` (expect missing module).
- [ ] **Step 3: Implement** `tools/dinov3/storage.py`:

```python
import fcntl
import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from tools.dinov3.identity import sha256_file


def verify_location(path: Path, checkout: Path) -> None:
    resolved, project = path.resolve(strict=False), checkout.resolve(strict=False)
    if resolved == project or project in resolved.parents:
        raise ValueError("model cache/output inside Git checkout")
    ancestor = resolved
    while not ancestor.exists():
        ancestor = ancestor.parent
    filesystem = subprocess.check_output(["findmnt", "-no", "FSTYPE", "-T", str(ancestor)],
                                         text=True, env={k: v for k, v in os.environ.items()
                                                         if k != "HF_TOKEN"}).strip()
    if filesystem not in {"ext4", "xfs", "btrfs", "overlay", "tmpfs", "zfs", "f2fs"}:
        raise ValueError("model cache/output must use Linux-native filesystem")


@contextmanager
def writer(root: Path):
    verify_location(root, Path(__file__).resolve().parents[2])
    if (root / "complete").is_dir() and not (root / ".writer.lock").is_file():
        raise ValueError("completed version missing writer lock")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another writer owns encoder key") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _atomic(path: Path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".partial-", delete=False) as handle:
            name = Path(handle.name)
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        dirfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def atomic_json(path: Path, payload: object) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    _atomic(path, lambda handle: handle.write(raw))
    if json.loads(path.read_text()) != payload:
        raise ValueError(f"JSON read-back mismatch: {path}")


def atomic_array(path: Path, array: np.ndarray) -> None:
    if array.ndim != 2 or array.shape[0] < 5 or array.shape[1] != 2048 or (
            array.dtype != np.float32 or not np.isfinite(array).all()):
        raise ValueError(f"invalid feature array: {path}")
    _atomic(path, lambda handle: np.save(handle, array, allow_pickle=False))
    saved = np.load(path, allow_pickle=False)
    if saved.shape != array.shape or saved.dtype != np.float32 or not np.array_equal(saved, array):
        raise ValueError(f"feature read-back mismatch: {path}")


def check_receipt(root: Path, split: str, clip_id: str, source: dict, key: str) -> dict | None:
    path = root / "receipts" / split / f"{clip_id}.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
        required = {"encoder_key", "source_hash", "paths", "source_indices",
                    "sampled_images", "resolutions", "feature_hash"}
        if (not isinstance(record, dict) or set(record) != required
                or not isinstance(record["paths"], list)
                or not isinstance(record["source_indices"], list)
                or not isinstance(record["sampled_images"], dict)
                or not isinstance(record["resolutions"], list)
                or not isinstance(record["feature_hash"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["feature_hash"])):
            raise ValueError("invalid receipt schema")
    except (ValueError, UnicodeError, OSError, TypeError, KeyError):
        raise ValueError(f"{split}/{clip_id}: malformed receipt") from None
    if record["encoder_key"] != key or record["source_hash"] != source["source_hash"] or (
            record["paths"] != source["paths"] or record["source_indices"] != source["source_indices"]
            or record["sampled_images"] != source["sampled_images"]):
        raise ValueError(f"{split}/{clip_id}: incompatible receipt (version/source changed)")
    feature = root / split / f"{clip_id}.npy"
    if not feature.is_file() or sha256_file(feature) != record["feature_hash"]:
        raise ValueError(f"{split}/{clip_id}: feature hash mismatch")
    try:
        array = np.load(feature, allow_pickle=False, mmap_mode="r")
    except (ValueError, OSError, EOFError):
        raise ValueError(f"{split}/{clip_id}: invalid feature file") from None
    if array.shape != (len(source["paths"]), 2048) or array.dtype != np.float32 or (
            not np.isfinite(array).all() or len(record["resolutions"]) != len(array)):
        raise ValueError(f"{split}/{clip_id}: feature shape/dtype/finiteness mismatch")
    return record


def publish_clip(root: Path, split: str, clip_id: str, source: dict, key: str,
                 array: np.ndarray) -> dict:
    if check_receipt(root, split, clip_id, source, key) is not None:
        raise ValueError(f"{split}/{clip_id}: validated version already exists")
    if array.shape[0] != len(source["paths"]) or len(source["resolutions"]) != len(array):
        raise ValueError(f"{split}/{clip_id}: source/feature row mismatch")
    feature = root / split / f"{clip_id}.npy"
    atomic_array(feature, array)
    record = dict(encoder_key=key, source_hash=source["source_hash"], paths=source["paths"],
                  source_indices=source["source_indices"],
                  sampled_images=source["sampled_images"], resolutions=source["resolutions"],
                  feature_hash=sha256_file(feature))
    atomic_json(root / "receipts" / split / f"{clip_id}.json", record)
    check_receipt(root, split, clip_id, source, key)
    return record
```

- [ ] **Step 4: Run** `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_storage.py -q` (expect 3 passed). Simulated corrupt `.npy` must never count as resumable.

### Task 5: Split Gate, Failure Journal, Factor Row Map

**Files:** Create `tools/dinov3/pipeline.py`, `tools/dinov3/tests/test_pipeline.py`.

**Interfaces:** `run(frame_root: Path, annotation_root: Path, clip_manifest: Path, output: Path, key: str, metadata: dict, model: object, device: str, *, only: tuple[str,str] | None = None) -> dict`; `validate_complete(frame_root: Path, output: Path, published: Path, clips: dict, key: str) -> dict`. `only` bounded smoke publishes receipts but never `complete/`; full run requires exact expected counts and zero failures. Existing `complete/` is verified read-only then returned or rejected; no mutation. For source drift/corrupt validated receipts, record failure on incomplete roots and leave old array/receipt untouched; require fresh external output base/key (archival), never implicit destructive repair.

**Exception policy:** `encoder.extract` leaves PNG decode/preprocess `ValueError` recoverable per clip. CUDA transfer and model forward exceptions (including `ValueError`, `TypeError`, and `AttributeError`) become sanitized fatal `RuntimeError` with `from None`. Pipeline journals only predefined static fatal categories from `FATAL_REASONS`, or `DINOv3 extraction runtime failure` for any other `RuntimeError`/`MemoryError`; never serialize raw fatal exception text. Shape/non-finite CLS errors remain fatal with static categories.

- [ ] **Step 1: Write failing tests** (`tools/dinov3/tests/test_pipeline.py`):

```python
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from tools.dinov3.pipeline import run


class Fake:
    def __call__(self, pixel_values):
        import torch
        from types import SimpleNamespace
        cls = torch.ones(1, 1024) * pixel_values.shape[-1]
        return SimpleNamespace(last_hidden_state=cls[:, None, :])


def fixture(tmp_path):
    frames, ann = tmp_path / "frames", tmp_path / "ann"
    ann.mkdir()
    records = []
    for split in ("train", "dev", "test"):
        folder = frames / split / split
        folder.mkdir(parents=True)
        for i in range(5):
            Image.new("RGB", (210, 260), "white").save(folder / f"{i:06}.png")
        np.save(ann / f"{split}_info_ml.npy", {0: {"fileid": split, "num_frames": 5}})
        records.append(dict(clip_id=split, split=split, path=f"{split}/{split}.npy",
                            length=5, width=2048, dtype="float32"))
    clip = tmp_path / "clip.json"
    clip.write_text(json.dumps(dict(schema_version=1, encoder="clip",
                                    expected_dim=2048, records=records)))
    return frames, ann, clip


def test_partial_and_failure_never_publish_complete_manifests(tmp_path):
    frames, ann, clip = fixture(tmp_path)
    output = tmp_path / "out"
    stats = run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(),
                "cpu", only=("train", "train"))
    assert stats["completed"] == 1 and not (output / "complete").exists()
    (frames / "dev/dev/000000.png").write_bytes(b"broken")
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    assert "dev/dev" in (output / "failures.json").read_text()
    assert not (output / "complete").exists()


def test_complete_map_matches_factor_contract_with_small_fixture(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", {"train": 1, "dev": 1, "test": 1})
    output = tmp_path / "out"
    run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    manifest = json.loads((output / "complete/manifest.json").read_text())
    mapping = json.loads((output / "complete/frame_rows.json").read_text())
    assert len(manifest["records"]) == 3 and manifest["expected_dim"] == 2048
    assert manifest["records"][0]["path"] in {"train/train.npy", "dev/dev.npy", "test/test.npy"}
    assert mapping["clips"]["train"]["source_indices"] == list(range(5))
    assert set(mapping["clips"]["train"]["sampled_images"]) == set(map(str, range(5)))
    assert mapping["spatial_manifest_hash"] == hashlib.sha256(
        (output / "complete/manifest.json").read_bytes()).hexdigest()
    assert mapping["clips"]["train"]["feature_hash"] == hashlib.sha256(
        (output / "train/train.npy").read_bytes()).hexdigest()
    assert json.loads((output / "failures.json").read_text()) == {}


def test_completed_version_rerun_validates_without_modifying_any_file(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", {"train": 1, "dev": 1, "test": 1})
    output = tmp_path / "out"
    run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    def snapshot():
        return {p.relative_to(output).as_posix():
                (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
                for p in (output, *output.rglob("*"))}

    before = snapshot()
    assert run(frames, ann, clip, output, "key", {"model_sha": "a" * 40},
               Fake(), "cpu")["complete"]
    assert snapshot() == before
    (frames / "train/train/000000.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="train/train"):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    assert snapshot() == before


def test_joint_publication_failure_never_exposes_half_manifest(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", {"train": 1, "dev": 1, "test": 1})
    from tools.dinov3 import pipeline
    write = pipeline.atomic_json

    def crash_on_row_map(path, payload):
        if path.name == "frame_rows.json":
            raise OSError("injected staging failure")
        return write(path, payload)

    monkeypatch.setattr(pipeline, "atomic_json", crash_on_row_map)
    with pytest.raises(OSError, match="injected"):
        run(frames, ann, clip, tmp_path / "out", "key", {"model_sha": "a" * 40},
            Fake(), "cpu")
    assert not (tmp_path / "out/complete").exists()
    assert not list((tmp_path / "out").glob(".complete-staging-*"))


def test_model_runtime_error_journaled_then_aborts_before_third_clip(tmp_path):
    frames, ann, clip = fixture(tmp_path)
    class Broken(Fake):
        calls = 0

        def __call__(self, pixel_values):
            self.calls += 1
            if self.calls == 11:
                raise RuntimeError("encoder incompatible")
            return super().__call__(pixel_values)

    model = Broken()
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="dev/dev.*model forward failed"):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, model, "cpu")
    assert model.calls == 11
    assert (output / "train/train.npy").is_file()
    assert not (output / "test/test.npy").exists()
    assert json.loads((output / "failures.json").read_text())["dev/dev"]["stage"] == "encode"
    assert not (output / "complete").exists()


def test_malformed_resume_receipt_is_journaled_with_clip_id(tmp_path):
    frames, ann, clip = fixture(tmp_path)
    output = tmp_path / "out"
    run(frames, ann, clip, output, "key", {"model_sha": "a" * 40},
        Fake(), "cpu", only=("train", "train"))
    (output / "receipts/train/train.json").write_text("{")
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40},
            Fake(), "cpu", only=("train", "train"))
    assert "train/train: malformed receipt" in (
        json.loads((output / "failures.json").read_text())["train/train"]["reason"])


def test_full_gate_rejects_partial_split_counts(tmp_path):
    frames, ann, clip = fixture(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, tmp_path / "out", "key",
            {"model_sha": "a" * 40}, Fake(), "cpu")
    assert not (tmp_path / "out/complete").exists()


def test_source_change_between_scan_and_encode_cannot_publish(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    from tools.dinov3 import pipeline
    real_extract = pipeline.extract

    def change_input(paths, model, device):
        result = real_extract(paths, model, device)
        Image.new("RGB", (210, 260), "black").save(paths[0])
        return result

    monkeypatch.setattr(pipeline, "extract", change_input)
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, tmp_path / "out", "key", {"model_sha": "a" * 40},
            Fake(), "cpu", only=("train", "train"))
    assert "source PNG changed" in (tmp_path / "out/failures.json").read_text()
    assert not (tmp_path / "out/train/train.npy").exists()


def test_resumed_source_change_before_final_promotion_fails(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    output = tmp_path / "out"
    run(frames, ann, clip, output, "key", {"model_sha": "a" * 40},
        Fake(), "cpu", only=("train", "train"))
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", {"train": 1, "dev": 1, "test": 1})
    from tools.dinov3 import pipeline
    original = pipeline.check_receipt

    def change_after_resume(root, split, clip_id, source, key):
        receipt = original(root, split, clip_id, source, key)
        if receipt is not None and split == "train":
            Image.new("RGB", (210, 260), "black").save(frames / "train/train/000000.png")
        return receipt

    monkeypatch.setattr(pipeline, "check_receipt", change_after_resume)
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    assert "train/train" in json.loads((output / "failures.json").read_text())
    assert not (output / "complete").exists()


def test_fresh_source_change_during_staging_is_caught_before_promotion(tmp_path, monkeypatch):
    frames, ann, clip = fixture(tmp_path)
    output = tmp_path / "out"
    monkeypatch.setattr("tools.dinov3.pipeline.EXPECTED", {"train": 1, "dev": 1, "test": 1})
    from tools.dinov3 import pipeline
    write = pipeline.atomic_json

    def mutate_after_manifest(path, payload):
        write(path, payload)
        if path.name == "manifest.json":
            Image.new("RGB", (210, 260), "black").save(frames / "dev/dev/000001.png")

    monkeypatch.setattr(pipeline, "atomic_json", mutate_after_manifest)
    with pytest.raises(ValueError, match="incomplete"):
        run(frames, ann, clip, output, "key", {"model_sha": "a" * 40}, Fake(), "cpu")
    assert "dev/dev" in json.loads((output / "failures.json").read_text())
    assert not (output / "complete").exists()
```

- [ ] **Step 2: Run** `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_pipeline.py -q` (expect missing module).
- [ ] **Step 3: Implement** `tools/dinov3/pipeline.py`:

```python
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from tools.dinov3.encoder import FATAL_REASONS, extract
from tools.dinov3.frames import annotations, scan
from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json, check_receipt, publish_clip, writer

EXPECTED = {"train": 7096, "dev": 519, "test": 642}


def validate_complete(frame_root: Path, output: Path, published: Path,
                      clips: dict, key: str) -> dict:
    failure_path = output / "failures.json"
    if not failure_path.is_file() or json.loads(failure_path.read_text()) != {}:
        raise ValueError("complete version requires zero recorded failures")
    manifest_path = published / "manifest.json"
    row_path = published / "frame_rows.json"
    manifest = json.loads(manifest_path.read_text())
    row_map = json.loads(row_path.read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("expected_dim") != 2048
            or manifest.get("encoder") != f"dinov3:{key}"
            or row_map.get("encoder_key") != key
            or row_map.get("spatial_manifest_hash") != sha256_file(manifest_path)):
        raise ValueError("complete manifest identity/hash mismatch")
    records = manifest["records"]
    counts = Counter(record["split"] for record in records)
    if (counts != EXPECTED or len(records) != sum(EXPECTED.values())
            or len(row_map["clips"]) != len(records)):
        raise ValueError("complete split/row-map count mismatch")
    indexed = {(record["split"], record["clip_id"]): record for record in records}
    if len(indexed) != len(records) or set(indexed) != {
            (split, clip_id) for split, entries in clips.items() for clip_id in entries}:
        raise ValueError("complete manifest duplicate/missing clip")
    for split, entries in clips.items():
        for clip_id, expected in entries.items():
            try:
                source = scan(frame_root, split, clip_id, expected)
                receipt = check_receipt(output, split, clip_id, source, key)
                if receipt is None:
                    raise ValueError("missing validated receipt")
                if scan(frame_root, split, clip_id, expected)["source_hash"] != source["source_hash"]:
                    raise ValueError("source changed during final validation")
                row = dict(feature_hash=receipt["feature_hash"],
                           source_indices=receipt["source_indices"],
                           sampled_images=receipt["sampled_images"],
                           frame_count=expected[0], paths=receipt["paths"],
                           source_hash=receipt["source_hash"],
                           resolutions=receipt["resolutions"])
                record = dict(clip_id=clip_id, split=split, path=f"{split}/{clip_id}.npy",
                              length=expected[0], width=2048, dtype="float32")
                if row_map["clips"][clip_id] != row or indexed[split, clip_id] != record:
                    raise ValueError("frame-row/feature record mismatch")
            except (ValueError, OSError, KeyError, TypeError) as exc:
                raise ValueError(f"{split}/{clip_id}: {exc}") from None
    return dict(completed=len(records), complete=True, counts=dict(counts))


def run(frame_root: Path, annotation_root: Path, clip_manifest: Path, output: Path,
        key: str, metadata: dict, model: object, device: str,
        *, only: tuple[str, str] | None = None) -> dict:
    clips = annotations(annotation_root, clip_manifest)
    if only is None:
        for split, entries in clips.items():
            actual = {p.name for p in (frame_root / split).iterdir() if p.is_dir()}
            if actual != set(entries):
                raise ValueError(f"{split}: missing/unexpected source clip directories")
    with writer(output):
        version = output / "version.json"
        if version.exists() and json.loads(version.read_text()) != metadata:
            raise ValueError("encoder key/version metadata mismatch")
        complete = output / "complete"
        if complete.exists():
            if not complete.is_dir() or not version.is_file():
                raise ValueError("malformed complete version")
            return validate_complete(frame_root, output, complete, clips, key)
        if not version.exists():
            atomic_json(version, metadata)
        failures = {}
        failure_path = output / "failures.json"
        if failure_path.exists():
            failures = json.loads(failure_path.read_text())
        rows = {}
        records = []
        completed = 0
        for split, entries in clips.items():
            for clip_id, expected in sorted(entries.items()):
                if only is not None and (split, clip_id) != only:
                    continue
                stage = "source"
                try:
                    source = scan(frame_root, split, clip_id, expected)
                    stage = "resume"
                    receipt = check_receipt(output, split, clip_id, source, key)
                    if receipt is None:
                        stage = "encode"
                        array, resolutions = extract([frame_root / p for p in source["paths"]],
                                                     model, device)
                        if scan(frame_root, split, clip_id, expected)["source_hash"] != source["source_hash"]:
                            raise ValueError("source PNG changed during extraction")
                        source["resolutions"] = resolutions
                        stage = "publish"
                        receipt = publish_clip(output, split, clip_id, source, key, array)
                    if scan(frame_root, split, clip_id, expected)["source_hash"] != source["source_hash"]:
                        raise ValueError("source PNG changed before row validation")
                    if len(receipt["source_indices"]) != expected[0] or (
                            len(receipt["sampled_images"]) != 5 or
                            len(set(receipt["source_indices"])) != expected[0] or
                            any(receipt["source_indices"].count(int(i)) != 1
                                for i in receipt["sampled_images"])):
                        raise ValueError("invalid source-to-row mapping")
                    rows[clip_id] = dict(feature_hash=receipt["feature_hash"],
                                         source_indices=receipt["source_indices"],
                                         sampled_images=receipt["sampled_images"],
                                         frame_count=expected[0], paths=receipt["paths"],
                                         source_hash=receipt["source_hash"],
                                         resolutions=receipt["resolutions"])
                    records.append(dict(clip_id=clip_id, split=split,
                                        path=f"{split}/{clip_id}.npy", length=expected[0],
                                        width=2048, dtype="float32"))
                    failures.pop(f"{split}/{clip_id}", None)
                    completed += 1
                except (ValueError, OSError, RuntimeError, MemoryError) as exc:
                    fatal = isinstance(exc, (RuntimeError, MemoryError))
                    reason = str(exc)
                    if fatal and reason not in FATAL_REASONS:
                        reason = "DINOv3 extraction runtime failure"
                    failures[f"{split}/{clip_id}"] = dict(stage=stage, reason=reason)
                    atomic_json(failure_path, failures)
                    if fatal:
                        raise RuntimeError(f"{split}/{clip_id}: {reason}; extraction stopped") from None
        atomic_json(failure_path, failures)
        if only is not None:
            if completed != 1 or failures.get(f"{only[0]}/{only[1]}"):
                raise ValueError("bounded extraction incomplete; see failures.json")
            return dict(completed=completed, complete=False)
        counts = Counter(record["split"] for record in records)
        if failures or counts != EXPECTED or len(rows) != sum(EXPECTED.values()):
            raise ValueError(f"extraction incomplete; completed={dict(counts)}, failures={len(failures)}")
        expected_png = {"train": 827354, "dev": 55775, "test": 64627}
        if EXPECTED == {"train": 7096, "dev": 519, "test": 642} and {
                split: sum(record["length"] for record in records if record["split"] == split)
                for split in EXPECTED} != expected_png:
            raise ValueError("PNG totals differ from approved annotation/CLIP baseline")
        manifest = dict(schema_version=1, encoder=f"dinov3:{key}", expected_dim=2048,
                        records=sorted(records, key=lambda record: (record["split"], record["clip_id"])))
        staging = Path(tempfile.mkdtemp(prefix=".complete-staging-", dir=output))
        try:
            atomic_json(staging / "manifest.json", manifest)
            row_map = dict(spatial_manifest_hash=sha256_file(staging / "manifest.json"),
                           encoder_key=key, clips=rows)
            atomic_json(staging / "frame_rows.json", row_map)
            try:
                result = validate_complete(frame_root, output, staging, clips, key)
            except ValueError as exc:
                clip_id = str(exc).split(":", 1)[0]
                if clip_id in {f"{split}/{clip}" for split, entries in clips.items()
                               for clip in entries}:
                    failures[clip_id] = dict(stage="final_validation", reason=str(exc))
                    atomic_json(failure_path, failures)
                raise ValueError("extraction incomplete; final validation failed") from exc
            if complete.exists():
                raise ValueError("complete version appeared during staging")
            os.replace(staging, complete)
            dirfd = os.open(output, os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
            return result
        finally:
            if staging.exists():
                shutil.rmtree(staging)
```

- [ ] **Step 4: Run** `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_pipeline.py -q` (expect 10 passed). In particular, injected second-JSON failure leaves no `complete/`; existing `complete/` read-only validation preserves bytes and mtimes; incompatible encoder stops at second clip; malformed receipt has `train/train` journal context.

### Task 6: Access CLI, Optional Live Smoke, Baseline Consumer

**Files:** Create `tools/dinov3/cli.py`, `tools/dinov3/tests/test_cli.py`, `tools/dinov3/tests/test_live.py`, `tests/integration/test_dinov3_baseline.py`.

**Interfaces:** CLI `--env-file`, `--revision`, `--cache`, `--output-base`, `--annotation-root`, `--clip-manifest`; `--check-access` fetches **only** config; `--one-clip SPLIT CLIP_ID` runs exactly one clip; `--all --confirm-full-extraction` requires separate authorization. No command requires/uses baseline runtime in extractor process.

- [ ] **Step 1: Write failing CLI test** (`tools/dinov3/tests/test_cli.py`):

```python
import sys

import pytest

from tools.dinov3.cli import main


def test_access_check_uses_pinned_sha_without_encoder_download(tmp_path, monkeypatch, capsys):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=hf_private_test\n")
    monkeypatch.setenv("HF_TOKEN", "")  # restore caller's environment after test
    calls = []
    monkeypatch.setattr("tools.dinov3.cli.resolve_sha", lambda token, revision: "a" * 40)
    monkeypatch.setattr("tools.dinov3.cli.hf_hub_download", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["cli", "--env-file", str(env),
                                   "--cache", str(tmp_path), "--check-access"])
    main()
    assert calls[0]["filename"] == "config.json"
    assert calls[0]["revision"] == "a" * 40
    assert "hf_private_test" not in capsys.readouterr().out


def test_full_corpus_requires_confirmation(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=secret\n")
    monkeypatch.setattr(sys, "argv", ["cli", "--env-file", str(env), "--all"])
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize("blocked", ["cache", "output"])
def test_checkout_paths_fail_before_download_or_creation(tmp_path, monkeypatch, blocked):
    from pathlib import Path
    checkout = Path(__file__).resolve().parents[3]
    inside = checkout / "not-created-by-extractor"
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=fake\n")
    monkeypatch.setattr("tools.dinov3.cli.hf_hub_download",
                        lambda **kwargs: pytest.fail("network reached"))
    args = ["cli", "--env-file", str(env)]
    if blocked == "cache":
        args += ["--cache", str(inside), "--check-access"]
    else:
        args += ["--cache", str(tmp_path), "--output-base", str(inside),
                 "--annotation-root", str(tmp_path),
                 "--clip-manifest", str(tmp_path), "--one-clip", "train", "c"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="inside Git checkout"):
        main()
    assert not inside.exists()


# Completed rerun test must supply valid three-split annotations, CLIP manifest,
# frame root, and selected clip directory. See tools/dinov3/tests/test_cli.py.
# It validates those inputs without write probe, CUDA initialization, or model load.
```

- [ ] **Step 2: Run** `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_cli.py -q` (expect import failure).
- [ ] **Step 3: Implement** `tools/dinov3/cli.py`:

```python
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from tools.dinov3.encoder import load_model
from tools.dinov3.frames import SPLITS, annotations
from tools.dinov3.identity import MODEL, identity, read_token, resolve_sha
from tools.dinov3.pipeline import run
from tools.dinov3.storage import atomic_array, verify_location, writer


def _preflight_inputs(annotation_root: Path, clip_manifest: Path,
                      only: tuple[str, str] | None) -> Path:
    frame_value = os.environ.get("PHOENIX14T_FRAME_ROOT")
    if not frame_value:
        raise ValueError("PHOENIX14T_FRAME_ROOT required for extraction")
    frame_root = Path(frame_value)
    if not frame_root.is_dir():
        raise ValueError("PHOENIX14T_FRAME_ROOT must be an existing directory")
    if not annotation_root.is_dir():
        raise ValueError("annotation root must be an existing directory")
    for filename in SPLITS.values():
        if not (annotation_root / filename).is_file():
            raise ValueError(f"missing annotation file: {filename}")
    if not clip_manifest.is_file():
        raise ValueError("CLIP manifest must be an existing file")
    clips = annotations(annotation_root, clip_manifest)  # metadata only; scan journals PNG failures
    if only is not None:
        split, clip_id = only
        if split not in clips or clip_id not in clips[split]:
            raise ValueError(f"selected clip missing from annotations/CLIP manifest: {split}/{clip_id}")
        if not (frame_root / split / clip_id).is_dir():
            raise ValueError(f"{split}/{clip_id}: missing source directory")
    return frame_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output-base", type=Path)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--clip-manifest", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-access", action="store_true")
    mode.add_argument("--one-clip", nargs=2, metavar=("SPLIT", "CLIP_ID"))
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--confirm-full-extraction", action="store_true")
    args = parser.parse_args()
    if args.all and not args.confirm_full_extraction:
        parser.error("--all requires --confirm-full-extraction and explicit user authorization")
    if not args.check_access and not all((args.cache, args.output_base,
                                          args.annotation_root, args.clip_manifest)):
        parser.error("extraction requires cache, output-base, annotation-root, clip-manifest")
    if args.check_access and args.cache is None:
        parser.error("access check requires --cache on external filesystem")
    checkout = Path(__file__).resolve().parents[2]
    verify_location(args.cache, checkout)
    if args.output_base is not None:
        verify_location(args.output_base, checkout)
    only = tuple(args.one_clip) if args.one_clip else None
    frame_root = None if args.check_access else _preflight_inputs(
        args.annotation_root, args.clip_manifest, only)
    token = read_token(args.env_file)
    sha = resolve_sha(token, args.revision)
    try:
        hf_hub_download(repo_id=MODEL, filename="config.json", revision=sha,
                        token=token, cache_dir=args.cache)
    except Exception:
        raise RuntimeError("DINOv3 gated config inaccessible; check model license and HF_TOKEN") from None
    if args.check_access:
        print(f"DINOv3 config accessible at revision {sha}")
        return
    lock = Path(__file__).resolve().parent / "uv.lock"
    key, metadata = identity(sha, lock)
    target = args.output_base / key
    if target.is_symlink() or target.resolve(strict=False).parent != args.output_base.resolve(
        strict=False
    ):
        raise ValueError("encoder key root must be an immediate child of output base, not a symlink")
    if not (target / "complete").exists():
        with writer(target):
            probe = target / ".write-probe.npy"
            try:
                atomic_array(probe, np.zeros((5, 2048), dtype=np.float32))
            finally:
                probe.unlink(missing_ok=True)
    if (target / "complete").exists():
        model, device = None, "cpu"  # run() validates completed version without encoding
    else:
        try:
            available = torch.cuda.is_available()
        except Exception:
            raise RuntimeError("DINOv3 CUDA initialization failed; check driver/runtime") from None
        if not available:
            raise RuntimeError("live DINOv3 extraction requires CUDA")
        try:
            torch.empty((1,), device="cuda")  # isolate driver failure before gated weights
        except Exception:
            raise RuntimeError("DINOv3 CUDA initialization failed; check driver/runtime") from None
        model, device = load_model(sha, token, args.cache, "cuda"), "cuda"
    stats = run(frame_root, args.annotation_root,
                args.clip_manifest, target, key, metadata, model, device, only=only)
    print(json.dumps(dict(encoder_key=key, model_sha=sha, **stats), sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests -q -k 'not live'` (expect offline unit tests pass). Run `uv run --project tools/dinov3 --locked python -m tools.dinov3.cli --help` (expect access/one-clip/all modes). `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_cli.py -q` (includes input preflight, symlink, process-env and CUDA initialization regression cases). Run root static-tool-only `uv run --locked ruff check tools/dinov3` (no extractor imports by baseline Python).
- [ ] **Step 5: Write optional real tests** (`tools/dinov3/tests/test_live.py` and `tests/integration/test_dinov3_baseline.py`):

```python
# tools/dinov3/tests/test_live.py
import os
from pathlib import Path

import numpy as np
import pytest

from tools.dinov3.encoder import extract, load_model
from tools.dinov3.identity import read_token, resolve_sha


@pytest.mark.integration
@pytest.mark.gpu
def test_one_gated_frame_cls_is_real_and_finite():
    if os.environ.get("DESPAMO_RUN_DINO_LIVE") != "1":
        pytest.skip("opt in after gate and pinned CUDA lock")
    token = read_token(Path(os.environ["DESPAMO_DINO_ENV_FILE"]))
    sha = resolve_sha(token, os.environ["DESPAMO_DINO_SHA"])
    assert sha == os.environ["DESPAMO_DINO_SHA"]
    model = load_model(sha, token, Path(os.environ["DESPAMO_DINO_CACHE"]), "cuda")
    image = Path(os.environ["PHOENIX14T_FRAME_ROOT"]) / os.environ["DESPAMO_DINO_FRAME"]
    feature, resolution = extract([image], model, "cuda")
    assert feature.shape == (1, 2048) and feature.dtype == np.float32
    assert np.isfinite(feature).all() and resolution == [[210, 260]]
```

```python
# tests/integration/test_dinov3_baseline.py
import os
from collections import Counter
from pathlib import Path

import pytest

from despamo.config import load_config
from despamo.data.manifest import FeatureManifest
from despamo.factory import build_data


@pytest.mark.integration
def test_external_dino_manifest_loads_without_new_transformers(tmp_path, monkeypatch):
    if os.environ.get("DESPAMO_RUN_DINO_BASELINE") != "1":
        pytest.skip("opt in only after full approved extraction")
    root = Path(os.environ["DESPAMO_DINO_ROOT"])
    monkeypatch.setenv("DESPAMO_HF_CACHE", str(tmp_path / "synthetic-hf-cache"))
    manifest = FeatureManifest.load(root / "complete/manifest.json")
    assert Counter(r.split for r in manifest.records) == {"train": 7096,
                                                           "dev": 519, "test": 642}
    configs = Path(__file__).resolve().parents[2] / "configs"
    config = load_config([configs / "data/phoenix14t.yaml",
                          configs / "model/spamo_flan_t5_xl.yaml",
                          configs / "experiment/phoenix14t_baseline.yaml"],
                         [f"data.spatial_root={root}",
                          f"data.spatial_manifest={root / 'complete/manifest.json'}",
                          "model.spatial_crop_mode=full"])
    assert config.model.cache_dir == str(tmp_path / "synthetic-hf-cache")
    batch = next(iter(build_data(config).test_dataloader()))
    assert batch.spatial.shape[-1] == 2048 and batch.motion.shape[-1] == 1024
```

- [ ] **Step 6: Small live gate only after access approval.** In shell set paths below (model cache/output must be Linux-native); no command prints token:

```bash
export PHOENIX14T_FRAME_ROOT=/mnt/e/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px
export DESPAMO_DINO_ENV_FILE=/home/kan/Research/DeSpaMo/.env
export DESPAMO_DINO_CACHE=/home/kan/.cache/huggingface
export DINO_OUTPUT=/home/kan/dinov3-features
: "${PHOENIX14T_ANNOTATION_ROOT:?set accepted baseline annotation root}"
: "${DESPAMO_FEATURE_ROOT:?set accepted baseline feature root}"
export DESPAMO_DINO_SHA="$(uv run --project tools/dinov3 --locked python -c 'from pathlib import Path; from tools.dinov3.identity import read_token,resolve_sha; print(resolve_sha(read_token(Path("/home/kan/Research/DeSpaMo/.env")), "main"))')"
uv run --project tools/dinov3 --locked python -m tools.dinov3.cli --env-file "$DESPAMO_DINO_ENV_FILE" --revision "$DESPAMO_DINO_SHA" --cache "$DESPAMO_DINO_CACHE" --check-access
```

Expect `config accessible at revision` followed by exact 40-hex SHA; 401 => **DONE_WITH_CONCERNS**, stop live gate, retain passing fake tests. Determine an existing train clip/frame without guessing a filename:

```bash
export DESPAMO_DINO_CLIP="$(uv run --project tools/dinov3 --locked python -c 'import json,os; from pathlib import Path; path=Path(os.environ["DESPAMO_FEATURE_ROOT"])/"manifests/phoenix14t_spatial.json"; records=json.loads(path.read_text())["records"]; print(next(r["clip_id"] for r in records if r["split"]=="train"))')"
export DESPAMO_DINO_FRAME="$(uv run --project tools/dinov3 --locked python -c 'import os; from pathlib import Path; root=Path(os.environ["PHOENIX14T_FRAME_ROOT"]); clip=os.environ["DESPAMO_DINO_CLIP"]; print(sorted((root/"train"/clip).glob("*.png"))[0].relative_to(root).as_posix())')"
export DESPAMO_RUN_DINO_LIVE=1
uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_live.py -q
uv run --project tools/dinov3 --locked python -m tools.dinov3.cli --env-file "$DESPAMO_DINO_ENV_FILE" --revision "$DESPAMO_DINO_SHA" --cache "$DESPAMO_DINO_CACHE" --annotation-root "$PHOENIX14T_ANNOTATION_ROOT" --clip-manifest "$DESPAMO_FEATURE_ROOT/manifests/phoenix14t_spatial.json" --output-base "$DINO_OUTPUT" --one-clip train "$DESPAMO_DINO_CLIP"
```

Expect **1 passed**, skip is NOT success; bounded CLI prints `"complete": false`, one receipt and `[T,2048]` array, no final manifests. Verify annotation/feature paths exist before launching live model. Do not run `--all` here.
- [ ] **Step 7: Authorized post-extraction gate (document only; execute later).** With separately authorized full corpus job, run:

```bash
uv run --project tools/dinov3 --locked python -m tools.dinov3.cli --env-file "$DESPAMO_DINO_ENV_FILE" --revision "$DESPAMO_DINO_SHA" --cache "$DESPAMO_DINO_CACHE" --annotation-root "$PHOENIX14T_ANNOTATION_ROOT" --clip-manifest "$DESPAMO_FEATURE_ROOT/manifests/phoenix14t_spatial.json" --output-base "$DINO_OUTPUT" --all --confirm-full-extraction
export DESPAMO_DINO_ROOT="$DINO_OUTPUT/$(uv run --project tools/dinov3 --locked python -c 'from pathlib import Path; import os; from tools.dinov3.identity import identity; print(identity(os.environ["DESPAMO_DINO_SHA"], Path("tools/dinov3/uv.lock"))[0])')"
export DESPAMO_RUN_DINO_BASELINE=1
DESPAMO_HF_CACHE="$DINO_OUTPUT/synthetic-hf-cache" uv run --locked python -m pytest tests/integration/test_dinov3_baseline.py -q
```

Expect `completed=8257`, `complete=true`, counts `{train:7096,dev:519,test:642}`, empty `failures.json`, and `complete/manifest.json` plus `complete/frame_rows.json` with matching hash; baseline test **1 passed**, skip not success. Re-run `uv run --project tools/dinov3 --locked python -m pytest tools/dinov3/tests -q -k 'not live'`, root `uv run --locked python -m pytest tests/unit -q` (baseline must not collect extractor tests), `git status --short`, `git diff --check`, `git diff -- tools/dinov3 tests/integration/test_dinov3_baseline.py`; baseline `pyproject.toml` and root `uv.lock` unchanged.

## Self-Review And Handoff

- Contracts: `FeatureManifest` schema 1 exact keys `schema_version`, `encoder`, `expected_dim`, `records[{clip_id,split,path,length,width,dtype}]`; `FeatureRecord.path` is relative to encoder-key root. `build_data` reads `data.spatial_root` at encoder-key root and `data.spatial_manifest` at `root/complete/manifest.json`; `factors.frame_rows` points to `root/complete/frame_rows.json`, with exact factor consumer keys. `spatial_manifest_hash` hashes staged/published manifest bytes, not semantic JSON. For five sampled frames, `source_indices` is full sorted 0-based row mapping; sampled image digests derive from original PNG bytes. Baseline consumer uses synthetic `DESPAMO_HF_CACHE` to satisfy config resolution without loading weights.
- Review gates: completed rerun validates without modifying any file; joint staging rename prevents half-published JSON; encoder RuntimeError journals once and stops; source errors remain clip-specific; malformed receipts include split/clip context; source hashes revalidated after resume and immediately before atomic promotion. Checkout/cache rejection occurs before network, directory creation, or model load. Extractor unit/live tests remain under `tools/dinov3/tests/`; root `tests/` imports only baseline stack. Task 5's count/source/corruption/publication failure tests are written **before** implementation; Step 2 requires observing red.
- Spec scope: model gate/lock and credentials Task 1/6; annotation/CLIP parity and exact float32 bicubic preprocessing Task 2; official CLS extraction Task 3; single-writer/atomic resume Task 4; zero-failure and complete split gate Task 5; live one-clip and baseline consumer Task 6. Controlled CLIP-vs-DINO training belongs to separate comparison plan. No source code, dependency installs, extraction, or commits occur while authoring this document.
- Live result status: if gated model remains HTTP 401, report `DONE_WITH_CONCERNS` and specific blocked live gate; offline fake-encoder tests remain runnable after implementation.
