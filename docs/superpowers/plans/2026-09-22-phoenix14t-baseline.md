# PHOENIX14T Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone DeSpaMo baseline that smoke-trains on PHOENIX14T and evaluates the released SpaMo checkpoint over all 642 test clips.

**Architecture:** Port only SpaMo's PHOENIX14T data path, spatial-motion adapter, VT-Align objective, and Flan-T5-XL LoRA integration. Keep external feature arrays and checkpoints outside Git, validate them through manifests, and isolate upstream-compatible behavior behind explicit configuration modes.

**Tech Stack:** Python 3.11, uv, PyTorch 2.0.1, PyTorch Lightning 1.9.5, Transformers 4.32.0, PEFT 0.7.1, OmegaConf 2.3.0, NumPy 1.26.4, SacreBLEU 2.2.0, rouge-score 0.1.2, pytest, Ruff.

## Global Constraints

- Source references: `/home/kan/Research/SpaMo` and `/home/kan/Research/DIFFER`.
- Production code must not import either source repository.
- PHOENIX14T baseline uses `google/flan-t5-xl` with LoRA rank 16, alpha 32, dropout 0.1, targeting `q` and `v`.
- Spatial inputs are CLIP ViT-L/14 S2 features with width 2048; motion inputs are VideoMAE-L/16 features with width 1024.
- Visual fusion concatenates valid spatial and motion tokens along time before `K5-P2-K5-P2` temporal convolution.
- External datasets, features, checkpoints, model caches, and run artifacts stay outside Git.
- Missing, empty, malformed, or dimensionally incompatible features must fail before model execution.
- Checkpoint loading is strict. Converted tensors must map exactly once.
- Upstream-compatible pooling and sampling remain explicit modes; corrected behavior must not silently change baseline evaluation.
- Use TDD for every behavioral task: failing test, observed failure, minimal implementation, passing test.
- Do not commit unless user explicitly requests commits. Commit snippets below are checkpoints, not authorization.

## File Map

- `.python-version`: select Python 3.11.
- `.gitignore`: protect secrets, local data, caches, checkpoints, and generated runs.
- `.env.example`: document required local path variables without secrets.
- `pyproject.toml`: package metadata, locked runtime dependencies, test and lint configuration.
- `src/despamo/config.py`: merge and validate OmegaConf files.
- `src/despamo/data/manifest.py`: index and validate offline feature files.
- `src/despamo/data/phoenix14t.py`: load PHOENIX annotations and paired features.
- `src/despamo/data/batch.py`: typed batch and variable-length collation.
- `src/despamo/data/datamodule.py`: Lightning dataloaders.
- `src/despamo/models/temporal.py`: SpaMo-compatible `K5-P2-K5-P2` encoder.
- `src/despamo/models/visual_adapter.py`: spatial and motion projection, temporal fusion, multimodal projection.
- `src/despamo/models/prompts.py`: language prompt and in-context example construction.
- `src/despamo/models/flan_t5.py`: Flan-T5-XL loading, LoRA, loss, target embeddings, and generation.
- `src/despamo/losses/vt_align.py`: legacy and mask-aware symmetric contrastive alignment.
- `src/despamo/training/baseline_module.py`: Lightning orchestration and optimizer.
- `src/despamo/evaluation/metrics.py`: BLEU-1 through BLEU-4 and ROUGE-L.
- `src/despamo/evaluation/artifact.py`: reproducibility and result JSON.
- `src/despamo/checkpoints.py`: released-checkpoint conversion and strict validation.
- `src/despamo/factory.py`: construct data and model objects from validated config.
- `scripts/index_features.py`: create spatial and motion manifests.
- `scripts/convert_spamo_checkpoint.py`: convert released Lightning checkpoint.
- `scripts/train.py`: run bounded or full baseline training.
- `scripts/evaluate.py`: run complete checkpoint evaluation.
- `configs/data/phoenix14t.yaml`: annotation and external feature locations.
- `configs/model/spamo_flan_t5_xl.yaml`: exact baseline model settings.
- `configs/experiment/phoenix14t_baseline.yaml`: baseline training and evaluation settings.
- `configs/experiment/phoenix14t_smoke.yaml`: bounded smoke overrides.
- `configs/local.example.yaml`: documented machine-local overrides.
- `tests/unit`: fast behavior tests without model downloads or GPU.
- `tests/parity`: comparisons against SpaMo source modules and released checkpoint schema.
- `tests/integration`: real-feature and optional GPU/model tests.

---

### Task 1: Secure Package Foundation

**Files:**
- Modify: `.python-version`
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Create: `.env.example`
- Create: `src/despamo/__init__.py`
- Create: `tests/unit/test_package.py`

**Interfaces:**
- Produces: importable package `despamo` with `__version__ == "0.1.0"`.
- Produces: `uv.lock` containing exact dependency resolution.

- [ ] **Step 1: Write failing package test**

```python
# tests/unit/test_package.py
import despamo


def test_package_exposes_version() -> None:
    assert despamo.__version__ == "0.1.0"
```

- [ ] **Step 2: Verify test fails before package exists**

Run: `uv run --with pytest pytest tests/unit/test_package.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'despamo'`.

- [ ] **Step 3: Replace scaffold metadata and create package**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "despamo"
version = "0.1.0"
description = "Signer-invariant gloss-free sign language translation"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
  "numpy==1.26.4",
  "omegaconf==2.3.0",
  "peft==0.7.1",
  "pytorch-lightning==1.9.5",
  "rouge-score==0.1.2",
  "sacrebleu==2.2.0",
  "setuptools>=65,<81",
  "torch==2.0.1",
  "torchmetrics==1.8.2",
  "transformers==4.32.0",
]

[dependency-groups]
dev = [
  "pytest>=8.3,<9",
  "ruff>=0.11,<1",
]

[tool.hatch.build.targets.wheel]
packages = ["src/despamo"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "integration: requires local PHOENIX14T data or model artifacts",
  "gpu: requires CUDA",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Lightning 1.9.5 imports `pkg_resources` via `lightning_fabric`; setuptools 81+ omits it, so pin setuptools below 81 at runtime.

```python
# src/despamo/__init__.py
__version__ = "0.1.0"
```

```text
# .python-version
3.11
```

```dotenv
# .env.example
PHOENIX14T_ANNOTATION_ROOT=/path/to/SpaMo/preprocess/Phoenix14T
DESPAMO_FEATURE_ROOT=/path/to/spamo/features
DESPAMO_CHECKPOINT=/path/to/spamo.ckpt
DESPAMO_HF_CACHE=/path/to/hf_cache
```

Append these entries to `.gitignore`:

```gitignore
.env
.env.*
!.env.example
.venv/
.pytest_cache/
.ruff_cache/
__pycache__/
*.py[cod]
*.ckpt
*.pt
*.pth
*.npy
*.npz
artifacts/
logs/
outputs/
configs/local.yaml
```

- [ ] **Step 4: Lock environment and run package test**

Run: `uv lock && uv sync --group dev && uv run pytest tests/unit/test_package.py -v`

Expected: lock succeeds and test reports `1 passed`.

- [ ] **Step 5: Run initial lint**

Run: `uv run ruff check src tests`

Expected: `All checks passed!`

- [ ] **Step 6: Checkpoint commit if explicitly authorized**

```bash
git add .python-version .gitignore .env.example pyproject.toml uv.lock src/despamo/__init__.py tests/unit/test_package.py
git commit -m "build: initialize DeSpaMo package"
```

---

### Task 2: Validated Layered Configuration

**Files:**
- Create: `src/despamo/config.py`
- Create: `configs/data/phoenix14t.yaml`
- Create: `configs/model/spamo_flan_t5_xl.yaml`
- Create: `configs/experiment/phoenix14t_baseline.yaml`
- Create: `configs/experiment/phoenix14t_smoke.yaml`
- Create: `configs/local.example.yaml`
- Create: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `load_config(paths: Sequence[Path], overrides: Sequence[str] = ()) -> DictConfig`.
- Produces: `validate_baseline_config(config: DictConfig) -> None`.
- Consumes: environment variables documented in `.env.example`.

- [ ] **Step 1: Write failing config tests**

```python
# tests/unit/test_config.py
from pathlib import Path

import pytest

from despamo.config import load_config, validate_baseline_config


def test_load_config_merges_override(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text("model:\n  spatial_dim: 2048\ntrainer:\n  max_epochs: 40\n")
    override.write_text("trainer:\n  max_epochs: 1\n")

    config = load_config([base, override])

    assert config.model.spatial_dim == 2048
    assert config.trainer.max_epochs == 1


def test_validate_rejects_wrong_baseline_width() -> None:
    config = load_config([], ["model.spatial_dim=1024", "model.motion_dim=1024"])

    with pytest.raises(ValueError, match="spatial_dim must be 2048"):
        validate_baseline_config(config)
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/unit/test_config.py -v`

Expected: FAIL because `despamo.config` does not exist.

- [ ] **Step 3: Implement merge and validation**

```python
# src/despamo/config.py
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


def load_config(
    paths: Sequence[Path], overrides: Sequence[str] = ()
) -> DictConfig:
    configs = [OmegaConf.load(path) for path in paths]
    if overrides:
        configs.append(OmegaConf.from_dotlist(list(overrides)))
    merged = OmegaConf.merge(*configs) if configs else OmegaConf.create()
    OmegaConf.resolve(merged)
    return merged


def validate_baseline_config(config: DictConfig) -> None:
    spatial_dim = OmegaConf.select(config, "model.spatial_dim")
    motion_dim = OmegaConf.select(config, "model.motion_dim")
    if spatial_dim != 2048:
        raise ValueError(f"model.spatial_dim must be 2048, got {spatial_dim}")
    if motion_dim != 1024:
        raise ValueError(f"model.motion_dim must be 1024, got {motion_dim}")
    mode = OmegaConf.select(config, "model.vt_pooling", default="legacy_mean")
    if mode not in {"legacy_mean", "masked_mean"}:
        raise ValueError(f"unsupported model.vt_pooling: {mode}")
    generation = OmegaConf.select(config, "evaluation.generation", default="upstream")
    if generation not in {"upstream", "deterministic"}:
        raise ValueError(f"unsupported evaluation.generation: {generation}")
```

- [ ] **Step 4: Add baseline YAML layers**

```yaml
# configs/data/phoenix14t.yaml
data:
  annotation_root: ${oc.env:PHOENIX14T_ANNOTATION_ROOT}
  spatial_root: ${oc.env:DESPAMO_FEATURE_ROOT}/vit_feat_Phoenix14T
  motion_root: ${oc.env:DESPAMO_FEATURE_ROOT}/mae_feat_Phoenix14T
  spatial_manifest: ${oc.env:DESPAMO_FEATURE_ROOT}/manifests/phoenix14t_spatial.json
  motion_manifest: ${oc.env:DESPAMO_FEATURE_ROOT}/manifests/phoenix14t_motion.json
  batch_size: 2
  num_workers: 0
```

```yaml
# configs/model/spamo_flan_t5_xl.yaml
model:
  name: google/flan-t5-xl
  cache_dir: ${oc.env:DESPAMO_HF_CACHE}
  spatial_dim: 2048
  motion_dim: 1024
  adapter_dim: 768
  language_dim: 2048
  max_text_length: 64
  prompt: Translate the given sentence into {}.
  use_in_context: true
  num_in_context: 3
  vt_pooling: legacy_mean
  vt_weight: 1.0
  warm_up_steps: 0
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1
```

```yaml
# configs/experiment/phoenix14t_baseline.yaml
seed: 0
trainer:
  accelerator: gpu
  devices: 1
  precision: bf16
  default_root_dir: artifacts/phoenix14t_baseline
  max_epochs: 500
  accumulate_grad_batches: 2
  gradient_clip_val: 1.0
  check_val_every_n_epoch: 2
optimizer:
  learning_rate: 6.0e-4
  weight_decay: 0.01
evaluation:
  generation: upstream
  beam_size: 5
  expected_test_items: 642
  output_dir: artifacts/phoenix14t_baseline
```

```yaml
# configs/experiment/phoenix14t_smoke.yaml
model:
  tuning_type: freeze
  use_in_context: false
  num_in_context: 0
  warm_up_steps: null
trainer:
  max_epochs: 1
  limit_train_batches: 0.01
  limit_val_batches: 1
  num_sanity_val_steps: 0
  accumulate_grad_batches: 1
  check_val_every_n_epoch: 1
```

```yaml
# configs/local.example.yaml
evaluation:
  checkpoint: ${oc.env:DESPAMO_CHECKPOINT}
```

- [ ] **Step 5: Run config tests**

Run: `uv run pytest tests/unit/test_config.py -v`

Expected: `2 passed`.

- [ ] **Step 6: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/config.py configs tests/unit/test_config.py
git commit -m "feat: add validated experiment config"
```

---

### Task 3: Offline Feature Manifests

**Files:**
- Create: `src/despamo/data/__init__.py`
- Create: `src/despamo/data/manifest.py`
- Create: `scripts/index_features.py`
- Create: `tests/unit/data/test_manifest.py`

**Interfaces:**
- Produces: immutable `FeatureRecord` and `FeatureManifest` dataclasses.
- Produces: `index_feature_tree(root: Path, expected_dim: int, encoder: str) -> FeatureManifest`.
- Produces: `FeatureManifest.require(split: str, clip_id: str) -> FeatureRecord`.

- [ ] **Step 1: Write failing manifest tests**

```python
# tests/unit/data/test_manifest.py
from pathlib import Path

import numpy as np
import pytest

from despamo.data.manifest import FeatureManifest, index_feature_tree


def test_index_feature_tree_records_shape(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    np.save(split / "clip-1.npy", np.zeros((7, 2048), dtype=np.float32))

    manifest = index_feature_tree(tmp_path, 2048, "openai/clip-vit-large-patch14")

    record = manifest.require("train", "clip-1")
    assert record.length == 7
    assert record.width == 2048
    assert record.path == "train/clip-1.npy"


def test_index_feature_tree_rejects_wrong_width(tmp_path: Path) -> None:
    split = tmp_path / "test"
    split.mkdir()
    np.save(split / "bad.npy", np.zeros((3, 1024), dtype=np.float32))

    with pytest.raises(ValueError, match="expected width 2048"):
        index_feature_tree(tmp_path, 2048, "clip")
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/unit/data/test_manifest.py -v`

Expected: FAIL because manifest module does not exist.

- [ ] **Step 3: Implement manifest indexing and JSON serialization**

```python
# src/despamo/data/manifest.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureRecord:
    clip_id: str
    split: str
    path: str
    length: int
    width: int
    dtype: str


@dataclass(frozen=True)
class FeatureManifest:
    schema_version: int
    encoder: str
    expected_dim: int
    records: tuple[FeatureRecord, ...]

    def require(self, split: str, clip_id: str) -> FeatureRecord:
        matches = [r for r in self.records if r.split == split and r.clip_id == clip_id]
        if len(matches) != 1:
            raise KeyError(f"expected one feature for {split}/{clip_id}, found {len(matches)}")
        return matches[0]

    def save(self, path: Path) -> None:
        payload = asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> FeatureManifest:
        payload = json.loads(path.read_text())
        records = tuple(FeatureRecord(**record) for record in payload.pop("records"))
        return cls(records=records, **payload)


def index_feature_tree(root: Path, expected_dim: int, encoder: str) -> FeatureManifest:
    records: list[FeatureRecord] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(root.glob("*/*.npy")):
        split = path.parent.name
        clip_id = path.stem
        key = (split, clip_id)
        if key in seen:
            raise ValueError(f"duplicate feature: {split}/{clip_id}")
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError(f"feature must be non-empty rank 2: {path}")
        if array.shape[1] != expected_dim:
            raise ValueError(
                f"expected width {expected_dim}, got {array.shape[1]}: {path}"
            )
        records.append(
            FeatureRecord(
                clip_id=clip_id,
                split=split,
                path=str(path.relative_to(root)),
                length=int(array.shape[0]),
                width=int(array.shape[1]),
                dtype=str(array.dtype),
            )
        )
        seen.add(key)
    if not records:
        raise ValueError(f"no .npy feature files found under {root}")
    return FeatureManifest(1, encoder, expected_dim, tuple(records))
```

```python
# scripts/index_features.py
import argparse
from pathlib import Path

from despamo.data.manifest import index_feature_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-dim", type=int, required=True)
    parser.add_argument("--encoder", required=True)
    args = parser.parse_args()
    manifest = index_feature_tree(args.root, args.expected_dim, args.encoder)
    manifest.save(args.output)
    print(f"indexed {len(manifest.records)} features into {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run manifest tests**

Run: `uv run pytest tests/unit/data/test_manifest.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Index existing SpaMo features**

Run:

```bash
mkdir -p /home/kan/datasets/spamo/features/manifests
uv run python scripts/index_features.py --root /home/kan/datasets/spamo/features/vit_feat_Phoenix14T --output /home/kan/datasets/spamo/features/manifests/phoenix14t_spatial.json --expected-dim 2048 --encoder openai/clip-vit-large-patch14-s2-1-2 --expected-count train=7096 --expected-count dev=519 --expected-count test=642
uv run python scripts/index_features.py --root /home/kan/datasets/spamo/features/mae_feat_Phoenix14T --output /home/kan/datasets/spamo/features/manifests/phoenix14t_motion.json --expected-dim 1024 --encoder MCG-NJU/videomae-large --expected-count train=7096 --expected-count dev=519 --expected-count test=642
```

Expected: spatial and motion manifests are created outside Git; indexing reports 8,257 records for each complete feature corpus: 7,096 train, 519 dev, 642 test.

- [ ] **Step 6: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/data scripts/index_features.py tests/unit/data/test_manifest.py
git commit -m "feat: validate offline feature manifests"
```

---

### Task 4: PHOENIX14T Dataset And Collation

**Files:**
- Create: `src/despamo/data/batch.py`
- Create: `src/despamo/data/phoenix14t.py`
- Create: `tests/unit/data/test_phoenix14t.py`
- Create: `tests/unit/data/test_batch.py`

**Interfaces:**
- Produces: `PhoenixSample` containing metadata plus spatial and motion tensors.
- Produces: `PhoenixBatch` containing padded tensors, boolean masks, text, signer, and multilingual context.
- Produces: `collate_phoenix(samples: Sequence[PhoenixSample]) -> PhoenixBatch`.

- [ ] **Step 1: Write failing dataset and collation tests**

```python
# tests/unit/data/test_phoenix14t.py
from pathlib import Path

import numpy as np

from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.data.phoenix14t import Phoenix14T


def test_dataset_loads_integer_annotation_records(tmp_path: Path) -> None:
    annotation = tmp_path / "train_info_ml.npy"
    np.save(
        annotation,
        {
            "prefix": "ignored",
            0: {
                "fileid": "clip-1",
                "signer": "Signer01",
                "gloss": "WIND",
                "text": "es bleibt windig",
                "en_text": "It remains windy.",
                "es_text": "Permanece ventoso.",
                "fr_text": "Il reste venteux.",
            },
        },
    )
    spatial_root = tmp_path / "spatial"
    motion_root = tmp_path / "motion"
    (spatial_root / "train").mkdir(parents=True)
    (motion_root / "train").mkdir(parents=True)
    np.save(spatial_root / "train/clip-1.npy", np.ones((20, 2048), dtype=np.float32))
    np.save(motion_root / "train/clip-1.npy", np.ones((4, 1024), dtype=np.float32))
    spatial_manifest = FeatureManifest(
        1, "clip", 2048, (FeatureRecord("clip-1", "train", "train/clip-1.npy", 20, 2048, "float32"),)
    )
    motion_manifest = FeatureManifest(
        1, "mae", 1024, (FeatureRecord("clip-1", "train", "train/clip-1.npy", 4, 1024, "float32"),)
    )

    dataset = Phoenix14T(
        annotation, "train", spatial_root, motion_root, spatial_manifest, motion_manifest
    )

    sample = dataset[0]
    assert len(dataset) == 1
    assert sample.clip_id == "clip-1"
    assert sample.signer == "Signer01"
    assert sample.text == "es bleibt windig."
    assert sample.spatial.shape == (20, 2048)
    assert sample.motion.shape == (4, 1024)
```

```python
# tests/unit/data/test_batch.py
import torch

from despamo.data.batch import PhoenixSample, collate_phoenix


def test_collate_builds_boolean_masks() -> None:
    samples = [
        PhoenixSample("a", "s1", "a.", "A", "e", "s", "f", torch.ones(20, 2048), torch.ones(4, 1024)),
        PhoenixSample("b", "s2", "b.", "B", "e", "s", "f", torch.ones(24, 2048), torch.ones(6, 1024)),
    ]

    batch = collate_phoenix(samples)

    assert batch.spatial.shape == (2, 24, 2048)
    assert batch.motion.shape == (2, 6, 1024)
    assert batch.spatial_mask.sum(dim=1).tolist() == [20, 24]
    assert batch.motion_mask.sum(dim=1).tolist() == [4, 6]
    assert batch.spatial_mask.dtype == torch.bool
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/unit/data/test_phoenix14t.py tests/unit/data/test_batch.py -v`

Expected: FAIL because dataset and batch modules do not exist.

- [ ] **Step 3: Implement typed sample and collation**

```python
# src/despamo/data/batch.py
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class PhoenixSample:
    clip_id: str
    signer: str
    text: str
    gloss: str
    en_text: str
    es_text: str
    fr_text: str
    spatial: torch.Tensor
    motion: torch.Tensor


@dataclass(frozen=True)
class PhoenixBatch:
    clip_ids: tuple[str, ...]
    signers: tuple[str, ...]
    texts: tuple[str, ...]
    glosses: tuple[str, ...]
    en_texts: tuple[str, ...]
    es_texts: tuple[str, ...]
    fr_texts: tuple[str, ...]
    spatial: torch.Tensor
    spatial_mask: torch.Tensor
    motion: torch.Tensor
    motion_mask: torch.Tensor


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    return torch.arange(max_length)[None, :] < lengths[:, None]


def collate_phoenix(samples: Sequence[PhoenixSample]) -> PhoenixBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    spatial_lengths = torch.tensor([sample.spatial.shape[0] for sample in samples])
    motion_lengths = torch.tensor([sample.motion.shape[0] for sample in samples])
    spatial = pad_sequence([sample.spatial for sample in samples], batch_first=True)
    motion = pad_sequence([sample.motion for sample in samples], batch_first=True)
    return PhoenixBatch(
        clip_ids=tuple(sample.clip_id for sample in samples),
        signers=tuple(sample.signer for sample in samples),
        texts=tuple(sample.text for sample in samples),
        glosses=tuple(sample.gloss for sample in samples),
        en_texts=tuple(sample.en_text for sample in samples),
        es_texts=tuple(sample.es_text for sample in samples),
        fr_texts=tuple(sample.fr_text for sample in samples),
        spatial=spatial,
        spatial_mask=lengths_to_mask(spatial_lengths, spatial.shape[1]),
        motion=motion,
        motion_mask=lengths_to_mask(motion_lengths, motion.shape[1]),
    )
```

- [ ] **Step 4: Implement strict PHOENIX loader**

```python
# src/despamo/data/phoenix14t.py
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from despamo.data.batch import PhoenixSample
from despamo.data.manifest import FeatureManifest


def normalize_text(text: str) -> str:
    stripped = text.strip().lower()
    return stripped if stripped.endswith(".") else f"{stripped}."


class Phoenix14T(Dataset[PhoenixSample]):
    def __init__(
        self,
        annotation_path: Path,
        split: str,
        spatial_root: Path,
        motion_root: Path,
        spatial_manifest: FeatureManifest,
        motion_manifest: FeatureManifest,
    ) -> None:
        raw = np.load(annotation_path, allow_pickle=True).item()
        self.records = [raw[key] for key in sorted(key for key in raw if isinstance(key, int))]
        self.split = split
        self.spatial_root = spatial_root
        self.motion_root = motion_root
        self.spatial_manifest = spatial_manifest
        self.motion_manifest = motion_manifest

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PhoenixSample:
        item = self.records[index]
        clip_id = item["fileid"]
        spatial_record = self.spatial_manifest.require(self.split, clip_id)
        motion_record = self.motion_manifest.require(self.split, clip_id)
        spatial = np.load(self.spatial_root / spatial_record.path)
        motion = np.load(self.motion_root / motion_record.path)
        if spatial.shape != (spatial_record.length, spatial_record.width):
            raise ValueError(f"spatial manifest mismatch for {clip_id}: {spatial.shape}")
        if motion.shape != (motion_record.length, motion_record.width):
            raise ValueError(f"motion manifest mismatch for {clip_id}: {motion.shape}")
        return PhoenixSample(
            clip_id=clip_id,
            signer=item["signer"],
            text=normalize_text(item["text"]),
            gloss=item["gloss"],
            en_text=item["en_text"],
            es_text=item["es_text"],
            fr_text=item["fr_text"],
            spatial=torch.from_numpy(spatial.astype(np.float32, copy=False)),
            motion=torch.from_numpy(motion.astype(np.float32, copy=False)),
        )
```

- [ ] **Step 5: Run dataset tests**

Run: `uv run pytest tests/unit/data/test_phoenix14t.py tests/unit/data/test_batch.py -v`

Expected: `2 passed`.

- [ ] **Step 6: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/data tests/unit/data
git commit -m "feat: load PHOENIX14T feature pairs"
```

---

### Task 5: SpaMo-Compatible Visual Adapter

**Files:**
- Create: `src/despamo/models/__init__.py`
- Create: `src/despamo/models/temporal.py`
- Create: `src/despamo/models/visual_adapter.py`
- Create: `tests/unit/models/test_temporal.py`
- Create: `tests/unit/models/test_visual_adapter.py`
- Create: `tests/parity/test_spamo_visual_adapter.py`

**Interfaces:**
- Produces: `TemporalConv.forward(features: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]`.
- Produces: `SpaMoVisualAdapter.forward(spatial, spatial_mask, motion, motion_mask) -> tuple[visual_tokens, visual_mask]`.
- Output visual tokens have shape `(batch, output_time, 2048)`.

- [ ] **Step 1: Write failing temporal length test**

```python
# tests/unit/models/test_temporal.py
import torch

from despamo.models.temporal import TemporalConv


def test_temporal_conv_updates_lengths() -> None:
    encoder = TemporalConv(768, 768)
    features = torch.randn(2, 768, 24)
    output, lengths = encoder(features, torch.tensor([20, 24]))

    assert output.shape == (2, 3, 768)
    assert lengths.tolist() == [2, 3]
```

- [ ] **Step 2: Verify temporal test fails**

Run: `uv run pytest tests/unit/models/test_temporal.py -v`

Expected: FAIL because temporal module does not exist.

- [ ] **Step 3: Implement exact temporal stack**

```python
# src/despamo/models/temporal.py
import torch
from torch import nn


class TemporalConv(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=5),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths - 4
        lengths = torch.div(lengths, 2, rounding_mode="floor")
        lengths = lengths - 4
        return torch.div(lengths, 2, rounding_mode="floor")

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.any(lengths < 16):
            raise ValueError(f"all fused sequences must contain at least 16 tokens: {lengths}")
        output = self.temporal_conv(features).transpose(1, 2)
        return output, self.output_lengths(lengths)
```

- [ ] **Step 4: Write failing visual adapter test**

```python
# tests/unit/models/test_visual_adapter.py
import io

import torch

from despamo.models.visual_adapter import SpaMoVisualAdapter


def test_visual_adapter_fuses_only_valid_tokens() -> None:
    adapter = SpaMoVisualAdapter(2048, 1024, 768, 2048)
    spatial = torch.randn(2, 20, 2048)
    motion = torch.randn(2, 4, 1024)
    spatial_mask = torch.ones(2, 20, dtype=torch.bool)
    motion_mask = torch.ones(2, 4, dtype=torch.bool)

    tokens, mask = adapter(spatial, spatial_mask, motion, motion_mask)

    assert tokens.shape == (2, 3, 2048)
    assert mask.shape == (2, 3)
    assert mask.all()


def test_visual_adapter_checkpoint_round_trip() -> None:
    torch.manual_seed(3)
    source = SpaMoVisualAdapter(2048, 1024, 768, 2048).eval()
    spatial = torch.randn(1, 20, 2048)
    motion = torch.randn(1, 4, 1024)
    spatial_mask = torch.ones(1, 20, dtype=torch.bool)
    motion_mask = torch.ones(1, 4, dtype=torch.bool)
    expected, _ = source(spatial, spatial_mask, motion, motion_mask)
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    restored = SpaMoVisualAdapter(2048, 1024, 768, 2048).eval()
    restored.load_state_dict(torch.load(buffer), strict=True)

    actual, _ = restored(spatial, spatial_mask, motion, motion_mask)

    torch.testing.assert_close(actual, expected)
```

- [ ] **Step 5: Implement visual adapter**

```python
# src/despamo/models/visual_adapter.py
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from despamo.models.temporal import TemporalConv


def lengths_to_mask(lengths: torch.Tensor, max_length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(max_length, device=device)[None, :] < lengths[:, None]


class SpaMoVisualAdapter(nn.Module):
    def __init__(
        self, spatial_dim: int, motion_dim: int, adapter_dim: int, language_dim: int
    ) -> None:
        super().__init__()
        self.spatial_projector = nn.Linear(spatial_dim, adapter_dim)
        self.motion_projector = nn.Linear(motion_dim, adapter_dim)
        self.temporal_encoder = TemporalConv(adapter_dim, adapter_dim)
        self.multimodal_projector = nn.Sequential(
            nn.Linear(adapter_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        spatial_mask: torch.Tensor,
        motion: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = self.spatial_projector(spatial)
        motion = self.motion_projector(motion)
        fused = [
            torch.cat((spatial[i, spatial_mask[i]], motion[i, motion_mask[i]]), dim=0)
            for i in range(spatial.shape[0])
        ]
        lengths = torch.tensor([item.shape[0] for item in fused], device=spatial.device)
        padded = pad_sequence(fused, batch_first=True)
        encoded, output_lengths = self.temporal_encoder(padded.transpose(1, 2), lengths)
        tokens = self.multimodal_projector(encoded)
        mask = lengths_to_mask(output_lengths, tokens.shape[1], tokens.device)
        return tokens, mask
```

- [ ] **Step 6: Run unit tests**

Run: `uv run pytest tests/unit/models/test_temporal.py tests/unit/models/test_visual_adapter.py -v`

Expected: `3 passed`.

- [ ] **Step 7: Add source parity test**

```python
# tests/parity/test_spamo_visual_adapter.py
import os
import sys

import pytest
import torch

from despamo.models.visual_adapter import SpaMoVisualAdapter


@pytest.mark.integration
def test_visual_weights_and_temporal_outputs_match_spamo() -> None:
    source = os.environ.get("SPAMO_PROJECT_PATH")
    if source is None:
        pytest.skip("SPAMO_PROJECT_PATH is not configured")
    sys.path.insert(0, source)
    from spamo.tconv import TemporalConv as SourceTemporalConv

    torch.manual_seed(7)
    target = SpaMoVisualAdapter(2048, 1024, 768, 2048).eval()
    source_temporal = SourceTemporalConv(768, 768).eval()
    source_temporal.load_state_dict(target.temporal_encoder.state_dict())
    features = torch.randn(2, 768, 24)
    lengths = torch.tensor([20, 24])

    target_output, target_lengths = target.temporal_encoder(features, lengths)
    source_result = source_temporal(features, lengths)

    torch.testing.assert_close(target_output.transpose(0, 1), source_result["visual_feat"])
    assert target_lengths.tolist() == source_result["feat_len"].to(torch.int).tolist()
```

- [ ] **Step 8: Run parity test with source path**

Run: `SPAMO_PROJECT_PATH=/home/kan/Research/SpaMo uv run pytest tests/parity/test_spamo_visual_adapter.py -v`

Expected: `1 passed`.

- [ ] **Step 9: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/models tests/unit/models tests/parity/test_spamo_visual_adapter.py
git commit -m "feat: port SpaMo visual adapter"
```

---

### Task 6: VT-Align Loss With Compatibility Modes

**Files:**
- Create: `src/despamo/losses/__init__.py`
- Create: `src/despamo/losses/vt_align.py`
- Create: `tests/unit/losses/test_vt_align.py`

**Interfaces:**
- Produces: `pooled_sequence(features, mask, mode) -> Tensor`.
- Produces: `VTAlignLoss.forward(visual, visual_mask, text, text_mask, mode) -> Tensor`.
- State key used by checkpoint conversion: `vt_align.logit_scale`.

- [ ] **Step 1: Write failing loss tests**

```python
# tests/unit/losses/test_vt_align.py
import torch

from despamo.losses.vt_align import VTAlignLoss, pooled_sequence


def test_masked_mean_ignores_padding() -> None:
    features = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    pooled = pooled_sequence(features, mask, "masked_mean")
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 0.0]]))


def test_vt_align_is_symmetric() -> None:
    loss_fn = VTAlignLoss(initial_logit_scale=0.0)
    visual = torch.eye(2).unsqueeze(1)
    text = torch.eye(2).unsqueeze(1)
    mask = torch.ones(2, 1, dtype=torch.bool)
    loss = loss_fn(visual, mask, text, mask, "masked_mean")
    expected = torch.nn.functional.cross_entropy(torch.eye(2), torch.arange(2))
    torch.testing.assert_close(loss, expected)
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/unit/losses/test_vt_align.py -v`

Expected: FAIL because loss module does not exist.

- [ ] **Step 3: Implement pooling and symmetric contrastive loss**

```python
# src/despamo/losses/vt_align.py
import torch
from torch import nn
from torch.nn import functional as F


def pooled_sequence(
    features: torch.Tensor, mask: torch.Tensor, mode: str
) -> torch.Tensor:
    if mode == "legacy_mean":
        return features.mean(dim=1)
    if mode == "masked_mean":
        weights = mask.unsqueeze(-1).to(features.dtype)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    raise ValueError(f"unsupported pooling mode: {mode}")


class VTAlignLoss(nn.Module):
    def __init__(self, initial_logit_scale: float = 2.6592) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(initial_logit_scale))

    def forward(
        self,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        visual_embedding = F.normalize(pooled_sequence(visual, visual_mask, mode), dim=-1)
        text_embedding = F.normalize(pooled_sequence(text, text_mask, mode), dim=-1)
        logits = text_embedding @ visual_embedding.T * self.logit_scale.exp()
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        ) / 2
```

- [ ] **Step 4: Run loss tests**

Run: `uv run pytest tests/unit/losses/test_vt_align.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/losses tests/unit/losses
git commit -m "feat: add VT-Align compatibility modes"
```

---

### Task 7: Prompt Construction And Flan-T5 Boundary

**Files:**
- Create: `src/despamo/models/prompts.py`
- Create: `src/despamo/models/flan_t5.py`
- Create: `tests/unit/models/test_prompts.py`
- Create: `tests/unit/models/test_flan_t5.py`

**Interfaces:**
- Produces: `build_prompts(batch, template, use_in_context, num_in_context, rng) -> list[str]`.
- Produces: `FlanT5Backbone.target_embeddings(texts) -> tuple[Tensor, Tensor]`.
- Produces: `FlanT5Backbone.translation_loss(visual, visual_mask, prompts, targets) -> Tensor`.
- Produces: `FlanT5Backbone.generate_text(visual, visual_mask, prompts, mode, beam_size, max_length) -> list[str]`.

- [ ] **Step 1: Write failing prompt tests**

```python
# tests/unit/models/test_prompts.py
import random

from despamo.data.batch import PhoenixBatch
from despamo.models.prompts import build_prompts


def test_in_context_prompts_use_other_sample() -> None:
    batch = PhoenixBatch(
        clip_ids=("a", "b"),
        signers=("s1", "s2"),
        texts=("eins.", "zwei."),
        glosses=("A", "B"),
        en_texts=("one", "two"),
        es_texts=("uno", "dos"),
        fr_texts=("un", "deux"),
        spatial=None,
        spatial_mask=None,
        motion=None,
        motion_mask=None,
    )

    prompts = build_prompts(
        batch, "Translate the given sentence into {}.", True, 3, random.Random(0)
    )

    assert prompts[0].startswith("Translate the given sentence into German.")
    assert "two=zwei." in prompts[0]
    assert "one=eins." in prompts[1]
```

- [ ] **Step 2: Implement bounded derangement and prompts**

```python
# src/despamo/models/prompts.py
import random

from despamo.data.batch import PhoenixBatch


def derange(values: list[str], rng: random.Random) -> list[str]:
    if len(values) < 2:
        raise ValueError("in-context prompting requires batch size >= 2")
    for _ in range(1000):
        shuffled = values.copy()
        rng.shuffle(shuffled)
        if all(original != candidate for original, candidate in zip(values, shuffled)):
            return shuffled
    raise ValueError("cannot derange duplicate in-context examples")


def build_prompts(
    batch: PhoenixBatch,
    template: str,
    use_in_context: bool,
    num_in_context: int,
    rng: random.Random,
) -> list[str]:
    prompts = [template.format("German") for _ in batch.texts]
    if not use_in_context:
        return prompts
    examples = [
        " ".join(
            (
                f"{batch.en_texts[i]}={batch.texts[i]}",
                f"{batch.fr_texts[i]}={batch.texts[i]}",
                f"{batch.es_texts[i]}={batch.texts[i]}",
            )[:num_in_context]
        )
        for i in range(len(batch.texts))
    ]
    return [f"{prompt} {example}" for prompt, example in zip(prompts, derange(examples, rng))]
```

- [ ] **Step 3: Run prompt test**

Run: `uv run pytest tests/unit/models/test_prompts.py -v`

Expected: `1 passed`.

- [ ] **Step 4: Write Flan boundary test using fakes**

```python
# tests/unit/models/test_flan_t5.py
import torch
from torch import nn

from despamo.models.flan_t5 import FlanT5Backbone


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, padding, return_tensors, truncation=False, max_length=None):
        width = max(len(text.split()) for text in texts)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, text in enumerate(texts):
            size = len(text.split())
            ids[row, :size] = torch.arange(1, size + 1)
            mask[row, :size] = 1
        return {"input_ids": ids, "attention_mask": mask}


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.encoder = type("Encoder", (), {"embed_tokens": self.embedding})()


def test_target_embeddings_return_mask() -> None:
    backbone = FlanT5Backbone(FakeModel(), FakeTokenizer(), max_text_length=8)
    embeddings, mask = backbone.target_embeddings(["one two", "three"])
    assert embeddings.shape == (2, 2, 4)
    assert mask.tolist() == [[True, True], [True, False]]


def test_force_eval_keeps_frozen_model_in_eval_mode() -> None:
    backbone = FlanT5Backbone(
        FakeModel(), FakeTokenizer(), max_text_length=8, force_eval=True
    )
    backbone.train()
    assert backbone.training
    assert not backbone.model.training
```

- [ ] **Step 5: Implement Flan-T5 loading and target embeddings first**

```python
# src/despamo/models/flan_t5.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, T5ForConditionalGeneration


@dataclass(frozen=True)
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class FlanT5Backbone(nn.Module):
    def __init__(
        self, model: nn.Module, tokenizer, max_text_length: int, force_eval: bool = False
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.force_eval = force_eval

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        cache_dir: str,
        max_text_length: int,
        tuning_type: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> FlanT5Backbone:
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, max_length=max_text_length
        )
        if tuning_type == "lora":
            model = get_peft_model(
                model,
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=["q", "v"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_2_SEQ_LM,
                ),
            )
        elif tuning_type == "freeze":
            model.requires_grad_(False)
            model.eval()
        else:
            raise ValueError(f"unsupported tuning_type: {tuning_type}")
        return cls(model, tokenizer, max_text_length, force_eval=tuning_type == "freeze")

    def train(self, mode: bool = True) -> "FlanT5Backbone":
        super().train(mode)
        if self.force_eval:
            self.model.eval()
        return self

    def _tokenize(
        self, texts: list[str] | tuple[str, ...], truncate: bool
    ) -> TokenBatch:
        encoded = self.tokenizer(
            list(texts),
            padding="longest",
            truncation=truncate,
            return_tensors="pt",
        )
        return TokenBatch(encoded["input_ids"], encoded["attention_mask"].bool())

    def target_embeddings(
        self, texts: list[str] | tuple[str, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokenize(texts, truncate=False)
        device = next(self.model.parameters()).device
        input_ids = tokens.input_ids.to(device)
        mask = tokens.attention_mask.to(device)
        return self.model.encoder.embed_tokens(input_ids), mask

    def _joint_inputs(
        self,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
        prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_tokens = self._tokenize(prompts, truncate=True)
        prompt_ids = prompt_tokens.input_ids.to(visual.device)
        prompt_mask = prompt_tokens.attention_mask.to(visual.device)
        prompt_embeddings = self.model.encoder.embed_tokens(prompt_ids)
        samples = [
            torch.cat((visual[i, visual_mask[i]], prompt_embeddings[i, prompt_mask[i]]), dim=0)
            for i in range(visual.shape[0])
        ]
        lengths = torch.tensor([sample.shape[0] for sample in samples], device=visual.device)
        joint = pad_sequence(samples, batch_first=True)
        mask = torch.arange(joint.shape[1], device=visual.device)[None, :] < lengths[:, None]
        return joint, mask

    def translation_loss(
        self,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
        prompts: list[str],
        targets: tuple[str, ...],
    ) -> torch.Tensor:
        joint, joint_mask = self._joint_inputs(visual, visual_mask, prompts)
        target_tokens = self._tokenize(targets, truncate=False)
        labels = target_tokens.input_ids.to(visual.device)
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        output = self.model(
            inputs_embeds=joint,
            attention_mask=joint_mask,
            decoder_attention_mask=target_tokens.attention_mask.to(visual.device),
            labels=labels,
            return_dict=True,
        )
        return output.loss

    @torch.no_grad()
    def generate_text(
        self,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
        prompts: list[str],
        mode: str,
        beam_size: int,
        max_length: int,
    ) -> list[str]:
        joint, joint_mask = self._joint_inputs(visual, visual_mask, prompts)
        if mode not in {"upstream", "deterministic"}:
            raise ValueError(f"unsupported generation mode: {mode}")
        generation_kwargs = {
            "inputs_embeds": joint,
            "attention_mask": joint_mask,
            "num_beams": beam_size,
            "max_length": max_length,
            "do_sample": mode == "upstream",
        }
        if mode == "upstream":
            generation_kwargs["top_p"] = 0.9
        generated = self.model.generate(
            **generation_kwargs,
        )
        return [text.lower() for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]
```

- [ ] **Step 6: Run model boundary tests**

Run: `uv run pytest tests/unit/models/test_prompts.py tests/unit/models/test_flan_t5.py -v`

Expected: `3 passed` without downloading Flan-T5.

- [ ] **Step 7: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/models tests/unit/models
git commit -m "feat: add Flan-T5 LoRA boundary"
```

---

### Task 8: Baseline Lightning Training Module

**Files:**
- Create: `src/despamo/training/__init__.py`
- Create: `src/despamo/training/baseline_module.py`
- Create: `tests/unit/training/test_baseline_module.py`

**Interfaces:**
- Produces: `SpaMoBaselineModule` with state prefixes `visual_adapter`, `language_model`, and `vt_align`.
- Consumes: `PhoenixBatch`, `SpaMoVisualAdapter`, `FlanT5Backbone`, and `VTAlignLoss`.
- Produces logs: `train/loss`, `train/contra_loss`, and `train/combined_loss`.

- [ ] **Step 1: Write failing warm-up behavior test with fakes**

```python
# tests/unit/training/test_baseline_module.py
import pytest
import torch

from despamo.training.baseline_module import combine_losses, ensure_finite_losses


def test_combine_losses_uses_only_vt_during_warmup() -> None:
    translation = torch.tensor(4.0)
    vt = torch.tensor(2.0)
    assert combine_losses(translation, vt, global_step=0, warm_up_steps=10, vt_weight=0.5) == 2.0
    assert combine_losses(translation, vt, global_step=11, warm_up_steps=10, vt_weight=0.5) == 5.0


def test_non_finite_loss_reports_clip_ids() -> None:
    with pytest.raises(FloatingPointError, match="clip-a"):
        ensure_finite_losses({"combined_loss": torch.tensor(float("nan"))}, ("clip-a",))
```

- [ ] **Step 2: Verify test fails**

Run: `uv run pytest tests/unit/training/test_baseline_module.py -v`

Expected: FAIL because training module does not exist.

- [ ] **Step 3: Implement loss schedule and Lightning module**

```python
# src/despamo/training/baseline_module.py
import random

import pytorch_lightning as pl
import torch

from despamo.data.batch import PhoenixBatch
from despamo.losses.vt_align import VTAlignLoss
from despamo.models.flan_t5 import FlanT5Backbone
from despamo.models.prompts import build_prompts
from despamo.models.visual_adapter import SpaMoVisualAdapter


def combine_losses(
    translation_loss: torch.Tensor,
    vt_loss: torch.Tensor,
    global_step: int,
    warm_up_steps: int | None,
    vt_weight: float,
) -> torch.Tensor:
    if warm_up_steps is not None and global_step <= warm_up_steps:
        return vt_loss
    return translation_loss + vt_weight * vt_loss


def ensure_finite_losses(
    losses: dict[str, torch.Tensor], clip_ids: tuple[str, ...]
) -> None:
    invalid = [name for name, value in losses.items() if not torch.isfinite(value).all()]
    if invalid:
        raise FloatingPointError(
            f"non-finite losses {invalid} for clips {list(clip_ids)}"
        )


class SpaMoBaselineModule(pl.LightningModule):
    def __init__(
        self,
        visual_adapter: SpaMoVisualAdapter,
        language_model: FlanT5Backbone,
        vt_align: VTAlignLoss,
        prompt_template: str,
        use_in_context: bool,
        num_in_context: int,
        vt_pooling: str,
        vt_weight: float,
        warm_up_steps: int | None,
        learning_rate: float,
        weight_decay: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.visual_adapter = visual_adapter
        self.language_model = language_model
        self.vt_align = vt_align
        self.prompt_template = prompt_template
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.vt_pooling = vt_pooling
        self.vt_weight = vt_weight
        self.warm_up_steps = warm_up_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rng = random.Random(seed)

    def compute_losses(self, batch: PhoenixBatch) -> dict[str, torch.Tensor]:
        visual, visual_mask = self.visual_adapter(
            batch.spatial.to(self.device),
            batch.spatial_mask.to(self.device),
            batch.motion.to(self.device),
            batch.motion_mask.to(self.device),
        )
        text_embeddings, text_mask = self.language_model.target_embeddings(batch.texts)
        vt_loss = self.vt_align(
            visual, visual_mask, text_embeddings, text_mask, self.vt_pooling
        )
        prompts = build_prompts(
            batch,
            self.prompt_template,
            self.use_in_context,
            self.num_in_context,
            self.rng,
        )
        translation_loss = self.language_model.translation_loss(
            visual, visual_mask, prompts, batch.texts
        )
        total = combine_losses(
            translation_loss,
            vt_loss,
            self.global_step,
            self.warm_up_steps,
            self.vt_weight,
        )
        losses = {"loss": translation_loss, "contra_loss": vt_loss, "combined_loss": total}
        ensure_finite_losses(losses, batch.clip_ids)
        return losses

    def training_step(self, batch: PhoenixBatch, batch_idx: int) -> torch.Tensor:
        losses = self.compute_losses(batch)
        self.log_dict({f"train/{key}": value for key, value in losses.items()}, batch_size=len(batch.texts))
        return losses["combined_loss"]

    def validation_step(self, batch: PhoenixBatch, batch_idx: int) -> None:
        losses = self.compute_losses(batch)
        self.log_dict({f"val/{key}": value for key, value in losses.items()}, batch_size=len(batch.texts))

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
```

- [ ] **Step 4: Run training module tests**

Run: `uv run pytest tests/unit/training/test_baseline_module.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/training tests/unit/training
git commit -m "feat: orchestrate SpaMo baseline losses"
```

---

### Task 9: Translation Metrics And Result Artifacts

**Files:**
- Create: `src/despamo/evaluation/__init__.py`
- Create: `src/despamo/evaluation/metrics.py`
- Create: `src/despamo/evaluation/artifact.py`
- Create: `src/despamo/utils/__init__.py`
- Create: `src/despamo/utils/hashing.py`
- Create: `tests/unit/evaluation/test_metrics.py`
- Create: `tests/unit/evaluation/test_artifact.py`

**Interfaces:**
- Produces: `evaluate_translations(predictions, references) -> dict[str, float]`.
- Produces: `write_result_artifact(path, clip_ids, predictions, references, metrics, metadata) -> None`.
- Produces: `collect_runtime_metadata(seed, package_lock, spatial_manifest, motion_manifest) -> dict[str, Any]`.
- Artifact contains exactly one prediction and reference per clip ID.

- [ ] **Step 1: Write failing metric and artifact tests**

```python
# tests/unit/evaluation/test_metrics.py
from despamo.evaluation.metrics import evaluate_translations


def test_exact_predictions_score_perfect_rouge() -> None:
    metrics = evaluate_translations(["es bleibt windig"], ["es bleibt windig"])
    assert metrics["rougeL_f1"] == 1.0
    assert metrics["bleu1"] == 100.0
```

```python
# tests/unit/evaluation/test_artifact.py
import json
from pathlib import Path

from despamo.evaluation.artifact import collect_runtime_metadata, write_result_artifact


def test_artifact_records_items_and_metadata(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    write_result_artifact(
        output,
        ["clip-1"],
        ["prediction"],
        ["reference"],
        {"bleu4": 1.5},
        {"generation": "deterministic"},
    )
    payload = json.loads(output.read_text())
    assert payload["items"][0]["clip_id"] == "clip-1"
    assert payload["metadata"]["generation"] == "deterministic"


def test_runtime_metadata_hashes_reproducibility_inputs(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    spatial = tmp_path / "spatial.json"
    motion = tmp_path / "motion.json"
    lock.write_text("lock")
    spatial.write_text("spatial")
    motion.write_text("motion")

    metadata = collect_runtime_metadata(7, lock, spatial, motion)

    assert metadata["seed"] == 7
    assert len(metadata["package_lock_sha256"]) == 64
    assert len(metadata["spatial_manifest_sha256"]) == 64
    assert len(metadata["motion_manifest_sha256"]) == 64
    assert metadata["git_revision"]
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/unit/evaluation -v`

Expected: FAIL because evaluation modules do not exist.

- [ ] **Step 3: Implement metrics and artifact writer**

```python
# src/despamo/evaluation/metrics.py
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


def evaluate_translations(
    predictions: list[str], references: list[str], tokenizer: str = "13a"
) -> dict[str, float]:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("predictions and references must have equal non-zero length")
    metrics = {
        f"bleu{order}": BLEU(max_ngram_order=order, tokenize=tokenizer)
        .corpus_score(predictions, [references])
        .score
        for order in range(1, 5)
    }
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"] for ref, pred in zip(references, predictions)]
    metrics["rougeL_precision"] = sum(score.precision for score in scores) / len(scores)
    metrics["rougeL_recall"] = sum(score.recall for score in scores) / len(scores)
    metrics["rougeL_f1"] = sum(score.fmeasure for score in scores) / len(scores)
    return metrics
```

```python
# src/despamo/utils/hashing.py
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

```python
# src/despamo/evaluation/artifact.py
import json
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
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
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
    if not (len(clip_ids) == len(predictions) == len(references)):
        raise ValueError("clip_ids, predictions, and references must have equal length")
    payload = {
        "metadata": metadata,
        "metrics": metrics,
        "items": [
            {"clip_id": clip_id, "prediction": prediction, "reference": reference}
            for clip_id, prediction, reference in zip(clip_ids, predictions, references)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run evaluation tests**

Run: `uv run pytest tests/unit/evaluation -v`

Expected: `3 passed`.

- [ ] **Step 5: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/evaluation src/despamo/utils tests/unit/evaluation
git commit -m "feat: record translation evaluation artifacts"
```

---

### Task 10: Released SpaMo Checkpoint Conversion

**Files:**
- Create: `src/despamo/checkpoints.py`
- Create: `scripts/convert_spamo_checkpoint.py`
- Create: `scripts/export_model_schema.py`
- Create: `tests/unit/test_checkpoints.py`
- Create: `tests/unit/test_checkpoint_schema.py`
- Create: `tests/unit/test_export_model_schema.py`
- Create: `tests/parity/test_released_checkpoint_schema.py`

**Interfaces:**
- Produces: `convert_spamo_state_dict(source: Mapping[str, Tensor]) -> dict[str, Tensor]`.
- Produces target prefixes: `visual_adapter`, `language_model.model`, and `vt_align`.
- Produces standalone checkpoint with `state_dict`, source SHA-256, and target-schema SHA-256 metadata; requires strict target `state_dict()` key/shape/dtype validation.

- [ ] **Step 1: Write failing key-mapping tests**

```python
# tests/unit/test_checkpoints.py
import pytest
import torch

from despamo.checkpoints import convert_spamo_state_dict


def test_checkpoint_key_mapping() -> None:
    source = {
        "logit_scale": torch.tensor(1.0),
        "spatio_proj.weight": torch.zeros(768, 2048),
        "spatiotemp_proj.weight": torch.zeros(768, 1024),
        "fusion_proj.0.weight": torch.zeros(2048, 768),
        "temporal_encoder.temporal_conv.0.weight": torch.zeros(768, 768, 5),
        "t5_model.shared.weight": torch.zeros(2, 2),
    }
    converted = convert_spamo_state_dict(source)
    assert "vt_align.logit_scale" in converted
    assert "visual_adapter.spatial_projector.weight" in converted
    assert "visual_adapter.motion_projector.weight" in converted
    assert "visual_adapter.multimodal_projector.0.weight" in converted
    assert "visual_adapter.temporal_encoder.temporal_conv.0.weight" in converted
    assert "language_model.model.shared.weight" in converted


def test_checkpoint_mapping_rejects_unknown_key() -> None:
    with pytest.raises(KeyError, match="unmapped SpaMo key"):
        convert_spamo_state_dict({"unknown.weight": torch.zeros(1)})
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/unit/test_checkpoints.py -v`

Expected: FAIL because checkpoint module does not exist.

- [ ] **Step 3: Implement exhaustive prefix mapping and checksum**

```python
# src/despamo/checkpoints.py
from collections.abc import Mapping

import torch


PREFIXES = (
    ("spatio_proj.", "visual_adapter.spatial_projector."),
    ("spatiotemp_proj.", "visual_adapter.motion_projector."),
    ("fusion_proj.", "visual_adapter.multimodal_projector."),
    ("temporal_encoder.", "visual_adapter.temporal_encoder."),
    ("t5_model.", "language_model.model."),
)


def convert_spamo_state_dict(
    source: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if key == "logit_scale":
            target = "vt_align.logit_scale"
        else:
            matches = [(old, new) for old, new in PREFIXES if key.startswith(old)]
            if len(matches) != 1:
                raise KeyError(f"unmapped SpaMo key: {key}")
            old, new = matches[0]
            target = new + key[len(old) :]
        if target in converted:
            raise KeyError(f"checkpoint mapping collision: {target}")
        converted[target] = value
    return converted
```

```python
# scripts/convert_spamo_checkpoint.py
import argparse
from pathlib import Path

import torch

from despamo.checkpoints import convert_spamo_state_dict
from despamo.utils.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = torch.load(args.input, map_location="cpu")
    converted = convert_spamo_state_dict(source["state_dict"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": converted,
            "metadata": {
                "source_path": str(args.input),
                "source_sha256": sha256_file(args.input),
                "source_tensor_count": len(source["state_dict"]),
                "target_tensor_count": len(converted),
            },
        },
        args.output,
    )
    print(f"converted {len(converted)} tensors into {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run key-mapping tests**

Run: `uv run pytest tests/unit/test_checkpoints.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Add released schema parity test**

```python
# tests/parity/test_released_checkpoint_schema.py
import os
from pathlib import Path

import pytest
import torch

from despamo.checkpoints import convert_spamo_state_dict


@pytest.mark.integration
def test_released_checkpoint_maps_all_871_tensors() -> None:
    checkpoint = os.environ.get("DESPAMO_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("DESPAMO_CHECKPOINT is not configured")
    source = torch.load(Path(checkpoint), map_location="meta")["state_dict"]
    converted = convert_spamo_state_dict(source)
    assert len(source) == 871
    assert len(converted) == 871
```

- [ ] **Step 6: Run schema parity test**

Run: `DESPAMO_CHECKPOINT=/home/kan/datasets/spamo/ckpt/spamo.ckpt uv run pytest tests/parity/test_released_checkpoint_schema.py -v`

Expected: `1 passed` and all 871 tensors map exactly once.

- [ ] **Step 7: Export actual target schema, then validate and convert released checkpoint outside Git**

After PHOENIX transfer finishes and local Flan-T5-XL weights are available, set `DESPAMO_HF_CACHE` to their cache location. Export schema from actual target `build_model(config).state_dict()`; run conversion only after inspecting exported schema:

```bash
uv run python scripts/export_model_schema.py --config configs/model/spamo_flan_t5_xl.yaml --config configs/experiment/phoenix14t_baseline.yaml --output /home/kan/datasets/spamo/ckpt/despamo-target-schema.json
uv run python scripts/convert_spamo_checkpoint.py --input /home/kan/datasets/spamo/ckpt/spamo.ckpt --target-schema /home/kan/datasets/spamo/ckpt/despamo-target-schema.json --output /home/kan/datasets/spamo/ckpt/despamo-spamo-baseline.pt
```

Expected: output reports `converted 871 tensors` only if every converted key, shape, and dtype exactly matches exported target schema; records source and schema SHA-256. Real 5.9 GB parity/conversion and Flan model schema export remain deferred during PHOENIX download.

- [ ] **Step 8: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/checkpoints.py scripts/convert_spamo_checkpoint.py scripts/export_model_schema.py tests/unit/test_checkpoints.py tests/unit/test_checkpoint_schema.py tests/unit/test_export_model_schema.py tests/parity/test_released_checkpoint_schema.py
git commit -m "feat: convert released SpaMo checkpoint"
```

---

### Task 11: DataModule, Factories, And Training CLI

**Files:**
- Create: `src/despamo/data/datamodule.py`
- Create: `src/despamo/factory.py`
- Create: `scripts/train.py`
- Create: `tests/unit/data/test_datamodule.py`
- Create: `tests/unit/test_factory.py`
- Create: `tests/integration/test_real_phoenix_batch.py`

**Interfaces:**
- Produces: `PhoenixDataModule` with train, validation, and test loaders.
- Produces: `build_data(config) -> PhoenixDataModule`.
- Produces: `build_model(config) -> SpaMoBaselineModule`.
- Training CLI merges data, model, experiment, smoke, and local configs in order.

- [ ] **Step 1: Write failing datamodule test**

```python
# tests/unit/data/test_datamodule.py
from torch.utils.data import Dataset

from despamo.data.datamodule import PhoenixDataModule


class TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return index


def test_datamodule_uses_configured_batch_size() -> None:
    module = PhoenixDataModule(TinyDataset(), TinyDataset(), TinyDataset(), batch_size=2, num_workers=0, collate_fn=list)
    assert len(next(iter(module.train_dataloader()))) == 2
```

- [ ] **Step 2: Implement datamodule**

```python
# src/despamo/data/datamodule.py
from collections.abc import Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class PhoenixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train: Dataset,
        validation: Dataset,
        test: Dataset,
        batch_size: int,
        num_workers: int,
        collate_fn: Callable,
    ) -> None:
        super().__init__()
        self.train_dataset = train
        self.validation_dataset = validation
        self.test_dataset = test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.validation_dataset, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, False)
```

- [ ] **Step 3: Implement explicit factories**

```python
# src/despamo/factory.py
from pathlib import Path

from omegaconf import DictConfig

from despamo.data.batch import collate_phoenix
from despamo.data.datamodule import PhoenixDataModule
from despamo.data.manifest import FeatureManifest
from despamo.data.phoenix14t import Phoenix14T
from despamo.losses.vt_align import VTAlignLoss
from despamo.models.flan_t5 import FlanT5Backbone
from despamo.models.visual_adapter import SpaMoVisualAdapter
from despamo.training.baseline_module import SpaMoBaselineModule


def build_data(config: DictConfig) -> PhoenixDataModule:
    spatial_manifest = FeatureManifest.load(Path(config.data.spatial_manifest))
    motion_manifest = FeatureManifest.load(Path(config.data.motion_manifest))
    datasets = {}
    for name, annotation_name in (("train", "train_info_ml.npy"), ("validation", "dev_info_ml.npy"), ("test", "test_info_ml.npy")):
        split = "dev" if name == "validation" else name
        datasets[name] = Phoenix14T(
            Path(config.data.annotation_root) / annotation_name,
            split,
            Path(config.data.spatial_root),
            Path(config.data.motion_root),
            spatial_manifest,
            motion_manifest,
        )
    return PhoenixDataModule(
        datasets["train"],
        datasets["validation"],
        datasets["test"],
        config.data.batch_size,
        config.data.num_workers,
        collate_phoenix,
    )


def build_model(config: DictConfig) -> SpaMoBaselineModule:
    language_model = FlanT5Backbone.from_pretrained(
        config.model.name,
        config.model.cache_dir,
        config.model.max_text_length,
        config.model.get("tuning_type", "lora"),
        config.model.lora_rank,
        config.model.lora_alpha,
        config.model.lora_dropout,
    )
    return SpaMoBaselineModule(
        visual_adapter=SpaMoVisualAdapter(
            config.model.spatial_dim,
            config.model.motion_dim,
            config.model.adapter_dim,
            config.model.language_dim,
        ),
        language_model=language_model,
        vt_align=VTAlignLoss(),
        prompt_template=config.model.prompt,
        use_in_context=config.model.use_in_context,
        num_in_context=config.model.num_in_context,
        vt_pooling=config.model.vt_pooling,
        vt_weight=config.model.vt_weight,
        warm_up_steps=config.model.warm_up_steps,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        seed=config.seed,
    )
```

```python
# scripts/train.py
import argparse
from pathlib import Path

import pytorch_lightning as pl

from despamo.config import load_config, validate_baseline_config
from despamo.factory import build_data, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    validate_baseline_config(config)
    pl.seed_everything(config.seed, workers=True)
    trainer = pl.Trainer(**dict(config.trainer))
    trainer.fit(build_model(config), datamodule=build_data(config))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add factory test without model downloads**

```python
# tests/unit/test_factory.py
from omegaconf import OmegaConf
from torch import nn

from despamo.factory import build_model
from despamo.models.flan_t5 import FlanT5Backbone


class DummyLanguageModel(nn.Module):
    pass


def test_build_model_uses_configured_visual_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        FlanT5Backbone,
        "from_pretrained",
        lambda *args, **kwargs: DummyLanguageModel(),
    )
    config = OmegaConf.create(
        {
            "seed": 0,
            "model": {
                "name": "google/flan-t5-xl",
                "cache_dir": "/tmp/cache",
                "spatial_dim": 2048,
                "motion_dim": 1024,
                "adapter_dim": 768,
                "language_dim": 2048,
                "max_text_length": 64,
                "tuning_type": "freeze",
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "prompt": "Translate the given sentence into {}.",
                "use_in_context": False,
                "num_in_context": 0,
                "vt_pooling": "legacy_mean",
                "vt_weight": 1.0,
                "warm_up_steps": None,
            },
            "optimizer": {"learning_rate": 6.0e-4, "weight_decay": 0.01},
        }
    )

    model = build_model(config)

    assert model.visual_adapter.spatial_projector.in_features == 2048
    assert model.visual_adapter.motion_projector.in_features == 1024
    assert model.visual_adapter.multimodal_projector[-1].out_features == 2048
```

- [ ] **Step 5: Run datamodule and factory tests**

Run: `uv run pytest tests/unit/data/test_datamodule.py tests/unit/test_factory.py -v`

Expected: `2 passed` without downloading Flan-T5.

- [ ] **Step 6: Add real feature integration test**

```python
# tests/integration/test_real_phoenix_batch.py
import os
from pathlib import Path

import pytest

from despamo.config import load_config
from despamo.factory import build_data


@pytest.mark.integration
def test_real_test_batch_has_expected_widths() -> None:
    required = ("PHOENIX14T_ANNOTATION_ROOT", "DESPAMO_FEATURE_ROOT")
    if any(name not in os.environ for name in required):
        pytest.skip("local PHOENIX14T paths are not configured")
    config = load_config(
        [Path("configs/data/phoenix14t.yaml"), Path("configs/model/spamo_flan_t5_xl.yaml"), Path("configs/experiment/phoenix14t_baseline.yaml")]
    )
    batch = next(iter(build_data(config).test_dataloader()))
    assert batch.spatial.shape[0] == 2
    assert batch.spatial.shape[2] == 2048
    assert batch.motion.shape[2] == 1024
```

- [ ] **Step 7: Run real data integration test**

Run:

```bash
PHOENIX14T_ANNOTATION_ROOT=/home/kan/Research/SpaMo/preprocess/Phoenix14T DESPAMO_FEATURE_ROOT=/home/kan/datasets/spamo/features DESPAMO_HF_CACHE=/home/kan/datasets/spamo/hf_cache uv run pytest tests/integration/test_real_phoenix_batch.py -v
```

Expected: `1 passed` without loading Flan-T5.

- [ ] **Step 8: Checkpoint commit if explicitly authorized**

```bash
git add src/despamo/data/datamodule.py src/despamo/factory.py scripts/train.py tests
git commit -m "feat: wire baseline training pipeline"
```

---

### Task 12: Smoke Training Gate

**Files:**
- Modify: `README.md`
- Create: `tests/integration/test_smoke_config.py`

**Interfaces:**
- Consumes all baseline configs and local artifacts.
- Produces at least one optimizer step and one reloadable Lightning checkpoint.

- [ ] **Step 1: Write smoke-config test**

```python
# tests/integration/test_smoke_config.py
from pathlib import Path

from despamo.config import load_config


def test_smoke_config_is_bounded() -> None:
    config = load_config(
        [
            Path("configs/data/phoenix14t.yaml"),
            Path("configs/model/spamo_flan_t5_xl.yaml"),
            Path("configs/experiment/phoenix14t_baseline.yaml"),
            Path("configs/experiment/phoenix14t_smoke.yaml"),
        ]
    )
    assert config.trainer.max_epochs == 1
    assert config.trainer.limit_train_batches == 0.01
    assert config.trainer.limit_val_batches == 1
    assert config.model.tuning_type == "freeze"
```

- [ ] **Step 2: Run smoke-config test**

Run: `uv run pytest tests/integration/test_smoke_config.py -v`

Expected: `1 passed`.

- [ ] **Step 3: Run unit, parity, and non-GPU integration suite first**

Run:

```bash
SPAMO_PROJECT_PATH=/home/kan/Research/SpaMo PHOENIX14T_ANNOTATION_ROOT=/home/kan/Research/SpaMo/preprocess/Phoenix14T DESPAMO_FEATURE_ROOT=/home/kan/datasets/spamo/features DESPAMO_HF_CACHE=/home/kan/datasets/spamo/hf_cache DESPAMO_CHECKPOINT=/home/kan/datasets/spamo/ckpt/spamo.ckpt uv run pytest -m "not gpu" -v
```

Expected: all selected tests pass; no model download occurs in unit tests.

- [ ] **Step 4: Run one bounded GPU training smoke**

Run:

```bash
PHOENIX14T_ANNOTATION_ROOT=/home/kan/Research/SpaMo/preprocess/Phoenix14T DESPAMO_FEATURE_ROOT=/home/kan/datasets/spamo/features DESPAMO_HF_CACHE=/home/kan/datasets/spamo/hf_cache uv run python scripts/train.py --config configs/data/phoenix14t.yaml --config configs/model/spamo_flan_t5_xl.yaml --config configs/experiment/phoenix14t_baseline.yaml --config configs/experiment/phoenix14t_smoke.yaml
```

Expected: finite `train/loss`, `train/contra_loss`, and `train/combined_loss`; at least one optimizer step; one validation batch; checkpoint written under configured artifacts directory. Stop rather than expanding workload if CUDA memory is insufficient.

- [ ] **Step 5: Document exact smoke command and observed artifact path**

Add to `README.md`:

```markdown
## PHOENIX14T Baseline Smoke

The bounded smoke run uses frozen `google/flan-t5-xl`, 1% of training batches,
one validation batch, and the original CLIP plus VideoMAE feature files. It is
a pipeline verification, not a quality result. See the approved command in
`docs/superpowers/plans/2026-09-22-phoenix14t-baseline.md`.
```

- [ ] **Step 6: Checkpoint commit if explicitly authorized**

```bash
git add README.md tests/integration/test_smoke_config.py
git commit -m "test: verify PHOENIX14T smoke training"
```

---

### Task 13: Complete Checkpoint Evaluation And Acceptance

**Files:**
- Create: `scripts/evaluate.py`
- Create: `src/despamo/evaluation/acceptance.py`
- Create: `tests/unit/evaluation/test_acceptance.py`
- Modify: `README.md`

**Interfaces:**
- Produces: one JSON artifact per generation mode.
- Produces: `validate_baseline_result(item_count, metrics) -> None`.
- Acceptance requires 642 clips, BLEU-4 within 1.0 of 25.08, and ROUGE-L F1 within 0.01 of 0.4698.

- [ ] **Step 1: Write failing acceptance test**

```python
# tests/unit/evaluation/test_acceptance.py
import pytest

from despamo.evaluation.acceptance import validate_baseline_result


def test_acceptance_allows_documented_tolerance() -> None:
    validate_baseline_result(642, {"bleu4": 24.5, "rougeL_f1": 0.465})


def test_acceptance_rejects_partial_split() -> None:
    with pytest.raises(ValueError, match="expected 642 test items"):
        validate_baseline_result(641, {"bleu4": 25.08, "rougeL_f1": 0.4698})
```

- [ ] **Step 2: Implement acceptance validator**

```python
# src/despamo/evaluation/acceptance.py
def validate_baseline_result(item_count: int, metrics: dict[str, float]) -> None:
    if item_count != 642:
        raise ValueError(f"expected 642 test items, got {item_count}")
    if abs(metrics["bleu4"] - 25.08) > 1.0:
        raise ValueError(f"BLEU-4 outside tolerance: {metrics['bleu4']}")
    if abs(metrics["rougeL_f1"] - 0.4698) > 0.01:
        raise ValueError(f"ROUGE-L F1 outside tolerance: {metrics['rougeL_f1']}")
```

- [ ] **Step 3: Implement evaluation entry point**

```python
# scripts/evaluate.py
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from despamo.config import load_config, validate_baseline_config
from despamo.evaluation.acceptance import validate_baseline_result
from despamo.evaluation.artifact import collect_runtime_metadata, write_result_artifact
from despamo.evaluation.metrics import evaluate_translations
from despamo.factory import build_data, build_model
from despamo.models.prompts import build_prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation", choices=("upstream", "deterministic"), required=True)
    parser.add_argument("--accept-baseline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.accept_baseline and args.generation != "upstream":
        parser.error("--accept-baseline requires --generation upstream")
    config = load_config(args.config)
    validate_baseline_config(config)
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cuda().eval()
    clip_ids: list[str] = []
    predictions: list[str] = []
    references: list[str] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in build_data(config).test_dataloader():
            visual, visual_mask = model.visual_adapter(
                batch.spatial.cuda(),
                batch.spatial_mask.cuda(),
                batch.motion.cuda(),
                batch.motion_mask.cuda(),
            )
            prompts = build_prompts(
                batch,
                model.prompt_template,
                model.use_in_context,
                model.num_in_context,
                model.rng,
            )
            predictions.extend(
                model.language_model.generate_text(
                    visual,
                    visual_mask,
                    prompts,
                    args.generation,
                    config.evaluation.beam_size,
                    config.model.max_text_length,
                )
            )
            references.extend(text.lower() for text in batch.texts)
            clip_ids.extend(batch.clip_ids)
    metrics = evaluate_translations(predictions, references)
    if args.accept_baseline:
        validate_baseline_result(len(clip_ids), metrics)
    metadata = {
        **collect_runtime_metadata(
            config.seed,
            Path("uv.lock"),
            Path(config.data.spatial_manifest),
            Path(config.data.motion_manifest),
        ),
        "generation": args.generation,
        "baseline_accepted": args.accept_baseline,
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": checkpoint["metadata"],
        "config": OmegaConf.to_container(config, resolve=True),
    }
    write_result_artifact(args.output, clip_ids, predictions, references, metrics, metadata)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run acceptance unit tests**

Run: `uv run pytest tests/unit/evaluation/test_acceptance.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Evaluate upstream-compatible generation**

Run:

```bash
PHOENIX14T_ANNOTATION_ROOT=/home/kan/Research/SpaMo/preprocess/Phoenix14T DESPAMO_FEATURE_ROOT=/home/kan/datasets/spamo/features DESPAMO_HF_CACHE=/home/kan/datasets/spamo/hf_cache uv run python scripts/evaluate.py --config configs/data/phoenix14t.yaml --config configs/model/spamo_flan_t5_xl.yaml --config configs/experiment/phoenix14t_baseline.yaml --checkpoint /home/kan/datasets/spamo/ckpt/despamo-spamo-baseline.pt --generation upstream --accept-baseline --output artifacts/phoenix14t_baseline/upstream.json
```

Expected: 642 items; BLEU-4 between 24.08 and 26.08; ROUGE-L F1 between 0.4598 and 0.4798. Sampling means reruns can differ.

- [ ] **Step 6: Evaluate deterministic generation twice**

Run the deterministic command twice, changing only output filename:

```bash
PHOENIX14T_ANNOTATION_ROOT=/home/kan/Research/SpaMo/preprocess/Phoenix14T DESPAMO_FEATURE_ROOT=/home/kan/datasets/spamo/features DESPAMO_HF_CACHE=/home/kan/datasets/spamo/hf_cache uv run python scripts/evaluate.py --config configs/data/phoenix14t.yaml --config configs/model/spamo_flan_t5_xl.yaml --config configs/experiment/phoenix14t_baseline.yaml --checkpoint /home/kan/datasets/spamo/ckpt/despamo-spamo-baseline.pt --generation deterministic --output artifacts/phoenix14t_baseline/deterministic-1.json
```

Expected: both deterministic artifact `items` and `metrics` sections are byte-equivalent after excluding timestamps; this plan adds no timestamps, so full files should match.

- [ ] **Step 7: Run final verification**

Run:

```bash
uv run ruff check src scripts tests
uv run pytest -m "not gpu" -v
git status --short
git diff --check
```

Expected: lint clean; all selected tests pass; only intended project files appear; no whitespace errors.

- [ ] **Step 8: Document measured results**

Add a `PHOENIX14T Baseline Evaluation` section to `README.md` with artifact paths, item count, BLEU-1 through BLEU-4, ROUGE-L F1, checkpoint SHA-256, generation mode, and exact command. Report measured values only after Step 5 and Step 6 complete.

- [ ] **Step 9: Checkpoint commit if explicitly authorized**

```bash
git add scripts/evaluate.py src/despamo/evaluation README.md tests/unit/evaluation/test_acceptance.py
git commit -m "feat: evaluate released SpaMo baseline"
```

---

## Baseline Completion Gate

- [ ] Python 3.11 environment installs from `uv.lock`.
- [ ] `.env` and all generated artifacts are ignored.
- [ ] Feature manifests contain complete train, dev, and test splits.
- [ ] Unit, parity, and non-GPU integration tests pass.
- [ ] Visual adapter parity test matches SpaMo source outputs.
- [ ] All 871 released checkpoint tensors map exactly once.
- [ ] Training smoke performs an optimizer step and writes a reloadable checkpoint.
- [ ] Upstream-compatible evaluation covers exactly 642 clips and passes metric tolerances.
- [ ] Deterministic evaluation repeats exactly.
- [ ] README records commands, artifacts, metrics, and checkpoint hash.

## Follow-Up Plans

After this gate passes, create separate plans in this order:

1. DINOv3 ViT-L/16 S2 offline extraction and CLIP-versus-DINOv3 comparison.
2. Local Ollama `qwen3-vl:8b` appearance-caption dataset and schema validation.
3. GRL appearance branch, loss scheduling, and ablations.
4. Signer-disjoint evaluation and signer/appearance leakage probes using PHOENIX signer metadata.
