# DINOv3 CLIP-vs-DINO Controlled Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare frozen CLIP S2 and frozen DINOv3 S2 spatial features on PHOENIX14T using matched SpaMo training/evaluation and report paired three-seed results.

**Architecture:** Keep existing baseline runtime and model intact. Two tiny spatial-source overlays sit on one common controlled protocol; a baseline-environment preflight checks *both* complete feature sets against all annotations before any translation model is built. Existing train/evaluate entry points gain comparison-only gates; a reporting script validates six full-test artifacts and summarizes seed-paired metrics.

**Tech Stack:** Python 3.11, existing root `uv.lock` (Torch 2.0.1, Transformers 4.32.0, PyTorch Lightning 1.9.5), OmegaConf, NumPy, pytest, sacrebleu and rouge-score already in the baseline. No extractor-runtime imports in translation code.

## Global Constraints

- Approved source: `docs/superpowers/specs/2026-09-24-dinov3-stage-design.md` §§1, 5–7; extraction contract: `docs/superpowers/plans/2026-09-24-dinov3-extraction.md` §§4–6. This plan covers the second deliverable only.
- Require completed `${DINO_OUTPUT}/<encoder_key>/{train,dev,test}/<clip_id>.npy`, `complete/manifest.json`, `complete/frame_rows.json`, `version.json`, and empty `failures.json`. `data.spatial_root` is **encoder-key root**, never `complete/`; manifest path is `root/complete/manifest.json`. Do not substitute one-clip receipts or synthetic features for a complete corpus.
- All three splits, not just test: exactly `train=7096`, `dev=519`, `test=642`; annotation ID sets and spatial row counts agree CLIP=DINO=`num_frames` for each clip. Expected row totals `train=827354`, `dev=55775`, `test=64627`. Bind resolved CLIP/DINO spatial root **and** manifest to their respective `comparison.*` sources before feature I/O or Flan-T5 weights; dotlist changes that point DINO at CLIP fail. Hash full bytes of `train_info_ml.npy`, `dev_info_ml.npy`, `test_info_ml.npy` in every comparison training run and carry same three hashes through checkpoint, deterministic evaluation and six-run report; reject text-only drift even with unchanged IDs/row counts. Baseline metadata remains unchanged.
- Both variants: frozen spatial feature arrays and original VideoMAE motion features; `google/flan-t5-xl` initialized from immutable, measured, cached model/tokenizer revision `7d6315df2c2fb742f0f5b556879d730926ca9001`; fail if loaded model `config._commit_hash` differs. LoRA `r=16`, `alpha=32`, dropout `0.1`, target modules `q,v`; `vt_pooling=masked_mean`, `spatial_crop_mode=full` (512 is ignored in full mode), same VT-Align, prompt, in-context policy, batch size 2, grad accumulation 2, AdamW `6e-4`/`0.01`, existing cosine/10% warmup, bf16, beam size 5, deterministic `do_sample=False`.
- Common comparison config limits **optimizer** steps to 1000 (not batches or epochs); independently seed CLIP and DINO with each of `0,1,2` *before constructing their models*. No `--resume`, no released CLIP projector, no checkpoint transplantation. Only source-specific overlay keys are `data.spatial_root` and `data.spatial_manifest`; artifact directory and `seed` vary operationally per run, in both variants identically.
- Run fake CPU tests first; after HF access and separately authorized full extraction, run one bounded **real translation smoke per source** with identical seed/budget and check finite losses, checkpoint step and strict reload. Request separate explicit authorization **before six full 1000-step GPU runs**. No extraction, weight download, training, or install during plan authoring. If gated weights/features unavailable, mark real gates pending; never count a skipped integration test as passed.
- Evaluate every finished full run on exactly 642 test IDs using `--generation deterministic` and no `--accept-baseline`. Report per-run BLEU-4 and ROUGE-L F1, checkpoint/manifest hashes, beam, seed, step count, per-variant mean/sample std and paired DINO-minus-CLIP mean/sample std; no improvement threshold. Existing released-checkpoint upstream acceptance belongs only to its converted SpaMo checkpoint.
- Root `pyproject.toml` and `uv.lock` unchanged; baseline translation environment only; external output roots under `${COMPARISON_OUTPUT}` (outside Git) and ignored by Git if ever placed inside it. No `.env` in this worktree; do not read, source, copy, or log any token. No commits unless separately requested.
- **Task 3 reviewer amendment (supersedes older Task 2/3 example snippets below):** comparison-only training saves `comparison_code_sha256` in run/checkpoint metadata, hashing sorted relative filenames and exact bytes from `src/despamo/**/*.py`, `scripts/*.py`, `configs/**/*.yaml`, root `pyproject.toml`, `uv.lock` (including uncommitted changes); evaluation checks checkpoint SHA against current code before model build and recopies/rechecks it before writing the artifact. Six-run summary requires code SHA, Git revision and dirty flag parity; rejects identical CLIP/DINO spatial roots, manifests or spatial manifest hashes; BLEU4 must be numeric 0–100, ROUGE-L F1 numeric 0–1, and all SHA256 fields lowercase 64-hex strings. Report writes a fully serialized, fsynced same-directory temporary file, publishes via no-overwrite hard link, fsyncs parent and removes temporary file on failure. Baseline metadata/CLI remains unchanged. Use current code and tests for updated examples rather than older inlined snippets.

## File Map And Interfaces

| File | Responsibility |
|---|---|
| `configs/experiment/phoenix14t_encoder_comparison.yaml` | Shared fixed-budget/model/decoding protocol and complete-root paths. |
| `configs/experiment/phoenix14t_clip_control.yaml`, `phoenix14t_dinov3.yaml` | Exactly two spatial path overrides each. |
| `src/despamo/comparison.py` | Named-source config binding/parity, full-split annotation/row/map gate, three annotation SHA256s, checkpoint provenance, six-run summary. |
| `src/despamo/models/flan_t5.py`, `src/despamo/factory.py` | Comparison-only immutable model/tokenizer revision; preserve baseline positional loader call. |
| `scripts/train.py` | Both-source preflight before `build_model`, comparison-only annotation hashes, independent seed, exact-step final checkpoint. |
| `scripts/evaluate.py` | Prevent experimental baseline acceptance; require matching annotation/feature checkpoint provenance and deterministic mode. |
| `scripts/compare_encoders.py` | CPU preflight command and offline six-artifact report command. |
| `tests/unit/test_comparison.py`, `tests/unit/test_comparison_cli.py`, `tests/unit/test_train_cli.py`, `tests/unit/evaluation/test_evaluate_cli.py`, `tests/unit/test_factory.py`, `tests/unit/models/test_flan_t5.py` | Fake small-split/metadata/CPU entry-point checks; never load pretrained weights. |

Public contracts: `load_pair(base: list[Path], overrides: Sequence[str] = ()) -> tuple[DictConfig, DictConfig]`; `validate_pair_configs(clip: DictConfig, dino: DictConfig) -> None`; `validate_comparison_protocol(config: DictConfig) -> None` (pure; no feature file I/O); `preflight_sources(config: DictConfig) -> dict[str, dict[str, int]]`; `preflight_run(config: DictConfig, paths: list[Path], overrides: Sequence[str]) -> dict[str, dict[str, int]]`; `annotation_hashes(config: DictConfig) -> dict[str, str]`; `validate_comparison_checkpoint(config: DictConfig, checkpoint: dict, generation: str) -> None`; `summarize(root: Path) -> dict`. The `base` list is ordered data, model, released experiment, common comparison; `load_pair` adds one source overlay to each and applies identical overrides (including per-run seed). `validate_pair_configs` binds both resolved variants to their named sources before normalizing the two spatial keys; `preflight_run` repeats selected-source binding, checks the selected fully resolved config equals its paired source config, then checks both complete manifests before `build_model`. `annotation_hashes` maps `train`, `dev`, `test` to SHA-256 of their exact annotation `.npy` bytes. Optional `comparison.enabled` defaults false for existing baseline paths. All snippets below are complete file contents when creating files; existing-file snippets explicitly identify insertions.

---

### Task 1: Matched Source Overlays And CPU Parity Gate

**Files:** Create `configs/experiment/phoenix14t_encoder_comparison.yaml`, `configs/experiment/phoenix14t_clip_control.yaml`, `configs/experiment/phoenix14t_dinov3.yaml`, `src/despamo/comparison.py`, `tests/unit/test_comparison.py`.

**Interfaces:** Produces `load_pair`, `validate_pair_configs`, `preflight_run`, `preflight_sources`; consumes baseline `load_config`, `FeatureManifest.load` and `sha256_file`. `validate_pair_configs` binds each resolved source before comparing other fields; `preflight_run` also binds selected config before `preflight_sources`, which checks both sources and returns split counts keyed by source. Use `EXPECTED` as module constant, overridden only in fake unit fixtures; never in live commands.

- [ ] **Step 1: Write failing fake test** (`tests/unit/test_comparison.py`):

```python
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from despamo.comparison import load_pair, preflight_run, preflight_sources, validate_pair_configs
from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.factory import build_model
from despamo.models.flan_t5 import FlanT5Backbone
from despamo.utils.hashing import sha256_file

FLAN_SHA = "7d6315df2c2fb742f0f5b556879d730926ca9001"


def make_pair(tmp_path, monkeypatch):
    import despamo.comparison as comparison

    monkeypatch.setattr(comparison, "EXPECTED", {"train": 1, "dev": 1, "test": 1})
    ann = tmp_path / "annotations"
    ann.mkdir()
    version = {"schema_version": 1, "model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
               "model_sha": "b" * 40}
    key = hashlib.sha256(json.dumps(version, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True).encode()).hexdigest()
    clip_root, dino_root = tmp_path / "clip", tmp_path / "dino" / key
    clip_records, dino_records, rows = [], [], {}
    for split in ("train", "dev", "test"):
        clip_id = f"{split}-id"
        np.save(ann / f"{split}_info_ml.npy",
                {0: {"fileid": clip_id, "num_frames": 5, "text": "original text"}})
        for root, records in ((clip_root, clip_records), (dino_root, dino_records)):
            (root / split).mkdir(parents=True)
            np.save(root / split / f"{clip_id}.npy", np.ones((5, 2048), np.float32))
            records.append(FeatureRecord(clip_id, split, f"{split}/{clip_id}.npy", 5, 2048, "float32"))
        rows[clip_id] = {"feature_hash": sha256_file(dino_root / split / f"{clip_id}.npy"),
                         "frame_count": 5,
                         "source_indices": list(range(5)),
                         "sampled_images": {str(i): "a" * 64 for i in range(5)}}
    clip_manifest = tmp_path / "clip.json"
    FeatureManifest(1, "clip", 2048, tuple(clip_records)).save(clip_manifest)
    (dino_root / "complete").mkdir()
    dino_manifest = dino_root / "complete/manifest.json"
    FeatureManifest(1, f"dinov3:{dino_root.name}", 2048, tuple(dino_records)).save(dino_manifest)
    (dino_root / "complete/frame_rows.json").write_text(json.dumps({
        "encoder_key": dino_root.name, "spatial_manifest_hash": sha256_file(dino_manifest),
        "clips": rows,
    }))
    (dino_root / "version.json").write_text(json.dumps(version))
    (dino_root / "failures.json").write_text("{}")
    common = {"seed": 0, "comparison": {
        "enabled": True, "smoke": False,
        "clip_root": str(clip_root), "clip_manifest": str(clip_manifest),
        "dino_root": str(dino_root), "dino_manifest": str(dino_manifest)},
        "data": {"annotation_root": str(ann), "batch_size": 2},
        "model": {"spatial_dim": 2048, "motion_dim": 1024,
                  "vt_pooling": "masked_mean", "spatial_crop_mode": "full",
                  "name": "google/flan-t5-xl", "lora_rank": 16, "lora_alpha": 32,
                  "lora_dropout": 0.1, "revision": FLAN_SHA,
                  "prompt": "Translate the given sentence into {}.",
                  "vt_weight": 1.0, "use_in_context": True, "num_in_context": 3,
                  "warm_up_steps": 0},
        "trainer": {"max_steps": 1000, "max_epochs": -1,
                    "accumulate_grad_batches": 2, "precision": "bf16",
                    "default_root_dir": str(tmp_path / "run")},
        "optimizer": {"learning_rate": 6e-4, "weight_decay": 0.01},
        "evaluation": {"generation": "deterministic", "beam_size": 5,
                       "expected_test_items": 642}}
    clip = OmegaConf.merge(common, {"data": {"spatial_root": str(clip_root),
                                    "spatial_manifest": str(clip_manifest)}})
    dino = OmegaConf.merge(common, {"data": {"spatial_root": str(dino_root),
                                    "spatial_manifest": str(dino_manifest)}})
    return clip, dino, ann, dino_root


def test_real_overlays_differ_only_by_spatial_source(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOENIX14T_ANNOTATION_ROOT", str(tmp_path))
    monkeypatch.setenv("DESPAMO_FEATURE_ROOT", str(tmp_path))
    monkeypatch.setenv("DESPAMO_HF_CACHE", str(tmp_path))
    monkeypatch.setenv("DINO_ROOT", str(tmp_path / "dino"))
    monkeypatch.setenv("COMPARISON_RUN_DIR", str(tmp_path / "run"))
    configs = Path(__file__).resolve().parents[2] / "configs"
    clip, dino = load_pair([
        configs / "data/phoenix14t.yaml", configs / "model/spamo_flan_t5_xl.yaml",
        configs / "experiment/phoenix14t_baseline.yaml",
        configs / "experiment/phoenix14t_encoder_comparison.yaml",
    ])
    validate_pair_configs(clip, dino)
    assert clip.model.vt_pooling == dino.model.vt_pooling == "masked_mean"
    assert clip.model.spatial_crop_mode == dino.model.spatial_crop_mode == "full"
    assert clip.trainer.max_steps == dino.trainer.max_steps == 1000
    assert clip.model.lora_rank == dino.model.lora_rank == 16
    assert clip.model.lora_alpha == dino.model.lora_alpha == 32
    assert clip.model.lora_dropout == dino.model.lora_dropout == 0.1
    assert clip.model.name == dino.model.name == "google/flan-t5-xl"
    assert clip.model.revision == dino.model.revision == FLAN_SHA
    assert clip.model.prompt == dino.model.prompt == "Translate the given sentence into {}."
    assert clip.model.vt_weight == dino.model.vt_weight == 1.0
    assert clip.data.batch_size == dino.data.batch_size == 2
    assert clip.trainer.accumulate_grad_batches == dino.trainer.accumulate_grad_batches == 2
    assert clip.optimizer.learning_rate == dino.optimizer.learning_rate == 6e-4
    assert clip.evaluation.generation == dino.evaluation.generation == "deterministic"
    assert clip.data.motion_root == dino.data.motion_root
    class FakeLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(1, 4, bias=False)
            self.lora_B = torch.nn.Linear(4, 1, bias=False)

    monkeypatch.setattr(FlanT5Backbone, "from_pretrained",
                        lambda *args, **kwargs: FakeLoRA())
    torch.manual_seed(0)
    clip_model = build_model(clip)
    torch.manual_seed(0)
    dino_model = build_model(dino)
    assert clip_model is not dino_model
    assert "language_model.lora_A.weight" in clip_model.state_dict()
    for key, value in clip_model.state_dict().items():
        torch.testing.assert_close(value, dino_model.state_dict()[key])
    seeded_clip, seeded_dino = load_pair([
        configs / "data/phoenix14t.yaml", configs / "model/spamo_flan_t5_xl.yaml",
        configs / "experiment/phoenix14t_baseline.yaml",
        configs / "experiment/phoenix14t_encoder_comparison.yaml",
    ], ["seed=2"])
    assert seeded_clip.seed == seeded_dino.seed == 2
    validate_pair_configs(seeded_clip, seeded_dino)


def test_all_splits_parity_and_corruption_before_training(tmp_path, monkeypatch):
    clip, dino, ann, root = make_pair(tmp_path, monkeypatch)
    validate_pair_configs(clip, dino)
    assert preflight_sources(clip) == preflight_sources(dino) == {
        "clip": {"train": 1, "dev": 1, "test": 1},
        "dino": {"train": 1, "dev": 1, "test": 1},
    }
    raw = np.load(ann / "dev_info_ml.npy", allow_pickle=True).item()
    raw[0]["num_frames"] = 6
    np.save(ann / "dev_info_ml.npy", raw)
    with pytest.raises(ValueError, match="dev/dev-id.*row"):
        preflight_sources(clip)
    raw[0]["num_frames"] = 5
    np.save(ann / "dev_info_ml.npy", raw)
    path = root / "test/test-id.npy"
    np.save(path, np.ones((4, 2048), np.float32))
    with pytest.raises(ValueError, match="test/test-id.*row"):
        preflight_sources(dino)


def test_incomplete_or_unmatched_configuration_fails_closed(tmp_path, monkeypatch):
    clip, dino, _, root = make_pair(tmp_path, monkeypatch)
    dino.optimizer = {"learning_rate": 0.0001}
    with pytest.raises(ValueError, match="only spatial"):
        validate_pair_configs(clip, dino)
    dino.optimizer = {"learning_rate": 6e-4, "weight_decay": 0.01}
    dino.model.prompt = "different prompt"
    with pytest.raises(ValueError, match="only spatial"):
        validate_pair_configs(clip, dino)
    (root / "version.json").write_text('{"model_sha":"main"}')
    with pytest.raises(ValueError, match="version key"):
        preflight_sources(clip)
    version = {"schema_version": 1, "model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
               "model_sha": "b" * 40}
    (root / "version.json").write_text(json.dumps(version))
    (root / "complete/frame_rows.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="frame_rows"):
        preflight_sources(clip)


def test_fake_full_length_batches_preserve_both_source_masks(tmp_path, monkeypatch):
    from despamo.factory import build_data

    clip, dino, ann, root = make_pair(tmp_path, monkeypatch)
    clip.model.max_frame_len = dino.model.max_frame_len = 512
    for split in ("train", "dev", "test"):
        path = ann / f"{split}_info_ml.npy"
        raw = np.load(path, allow_pickle=True).item()
        raw[0].update(signer="Signer01", gloss="WIND", text="wind", en_text="wind",
                      es_text="viento", fr_text="vent")
        if split == "train":
            raw[0]["num_frames"] = 520
        np.save(path, raw)
    train_id = "train-id"
    for config in (clip, dino):
        source = Path(config.data.spatial_root)
        np.save(source / "train" / f"{train_id}.npy", np.ones((520, 2048), np.float32))
        path = Path(config.data.spatial_manifest)
        manifest = FeatureManifest.load(path)
        records = tuple(FeatureRecord(r.clip_id, r.split, r.path,
                                      520 if r.split == "train" else r.length, 2048, "float32")
                        for r in manifest.records)
        FeatureManifest(1, manifest.encoder, 2048, records).save(path)
    rows_path = root / "complete/frame_rows.json"
    rows = json.loads(rows_path.read_text())
    train_row = rows["clips"][train_id]
    train_row["frame_count"] = 520
    train_row["source_indices"] = list(range(520))
    train_row["sampled_images"] = {
        str(int(519 * fraction + 0.5)): "a" * 64
        for fraction in (0.1, 0.3, 0.5, 0.7, 0.9)}
    train_row["feature_hash"] = sha256_file(root / "train/train-id.npy")
    rows["spatial_manifest_hash"] = sha256_file(root / "complete/manifest.json")
    rows_path.write_text(json.dumps(rows))
    motion_root = tmp_path / "motion"
    motion_records = []
    for split in ("train", "dev", "test"):
        (motion_root / split).mkdir(parents=True)
        length = 5
        np.save(motion_root / split / f"{split}-id.npy", np.ones((length, 1024), np.float32))
        motion_records.append(FeatureRecord(f"{split}-id", split, f"{split}/{split}-id.npy",
                                            length, 1024, "float32"))
    motion_manifest = tmp_path / "motion.json"
    FeatureManifest(1, "videomae", 1024, tuple(motion_records)).save(motion_manifest)
    for config in (clip, dino):
        config.data.motion_root = str(motion_root)
        config.data.motion_manifest = str(motion_manifest)
        config.data.num_workers = 0
    validate_pair_configs(clip, dino)
    assert preflight_sources(clip) == preflight_sources(dino)
    clip_batch = next(iter(build_data(clip).train_dataloader()))
    dino_batch = next(iter(build_data(dino).train_dataloader()))
    assert clip_batch.spatial.shape == dino_batch.spatial.shape == (1, 520, 2048)
    assert clip_batch.spatial_mask.equal(dino_batch.spatial_mask)
    assert clip_batch.spatial_mask.all()
    assert clip_batch.motion_mask.equal(dino_batch.motion_mask)


def test_run_preflight_checks_pair_selected_config_and_both_sources(tmp_path, monkeypatch):
    import despamo.comparison as comparison

    clip, dino, _, root = make_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(comparison, "load_pair", lambda base, overrides=(): (clip, dino))
    paths = [tmp_path / "base.yaml", comparison.CONFIGS / "phoenix14t_clip_control.yaml"]
    counts = {"clip": {"train": 1, "dev": 1, "test": 1},
              "dino": {"train": 1, "dev": 1, "test": 1}}
    assert preflight_run(clip, paths, ["seed=0"]) == counts
    dino.model.vt_weight = 0.5
    with pytest.raises(ValueError, match="only spatial"):
        preflight_run(clip, paths, ["seed=0"])
    dino.model.vt_weight = 1.0
    changed = OmegaConf.merge(clip, {"model": {"prompt": "wrong prompt"}})
    with pytest.raises(ValueError, match="selected config"):
        preflight_run(changed, paths, ["seed=0"])
    (root / "complete/frame_rows.json").unlink()
    with pytest.raises(FileNotFoundError, match="frame_rows"):
        preflight_run(clip, paths, ["seed=0"])


def test_dotlist_cannot_bind_dino_overlay_to_clip_features(tmp_path, monkeypatch):
    import despamo.comparison as comparison

    clip, dino, _, _ = make_pair(tmp_path, monkeypatch)
    overrides = [f"data.spatial_root={clip.data.spatial_root}",
                 f"data.spatial_manifest={clip.data.spatial_manifest}"]
    swapped_clip = OmegaConf.merge(clip, OmegaConf.from_dotlist(overrides))
    swapped_dino = OmegaConf.merge(dino, OmegaConf.from_dotlist(overrides))
    with pytest.raises(ValueError, match="dino spatial source binding"):
        validate_pair_configs(swapped_clip, swapped_dino)
    monkeypatch.setattr(comparison, "load_pair",
                        lambda base, overrides=(): (swapped_clip, swapped_dino))
    monkeypatch.setattr(comparison, "preflight_sources",
                        lambda config: pytest.fail("feature I/O started before source binding"))
    paths = [tmp_path / "base.yaml", comparison.CONFIGS / "phoenix14t_dinov3.yaml"]
    with pytest.raises(ValueError, match="dino spatial source binding"):
        preflight_run(swapped_dino, paths, overrides)
```

- [ ] **Step 2: Verify red.** Run `uv run --locked python -m pytest tests/unit/test_comparison.py -q`; expect missing `despamo.comparison` on collection. When iterating on `validate_pair_configs`, rerun `uv run --locked python -m pytest tests/unit/test_comparison.py::test_dotlist_cannot_bind_dino_overlay_to_clip_features -q` before adding `_require_source_binding`; expect red with two dotlist overrides because both variants otherwise appear equal after spatial-field deletion.
- [ ] **Step 3: Add config files** (full contents):

```yaml
# configs/experiment/phoenix14t_encoder_comparison.yaml
trainer:
  default_root_dir: ${oc.env:COMPARISON_RUN_DIR}
  max_epochs: -1
  max_steps: 1000
model:
  vt_pooling: masked_mean
  spatial_crop_mode: full
  revision: 7d6315df2c2fb742f0f5b556879d730926ca9001
evaluation:
  generation: deterministic
comparison:
  enabled: true
  smoke: false
  clip_root: ${oc.env:DESPAMO_FEATURE_ROOT}/vit_feat_Phoenix14T
  clip_manifest: ${oc.env:DESPAMO_FEATURE_ROOT}/manifests/phoenix14t_spatial.json
  dino_root: ${oc.env:DINO_ROOT}
  dino_manifest: ${oc.env:DINO_ROOT}/complete/manifest.json
```

```yaml
# configs/experiment/phoenix14t_clip_control.yaml
data:
  spatial_root: ${oc.env:DESPAMO_FEATURE_ROOT}/vit_feat_Phoenix14T
  spatial_manifest: ${oc.env:DESPAMO_FEATURE_ROOT}/manifests/phoenix14t_spatial.json
```

```yaml
# configs/experiment/phoenix14t_dinov3.yaml
data:
  spatial_root: ${oc.env:DINO_ROOT}
  spatial_manifest: ${oc.env:DINO_ROOT}/complete/manifest.json
```

- [ ] **Step 4: Implement parity module** (`src/despamo/comparison.py`; Task 3 adds `summarize` to same file):

```python
import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from despamo.config import load_config
from despamo.data.manifest import FeatureManifest
from despamo.utils.hashing import sha256_file

EXPECTED = {"train": 7096, "dev": 519, "test": 642}
EXPECTED_ROWS = {"train": 827354, "dev": 55775, "test": 64627}
FLAN_SHA = "7d6315df2c2fb742f0f5b556879d730926ca9001"
CONFIGS = Path(__file__).resolve().parents[2] / "configs/experiment"


def load_pair(base: list[Path], overrides: Sequence[str] = ()) -> tuple[DictConfig, DictConfig]:
    return (
        load_config([*base, CONFIGS / "phoenix14t_clip_control.yaml"], overrides),
        load_config([*base, CONFIGS / "phoenix14t_dinov3.yaml"], overrides),
    )


def _require_source_binding(config: DictConfig, source: str) -> None:
    actual = (config.data.spatial_root, config.data.spatial_manifest)
    expected = (config.comparison[f"{source}_root"], config.comparison[f"{source}_manifest"])
    if actual != expected:
        raise ValueError(f"{source} spatial source binding mismatch: root/manifest")


def validate_pair_configs(clip: DictConfig, dino: DictConfig) -> None:
    _require_source_binding(clip, "clip")
    _require_source_binding(dino, "dino")
    left, right = (OmegaConf.to_container(config, resolve=True) for config in (clip, dino))
    for config in (left, right):
        for name in ("spatial_root", "spatial_manifest"):
            del config["data"][name]
    if left != right:
        raise ValueError("only spatial root/manifest may differ between CLIP and DINO")
    if (not clip.comparison.enabled or clip.seed != dino.seed
            or clip.comparison.smoke != dino.comparison.smoke):
        raise ValueError("paired comparison requires same seed and protocol")


def preflight_run(config: DictConfig, paths: list[Path],
                  overrides: Sequence[str]) -> dict[str, dict[str, int]]:
    if len(paths) < 2:
        raise ValueError("comparison needs common config followed by source overlay")
    overlay = paths[-1].resolve()
    overlays = {
        (CONFIGS / "phoenix14t_clip_control.yaml").resolve(): "clip",
        (CONFIGS / "phoenix14t_dinov3.yaml").resolve(): "dino",
    }
    if overlay not in overlays:
        raise ValueError("comparison source overlay must be last --config layer")
    clip, dino = load_pair(paths[:-1], overrides)
    validate_pair_configs(clip, dino)
    source = overlays[overlay]
    expected = {"clip": clip, "dino": dino}[source]
    _require_source_binding(expected, source)
    _require_source_binding(config, source)
    if OmegaConf.to_container(config, resolve=True) != OmegaConf.to_container(
            expected, resolve=True):
        raise ValueError("selected config differs from paired source overlay")
    return preflight_sources(expected)


def _annotations(root: Path) -> dict[str, dict[str, int]]:
    result = {}
    all_ids = set()
    for split in EXPECTED:
        raw = np.load(root / f"{split}_info_ml.npy", allow_pickle=True).item()
        if not isinstance(raw, dict):
            raise ValueError(f"{split}: annotation is not a mapping")
        entries = {}
        for index in sorted(key for key in raw if type(key) is int):
            item = raw[index]
            clip_id, count = item["fileid"], item["num_frames"]
            if (not isinstance(clip_id, str) or not clip_id or "/" in clip_id
                    or "\\" in clip_id or type(count) is not int or count < 5
                    or clip_id in all_ids):
                raise ValueError(f"{split}/{clip_id}: invalid/duplicate annotation")
            entries[clip_id] = count
            all_ids.add(clip_id)
        if len(entries) != EXPECTED[split]:
            raise ValueError(f"{split}: expected {EXPECTED[split]} annotation IDs")
        if EXPECTED == {"train": 7096, "dev": 519, "test": 642} and (
                sum(entries.values()) != EXPECTED_ROWS[split]):
            raise ValueError(f"{split}: approved annotation row total mismatch")
        result[split] = entries
    return result


def preflight_sources(config: DictConfig) -> dict[str, dict[str, int]]:
    if not config.comparison.enabled:
        raise ValueError("comparison preflight requires comparison.enabled")
    if (config.model.vt_pooling != "masked_mean" or config.model.spatial_crop_mode != "full"
            or config.evaluation.generation != "deterministic" or config.evaluation.beam_size != 5
            or config.model.name != "google/flan-t5-xl" or config.model.revision != FLAN_SHA
            or config.model.prompt != "Translate the given sentence into {}."
            or config.model.vt_weight != 1.0 or config.model.use_in_context is not True
            or config.model.num_in_context != 3 or config.model.warm_up_steps != 0
            or config.model.lora_rank != 16
            or config.model.lora_alpha != 32 or config.model.lora_dropout != 0.1
            or config.model.spatial_dim != 2048
            or config.model.motion_dim != 1024 or config.trainer.max_epochs != -1
            or config.trainer.max_steps != (1 if config.comparison.smoke else 1000)):
        raise ValueError("comparison protocol mismatch")
    if (config.data.batch_size != 2 or config.trainer.accumulate_grad_batches != 2
            or config.trainer.precision != "bf16" or config.optimizer.learning_rate != 6e-4
            or config.optimizer.weight_decay != 0.01):
        raise ValueError("comparison optimizer/batch/precision protocol mismatch")
    sources = {
        "clip": (Path(config.comparison.clip_root), Path(config.comparison.clip_manifest)),
        "dino": (Path(config.comparison.dino_root), Path(config.comparison.dino_manifest)),
    }
    if (Path(config.data.spatial_root), Path(config.data.spatial_manifest)) not in sources.values():
        raise ValueError("selected spatial root/manifest must match one comparison source")
    dino_root, dino_manifest = sources["dino"]
    if dino_manifest != dino_root / "complete/manifest.json":
        raise ValueError("DINO manifest must be complete/manifest.json at encoder-key root")
    for file in (dino_root / "version.json", dino_root / "complete/frame_rows.json",
                 dino_root / "failures.json"):
        if not file.is_file():
            raise FileNotFoundError(f"DINO complete file missing: {file}")
    version = json.loads((dino_root / "version.json").read_text())
    canonical = json.dumps(version, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if (version.get("model") != "facebook/dinov3-vitl16-pretrain-lvd1689m"
            or not re.fullmatch(r"[0-9a-f]{40}", str(version.get("model_sha")))
            or hashlib.sha256(canonical.encode()).hexdigest() != dino_root.name):
        raise ValueError("DINO version key/model revision mismatch")
    if json.loads((dino_root / "failures.json").read_text()) != {}:
        raise ValueError("DINO extraction failures.json must be empty")
    rows = json.loads((dino_root / "complete/frame_rows.json").read_text())
    if (rows.get("encoder_key") != dino_root.name
            or rows.get("spatial_manifest_hash") != sha256_file(dino_manifest)):
        raise ValueError("DINO frame_rows manifest hash/encoder_key mismatch")
    if not isinstance(rows.get("clips"), dict):
        raise ValueError("DINO frame_rows clips mapping missing")
    annotations = _annotations(Path(config.data.annotation_root))
    manifests = {name: FeatureManifest.load(path) for name, (_, path) in sources.items()}
    if manifests["dino"].encoder != f"dinov3:{dino_root.name}":
        raise ValueError("DINO manifest encoder key mismatch")
    seen = {name: {} for name in sources}
    for name, manifest in manifests.items():
        if manifest.expected_dim != 2048:
            raise ValueError(f"{name}: expected spatial width 2048")
        counts = Counter(record.split for record in manifest.records)
        if counts != EXPECTED:
            raise ValueError(f"{name}: complete split counts mismatch: {counts}")
        for record in manifest.records:
            split, clip_id = record.split, record.clip_id
            if split not in EXPECTED or clip_id not in annotations[split]:
                raise ValueError(f"{name} {split}/{clip_id}: extra feature ID")
            expected = annotations[split][clip_id]
            if record.length != expected:
                raise ValueError(f"{name} {split}/{clip_id}: annotation/feature row mismatch")
            feature = sources[name][0] / record.path
            try:
                array = np.load(feature, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError, EOFError) as exc:
                raise ValueError(f"{name} {split}/{clip_id}: invalid feature") from exc
            if array.shape != (expected, 2048) or str(array.dtype) != record.dtype:
                raise ValueError(f"{name} {split}/{clip_id}: file/manifest row or dtype mismatch")
            if name == "dino" and (str(array.dtype) != "float32" or not np.isfinite(array).all()):
                raise ValueError(f"dino {split}/{clip_id}: feature dtype/finiteness mismatch")
            del array
            seen[name][split, clip_id] = expected
            if name == "dino":
                row = rows["clips"].get(clip_id)
                if (not isinstance(row, dict) or row.get("frame_count") != expected
                        or row.get("source_indices") != list(range(expected))
                        or not isinstance(row.get("sampled_images"), dict)
                        or set(row["sampled_images"]) != {
                            str(int((expected - 1) * position + 0.5))
                            for position in (0.1, 0.3, 0.5, 0.7, 0.9)}
                        or row.get("feature_hash") != sha256_file(feature)):
                    raise ValueError(f"dino {split}/{clip_id}: frame_rows/feature hash mismatch")
    reference = {(split, clip_id): count for split, items in annotations.items()
                 for clip_id, count in items.items()}
    if seen["clip"] != reference or seen["dino"] != reference:
        raise ValueError("CLIP/DINO full split ID/row parity mismatch")
    if set(rows["clips"]) != {clip for _, clip in reference}:
        raise ValueError("DINO frame_rows extra/missing IDs")
    return {name: {split: sum(key[0] == split for key in records) for split in EXPECTED}
            for name, records in seen.items()}
```

- [ ] **Step 5: Run** `uv run --locked python -m pytest tests/unit/test_comparison.py -q`; expect 6 passed, including malicious two-field dotlist test. Run `uv run --locked ruff format src/despamo/comparison.py tests/unit/test_comparison.py`, then `uv run --locked ruff check src/despamo/comparison.py tests/unit/test_comparison.py`; expect no findings. Review new untracked files directly; verify overlays each contain **only** two spatial keys.

### Task 2: Fresh Training, Exact Checkpoint, Evaluation Provenance

**Files:** Modify `src/despamo/comparison.py`, `src/despamo/models/flan_t5.py`, `src/despamo/factory.py`, `scripts/train.py`, `scripts/evaluate.py`, `tests/unit/test_comparison.py`, `tests/unit/test_factory.py`, `tests/unit/models/test_flan_t5.py`, `tests/unit/test_train_cli.py`, `tests/unit/evaluation/test_evaluate_cli.py`.

**Interfaces:** `annotation_hashes(config: DictConfig) -> dict[str, str]` hashes exact bytes of all three annotation split files; `validate_comparison_checkpoint(config: DictConfig, checkpoint: dict, generation: str) -> None` checks exact step, run metadata, current annotation/feature hashes and protocol; returns nothing; must be called after safe checkpoint snapshot/load and before `build_model` or scoring. Training comparison only: preflight before model, `run_metadata["annotation_sha256"]` before provisional manifest, save `checkpoints/final.ckpt` **after** `trainer.global_step == trainer.max_steps`; `SpaMoBaselineModule.on_save_checkpoint` already copies run metadata. Use `trainer.save_checkpoint` (full Lightning state), never an existing/released checkpoint.

- [ ] **Step 1: Add failing fake entry-point tests** (append blocks):

```python
# tests/unit/test_train_cli.py
def _fake_annotation_root(config_path, tmp_path):
    ann = tmp_path / "ann"
    ann.mkdir()
    for split in ("train", "dev", "test"):
        np.save(ann / f"{split}_info_ml.npy",
                {0: {"fileid": split, "num_frames": 5, "text": "original text"}})
    config = OmegaConf.load(config_path)
    config.data.annotation_root = str(ann)
    OmegaConf.save(config, config_path)
    return ann


def test_comparison_preflight_stops_before_model_and_training(tmp_path, monkeypatch):
    root = tmp_path / "run"
    train, _, config_path, _, events = _fake_run(
        tmp_path, monkeypatch, root=root,
        extra_config={"comparison": {"enabled": True, "smoke": True},
                      "trainer": {"default_root_dir": str(tmp_path / "run"), "max_steps": 1}},
    )
    monkeypatch.setattr(train, "preflight_run", lambda config, paths, overrides: (_ for _ in ()).throw(
        ValueError("DINO complete manifest missing")))
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    with pytest.raises(ValueError, match="DINO complete manifest missing"):
        train.main()
    assert "model" not in events
    assert not (root / "run_metadata.json").exists()


def test_train_cli_rejects_dino_overridden_to_clip_before_model(tmp_path, monkeypatch):
    train = _train_script()
    configs = Path(__file__).resolve().parents[2] / "configs"
    feature_root = tmp_path / "features"
    monkeypatch.setenv("PHOENIX14T_ANNOTATION_ROOT", str(tmp_path / "ann"))
    monkeypatch.setenv("DESPAMO_FEATURE_ROOT", str(feature_root))
    monkeypatch.setenv("DESPAMO_HF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("DINO_ROOT", str(tmp_path / "dino-key"))
    monkeypatch.setenv("COMPARISON_RUN_DIR", str(tmp_path / "run"))
    layers = [configs / "data/phoenix14t.yaml", configs / "model/spamo_flan_t5_xl.yaml",
              configs / "experiment/phoenix14t_baseline.yaml",
              configs / "experiment/phoenix14t_encoder_comparison.yaml",
              configs / "experiment/phoenix14t_dinov3.yaml"]
    spatial = feature_root / "manifests/phoenix14t_spatial.json"
    monkeypatch.setattr("despamo.comparison.preflight_sources",
                        lambda config: pytest.fail("feature I/O started before binding"))
    monkeypatch.setattr(train, "build_data", lambda config: pytest.fail("data built"))
    monkeypatch.setattr(train, "build_model", lambda config: pytest.fail("model built"))
    monkeypatch.setattr(train.pl, "Trainer", lambda **kwargs: pytest.fail("trainer built"))
    monkeypatch.setattr(sys, "argv", ["train.py",
                                    *(part for path in layers for part in ("--config", str(path))),
                                    f"data.spatial_root={feature_root / 'vit_feat_Phoenix14T'}",
                                    f"data.spatial_manifest={spatial}"])
    with pytest.raises(ValueError, match="dino spatial source binding"):
        train.main()
    assert not (tmp_path / "run/run_metadata.json").exists()


def test_comparison_saves_only_exact_step_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "run"
    train, _, config_path, model, _ = _fake_run(
        tmp_path, monkeypatch, root=root, revision=FLAN_SHA,
        extra_config={"comparison": {"enabled": True, "smoke": True},
                      "trainer": {"default_root_dir": str(root), "max_steps": 1},
                      "model": {"spatial_dim": 2048, "motion_dim": 1024,
                                "name": "google/flan-t5-xl", "revision": FLAN_SHA}},
    )
    ann = _fake_annotation_root(config_path, tmp_path)
    monkeypatch.setattr(train, "preflight_run", lambda config, paths, overrides: {})
    saved = []
    class FakeTrainer:
        global_step = 1
        def __init__(self, **kwargs):
            assert kwargs["max_steps"] == 1
        def fit(self, model, *, datamodule):
            pass
        def save_checkpoint(self, path):
            saved.append(Path(path))
            Path(path).write_bytes(b"fake full-state checkpoint")
    monkeypatch.setattr(train.pl, "Trainer", FakeTrainer)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    assert saved == [root / "checkpoints/final.ckpt"]
    assert saved[0].read_bytes() == b"fake full-state checkpoint"
    assert model.run_metadata["annotation_sha256"] == {
        split: sha256_file(ann / f"{split}_info_ml.npy") for split in ("train", "dev", "test")
    }


def test_comparison_rejects_wrong_resolved_flan_before_fit(tmp_path, monkeypatch):
    root = tmp_path / "run"
    train, _, config_path, _, events = _fake_run(
        tmp_path, monkeypatch, root=root, revision="c" * 40,
        extra_config={"comparison": {"enabled": True, "smoke": True},
                      "trainer": {"default_root_dir": str(root), "max_steps": 1},
                      "model": {"spatial_dim": 2048, "motion_dim": 1024,
                                "name": "google/flan-t5-xl", "revision": FLAN_SHA}},
    )
    _fake_annotation_root(config_path, tmp_path)
    monkeypatch.setattr(train, "preflight_run", lambda config, paths, overrides: {})
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    with pytest.raises(ValueError, match="comparison Flan revision mismatch"):
        train.main()
    assert not (root / "run_metadata.json").exists()
    assert not any(isinstance(event, tuple) and event[0] == "fit" for event in events)


def test_baseline_train_does_not_hash_annotations_or_add_metadata(tmp_path, monkeypatch):
    train, root, config_path, _, _ = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(train, "annotation_hashes",
                        lambda config: pytest.fail("baseline hashed annotations"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    assert "annotation_sha256" not in json.loads((root / "run_metadata.json").read_text())


# tests/unit/evaluation/test_evaluate_cli.py
def _mark_converted_released(checkpoint):
    torch.save({"state_dict": FakeModel().state_dict(), "metadata": {
        "source_path": "spamo.ckpt",
        "source_sha256": "06a432cdd8e1da4ce0b0e4cff246b20ad7d6a60406f32dfdbdfd974f94d3eee6",
        "target_schema_sha256": "b" * 64, "source_tensor_count": 871,
        "target_tensor_count": 871,
    }}, checkpoint)


def test_comparison_rejects_baseline_acceptance_even_with_upstream(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    config.comparison = {"enabled": True, "smoke": False}
    monkeypatch.setattr(script, "build_data", lambda config: pytest.fail("data built"))
    with pytest.raises(ValueError, match="released SpaMo"):
        script.evaluate(config, checkpoint, "upstream", tmp_path / "result.json",
                        device="cpu", accept_baseline=True)


def test_comparison_rejects_incomplete_checkpoint_before_model(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    config.comparison = {"enabled": True, "smoke": False}
    config.trainer = {"max_steps": 1000}
    config.evaluation.generation = "deterministic"
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model built"))
    with pytest.raises(ValueError, match="comparison.*step"):
        script.evaluate(config, checkpoint, "deterministic", tmp_path / "result.json",
                        device="cpu")


def test_baseline_acceptance_rejects_other_converted_source(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    _mark_converted_released(checkpoint)
    content = torch.load(checkpoint, map_location="cpu", weights_only=True)
    content["metadata"]["source_sha256"] = "c" * 64
    torch.save(content, checkpoint)
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model built"))
    with pytest.raises(ValueError, match="converted released SpaMo"):
        script.evaluate(config, checkpoint, "upstream", tmp_path / "result.json",
                        device="cpu", accept_baseline=True)


def test_evaluation_cli_applies_seed_override_to_same_layers(tmp_path, monkeypatch):
    script = _script()
    common_path = tmp_path / "comparison.yaml"
    common_path.write_text(
        "seed: 0\nmodel:\n  spatial_dim: 2048\n  motion_dim: 1024\n"
        "trainer:\n  default_root_dir: ${oc.env:COMPARISON_RUN_DIR}\n"
    )
    source_path = tmp_path / "dino.yaml"
    source_path.write_text("data:\n  spatial_root: dino\n")
    monkeypatch.setenv("COMPARISON_RUN_DIR", str(tmp_path / "dino/seed-2"))
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: True)
    seen = []
    monkeypatch.setattr(script, "evaluate", lambda config, *args, **kwargs: (
        seen.append((config.seed, config.trainer.default_root_dir, config.data.spatial_root))
        or {"bleu4": 1.0}))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--config", str(common_path),
                                    "--config", str(source_path),
                                    "--checkpoint", str(tmp_path / "model.ckpt"),
                                    "--generation", "deterministic", "--output",
                                    str(tmp_path / "test.json"), "seed=2"])
    script.main()
    assert seen == [(2, str(tmp_path / "dino/seed-2"), "dino")]


def test_comparison_text_only_annotation_drift_blocks_score_and_artifact(tmp_path, monkeypatch):
    import numpy as np
    from despamo.comparison import annotation_hashes
    from despamo.utils.hashing import sha256_file

    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    ann = tmp_path / "annotations"
    ann.mkdir()
    for split in ("train", "dev", "test"):
        np.save(ann / f"{split}_info_ml.npy",
                {0: {"fileid": f"{split}-id", "num_frames": 5, "text": "original"}})
    config.data.annotation_root = str(ann)
    config.data.spatial_root = str(tmp_path / "clip")
    config.data.batch_size = 2
    config.model.spatial_crop_mode = "full"
    config.model.name = "google/flan-t5-xl"
    config.model.revision = "7d6315df2c2fb742f0f5b556879d730926ca9001"
    config.model.vt_pooling = "masked_mean"
    config.model.prompt = "Translate the given sentence into {}."
    config.model.vt_weight = 1.0
    config.model.use_in_context = True
    config.model.num_in_context = 3
    config.model.warm_up_steps = 0
    config.model.lora_rank = 16
    config.model.lora_alpha = 32
    config.model.lora_dropout = 0.1
    config.model.spatial_dim = 2048
    config.model.motion_dim = 1024
    config.evaluation.generation = "deterministic"
    config.comparison = {"enabled": True, "smoke": False,
                         "clip_root": config.data.spatial_root,
                         "clip_manifest": config.data.spatial_manifest,
                         "dino_root": str(tmp_path / "dino"),
                         "dino_manifest": str(tmp_path / "dino/complete/manifest.json")}
    config.trainer = {"max_steps": 1000, "max_epochs": -1,
                      "accumulate_grad_batches": 2, "precision": "bf16"}
    config.optimizer = {"learning_rate": 6e-4, "weight_decay": 0.01}
    for modality in ("spatial", "motion"):
        Path(config.data[f"{modality}_manifest"]).write_text(modality)
    saved = {"seed": config.seed, "config": OmegaConf.to_container(config, resolve=True),
             "resume_reproducibility": "fresh",
             "spatial_manifest_sha256": sha256_file(Path(config.data.spatial_manifest)),
             "motion_manifest_sha256": sha256_file(Path(config.data.motion_manifest)),
             "annotation_sha256": annotation_hashes(config),
             "model_source": {"identifier": "google/flan-t5-xl", "tuning_type": "lora",
                              "revision_status": "resolved", "revision": config.model.revision}}
    torch.save({"pytorch-lightning_version": "1.9.5",
                "state_dict": FakeModel().state_dict(), "run_metadata": saved,
                "global_step": 1000,
                "optimizer_states": [{"state": {0: {"step": torch.tensor(1000.),
                                                    "exp_avg": torch.ones(1)}},
                                      "param_groups": [{"params": [0]}]}],
                "lr_schedulers": [{"last_epoch": 1000}]}, checkpoint)
    raw = np.load(ann / "train_info_ml.npy", allow_pickle=True).item()
    raw[0]["text"] = "changed text without changing clip ID or num_frames"
    np.save(ann / "train_info_ml.npy", raw)
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model built"))
    monkeypatch.setattr(script, "evaluate_translations",
                        lambda *args: pytest.fail("score calculated"))
    monkeypatch.setattr(script, "write_result_artifact",
                        lambda *args: pytest.fail("result recorded"))
    output = tmp_path / "drifted.json"
    with pytest.raises(ValueError, match="annotation SHA256 mismatch"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")
    assert not output.exists()
```

Insert this module-level constant in `tests/unit/test_train_cli.py` after existing imports (before test functions); do not import it from pending production code:

```python
FLAN_SHA = "7d6315df2c2fb742f0f5b556879d730926ca9001"
```

Existing `_fake_run(root=...)` already accepts explicit root and stores that root in generated config; use same `root` as comparison override in both fake CLI tests.

- [ ] **Step 1a: Add focused pinned-model and provenance red tests** (append to indicated files; existing fake Transformers/PEFT tests stay unchanged):

```python
# tests/unit/models/test_flan_t5.py
@pytest.mark.parametrize("loaded", ["7d6315df2c2fb742f0f5b556879d730926ca9001", "c" * 40])
def test_pinned_cached_revision_reaches_model_and_tokenizer_or_fails(monkeypatch, loaded):
    revision = "7d6315df2c2fb742f0f5b556879d730926ca9001"
    seen = []
    transformers = ModuleType("transformers")
    model = FakeModel()
    model.config = SimpleNamespace(_commit_hash=loaded)

    class T5:
        @staticmethod
        def from_pretrained(name, **kwargs):
            seen.append(("model", name, kwargs))
            return model

    class Tokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            seen.append(("tokenizer", name, kwargs))
            return FakeTokenizer()

    transformers.T5ForConditionalGeneration = T5
    transformers.AutoTokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    if loaded != revision:
        with pytest.raises(ValueError, match="Flan revision mismatch"):
            FlanT5Backbone.from_pretrained("flan", "/cache", 32, "freeze", 16, 32, .1,
                                           revision=revision)
        assert len(seen) == 1
    else:
        FlanT5Backbone.from_pretrained("flan", "/cache", 32, "freeze", 16, 32, .1,
                                       revision=revision)
        assert seen == [
            ("model", "flan", {"cache_dir": "/cache", "torch_dtype": torch.bfloat16,
                               "revision": revision, "local_files_only": True}),
            ("tokenizer", "flan", {"cache_dir": "/cache", "max_length": 32,
                                   "revision": revision, "local_files_only": True}),
        ]


def test_pinned_loader_rejects_mutable_revision_before_transformers_import():
    with pytest.raises(ValueError, match="immutable.*revision"):
        FlanT5Backbone.from_pretrained("flan", "/cache", 32, "freeze", 16, 32, .1,
                                       revision="main")
```

```python
# tests/unit/test_factory.py
def test_comparison_only_passes_immutable_revision_to_loader(monkeypatch):
    sha = "7d6315df2c2fb742f0f5b556879d730926ca9001"
    seen = []
    monkeypatch.setattr(FlanT5Backbone, "from_pretrained",
                        lambda *args, **kwargs: (seen.append((args, kwargs))
                                                 or DummyLanguageModel()))
    config = OmegaConf.create({
        "seed": 0,
        "comparison": {"enabled": True},
        "model": {"name": "google/flan-t5-xl", "cache_dir": "/cache", "revision": sha,
                  "spatial_dim": 2048, "motion_dim": 1024, "adapter_dim": 768,
                  "language_dim": 2048, "max_text_length": 64, "lora_rank": 16,
                  "lora_alpha": 32, "lora_dropout": .1, "vt_pooling": "masked_mean",
                  "vt_weight": 1.0, "warm_up_steps": 0,
                  "prompt": "Translate the given sentence into {}.",
                  "use_in_context": True, "num_in_context": 3},
        "optimizer": {"learning_rate": 6e-4, "weight_decay": .01},
    })
    build_model(config)
    assert seen[-1] == (("google/flan-t5-xl", "/cache", 64, "lora", 16, 32, .1),
                        {"revision": sha})
    config.comparison.enabled = False
    build_model(config)
    assert seen[-1][1] == {}
```

```python
# tests/unit/test_comparison.py (append; use existing FLAN_SHA and make_pair)
def test_comparison_checkpoint_requires_measured_language_revision(tmp_path, monkeypatch):
    from despamo.comparison import annotation_hashes, validate_comparison_checkpoint
    from omegaconf import OmegaConf

    clip, _, ann, _ = make_pair(tmp_path, monkeypatch)
    motion = tmp_path / "motion.json"
    motion.write_text("motion")
    clip.data.motion_manifest = str(motion)
    metadata = {"seed": 0, "config": OmegaConf.to_container(clip, resolve=True),
                "spatial_manifest_sha256": sha256_file(Path(clip.data.spatial_manifest)),
                "motion_manifest_sha256": sha256_file(motion),
                "annotation_sha256": annotation_hashes(clip),
                "resume_reproducibility": "fresh",
                "model_source": {"identifier": "google/flan-t5-xl", "tuning_type": "lora",
                                 "revision_status": "resolved", "revision": FLAN_SHA}}
    checkpoint = {"pytorch-lightning_version": "1.9.5",
                  "global_step": 1000, "run_metadata": metadata,
                  "state_dict": {"weight": torch.ones(1)},
                  "optimizer_states": [{"state": {0: {"step": torch.tensor(1000.),
                                                      "exp_avg": torch.ones(1)}},
                                        "param_groups": [{"params": [0]}]}],
                  "lr_schedulers": [{"last_epoch": 1000}]}
    validate_comparison_checkpoint(clip, checkpoint, "deterministic")
    checkpoint["run_metadata"]["model_source"]["revision"] = "c" * 40
    with pytest.raises(ValueError, match="revision"):
        validate_comparison_checkpoint(clip, checkpoint, "deterministic")
    checkpoint["run_metadata"]["model_source"]["revision"] = FLAN_SHA
    original = np.load(ann / "train_info_ml.npy", allow_pickle=True).item()
    original[0]["text"] = "different text"
    np.save(ann / "train_info_ml.npy", original)
    with pytest.raises(ValueError, match="annotation SHA256 mismatch"):
        validate_comparison_checkpoint(clip, checkpoint, "deterministic")
```

In `tests/unit/evaluation/test_evaluate_cli.py`, insert `_mark_converted_released(checkpoint)` immediately after `_fixture(...)` in **both** existing tests `test_cpu_fake_rejects_upstream_metrics_before_writing_success_artifact` and `test_explicit_baseline_acceptance_records_only_valid_upstream_result`. For each fake accepted file, pin that fixture's observed bytes to avoid treating synthetic bytes as the actual released artifact:

```python
monkeypatch.setattr(script, "RELEASED_CONVERTED_SHA256",
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest())
```

For a test isolating wrong source SHA256, pin the mutated fixture's digest too; the separately tested default digest guard must reject every other converted file before `build_model`.

- [ ] **Step 2: Verify red** with `uv run --locked python -m pytest tests/unit/models/test_flan_t5.py::test_pinned_loader_rejects_mutable_revision_before_transformers_import tests/unit/test_factory.py::test_comparison_only_passes_immutable_revision_to_loader tests/unit/test_train_cli.py::test_train_cli_rejects_dino_overridden_to_clip_before_model tests/unit/test_train_cli.py::test_comparison_saves_only_exact_step_checkpoint tests/unit/evaluation/test_evaluate_cli.py::test_evaluation_cli_applies_seed_override_to_same_layers tests/unit/evaluation/test_evaluate_cli.py::test_comparison_text_only_annotation_drift_blocks_score_and_artifact -q`; expect failures, not skips. Mutating only annotation `text` must fail before `build_model`, `evaluate_translations`, or `write_result_artifact`.
- [ ] **Step 3: Extend `src/despamo/comparison.py`** (reuse pure protocol checks in training and evaluation; no feature-array reread during checkpoint validation):

Move the protocol checks already in `preflight_sources` (comparison enabled; distinct/bound roots and manifests; DINO manifest location; fixed model, LoRA, trainer, optimizer, and decoding settings) into `validate_comparison_protocol(config: DictConfig) -> None`; add `config.model.get("tuning_type", "lora") == "lora"`. Keep feature/annotation/DINO version/file checks in `preflight_sources`, which calls the pure validator first. `validate_comparison_checkpoint` calls it too, before model construction. A config/checkpoint pair agreeing on a drifted protocol must still fail.

Use `Mapping` from `collections.abc` and `torch` for the full-state gate:

```python
def annotation_hashes(config: DictConfig) -> dict[str, str]:
    root = Path(config.data.annotation_root)
    return {split: sha256_file(root / f"{split}_info_ml.npy")
             for split in ("train", "dev", "test")}


def _require_full_lightning_checkpoint(checkpoint: dict) -> None:
    version = checkpoint.get("pytorch-lightning_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("comparison requires full Lightning checkpoint version")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()):
        raise ValueError("comparison checkpoint state_dict must contain tensors")
    optimizers = checkpoint.get("optimizer_states")
    if not isinstance(optimizers, list) or not optimizers:
        raise ValueError("comparison checkpoint requires optimizer_states")
    for optimizer in optimizers:
        if not isinstance(optimizer, Mapping):
            raise ValueError("comparison checkpoint optimizer state invalid")
        groups, state = optimizer.get("param_groups"), optimizer.get("state")
        if (not isinstance(groups, list) or not groups
                or any(not isinstance(group, Mapping)
                       or not isinstance(group.get("params"), list) or not group["params"]
                       or any(type(param) is not int for param in group["params"])
                       for group in groups)
                or not isinstance(state, Mapping) or not state):
            raise ValueError("comparison checkpoint optimizer state/param_groups missing")
        if not any(
                isinstance(moment, Mapping) and (
                    (isinstance(moment.get("step"), torch.Tensor)
                     and moment["step"].numel() == 1 and bool(moment["step"] == 1000))
                    or (type(moment.get("step")) in (int, float) and moment["step"] == 1000))
                for group in groups for param in group["params"]
                for moment in (state.get(param),)):
            raise ValueError("comparison checkpoint optimizer step must equal 1000")
    schedulers = checkpoint.get("lr_schedulers")
    if not isinstance(schedulers, list) or not schedulers or not all(
            isinstance(scheduler, Mapping) and scheduler for scheduler in schedulers):
        raise ValueError("comparison checkpoint requires lr_schedulers")


def validate_comparison_checkpoint(config: DictConfig, checkpoint: dict, generation: str) -> None:
    if generation != "deterministic" or config.evaluation.generation != "deterministic":
        raise ValueError("comparison requires deterministic generation")
    if config.comparison.smoke:
        raise ValueError("comparison test evaluation requires full 1000-step checkpoint")
    if type(checkpoint.get("global_step")) is not int or checkpoint["global_step"] != 1000:
        raise ValueError("comparison checkpoint step must equal 1000")
    _require_full_lightning_checkpoint(checkpoint)
    validate_comparison_protocol(config)
    saved = checkpoint.get("run_metadata")
    if not isinstance(saved, dict) or saved.get("resume_reproducibility") != "fresh":
        raise ValueError("comparison requires fresh training run_metadata")
    source = saved.get("model_source")
    if (not isinstance(source, dict) or source.get("identifier") != "google/flan-t5-xl"
            or source.get("tuning_type") != "lora" or source.get("revision_status") != "resolved"
            or source.get("revision") != FLAN_SHA
            or config.model.revision != FLAN_SHA):
        raise ValueError("comparison Flan revision mismatch in checkpoint provenance")
    if (saved.get("seed") != config.seed or saved.get("config") != OmegaConf.to_container(
            config, resolve=True)
            or saved.get("spatial_manifest_sha256") != sha256_file(
                Path(config.data.spatial_manifest))
            or saved.get("motion_manifest_sha256") != sha256_file(
                Path(config.data.motion_manifest))):
        raise ValueError("comparison checkpoint/config/manifest provenance mismatch")
    if saved.get("annotation_sha256") != annotation_hashes(config):
        raise ValueError("comparison annotation SHA256 mismatch")
```

- [ ] **Step 3a: Pin model and tokenizer only for comparison** (`src/despamo/models/flan_t5.py`, `src/despamo/factory.py` insertions; existing baseline positional calls remain untouched):

```python
# src/despamo/models/flan_t5.py: add next to future import
import re

# Add after lora_dropout: float in FlanT5Backbone.from_pretrained signature:
        *, revision: str | None = None,

# After tuning_type validation, before lazy imports:
        if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("immutable 40-hex Flan revision required")
        pinned = {"revision": revision, "local_files_only": True} if revision is not None else {}

# Replace existing model and tokenizer load statements (leave remaining PEFT code unchanged):
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16, **pinned
        )
        if revision is not None and getattr(model.config, "_commit_hash", None) != revision:
            raise ValueError("Flan revision mismatch in loaded model config")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, max_length=max_text_length, **pinned
        )
```

```python
# src/despamo/factory.py: in build_model(...) only, after config.model.lora_dropout,
# and before the closing parenthesis of FlanT5Backbone.from_pretrained(...):
        **({"revision": config.model.revision}
           if config.get("comparison", {}).get("enabled", False) else {}),
```

`Transformers` baseline 4.32.0 supports `revision` and `local_files_only` in model/tokenizer `from_pretrained`. The `revision is None` path supplies no new kwargs and retains existing frozen/released baseline call semantics. Comparing `_commit_hash` occurs **before** PEFT wrapping and tokenizer instantiation; a missing/incorrect cached revision fails closed without substituting `main` or downloading weights. `factory.build_model` passes the revision only when `comparison.enabled`.

- [ ] **Step 4: Change `scripts/train.py` at imports and `main`** (all insertions exact; leave existing baseline branches unchanged):

```python
# Add with existing despamo imports:
from despamo.comparison import annotation_hashes, preflight_run

# After validate_baseline_config(config), before pl.seed_everything(...):
    comparison = bool(config.get("comparison", {}).get("enabled", False))
    if comparison and args.resume is not None:
        raise ValueError("comparison runs must start fresh; --resume forbidden")
    if comparison:
        preflight_run(config, args.config, args.override)

# Inside existing try, immediately after `revision, status = _model_revision(model)`,
# before metadata["model_source"].update(...):
            if comparison and (status != "resolved" or revision != config.model.revision):
                raise ValueError("comparison Flan revision mismatch")

# After require_ignored_git_output(root, PROJECT_ROOT), before metadata creation:
    if comparison and (root.resolve() == PROJECT_ROOT or PROJECT_ROOT in root.resolve().parents):
        raise ValueError("comparison artifacts must live outside Git checkout")
    if comparison and (root / "checkpoints/final.ckpt").exists():
        raise ValueError("comparison checkpoint already exists; use a new run directory")

# Immediately after existing metadata = {...} construction, before entering
# snapshot_context (and before provisional run_metadata.json write):
    if comparison:
        metadata["annotation_sha256"] = annotation_hashes(config)

# After existing trainer.fit(...) branch finishes inside snapshot context:
        if comparison:
            if trainer.global_step != config.trainer.max_steps:
                raise ValueError("comparison stopped before exact optimizer-step budget")
            target = root / "checkpoints/final.ckpt"
            require_ignored_git_output(target, PROJECT_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(target)
            if not target.is_file():
                raise ValueError("comparison trainer did not write final checkpoint")
```

Note: `preflight_run(config, args.config, args.override)` reconstructs **both** resolved variants from the same ordered layers and seed override, checks pair equality (only two spatial keys differ), checks selected config exactly matches its overlay, then validates both feature sets before `build_model`. Training `pl.seed_everything(config.seed, workers=True)` precedes model construction; there is no implicit checkpoint import without `--resume`. Existing `SpaMoBaselineModule.ensure_finite_losses` aborts on NaN/Inf with clip IDs; `configure_optimizers` uses Lightning `estimated_stepping_batches` (1000 with `trainer.max_steps=1000`) for 100 warmup steps and cosine schedule. Do not edit `baseline_module.py`.

- [ ] **Step 5: Change `scripts/evaluate.py`** (imports plus guarded insertions):

```python
# Add with top-level standard library imports:
import re

# Add with existing despamo imports:
from despamo.comparison import annotation_hashes, validate_comparison_checkpoint

# Add after PROJECT_ROOT:
RELEASED_CONVERTED_SHA256 = "44ace3e8536817691f6c6b3104f8881fd65e6e1ac8ff92364a4af16efd93f45f"

# After original `if accept_baseline and generation != "upstream"` check:
    comparison = bool(config.get("comparison", {}).get("enabled", False))
    if accept_baseline and comparison:
        raise ValueError("--accept-baseline is only for released SpaMo upstream checkpoint")
    if comparison and generation != "deterministic":
        raise ValueError("comparison requires deterministic generation")

# After snapshot load + snapshot SHA256, select provenance:
        if comparison:
            checkpoint_metadata = checkpoint.get("run_metadata")
        else:
            checkpoint_metadata = (
                checkpoint["metadata"] if "metadata" in checkpoint else checkpoint["run_metadata"]
            )
        reject_secret_keys(checkpoint_metadata, "checkpoint_metadata")
        if comparison:
            validate_comparison_checkpoint(config, checkpoint, generation)
            if "metadata" in checkpoint and checkpoint["metadata"] != checkpoint_metadata:
                raise ValueError("comparison checkpoint has conflicting metadata")
        converted = checkpoint.get("metadata")
        if accept_baseline and (
            checkpoint_sha256 != RELEASED_CONVERTED_SHA256
            or "run_metadata" in checkpoint or not isinstance(checkpoint.get("metadata"), dict)
            or converted.get("source_tensor_count") != 871
            or converted.get("target_tensor_count") != 871
            or not isinstance(converted.get("source_path"), str)
            or converted.get("source_sha256") != (
                "06a432cdd8e1da4ce0b0e4cff246b20ad7d6a60406f32dfdbdfd974f94d3eee6"
            )
            or not re.fullmatch(r"[0-9a-f]{64}", str(converted.get("target_schema_sha256")))
            or config.model.get("vt_pooling", "legacy_mean") != "legacy_mean"
            or config.model.spatial_crop_mode != "upstream_random"
        ):
            raise ValueError("--accept-baseline requires converted released SpaMo "
                             "checkpoint SHA256 and provenance")

# After metadata dict is built and before write_result_artifact(...):
        if comparison:
            metadata["checkpoint_global_step"] = checkpoint["global_step"]
            metadata["comparison_source"] = (
                "clip" if Path(config.data.spatial_manifest) == Path(config.comparison.clip_manifest)
                else "dino"
            )
            if Path(config.data.spatial_manifest) not in {
                Path(config.comparison.clip_manifest), Path(config.comparison.dino_manifest)
            }:
                raise ValueError("comparison spatial manifest is not either controlled source")
            current_annotations = annotation_hashes(config)
            if current_annotations != checkpoint_metadata["annotation_sha256"]:
                raise ValueError("comparison annotation SHA256 mismatch before result write")
            metadata["annotation_sha256"] = current_annotations

# In main(), directly after parser.add_argument("--output", ...):
    parser.add_argument("override", nargs="*", help="OmegaConf key=value overrides")

# In main(), replace config = load_config(args.config):
    config = load_config(args.config, args.override)
```

Existing evaluation keeps `model.load_state_dict(checkpoint["state_dict"], strict=True)`, all-test ordered-ID check, frozen-feature forward, normal score calculation, no checkpoint migration. Comparison checkpoint preflight checks full Lightning state, optimizer step, pure controlled protocol, and all three current annotation file hashes before model construction or scoring; another check after inference and before result write prevents a changed annotation from producing a success artifact. Comparison result metadata always uses validated `run_metadata`; any conflicting `metadata` is rejected. The approved source SHA and **converted snapshot SHA256** above both must match for released acceptance; other 64-hex source hashes or converted bytes cannot claim acceptance. Baseline CLI behavior with zero overrides and baseline metadata remain unchanged. Comparison CLI passes same per-run `seed` override as training, so saved config/seed/annotation/manifest provenance matches checkpoint and report.

- [ ] **Step 6: Run** `uv run --locked python -m pytest tests/unit/test_comparison.py tests/unit/models/test_flan_t5.py tests/unit/test_factory.py tests/unit/test_train_cli.py tests/unit/evaluation/test_evaluate_cli.py -q`; expect pass. Run `uv run --locked ruff format scripts/train.py scripts/evaluate.py src/despamo/comparison.py src/despamo/models/flan_t5.py src/despamo/factory.py tests/unit/test_comparison.py tests/unit/test_factory.py tests/unit/models/test_flan_t5.py tests/unit/test_train_cli.py tests/unit/evaluation/test_evaluate_cli.py`, then `uv run --locked ruff check scripts/train.py scripts/evaluate.py src/despamo/comparison.py src/despamo/models/flan_t5.py src/despamo/factory.py tests/unit/test_comparison.py tests/unit/test_factory.py tests/unit/models/test_flan_t5.py tests/unit/test_train_cli.py tests/unit/evaluation/test_evaluate_cli.py`; expect no findings. Existing `tests/unit/test_train_cli.py::test_tiny_lightning_trainer_checkpoint_round_trip_through_cli` exercises full-state `trainer.save_checkpoint` with a fake CPU model; run it explicitly. No GPU needed to verify checkpoint-save semantics.

### Task 3: Offline Paired Six-Run Summary And Preflight CLI

**Files:** Extend `src/despamo/comparison.py`; create `scripts/compare_encoders.py`, `tests/unit/test_comparison_cli.py`.

**Interfaces:** `summarize(root: Path) -> dict` consumes `root/{clip,dino}/seed-{0,1,2}/test.json` from `write_result_artifact`. It rejects missing/incomplete/non-deterministic/accepted/swapped-ID runs and any train/dev/test annotation SHA drift (including text-only changes with same IDs) before producing a summary. Returns JSON-safe `runs`, shared `annotation_sha256`, per-source `mean`/`std`, and `paired_delta` (DINO minus CLIP); std is sample std (`statistics.stdev`, n=3). `scripts/compare_encoders.py preflight` loads both overlays and checks parity; `report --results-root ... --output ...` writes only to an external/ignored destination.

- [ ] **Step 1: Add failing fake report test** (`tests/unit/test_comparison_cli.py`):

```python
import hashlib
import json

import pytest

from despamo.comparison import summarize


def write_run(root, source, seed, ids, bleu4, rouge):
    revision = "7d6315df2c2fb742f0f5b556879d730926ca9001"
    protocol = {"enabled": True, "smoke": False, "clip_root": "clip",
                "clip_manifest": "clip.json", "dino_root": "dino",
                "dino_manifest": "dino.json"}
    config = {"seed": seed, "comparison": protocol,
              "data": {"spatial_root": source, "spatial_manifest": source + ".json",
                       "motion_manifest": "motion.json"},
              "trainer": {"max_steps": 1000, "default_root_dir": str(root / source / f"seed-{seed}")},
              "evaluation": {"generation": "deterministic", "beam_size": 5},
              "model": {"vt_pooling": "masked_mean", "spatial_crop_mode": "full",
                        "revision": revision}}
    digest = hashlib.sha256(source.encode()).hexdigest()
    annotations = {split: hashlib.sha256(f"{split}-annotations".encode()).hexdigest()
                   for split in ("train", "dev", "test")}
    metadata = {"seed": seed, "generation": "deterministic", "baseline_accepted": False,
                "spatial_crop_mode": "full", "checkpoint_global_step": 1000,
                "comparison_source": source,
                "checkpoint_sha256": hashlib.sha256(f"{source}-{seed}".encode()).hexdigest(),
                "spatial_manifest_sha256": digest,
                "motion_manifest_sha256": hashlib.sha256(b"motion").hexdigest(),
                "package_lock_sha256": hashlib.sha256(b"lock").hexdigest(),
                "annotation_sha256": dict(annotations),
                "checkpoint_metadata": {"config": config, "seed": seed,
                                        "spatial_manifest_sha256": digest,
                                        "motion_manifest_sha256": hashlib.sha256(b"motion").hexdigest(),
                                        "package_lock_sha256": hashlib.sha256(b"lock").hexdigest(),
                                        "annotation_sha256": dict(annotations),
                                        "resume_reproducibility": "fresh",
                                        "model_source": {"identifier": "google/flan-t5-xl",
                                                         "tuning_type": "lora",
                                                         "revision_status": "resolved",
                                                         "revision": revision}},
                "config": config}
    path = root / source / f"seed-{seed}/test.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"metadata": metadata,
        "metrics": {"bleu4": bleu4, "rougeL_f1": rouge},
        "items": [{"clip_id": item, "reference": "same text", "prediction": "prediction"}
                  for item in ids]}))
    return path


def test_pairwise_scores_use_three_matching_seeds_and_all_ids(tmp_path, monkeypatch):
    import despamo.comparison as comparison
    monkeypatch.setattr(comparison, "EXPECTED", {"train": 1, "dev": 1, "test": 2})
    for seed in (0, 1, 2):
        write_run(tmp_path, "clip", seed, ["a", "b"], 10.0 + seed, .3 + seed * .01)
        write_run(tmp_path, "dino", seed, ["a", "b"], 12.0 + seed, .4 + seed * .01)
    result = summarize(tmp_path)
    assert result["paired_delta"]["bleu4"] == {"mean": 2.0, "std": 0.0}
    assert result["paired_delta"]["rougeL_f1"]["mean"] == pytest.approx(.1)
    assert result["sources"]["clip"]["bleu4"] == {"mean": 11.0, "std": 1.0}
    assert len(result["runs"]) == 6
    (tmp_path / "dino/seed-1/test.json").unlink()
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path)


def test_rejects_unpaired_references_and_unfinished_checkpoint(tmp_path, monkeypatch):
    import despamo.comparison as comparison
    monkeypatch.setattr(comparison, "EXPECTED", {"train": 1, "dev": 1, "test": 2})
    for source in ("clip", "dino"):
        for seed in (0, 1, 2):
            write_run(tmp_path, source, seed, ["a", "b"], 10.0, .3)
    path = tmp_path / "dino/seed-0/test.json"
    payload = json.loads(path.read_text())
    payload["items"][0]["reference"] = "different"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="paired test IDs/references"):
        summarize(tmp_path)
    payload["items"][0]["reference"] = "same text"
    payload["metadata"]["checkpoint_global_step"] = 999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="1000 steps"):
        summarize(tmp_path)


def test_rejects_manifest_drift_across_seeds(tmp_path, monkeypatch):
    import despamo.comparison as comparison
    monkeypatch.setattr(comparison, "EXPECTED", {"train": 1, "dev": 1, "test": 2})
    for source in ("clip", "dino"):
        for seed in (0, 1, 2):
            write_run(tmp_path, source, seed, ["a", "b"], 10.0, .3)
    path = tmp_path / "dino/seed-2/test.json"
    payload = json.loads(path.read_text())
    original = json.loads(path.read_text())
    new_hash = hashlib.sha256(b"changed").hexdigest()
    payload["metadata"]["spatial_manifest_sha256"] = new_hash
    payload["metadata"]["checkpoint_metadata"]["spatial_manifest_sha256"] = new_hash
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="manifest drift"):
        summarize(tmp_path)
    original["metadata"]["checkpoint_metadata"]["model_source"]["revision"] = "c" * 40
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="Flan revision"):
        summarize(tmp_path)


def test_rejects_text_only_train_annotation_drift_across_six_runs(tmp_path, monkeypatch):
    import despamo.comparison as comparison

    monkeypatch.setattr(comparison, "EXPECTED", {"train": 1, "dev": 1, "test": 2})
    for source in ("clip", "dino"):
        for seed in (0, 1, 2):
            write_run(tmp_path, source, seed, ["a", "b"], 10.0, .3)
    path = tmp_path / "dino/seed-2/test.json"
    result = json.loads(path.read_text())
    changed = hashlib.sha256(b"same IDs and frame counts, changed train text").hexdigest()
    result["metadata"]["annotation_sha256"]["train"] = changed
    result["metadata"]["checkpoint_metadata"]["annotation_sha256"]["train"] = changed
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="train annotation SHA256 drift"):
        summarize(tmp_path)
```

- [ ] **Step 2: Run** `uv run --locked python -m pytest tests/unit/test_comparison_cli.py -q`; expect import error for `summarize`. On a skeleton `summarize` without annotation checks, rerun `uv run --locked python -m pytest tests/unit/test_comparison_cli.py::test_rejects_text_only_train_annotation_drift_across_six_runs -q`; expect red until all three hashes are enforced across six artifacts.
- [ ] **Step 3: Append to `src/despamo/comparison.py`** (add `import math` and `import statistics` with top imports):

```python
def summarize(root: Path) -> dict:
    runs = []
    reference_items = None
    reference_config = None
    spatial_hashes = {}
    motion_hash = None
    lock_hash = None
    annotation_reference = None
    for seed in (0, 1, 2):
        for source in ("clip", "dino"):
            path = root / source / f"seed-{seed}" / "test.json"
            payload = json.loads(path.read_text())
            meta, metrics, items = payload["metadata"], payload["metrics"], payload["items"]
            identities = [(item["clip_id"], item["reference"]) for item in items]
            if (len(identities) != EXPECTED["test"]
                    or len({item[0] for item in identities}) != len(identities)):
                raise ValueError(f"{path}: test must contain unique full split IDs")
            if reference_items is None:
                reference_items = identities
            elif identities != reference_items:
                raise ValueError(f"{path}: paired test IDs/references differ")
            if (meta["seed"] != seed or meta["comparison_source"] != source
                    or meta["generation"] != "deterministic" or meta["baseline_accepted"]
                    or meta["spatial_crop_mode"] != "full"
                    or meta.get("checkpoint_global_step") != 1000):
                raise ValueError(f"{path}: comparison requires deterministic 1000 steps")
            config = meta["config"]
            if (config["seed"] != seed or config["trainer"]["max_steps"] != 1000
                    or config["evaluation"]["beam_size"] != 5
                    or config["model"]["vt_pooling"] != "masked_mean"
                    or config["model"]["spatial_crop_mode"] != "full"
                    or config["model"]["revision"] != FLAN_SHA
                    or config["comparison"].get("enabled") is not True
                    or config["comparison"].get("smoke") is not False):
                raise ValueError(f"{path}: protocol mismatch")
            expected_root = config["comparison"][f"{source}_root"]
            expected_manifest = config["comparison"][f"{source}_manifest"]
            if (config["data"]["spatial_root"] != expected_root
                    or config["data"]["spatial_manifest"] != expected_manifest):
                raise ValueError(f"{path}: spatial source/config mismatch")
            if Path(config["trainer"]["default_root_dir"]).resolve() != (
                    root / source / f"seed-{seed}").resolve():
                raise ValueError(f"{path}: run root differs from seed/source output layout")
            saved = meta["checkpoint_metadata"]
            if (saved["seed"] != seed or saved["config"] != config
                    or saved["resume_reproducibility"] != "fresh"
                    or saved["spatial_manifest_sha256"] != meta["spatial_manifest_sha256"]
                    or saved["motion_manifest_sha256"] != meta["motion_manifest_sha256"]
                    or saved["package_lock_sha256"] != meta["package_lock_sha256"]):
                raise ValueError(f"{path}: checkpoint provenance mismatch")
            current_annotations = meta.get("annotation_sha256")
            if (not isinstance(current_annotations, dict)
                    or set(current_annotations) != {"train", "dev", "test"}
                    or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                           for value in current_annotations.values())
                    or saved.get("annotation_sha256") != current_annotations):
                raise ValueError(f"{path}: checkpoint annotation SHA256 mismatch")
            if annotation_reference is None:
                annotation_reference = current_annotations.copy()
            else:
                for split in ("train", "dev", "test"):
                    if current_annotations[split] != annotation_reference[split]:
                        raise ValueError(f"{path}: {split} annotation SHA256 drift across runs")
            model_source = saved["model_source"]
            if (model_source.get("identifier") != "google/flan-t5-xl"
                    or model_source.get("tuning_type") != "lora"
                    or model_source.get("revision_status") != "resolved"
                    or model_source.get("revision") != FLAN_SHA):
                raise ValueError(f"{path}: Flan revision mismatch")
            if source in spatial_hashes and spatial_hashes[source] != meta["spatial_manifest_sha256"]:
                raise ValueError(f"{path}: spatial manifest drift across seeds")
            spatial_hashes[source] = meta["spatial_manifest_sha256"]
            if motion_hash is not None and motion_hash != meta["motion_manifest_sha256"]:
                raise ValueError(f"{path}: motion manifest drift across runs")
            motion_hash = meta["motion_manifest_sha256"]
            if lock_hash is not None and lock_hash != meta["package_lock_sha256"]:
                raise ValueError(f"{path}: baseline lock drift across runs")
            lock_hash = meta["package_lock_sha256"]
            matched = json.loads(json.dumps(config))
            for key in ("spatial_root", "spatial_manifest"):
                del matched["data"][key]
            del matched["trainer"]["default_root_dir"]
            del matched["seed"]
            if reference_config is None:
                reference_config = matched
            elif matched != reference_config:
                raise ValueError(f"{path}: training configs differ beyond source/seed/output")
            for metric in ("bleu4", "rougeL_f1"):
                if (type(metrics.get(metric)) not in (float, int)
                        or not math.isfinite(metrics[metric])):
                    raise ValueError(f"{path}: invalid {metric}")
            if not isinstance(meta.get("checkpoint_sha256"), str) or not meta["checkpoint_sha256"]:
                raise ValueError(f"{path}: missing checkpoint hash")
            runs.append({"source": source, "seed": seed, "budget_optimizer_steps": 1000,
                         "generation": "deterministic", "beam_size": 5,
                         "test_items": len(items), "bleu4": metrics["bleu4"],
                         "rougeL_f1": metrics["rougeL_f1"],
                         "checkpoint_sha256": meta["checkpoint_sha256"],
                         "spatial_manifest_sha256": meta["spatial_manifest_sha256"],
                         "motion_manifest_sha256": meta["motion_manifest_sha256"],
                         "package_lock_sha256": meta["package_lock_sha256"],
                         "flan_revision": model_source["revision"],
                         "annotation_sha256": current_annotations.copy(),
                         "result_path": str(path)})
    summary = {}
    for source in ("clip", "dino"):
        summary[source] = {}
        for metric in ("bleu4", "rougeL_f1"):
            values = [run[metric] for run in runs if run["source"] == source]
            summary[source][metric] = {"mean": statistics.mean(values),
                                        "std": statistics.stdev(values)}
    differences = {}
    for metric in ("bleu4", "rougeL_f1"):
        values = [next(run[metric] for run in runs if run["source"] == "dino"
                       and run["seed"] == seed) -
                  next(run[metric] for run in runs if run["source"] == "clip"
                       and run["seed"] == seed) for seed in (0, 1, 2)]
        differences[metric] = {"mean": statistics.mean(values),
                               "std": statistics.stdev(values)}
    return {"runs": runs, "annotation_sha256": annotation_reference,
            "sources": summary, "paired_delta": differences}
```

- [ ] **Step 4: Create CLI** (`scripts/compare_encoders.py`, full contents):

```python
import argparse
import json
from pathlib import Path

from despamo.comparison import load_pair, preflight_sources, summarize, validate_pair_configs
from despamo.utils.output_policy import require_ignored_git_output

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled PHOENIX14T spatial-source comparison")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    report = commands.add_parser("report")
    report.add_argument("--results-root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        base = [ROOT / "configs/data/phoenix14t.yaml",
                ROOT / "configs/model/spamo_flan_t5_xl.yaml",
                ROOT / "configs/experiment/phoenix14t_baseline.yaml",
                ROOT / "configs/experiment/phoenix14t_encoder_comparison.yaml"]
        clip, dino = load_pair(base)
        validate_pair_configs(clip, dino)
        print(json.dumps(preflight_sources(clip), sort_keys=True))
        return
    require_ignored_git_output(args.output, ROOT)
    if args.output.exists():
        raise ValueError("comparison report exists; select a new output path")
    report_data = summarize(args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report_data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report_data["paired_delta"], sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run** `uv run --locked python -m pytest tests/unit/test_comparison_cli.py tests/unit/test_comparison.py -q` (expect pass); `uv run --locked ruff format scripts/compare_encoders.py src/despamo/comparison.py tests/unit/test_comparison_cli.py`, then `uv run --locked ruff check scripts/compare_encoders.py src/despamo/comparison.py tests/unit/test_comparison_cli.py` (expect no findings). Inspect status and diff; ensure report errors do not write partial success file.

### Task 4: Gate Execution And Reproducible Commands (documentation for later authorized session)

**Files:** No extra code. Review existing `configs/data/phoenix14t.yaml`, `configs/model/spamo_flan_t5_xl.yaml`, `configs/experiment/phoenix14t_baseline.yaml`, `scripts/train.py`, `scripts/evaluate.py` and the files above.

**Interfaces:** Comparison output layout `${COMPARISON_OUTPUT}/{clip,dino}/seed-{0,1,2}/checkpoints/final.ckpt` and `test.json`; bounded smoke layout `${COMPARISON_OUTPUT}/smoke/{clip,dino}/seed-0`. Six full training runs must be **fresh**; no `--resume` or pretrained visual-projector load.

- [ ] **Step 1: Run fake-only verification in baseline environment:**

```bash
uv run --locked python -m pytest tests/unit -q
uv run --locked ruff check src/despamo/comparison.py src/despamo/models/flan_t5.py src/despamo/factory.py scripts/compare_encoders.py scripts/train.py scripts/evaluate.py tests/unit/test_comparison.py tests/unit/test_comparison_cli.py tests/unit/test_factory.py tests/unit/models/test_flan_t5.py tests/unit/test_train_cli.py tests/unit/evaluation/test_evaluate_cli.py
git diff --check
git status --short
git diff -- configs/experiment src/despamo/comparison.py src/despamo/models/flan_t5.py src/despamo/factory.py scripts/train.py scripts/evaluate.py scripts/compare_encoders.py tests/unit
```

Expected: focused tests pass without downloads; baseline tests still pass; root locks unmodified; inspect newly created untracked files directly (`git diff` does not show them).

- [ ] **Step 2: After separate full extraction authorization/completion**, use already exported baseline `PHOENIX14T_ANNOTATION_ROOT`, `DESPAMO_FEATURE_ROOT`, existing Flan-T5-XL `DESPAMO_HF_CACHE`, and `DESPAMO_DINO_ROOT` produced by extraction plan Task 6 Step 7. Set `COMPARISON_OUTPUT` to an external Linux-native directory; no `.env` or token is read by translation. Run:

```bash
: "${PHOENIX14T_ANNOTATION_ROOT:?set approved baseline annotation root}"
: "${DESPAMO_FEATURE_ROOT:?set approved baseline feature root}"
: "${DESPAMO_HF_CACHE:?set existing Flan-T5-XL cache}"
: "${DESPAMO_DINO_ROOT:?complete extraction first}"
: "${COMPARISON_OUTPUT:?set external comparison output root}"
export DINO_ROOT="$DESPAMO_DINO_ROOT"
COMPARISON_RUN_DIR="$COMPARISON_OUTPUT/preflight" uv run --locked python scripts/compare_encoders.py preflight
```

Expected JSON: `{"clip":{"dev":519,"test":642,"train":7096},"dino":{"dev":519,"test":642,"train":7096}}`; 8257 CLIP and DINO ID/row/file checks, DINO file hashes, complete frame map and zero failures. Before reading features, `validate_pair_configs` binds each resolved source to its named root/manifest; `preflight_run` repeats the selected-source binding for every training command. Never pass `data.spatial_root` or `data.spatial_manifest` dotlist overrides to switch encoders: even if both variants resolve to CLIP, run stops before model build. Missing gated DINO weights/full features => stop, report comparison gate pending. This command reads metadata/features only; never invokes `build_model`.

- [ ] **Step 3: Only after separate permission for bounded real translation smoke** (one per source, same seed and two optimizer steps to exercise both VT-only warm-up and translation backward; do not treat as full score):

```bash
for source in clip dino; do
  export COMPARISON_RUN_DIR="$COMPARISON_OUTPUT/smoke/$source/seed-0"
  if [ "$source" = clip ]; then overlay=configs/experiment/phoenix14t_clip_control.yaml; else overlay=configs/experiment/phoenix14t_dinov3.yaml; fi
  uv run --locked python scripts/train.py \
    --config configs/data/phoenix14t.yaml \
    --config configs/model/spamo_flan_t5_xl.yaml \
    --config configs/experiment/phoenix14t_baseline.yaml \
    --config configs/experiment/phoenix14t_encoder_comparison.yaml \
    --config "$overlay" seed=0 comparison.smoke=true trainer.max_steps=2
  export COMPARISON_OVERLAY="$overlay"
  uv run --locked python -c '
import os
from pathlib import Path
import torch
from despamo.comparison import annotation_hashes
from despamo.config import load_config
from despamo.factory import build_model

root = Path(os.environ["COMPARISON_RUN_DIR"])
checkpoint = torch.load(root / "checkpoints/final.ckpt", map_location="cpu", weights_only=True)
assert checkpoint["global_step"] == 2
assert checkpoint["run_metadata"]["resume_reproducibility"] == "fresh"
assert checkpoint["run_metadata"]["model_source"]["revision"] == "7d6315df2c2fb742f0f5b556879d730926ca9001"
assert checkpoint["optimizer_states"] and checkpoint["lr_schedulers"]
config = load_config([Path(p) for p in (
    "configs/data/phoenix14t.yaml", "configs/model/spamo_flan_t5_xl.yaml",
    "configs/experiment/phoenix14t_baseline.yaml",
    "configs/experiment/phoenix14t_encoder_comparison.yaml",
    os.environ["COMPARISON_OVERLAY"],
    )], ["seed=0", "comparison.smoke=true", "trainer.max_steps=2"])
assert checkpoint["run_metadata"]["annotation_sha256"] == annotation_hashes(config)
build_model(config).load_state_dict(checkpoint["state_dict"], strict=True)
'
done
```

Expected: both `run_metadata.json` say `fresh`, `checkpoints/final.ckpt` load safely, report `global_step == 2`, finite training losses (otherwise existing `ensure_finite_losses` raises), LoRA optimizer moments after translation backward, full optimizer state and strict model reload. Do not run test evaluation on two-step checkpoints or reuse these checkpoints for full runs.

- [ ] **Step 4: Ask separately before six full GPU training runs.** Once authorized, execute exactly six **fresh** runs. `COMPARISON_RUN_DIR` is external; pair CLIP and DINO with same seed. Per-run command is fully specified by this loop:

```bash
for seed in 0 1 2; do
  for source in clip dino; do
    export COMPARISON_RUN_DIR="$COMPARISON_OUTPUT/$source/seed-$seed"
    if [ "$source" = clip ]; then overlay=configs/experiment/phoenix14t_clip_control.yaml; else overlay=configs/experiment/phoenix14t_dinov3.yaml; fi
    uv run --locked python scripts/train.py \
      --config configs/data/phoenix14t.yaml \
      --config configs/model/spamo_flan_t5_xl.yaml \
      --config configs/experiment/phoenix14t_baseline.yaml \
      --config configs/experiment/phoenix14t_encoder_comparison.yaml \
      --config "$overlay" seed="$seed"
  done
done
```

Expected per run: both named spatial roots/manifests bound before feature I/O; full resolved CLIP/DINO config equality except those two spatial keys, both full-split feature preflights before Flan weight load, fresh adapter/LoRA initialization after `pl.seed_everything(seed, workers=True)`, pinned model/tokenizer commit and run metadata revision, full-byte SHA256 of train/dev/test annotation files recorded in run metadata/checkpoint, exactly 1000 **optimizer** steps at accumulation 2, one final full-state checkpoint (`global_step=1000`) and finite losses. `max_epochs=-1` prevents 500-epoch cap from ending early. A crash, mismatched hash, OOM or non-finite loss stops that run; do not silently change precision/batch/step budget or count it as completed.

- [ ] **Step 5: After all six checkpoints exist, evaluate deterministic full test using same per-run layers (never pass `--accept-baseline`):**

```bash
for seed in 0 1 2; do
  for source in clip dino; do
    export COMPARISON_RUN_DIR="$COMPARISON_OUTPUT/$source/seed-$seed"
    if [ "$source" = clip ]; then overlay=configs/experiment/phoenix14t_clip_control.yaml; else overlay=configs/experiment/phoenix14t_dinov3.yaml; fi
    uv run --locked python scripts/evaluate.py \
      --config configs/data/phoenix14t.yaml \
      --config configs/model/spamo_flan_t5_xl.yaml \
      --config configs/experiment/phoenix14t_baseline.yaml \
      --config configs/experiment/phoenix14t_encoder_comparison.yaml \
      --config "$overlay" \
      --checkpoint "$COMPARISON_RUN_DIR/checkpoints/final.ckpt" \
      --generation deterministic --output "$COMPARISON_RUN_DIR/test.json" seed="$seed"
  done
done
uv run --locked python scripts/compare_encoders.py report \
  --results-root "$COMPARISON_OUTPUT" --output "$COMPARISON_OUTPUT/comparison-report.json"
```

Expected: evaluation reuses per-run resolved seed and `COMPARISON_RUN_DIR`; before scoring, current train/dev/test annotation SHA256s must match training checkpoint, including train/dev text unseen by test-ID parity. Strict checkpoint load; 642 identical ordered IDs/references per run, beam 5, no sampling, no acceptance flag, result artifacts with `annotation_sha256`, `checkpoint_sha256`, `spatial_manifest_sha256`, `motion_manifest_sha256`, `checkpoint_global_step=1000`. Offline report requires each of three annotation hashes equal across **all six** jobs, includes six per-run metrics, locked Flan revision and baseline-lock hash, and paired `dino - clip` means/sample std; existing accepted released upstream checkpoint/score is reported separately, never used as paired CLIP control. No improvement threshold.

## Self-Review And Handoff

- **Spec/reviewer coverage:** §1 objective/source isolation Task 1/2; §2 separate extractor environment/no root lock changes global constraints; §§3–4 complete feature version/row mapping Task 1; §5 matched overlays, prompt/VT weight/optimizer and model+tokenizer revision checked in Tasks 1–3; independent full-state seeded adapter/**fake LoRA** parity test Task 1; full resolved CLIP and DINO source binding before stripping spatial keys or reading files, including a dotlist-override fake training CLI rejection before `build_model` Task 1/2; both full-split manifests and both resolved configs before every model build via `preflight_run` Task 1/2; exactly 1000 steps × three seeds and paired full-test means/std Tasks 2–4; evaluation positional `seed` override red test, parser insertion and six loop commands Task 2/4; **train/dev/test full-file annotation SHA256** in training metadata/checkpoint, before-score evaluation validation, before-write recheck, all-six report gate, and text-only mutation fake tests Tasks 2–4; §6 strict load, finite loss, exact released source SHA and wrong-converted-source rejection Task 2; §7 bounded smoke, explicit authorization and deferred full GPU gate Task 4. `uv run --locked python -m pytest tests/unit -q` checks baseline alongside new fake tests. No seven-factor work.
- **Signature/contract review:** `load_pair(base, overrides)` returns (clip,dino), applying same CLI seed/smoke overrides after source overlay; `_require_source_binding(config, source)` checks **both** root and manifest against named `comparison.*` keys; `validate_pair_configs` calls it for both variants before normalizing only two spatial fields; `preflight_run(config, paths, overrides)` requires final source overlay, binds selected config and paired config, then calls `preflight_sources`. `annotation_hashes(config: DictConfig) -> dict[str,str]` maps `train|dev|test` to exact `.npy` SHA256 and is used by `scripts/train.py`, `scripts/evaluate.py`, and `validate_comparison_checkpoint`; only comparison metadata receives `annotation_sha256`. `FlanT5Backbone.from_pretrained(..., *, revision=None)` keeps baseline positional call intact; factory supplies revision only for comparison. `_model_revision` verifies post-PEFT base commit before `fit`; `validate_comparison_checkpoint` checks resolved model revision, seed/config/annotation/feature manifest hashes and 1000 steps before model/score; `summarize` checks revision and three annotation hashes in all six checkpoint/result records, normalizing only verified `trainer.default_root_dir`, source paths and seed. Extraction `FeatureRecord.path` is relative to encoder root; row-map SHA references `complete/manifest.json` bytes. Baseline env never imports `tools.dinov3`. All Python fences are complete new files or explicitly labelled insertions with defined imports; no placeholders. Report path: `${COMPARISON_OUTPUT}/comparison-report.json`.
- **Remaining live gates/concerns:** fake row map does not independently verify raw PNG content (offline extractor owns that); completed DINO version and content hashes are checked in baseline environment. `preflight_sources` hashes every DINO `.npy` on every training run; estimate CPU I/O before scheduling full GPU jobs. Pinned Flan cache must actually expose exact `_commit_hash=7d6315df2c2fb742f0f5b556879d730926ca9001`; missing model/tokenizer snapshots cause explicit local-only failure, not `main` fallback. Existing released converter metadata includes approved source digest and 871-tensor counts. Final checkpoint is written after `fit`, so stopped-early run cannot publish passing 1000-step artifact. No real feature/weight tests, GPU extraction or six training runs without respective access and authorizations.
- **Completion criteria for plan execution:** fake suite/lint pass; real preflight zero mismatch; two authorized bounded smokes complete with finite losses; separately authorized six 1000-step fresh checkpoints, six deterministic full-test artifacts; complete paired report and reviewed status/diff. Until HF license/weights/full extraction and GPU authorizations are available, status is `DONE_WITH_CONCERNS` with live/full gates pending.
