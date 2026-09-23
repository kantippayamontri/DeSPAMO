# Dual-Path Seven-Factor Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four nuisance GRLs, three positive articulator heads, independent probes, strict inference export, and reproducible 11-variant/three-seed experiments.

**Architecture:** Attach all auxiliary supervision to the existing trainable spatial projector. Clip means feed nuisance heads; exactly mapped sampled rows feed articulator heads. Retain baseline checkpoint names for shared translation tensors, and keep captions/text encoders outside inference.

**Tech Stack:** Baseline Python 3.11, PyTorch 2.0.1, Lightning 1.9.5, Transformers 4.32.0, OmegaConf 2.3.0, NumPy 1.26.4, pytest/Ruff. No new runtime framework.

## Global Constraints

- Execute after baseline, DINOv3, and `2026-09-23-seven-factor-dataset.md` gates. Recheck planned baseline interfaces against implemented code before applying insertions.
- `STABLE = (biometric, clothing, hair, background)` uses GRL; `ARTICULATORS = (left_hand, right_hand, mouth)` never does.
- Seven independent linear heads map adapter width 768 to manifest-declared CLIP text width. No shared head parameters.
- Frozen visual/text encoders; shared spatial projector receives both translation and auxiliary gradients.
- Masked pooling, exact source-index mapping, full unflipped spatial sequences (`model.spatial_crop_mode: full`). Never use caption frame ordinal directly as feature row without a mapping.
- Symmetric uniform multi-positive cosine contrast, temperature 0.07; skip <2 valid targets or <2 distinct strings.
- Positive weights 1.0 initially. Only GRL reverses gradients. Ramp after baseline VT warm-up over 10% of joint optimizer steps.
- One device, no cross-rank negatives. Three seeds: 0, 1, 2. No experiment launch is authorized by writing this plan.
- Fresh probes only; clip-disjoint partitions; fixed unique-canonical gallery. Signer classification is closed-set; unseen-signer translation is separate.
- Keep a nuisance factor only when matching Recall@1 or signer top-1 decreases and mean BLEU-4 loss <=0.5 points. Articulators require increased Recall@1, no signer top-1 increase, and BLEU-4 loss <=0.5.
- No commits without explicit request. Checkpoint files and generated reports stay outside Git.

## File Map

| Files | Responsibility |
|---|---|
| `src/despamo/data/factors.py` | Validate dataset/encoder/audit identity, source-to-row mapping, cached targets, collation |
| `src/despamo/losses/factor_contrast.py` | Stable masked multi-positive objective |
| `src/despamo/models/factor_heads.py` | GRL and seven independent heads |
| `src/despamo/models/factor_adapter.py` | Baseline-compatible adapter exposing projected spatial rows |
| `src/despamo/training/factor_module.py` | Warm-up/ramp, weighted losses, metadata |
| `src/despamo/training/factor_factory.py` | Compose baseline factory with factor dataset/module |
| `src/despamo/evaluation/factor_probes.py` | Split validation, fresh probes, retrieval metrics |
| `src/despamo/evaluation/factor_experiments.py` | Variant matrix, paired summaries and retention |
| `src/despamo/evaluation/factor_export.py` | Strict shared-only inference checkpoint |
| `scripts/train_factors.py`, `scripts/probe_factors.py`, `scripts/factor_experiments.py`, `scripts/export_factors.py` | Explicit research commands |
| `configs/experiment/seven_factors.yaml` | Local-path environment interpolations and factor weights |

## Entry Contract: DINOv3 Frame-Row Manifest

Baseline `FeatureManifest` does not contain source indices. DINOv3 prerequisite must provide an additional JSON manifest at `factors.frame_rows`. It is a required artifact, not guessed retrospectively from feature lengths:

```text
spatial_manifest_hash: SHA-256 of baseline-compatible DINO feature manifest
clips: map keyed by clip_id
  feature_hash: SHA-256 of that clip's .npy spatial file
  source_indices: ordered unique source ordinals, one per feature row
  sampled_images: map from sampled source ordinal string to image SHA-256
```

Only training split entries are needed for auxiliary training/probes. A changed frame map or spatial feature hash invalidates the experiment identity. The code below rejects missing mappings; it does not implement DINOv3 extraction in this stage.

## Task 1: Validated Targets And Exact Frame Alignment

**Files:** Create `data/factors.py`, `tests/unit/data/test_factors.py`.

**Interfaces:** `FactorBatch(base: PhoenixBatch, rows: Tensor[B,5], vectors: dict[str,Tensor], masks: dict[str,Tensor], labels: dict[str,tuple[str,...]])`; `FactorDataset(base, dataset, text_manifest, frame_rows, spatial_manifest_path, enabled)`; `collate_factors(samples) -> FactorBatch`. Stable tensors `[B,D]`, frame tensors `[B,5,D]`; labels follow row-major flattening. Invalid positions use -1 and all corresponding masks false.

- [ ] **Step 1: Add alignment regression tests.**

```python
# tests/unit/data/test_factors.py
import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from despamo.appearance.audit import review_template, select_units, summarize
from despamo.appearance.provenance import atomic_json, file_hash, load_dataset
from despamo.appearance.schema import FACTORS
from despamo.appearance.text_features import encode_dataset
from despamo.data.batch import PhoenixSample
from despamo.data.factors import FactorDataset, collate_factors, map_rows
from tests.unit.appearance.helpers import prepared_dataset


def test_source_index_is_not_assumed_to_be_row():
    assert map_rows([3, 7], [1, 3, 5, 7], 4) == [1, 3]


@pytest.mark.parametrize("requested,source,length", [
    ([4], [1, 3, 5], 3), ([3], [1, 3, 3], 3), ([3], [1, 3], 3),
])
def test_missing_duplicate_or_wrong_length_mapping_fails(requested, source, length):
    with pytest.raises(ValueError):
        map_rows(requested, source, length)


def test_cached_targets_join_and_failed_clip_stays_for_translation(tmp_path):
    dataset = prepared_dataset(tmp_path)
    source, _ = load_dataset(dataset)
    units = select_units(dataset)
    reviews = review_template(units)
    for row in reviews:
        hand = row["factor"] in {"left_hand", "right_hand"}
        row.update(actual_visible="1", correct="1", visibility_correct="1", unsupported="0",
                   forbidden="0", side_auditable="1" if hand else "0", side_correct="1" if hand else "")
    audit = dataset / "audit"
    atomic_json(audit / "selection.json", units)
    with (audit / "review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviews[0]))
        writer.writeheader()
        writer.writerows(reviews)
    atomic_json(audit / "gate.json", summarize(dataset, units, reviews))
    encoder = SimpleNamespace(width=4, revision="0" * 40,
                              encode=lambda texts: np.ones((len(texts), 4), dtype=np.float32))
    text_manifest = encode_dataset(dataset, encoder)
    feature = tmp_path / "spatial.npy"
    np.save(feature, np.ones((11, 4), dtype=np.float32))
    spatial_manifest = tmp_path / "spatial.json"
    atomic_json(spatial_manifest, {"fixture": True})
    mappings = {c["clip_id"]: dict(feature_hash=file_hash(feature), source_indices=list(range(11)),
                sampled_images={str(f["index"]): f["sha256"] for f in c["frames"]}) for c in source["clips"]}
    frame_rows = tmp_path / "rows.json"
    atomic_json(frame_rows, dict(spatial_manifest_hash=file_hash(spatial_manifest), clips=mappings))

    class Base:
        split = "train"
        spatial_root = tmp_path
        records = [{"fileid": c["clip_id"]} for c in source["clips"]]
        spatial_manifest = SimpleNamespace(require=lambda *args: SimpleNamespace(path="spatial.npy"))

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return PhoenixSample(self.records[index]["fileid"], "s", "text", "", "", "", "",
                                 torch.ones(11, 4), torch.ones(5, 3))

    result = FactorDataset(Base(), dataset, text_manifest, frame_rows, spatial_manifest, list(FACTORS))
    batch = collate_factors([result[0], result[len(result) - 1]])
    assert len(batch.base.clip_ids) == 2
    assert all(not batch.masks[f][-1].any() for f in FACTORS)
    assert batch.rows[0].tolist() == [1, 3, 5, 7, 9]
    assert batch.vectors["mouth"].shape == (2, 5, 4)
```

- [ ] **Step 2: Run `uv run pytest tests/unit/data/test_factors.py -q`; expect missing module.**
- [ ] **Step 3: Implement target dataset and collation.**

```python
# src/despamo/data/factors.py
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from despamo.appearance.audit import select_units, summarize
from despamo.appearance.canonical import targets
from despamo.appearance.generation import load_records
from despamo.appearance.provenance import digest, file_hash, load_dataset, read_json
from despamo.appearance.schema import ARTICULATORS, FACTORS, STABLE, Record
from despamo.appearance.text_features import MODEL
from despamo.data.batch import PhoenixBatch, collate_phoenix


@dataclass(frozen=True)
class FactorBatch:
    base: PhoenixBatch
    rows: torch.Tensor
    vectors: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    labels: dict[str, tuple[str, ...]]


def map_rows(requested: list[int], source: list[int], length: int) -> list[int]:
    if len(source) != length or len(set(source)) != length:
        raise ValueError("invalid source-to-feature-row mapping")
    lookup = {index: row for row, index in enumerate(source)}
    if any(index not in lookup for index in requested):
        raise ValueError("sampled source index has no feature row")
    return [lookup[index] for index in requested]


class FactorDataset(Dataset):
    def __init__(self, base, dataset: Path, text_manifest: Path, frame_rows: Path,
                 spatial_manifest_path: Path, enabled: list[str]):
        import csv

        self.base = base
        self.source, _ = load_dataset(dataset)
        if self.source["split"] != "train":
            raise ValueError("factor training requires training-split captions")
        self.clips = {c["clip_id"]: c for c in self.source["clips"]}
        if set(self.clips) != {r["fileid"] for r in base.records}:
            raise ValueError("caption/translation split mismatch")
        journals = load_records(dataset)
        journal_by_id = {r["clip_id"]: r for r in journals}
        gate = read_json(dataset / "audit/gate.json")
        selection = read_json(dataset / "audit/selection.json")
        with (dataset / "audit/review.csv").open(newline="") as handle:
            reviewed = list(csv.DictReader(handle))
        if selection != select_units(dataset) or gate != summarize(dataset, selection, reviewed):
            raise ValueError("audit gate is stale")
        if not gate["global_pass"] or any(not gate["factor_pass"][f] for f in enabled):
            raise ValueError("requested factors failed audit/coverage")
        self.manifest = read_json(text_manifest)
        meta = self.manifest["metadata"]
        if (self.manifest["dataset_key"] != dataset.name or meta["schema_version"] != 2
                or meta["model"] != MODEL or meta["revision"] != meta["tokenizer_revision"]
                or self.manifest["encoder_key"] != digest(meta)):
            raise ValueError("text manifest identity mismatch")
        self.width = meta["width"]
        self.text_root = text_manifest.parent
        self.entries = {r["clip_id"]: r for r in self.manifest["records"]}
        if len(self.entries) != len(self.manifest["records"]) or set(self.entries) != set(self.clips):
            raise ValueError("missing/duplicate/extra text clips")
        for clip_id, entry in self.entries.items():
            if entry["record_hash"] != digest(journal_by_id[clip_id]):
                raise ValueError("text source-record hash mismatch")
            raw = journal_by_id[clip_id]["record"]
            expected_targets = targets(Record.model_validate(raw)) if raw else []
            if entry["targets"] != expected_targets:
                raise ValueError("text targets differ from validated source record")
        mapping = read_json(frame_rows)
        if mapping["spatial_manifest_hash"] != file_hash(spatial_manifest_path):
            raise ValueError("DINO source map belongs to another manifest")
        self.mapping = mapping["clips"]
        self.provenance = dict(dataset_key=dataset.name, text_manifest_hash=file_hash(text_manifest),
                               frame_rows_hash=file_hash(frame_rows), audit_hash=digest(gate))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        clip = self.clips[sample.clip_id]
        entry = self.entries[sample.clip_id]
        spatial = self.base.spatial_manifest.require(self.base.split, sample.clip_id)
        mapping = self.mapping[sample.clip_id]
        if file_hash(self.base.spatial_root / spatial.path) != mapping["feature_hash"]:
            raise ValueError(f"DINO feature hash mismatch: {sample.clip_id}")
        requested = [f["index"] for f in clip["frames"]]
        for frame in clip["frames"]:
            if mapping["sampled_images"].get(str(frame["index"])) != frame["sha256"]:
                raise ValueError("caption/DINO source-image mismatch")
        rows = map_rows(requested, mapping["source_indices"], sample.spatial.shape[0])
        rows = rows if rows else [-1] * 5
        vectors = {f: torch.zeros((1 if f in STABLE else 5, self.width)) for f in FACTORS}
        masks = {f: torch.zeros(1 if f in STABLE else 5, dtype=torch.bool) for f in FACTORS}
        labels = {f: [""] * (1 if f in STABLE else 5) for f in FACTORS}
        path = self.text_root / entry["path"]
        if file_hash(path) != entry["file_hash"]:
            raise ValueError("text feature hash mismatch")
        with np.load(path, allow_pickle=False) as saved:
            values, valid = saved["vectors"], saved["valid"]
        if values.shape != (len(entry["targets"]), self.width) or not np.isfinite(values).all():
            raise ValueError("invalid target dimensions/values")
        if values.dtype != np.float32 or valid.dtype != np.bool_ or valid.shape != values.shape[:1]:
            raise ValueError("invalid target dtype/mask")
        seen = set()
        for i, target in enumerate(entry["targets"]):
            factor, frame = target["factor"], target["frame_index"]
            identity = (factor, frame)
            if factor not in FACTORS or identity in seen:
                raise ValueError("unknown/duplicate factor target")
            seen.add(identity)
            slot = 0 if factor in STABLE and frame == -1 else requested.index(frame)
            if bool(valid[i]) != (target["text"] is not None):
                raise ValueError("canonical/feature validity mismatch")
            vectors[factor][slot] = torch.from_numpy(values[i].copy())
            masks[factor][slot] = bool(valid[i])
            labels[factor][slot] = target["text"] or ""
        expected = {(f, -1) for f in STABLE} | {(f, i) for i in requested for f in ARTICULATORS}
        if seen and seen != expected:
            raise ValueError("incomplete factor target set")
        return sample, torch.tensor(rows), vectors, masks, labels


def collate_factors(samples) -> FactorBatch:
    base, rows, vectors, masks, labels = zip(*samples, strict=True)
    output_vectors, output_masks, output_labels = {}, {}, {}
    for factor in FACTORS:
        output_vectors[factor] = torch.stack([v[factor] for v in vectors])
        output_masks[factor] = torch.stack([m[factor] for m in masks])
        output_labels[factor] = tuple(s for item in labels for s in item[factor])
        if factor in STABLE:
            output_vectors[factor] = output_vectors[factor][:, 0]
            output_masks[factor] = output_masks[factor][:, 0]
    return FactorBatch(collate_phoenix(base), torch.stack(rows),
                       output_vectors, output_masks, output_labels)
```

- [ ] **Step 4: Repeat mapping/join tests and baseline dataset tests; expect pass. The offline join fixture builds 1,001 synthetic clip records so the real coverage floor is exercised, not bypassed. Task 4 separately verifies mixed-mask backward behavior.**

## Task 2: Multi-Positive Objective And Seven Gradient Paths

**Files:** Create `losses/factor_contrast.py`, `models/factor_heads.py`, `tests/unit/models/test_factor_heads.py`.

**Interfaces:** `multi_positive(visual[N,D], text[N,D], labels, valid[N]) -> Tensor`; `FactorHeads.forward(projected[B,T,H], mask[B,T], batch, alpha) -> (losses, counts)`. `alpha` changes shared-feature gradient only, never head gradient.

- [ ] **Step 1: Write per-factor gradient-sign regression tests.**

```python
# tests/unit/models/test_factor_heads.py
import pytest
import torch

from despamo.appearance.schema import FACTORS, STABLE
from despamo.losses.factor_contrast import multi_positive
from despamo.models.factor_heads import reverse


@pytest.mark.parametrize("factor", FACTORS)
def test_each_factor_has_expected_shared_gradient(factor):
    x = torch.tensor([[1.0, 0.2], [0.3, 1.0]], requires_grad=True)
    weight = torch.eye(2, requires_grad=True)
    target = torch.tensor([[0.2, 1.0], [1.0, 0.3]])
    labels, valid = ("a", "b"), torch.ones(2, dtype=torch.bool)
    ordinary = multi_positive(x @ weight, target, labels, valid)
    gx, gw = torch.autograd.grad(ordinary, (x, weight))
    routed = reverse(x, 0.5) if factor in STABLE else x
    actual = multi_positive(routed @ weight, target, labels, valid)
    actual_x, actual_w = torch.autograd.grad(actual, (x, weight))
    torch.testing.assert_close(actual_x, gx * (-0.5 if factor in STABLE else 1))
    torch.testing.assert_close(actual_w, gw)


def test_all_positive_or_masked_batches_skip_without_nan():
    x = torch.randn(3, 4, requires_grad=True)
    for labels, mask in [(('a', 'a', 'a'), torch.ones(3, dtype=torch.bool)),
                         (('a', 'b', 'c'), torch.zeros(3, dtype=torch.bool))]:
        loss = multi_positive(x, x.detach(), labels, mask)
        assert loss.item() == 0
        assert torch.isfinite(loss)
```

- [ ] **Step 2: Run `uv run pytest tests/unit/models/test_factor_heads.py -q`; expect import failure.**
- [ ] **Step 3: Implement stable contrast and heads.**

```python
# src/despamo/losses/factor_contrast.py
import torch
from torch.nn import functional as F


def multi_positive(visual, text, labels, valid):
    if visual.shape != text.shape or visual.ndim != 2 or len(labels) != len(visual):
        raise ValueError("factor contrast shape mismatch")
    if valid.shape != visual.shape[:1]:
        raise ValueError("factor validity shape mismatch")
    indices = valid.nonzero(as_tuple=True)[0]
    active = [labels[i] for i in indices.tolist()]
    if len(active) < 2 or len(set(active)) < 2:
        return visual.sum() * 0
    v, t = visual[indices].float(), text[indices].detach().float()
    if not torch.isfinite(v).all() or not torch.isfinite(t).all():
        raise FloatingPointError("non-finite valid contrast target")
    logits = F.normalize(v, dim=-1) @ F.normalize(t, dim=-1).T / 0.07
    positives = torch.tensor([[a == b for b in active] for a in active],
                             device=logits.device, dtype=logits.dtype)
    row_targets = positives / positives.sum(1, keepdim=True)
    col_targets = positives.T / positives.T.sum(1, keepdim=True)
    return -0.5 * ((row_targets * F.log_softmax(logits, dim=1)).sum(1).mean()
                   + (col_targets * F.log_softmax(logits.T, dim=1)).sum(1).mean())
```

```python
# src/despamo/models/factor_heads.py
import torch
from torch import nn

from despamo.appearance.schema import FACTORS, STABLE
from despamo.losses.factor_contrast import multi_positive


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, alpha):
        ctx.alpha = alpha
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.alpha * gradient, None


def reverse(value, alpha):
    return Reverse.apply(value, alpha)


def select_representations(projected, mask, rows):
    mask = mask.to(projected.device)
    rows = rows.to(projected.device)
    if (mask.sum(1) == 0).any():
        raise ValueError("empty spatial sequence")
    clip = (projected * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    if ((rows < -1) | (rows >= projected.shape[1])).any():
        raise ValueError("frame row outside projected sequence")
    safe = rows.clamp_min(0)
    if ((rows >= 0) & ~mask.gather(1, safe)).any():
        raise ValueError("frame target aligned to padding")
    frame = projected.gather(1, safe.unsqueeze(-1).expand(-1, -1, projected.shape[-1]))
    return clip, frame


class FactorHeads(nn.Module):
    def __init__(self, width: int, text_width: int, enabled: list[str]):
        super().__init__()
        if not set(enabled) <= set(FACTORS) or len(set(enabled)) != len(enabled):
            raise ValueError("unknown/duplicate factor")
        self.enabled = tuple(enabled)
        self.heads = nn.ModuleDict({f: nn.Linear(width, text_width) for f in FACTORS})
        for factor in FACTORS:
            self.heads[factor].requires_grad_(factor in self.enabled)

    def forward(self, projected, mask, batch, alpha):
        clip, frames = select_representations(projected, mask, batch.rows)
        losses, counts = {}, {}
        for factor in self.enabled:
            if factor not in STABLE and alpha == 0:
                continue  # no optimizer update, including weight decay, before positive ramp
            representation = reverse(clip, alpha) if factor in STABLE else frames
            output = self.heads[factor](representation).reshape(-1, self.heads[factor].out_features)
            target = batch.vectors[factor].to(output.device).reshape_as(output)
            valid = batch.masks[factor].to(output.device).reshape(-1)
            if factor not in STABLE:
                valid = valid & (batch.rows.to(output.device).reshape(-1) >= 0)
            try:
                losses[factor] = multi_positive(output, target, batch.labels[factor], valid)
            except (ValueError, FloatingPointError) as error:
                raise type(error)(f"{factor}: clips={batch.base.clip_ids}, rows={batch.rows.tolist()}") from error
            counts[factor] = int(valid.sum())
        return losses, counts
```

- [ ] **Step 4: Repeat gradient tests; expect all seven factors pass sign checks and skipped losses stay finite.**

## Task 3: Shared Adapter Tap And Optimizer-Step Schedule

**Files:** Create `models/factor_adapter.py`, `training/factor_module.py`, `tests/unit/training/test_factor_module.py`.

**Interfaces:** `FactorVisualAdapter.forward_with_spatial(...) -> (tokens, token_mask, projected_spatial)`; inherited `forward(...)` retains two outputs and identical state keys. `ramp(step,warm,total) -> float | None`; `DualPathModule(base, enabled, text_width, total_steps, weights, provenance)` inherits baseline validation behavior.

- [ ] **Step 1: Add boundary and adapter parity tests.**

```python
# tests/unit/training/test_factor_module.py
import torch

from despamo.models.factor_adapter import FactorVisualAdapter
from despamo.models.visual_adapter import SpaMoVisualAdapter
from despamo.training.factor_module import ramp


def test_ramp_matches_baseline_warmup_and_resume():
    assert ramp(10, 10, 111) is None
    assert ramp(11, 10, 111) == 0
    assert ramp(16, 10, 111) == 0.5
    assert ramp(21, 10, 111) == 1


def test_adapter_tap_preserves_translation_and_checkpoint_names():
    base = SpaMoVisualAdapter(4, 3, 8, 6).eval()
    target = FactorVisualAdapter(4, 3, 8, 6).eval()
    target.load_state_dict(base.state_dict(), strict=True)
    inputs = (torch.randn(2, 20, 4), torch.ones(2, 20, dtype=torch.bool),
              torch.randn(2, 4, 3), torch.ones(2, 4, dtype=torch.bool))
    expected, mask = base(*inputs)
    actual, actual_mask, spatial = target.forward_with_spatial(*inputs)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_mask, mask)
    assert spatial.shape == (2, 20, 8)
    assert set(base.state_dict()) == set(target.state_dict())
```

- [ ] **Step 2: Run `uv run pytest tests/unit/training/test_factor_module.py -q`; expect missing module.**
- [ ] **Step 3: Implement adapter without changing baseline module or shared parameter names.**

```python
# src/despamo/models/factor_adapter.py
import torch
from torch.nn.utils.rnn import pad_sequence

from despamo.models.visual_adapter import SpaMoVisualAdapter, lengths_to_mask


class FactorVisualAdapter(SpaMoVisualAdapter):
    def forward_with_spatial(self, spatial, spatial_mask, motion, motion_mask):
        projected = self.spatial_projector(spatial)
        motion = self.motion_projector(motion)
        fused = [torch.cat((projected[i, spatial_mask[i]], motion[i, motion_mask[i]]), dim=0)
                 for i in range(projected.shape[0])]
        lengths = torch.tensor([len(x) for x in fused], device=projected.device)
        encoded, output_lengths = self.temporal_encoder(
            pad_sequence(fused, batch_first=True).transpose(1, 2), lengths)
        tokens = self.multimodal_projector(encoded)
        mask = lengths_to_mask(output_lengths, tokens.shape[1], tokens.device)
        return tokens, mask, projected

    def forward(self, spatial, spatial_mask, motion, motion_mask):
        tokens, mask, _ = self.forward_with_spatial(spatial, spatial_mask, motion, motion_mask)
        return tokens, mask
```

- [ ] **Step 4: Implement schedule and Lightning extension.**

```python
# src/despamo/training/factor_module.py
import math

from despamo.appearance.schema import FACTORS, STABLE
from despamo.data.factors import FactorBatch
from despamo.models.factor_adapter import FactorVisualAdapter
from despamo.models.factor_heads import FactorHeads
from despamo.models.prompts import build_prompts
from despamo.training.baseline_module import SpaMoBaselineModule, combine_losses, ensure_finite_losses


def ramp(step: int, warm: int | None, total: int) -> float | None:
    first_joint = 0 if warm is None else warm + 1
    if total <= first_joint:
        raise ValueError("training has no joint optimizer steps")
    if step < first_joint:
        return None
    duration = max(1, math.ceil(0.1 * (total - first_joint)))
    return min(1.0, (step - first_joint) / duration)


class DualPathModule(SpaMoBaselineModule):
    def __init__(self, base, enabled, text_width, total_steps, weights, provenance):
        old = base.visual_adapter
        adapter = FactorVisualAdapter(old.spatial_projector.in_features,
                                      old.motion_projector.in_features,
                                      old.spatial_projector.out_features,
                                      old.multimodal_projector[-1].out_features)
        adapter.load_state_dict(old.state_dict(), strict=True)
        super().__init__(adapter, base.language_model, base.vt_align, base.prompt_template,
                         base.use_in_context, base.num_in_context, base.vt_pooling, base.vt_weight,
                         base.warm_up_steps, base.learning_rate, base.weight_decay, seed=0)
        self.rng = base.rng
        self.auxiliary = FactorHeads(old.spatial_projector.out_features, text_width, enabled)
        if set(weights) != set(FACTORS) or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
            raise ValueError("seven positive finite factor weights required")
        self.factor_weights = weights
        self.total_steps = total_steps
        self.factor_provenance = {**provenance, "enabled": list(enabled), "weights": weights,
                                  "total_steps": total_steps}
        ramp(0, self.warm_up_steps, total_steps)

    def compute_losses(self, batch):
        if not isinstance(batch, FactorBatch):
            return super().compute_losses(batch)
        base = batch.base
        visual, visual_mask, spatial = self.visual_adapter.forward_with_spatial(
            base.spatial.to(self.device), base.spatial_mask.to(self.device),
            base.motion.to(self.device), base.motion_mask.to(self.device))
        text, text_mask = self.language_model.target_embeddings(base.texts)
        vt = self.vt_align(visual, visual_mask, text, text_mask, self.vt_pooling)
        prompts = build_prompts(base, self.prompt_template, self.use_in_context,
                                self.num_in_context, self.rng)
        translation = self.language_model.translation_loss(visual, visual_mask, prompts, base.texts)
        total = combine_losses(translation, vt, self.global_step, self.warm_up_steps, self.vt_weight)
        losses = dict(loss=translation, contra_loss=vt)
        scale = ramp(self.global_step, self.warm_up_steps, self.total_steps)
        if scale is not None:
            auxiliary, counts = self.auxiliary(spatial, base.spatial_mask, batch, scale)
            for factor, loss in auxiliary.items():
                weight = self.factor_weights[factor] * (1.0 if factor in STABLE else scale)
                total = total + weight * loss
                losses[f"factor/{factor}"] = loss
                losses[f"valid/{factor}"] = loss.new_tensor(counts[factor])
        losses["combined_loss"] = total
        ensure_finite_losses(losses, base.clip_ids)
        return losses

    def training_step(self, batch, batch_idx):
        losses = self.compute_losses(batch)
        self.log_dict({f"train/{k}": v for k, v in losses.items()}, batch_size=len(batch.base.texts))
        return losses["combined_loss"]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["factor_provenance"] = self.factor_provenance
        checkpoint["prompt_rng_state"] = self.rng.getstate()

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("factor_provenance") != self.factor_provenance:
            raise ValueError("resume factor configuration/provenance mismatch")
        self.rng.setstate(checkpoint["prompt_rng_state"])
```

Disabled heads have `requires_grad=False`; AdamW will not update them. Positive heads are omitted at ramp 0, so weight decay cannot update them before their loss starts.

- [ ] **Step 5: Repeat tests, then baseline adapter/parity tests. Require identical translation tensors with factors disabled.**

## Task 4: Factory, Training CLI, And Mixed-Target Smoke Test

**Files:** Create `training/factor_factory.py`, `scripts/train_factors.py`, config, `tests/unit/training/test_factor_smoke.py`.

**Interfaces:** `build_factor_system(config) -> (DualPathModule, PhoenixDataModule)`; train uses factor collator, validation/test remain plain baseline batches without caption dependency. `config.trainer.max_steps` fixes schedule duration and counts optimizer steps; reject distributed mode.

- [ ] **Step 1: Create factory and command.**

```python
# src/despamo/training/factor_factory.py
from pathlib import Path

import torch
from omegaconf import OmegaConf

from despamo.appearance.provenance import file_hash
from despamo.data.datamodule import PhoenixDataModule
from despamo.data.factors import FactorDataset, collate_factors
from despamo.factory import build_data, build_model
from despamo.training.factor_module import DualPathModule


class FactorDataModule(PhoenixDataModule):
    def __init__(self, source, train):
        super().__init__(train, source.validation_dataset, source.test_dataset,
                         source.batch_size, source.num_workers, source.collate_fn)
        self.original = source

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=collate_factors)


def build_factor_system(config):
    if config.trainer.devices != 1 or config.trainer.get("num_nodes", 1) != 1:
        raise ValueError("initial factor stage requires one device")
    if config.model.vt_pooling != "masked_mean":
        raise ValueError("factor stage requires masked spatial/VT pooling")
    if config.model.spatial_crop_mode != "full":
        raise ValueError("factor stage requires full spatial sequences")
    if config.trainer.max_steps <= 0:
        raise ValueError("explicit optimizer-step budget required")
    source = build_data(config)
    enabled = list(config.factors.enabled)
    train = FactorDataset(source.train_dataset, Path(config.factors.dataset),
                          Path(config.factors.text_manifest), Path(config.factors.frame_rows),
                          Path(config.data.spatial_manifest), enabled)
    initial = Path(config.factors.initial_checkpoint)
    base = build_model(config)
    base.load_state_dict(torch.load(initial, map_location="cpu")["state_dict"], strict=True)
    train.provenance["initial_checkpoint_hash"] = file_hash(initial)
    resolved = OmegaConf.to_container(config, resolve=True)
    train.provenance["model_policy"] = {k: v for k, v in resolved["model"].items() if k != "cache_dir"}
    train.provenance["optimizer_policy"] = resolved["optimizer"]
    train.provenance["trainer_policy"] = {
        k: v for k, v in resolved["trainer"].items() if k not in {"default_root_dir", "logger"}}
    train.provenance["batch_size"] = config.data.batch_size
    train.provenance["motion_manifest_hash"] = file_hash(Path(config.data.motion_manifest))
    model = DualPathModule(base, enabled, train.width, config.trainer.max_steps,
                           dict(config.factors.weights), train.provenance)
    return model, FactorDataModule(source, train)
```

```python
# scripts/train_factors.py
import argparse
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf

from despamo.appearance.provenance import atomic_json, file_hash, read_json
from despamo.config import load_config
from despamo.evaluation.artifact import collect_runtime_metadata
from despamo.training.factor_factory import build_factor_system


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    pl.seed_everything(config.seed, workers=True)
    gate = Path(config.factors.probe_gate)
    if not args.smoke and not read_json(gate)["passed"]:
        raise ValueError("baseline probe gate has not passed")
    model, data = build_factor_system(config)
    if not args.smoke:
        probe = read_json(gate)
        if (not probe["baseline"] or probe["provenance"] != data.train_dataset.provenance
                or probe["checkpoint_hash"] != data.train_dataset.provenance["initial_checkpoint_hash"]):
            raise ValueError("baseline probe gate belongs to different artifacts")
    root = Path(config.trainer.default_root_dir)
    atomic_json(root / "factor_run.json", dict(seed=config.seed,
                provenance=model.factor_provenance, probe_gate_hash=file_hash(gate) if gate.exists() else None,
                config=OmegaConf.to_container(config, resolve=True), smoke=args.smoke,
                runtime=collect_runtime_metadata(config.seed, Path("uv.lock"),
                         Path(config.data.spatial_manifest), Path(config.data.motion_manifest))))
    trainer_options = dict(config.trainer)
    if args.smoke:
        trainer_options.update(fast_dev_run=1, logger=False, enable_checkpointing=False)
    pl.Trainer(**trainer_options).fit(model, datamodule=data,
                                     ckpt_path=str(args.resume) if args.resume else None)


if __name__ == "__main__":
    main()
```

```yaml
# configs/experiment/seven_factors.yaml
model:
  vt_pooling: masked_mean
  spatial_crop_mode: full
factors:
  initial_checkpoint: ${oc.env:DESPAMO_DINO_CHECKPOINT}
  dataset: ${oc.env:DESPAMO_DATASET}
  text_manifest: ${oc.env:DESPAMO_TEXT_MANIFEST}
  frame_rows: ${oc.env:DESPAMO_FRAME_ROWS}
  probe_gate: ${oc.env:DESPAMO_PROBE_GATE}
  enabled: [biometric, clothing, hair, background, left_hand, right_hand, mouth]
  weights:
    biometric: 1.0
    clothing: 1.0
    hair: 1.0
    background: 1.0
    left_hand: 1.0
    right_hand: 1.0
    mouth: 1.0
trainer:
  devices: 1
  num_nodes: 1
```

The preceding verified DINO experiment config must supply an explicit `trainer.max_steps` and frozen feature paths. No arbitrary training duration is invented here. Every variant, including the zero-auxiliary DINO control, starts from the same verified shared-only DINO checkpoint and receives the same additional optimization budget. The existing DINO comparison checkpoint alone does not count as one of the three matched control runs.

- [ ] **Step 2: Add offline mixed-mask smoke test using synthetic heads and real adapter dimensions reduced for CPU.**

```python
# tests/unit/training/test_factor_smoke.py
from types import SimpleNamespace

import pytest
import torch

from despamo.appearance.schema import FACTORS, STABLE
from despamo.models.factor_heads import FactorHeads


@pytest.mark.parametrize("enabled", [[f] for f in FACTORS] + [list(STABLE), list(FACTORS[4:]), list(FACTORS)])
def test_factor_batch_backward_keeps_failed_clip_masked(enabled):
    torch.manual_seed(0)
    projected = torch.randn(3, 12, 8, requires_grad=True)
    masks, vectors, labels = {}, {}, {}
    for factor in FACTORS:
        shape = (3,) if factor in STABLE else (3, 5)
        masks[factor] = torch.ones(shape, dtype=torch.bool)
        masks[factor][-1] = False
        vectors[factor] = torch.randn(*shape, 4)
        labels[factor] = tuple(str(i % 2) for i in range(masks[factor].numel()))
    batch = SimpleNamespace(rows=torch.tensor([[1, 3, 5, 7, 9]] * 3),
                            base=SimpleNamespace(clip_ids=("a", "b", "failed")),
                            masks=masks, vectors=vectors, labels=labels)
    heads = FactorHeads(8, 4, enabled)
    losses, counts = heads(projected, torch.ones(3, 12, dtype=torch.bool), batch, 0.5)
    sum(losses.values()).backward()
    assert torch.isfinite(projected.grad).all()
    assert projected.grad[-1].count_nonzero() == 0
    assert set(losses) == set(enabled)
    for factor in set(FACTORS) - set(enabled):
        assert all(p.grad is None for p in heads.heads[factor].parameters())
    assert all(counts[f] == (2 if f in STABLE else 10) for f in enabled)
```

- [ ] **Step 3: Run `uv run pytest tests/unit/training/test_factor_smoke.py -q` and `uv run python scripts/train_factors.py --help`; expect 10 parameterized cases pass and CLI help.**
- [ ] **Step 4: After prerequisite acceptance, run one authorized real-batch smoke with `--smoke`. This verifies I/O and finite losses only; gradient-direction tests above exercise joint auxiliary behavior even when the smoke optimizer step lies in warm-up.**

## Task 5: Strict Inference Export

**Files:** Create `evaluation/factor_export.py`, `scripts/export_factors.py`, `tests/unit/evaluation/test_factor_export.py`.

**Interfaces:** `export_shared(checkpoint, output, baseline) -> None`; allow removal only under `auxiliary.heads.`; shared keys must exactly match baseline and load strictly. Preserve trained shared tensors bit-for-bit.

- [ ] **Step 1: Add export key/integrity test.**

```python
# tests/unit/evaluation/test_factor_export.py
import pytest
import torch

from despamo.evaluation.factor_export import export_shared


def test_export_rejects_unexpected_shared_key_and_preserves_weights(tmp_path):
    baseline = torch.nn.Linear(3, 2)
    source = tmp_path / "source.ckpt"
    output = tmp_path / "inference.pt"
    state = {**baseline.state_dict(), "auxiliary.heads.mouth.weight": torch.ones(2, 3)}
    torch.save({"state_dict": state}, source)
    export_shared(source, output, baseline)
    saved = torch.load(output, map_location="cpu")
    assert set(saved["state_dict"]) == set(baseline.state_dict())
    torch.testing.assert_close(saved["state_dict"]["weight"], baseline.weight)
    state["unknown.weight"] = torch.ones(1)
    torch.save({"state_dict": state}, source)
    with pytest.raises(ValueError, match="shared keys"):
        export_shared(source, output, baseline)
```

- [ ] **Step 2: Run `uv run pytest tests/unit/evaluation/test_factor_export.py -q`; expect missing module.**
- [ ] **Step 3: Implement strict export and CLI.**

```python
# src/despamo/evaluation/factor_export.py
import os

import torch

from despamo.appearance.provenance import file_hash


def export_shared(checkpoint, output, baseline):
    payload = torch.load(checkpoint, map_location="cpu")  # trusted local checkpoint
    shared = {k: v for k, v in payload["state_dict"].items() if not k.startswith("auxiliary.heads.")}
    if set(shared) != set(baseline.state_dict()):
        raise ValueError("inference export shared keys mismatch")
    baseline.load_state_dict(shared, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(dict(state_dict=shared, metadata=dict(source_sha256=file_hash(checkpoint),
                    factor_provenance=payload.get("factor_provenance"))), temporary)
    os.replace(temporary, output)
```

```python
# scripts/export_factors.py
import argparse
from pathlib import Path

from despamo.config import load_config
from despamo.evaluation.factor_export import export_shared
from despamo.factory import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, action="append", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    export_shared(args.checkpoint, args.output, build_model(load_config(args.config)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Repeat tests. On bounded real fixture, compare baseline generation from exported checkpoint with auxiliary model's shared path under identical deterministic decoding; require identical generated token IDs.**

## Task 6: Fresh Baseline And Post-Training Probes

**Files:** Create `evaluation/factor_probes.py`, `scripts/probe_factors.py`, `tests/unit/evaluation/test_factor_probes.py`.

**Interfaces:** `make_probe_split(clips, seed=0) -> dict[clip_id, partition]`; `fit_probe(train, validation, classification, seed) -> nn.Linear`; `retrieval(projected, targets, labels, gallery) -> dict`; `collect_probe_rows(model,dataset) -> rows`. Each row records `clip_id`, `signer`, `factor`, `frame_index`, frozen `x`, text `y`, canonical `label`. A signer row uses pooled clip representation and factor `signer`.

No probe uses adversarial-head output. Partition clips before creating frame rows. Require >=3 clips per signer; train/validation/test each contain that signer. Gallery is a sorted, unique list of canonical strings for each factor using cached text vectors; identical across variants because dataset and encoder manifests match. Keep gallery metadata and split in result provenance.

- [ ] **Step 1: Write split and retrieval tests.**

```python
# tests/unit/evaluation/test_factor_probes.py
import torch

from despamo.evaluation.factor_probes import make_probe_split, retrieval


def test_probe_split_is_clip_disjoint_and_closed_set():
    clips = [dict(clip_id=f"{s}-{i}", signer=s) for s in ("s1", "s2") for i in range(9)]
    split = make_probe_split(clips)
    assert len(split) == 18
    for s in ("s1", "s2"):
        assert {split[c["clip_id"]] for c in clips if c["signer"] == s} == {"train", "val", "test"}
    assert split == make_probe_split(clips)


def test_duplicate_descriptions_share_positive_gallery_identity():
    gallery = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    metrics = retrieval(x, x, ["a", "a", "b"], gallery)
    assert metrics["recall1"] == 1
    assert metrics["recall5"] == 1
    assert metrics["gallery_size"] == 2
```

- [ ] **Step 2: Run `uv run pytest tests/unit/evaluation/test_factor_probes.py -q`; expect missing module.**
- [ ] **Step 3: Implement frozen-row collection and probes.**

```python
# src/despamo/evaluation/factor_probes.py
import copy
import random
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from despamo.appearance.schema import FACTORS, STABLE
from despamo.data.factors import collate_factors
from despamo.models.factor_heads import select_representations


def make_probe_split(clips: list[dict], seed: int = 0) -> dict:
    groups = defaultdict(list)
    for clip in clips:
        groups[clip["signer"]].append(clip["clip_id"])
    output = {}
    rng = random.Random(seed)
    for signer in sorted(groups):
        ids = sorted(groups[signer])
        if len(ids) < 3:
            raise ValueError("closed-set probe needs at least three clips per signer")
        rng.shuffle(ids)
        held = max(1, len(ids) // 5)
        for i, clip_id in enumerate(ids):
            if clip_id in output:
                raise ValueError("duplicate clip in probe split")
            output[clip_id] = "test" if i < held else "val" if i < 2 * held else "train"
    return output


@torch.no_grad()
def collect_probe_rows(model, dataset):
    model.eval()
    rows = []
    for index in range(len(dataset)):
        batch = collate_factors([dataset[index]])
        projected = model.visual_adapter.spatial_projector(batch.base.spatial.to(model.device))
        clip, frame = select_representations(projected, batch.base.spatial_mask, batch.rows)
        common = dict(clip_id=batch.base.clip_ids[0], signer=batch.base.signers[0])
        rows.append(dict(**common, factor="signer", frame_index=-1, x=clip[0].cpu(),
                         label=batch.base.signers[0], y=None))
        for factor in FACTORS:
            values = clip if factor in STABLE else frame.reshape(-1, frame.shape[-1])
            valid = batch.masks[factor].reshape(-1)
            target = batch.vectors[factor].reshape(len(values), -1)
            for slot in valid.nonzero(as_tuple=True)[0].tolist():
                source = dataset.clips[common["clip_id"]]["frames"]
                rows.append(dict(**common, factor=factor,
                                 frame_index=-1 if factor in STABLE else source[slot]["index"],
                                 x=values[slot].cpu(), y=target[slot].cpu(),
                                 label=batch.labels[factor][slot]))
    return rows


def retrieval(projected, targets, labels, gallery):
    keys = sorted(gallery)
    if not labels or not keys or any(label not in gallery for label in labels):
        raise ValueError("empty/foreign retrieval target")
    g = F.normalize(torch.stack([gallery[k] for k in keys]).float(), dim=-1)
    p = F.normalize(projected.float(), dim=-1)
    rank = (p @ g.T).argsort(dim=1, descending=True, stable=True)
    expected = torch.tensor([keys.index(label) for label in labels])
    k = min(5, len(keys))
    return dict(recall1=(rank[:, 0] == expected).float().mean().item(),
                recall5=(rank[:, :k] == expected[:, None]).any(1).float().mean().item(),
                cosine=F.cosine_similarity(p, targets.float(), dim=-1).mean().item(),
                gallery_size=len(keys), recall5_k=k)


def fit_probe(train, validation, classification: bool, seed: int):
    torch.manual_seed(seed)
    x, y = train
    vx, vy = validation
    if not len(x) or not len(vx):
        raise ValueError("empty probe partition")
    width = int(y.max()) + 1 if classification else y.shape[-1]
    probe = nn.Linear(x.shape[-1], width)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = float("inf"), None
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        prediction = probe(x)
        loss = F.cross_entropy(prediction, y) if classification else (
            1 - F.cosine_similarity(prediction, y, dim=-1)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite probe fit")
        loss.backward()
        optimizer.step()
        if step + 1 in (20, 50, 100):
            with torch.no_grad():
                prediction = probe(vx)
                score = F.cross_entropy(prediction, vy) if classification else (
                    1 - F.cosine_similarity(prediction, vy, dim=-1)).mean()
                if torch.isfinite(score) and score.item() < best:
                    best, best_state = score.item(), copy.deepcopy(probe.state_dict())
    if best_state is None:
        raise FloatingPointError("probe validation never finite")
    probe.load_state_dict(best_state)
    return probe.eval()


def run_probes(rows, split, seed: int) -> dict:
    results = {}
    for factor in ("signer",) + FACTORS:
        selected = [r for r in rows if r["factor"] == factor]
        parts = {p: [r for r in selected if split[r["clip_id"]] == p] for p in ("train", "val", "test")}
        if any(not group for group in parts.values()):
            results[factor] = dict(available=False, reason="empty valid partition")
            continue
        classes = sorted({r["label"] for r in parts["train"]})
        classification = factor == "signer"
        if classification and any(r["label"] not in classes for r in selected):
            raise ValueError("signer probe is not closed-set")
        tensors = {}
        for p, group in parts.items():
            x = torch.stack([r["x"] for r in group]).float()
            y = torch.tensor([classes.index(r["label"]) for r in group]) if classification else (
                torch.stack([r["y"] for r in group]).float())
            tensors[p] = (x, y)
        probe = fit_probe(tensors["train"], tensors["val"], classification, seed)
        with torch.no_grad():
            predicted = probe(tensors["test"][0])
            if classification:
                metrics = dict(top1=(predicted.argmax(1) == tensors["test"][1]).float().mean().item())
            else:
                gallery = {r["label"]: r["y"] for r in selected}
                metrics = retrieval(predicted, tensors["test"][1],
                                    [r["label"] for r in parts["test"]], gallery)
        results[factor] = dict(available=True, **metrics,
                               partition_counts={p: len(g) for p, g in parts.items()})
    return results
```

- [ ] **Step 4: Add CLI that fixes split before fitting and records baseline gate.**

```python
# scripts/probe_factors.py
import argparse
from pathlib import Path

import torch

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.config import load_config
from despamo.evaluation.factor_probes import collect_probe_rows, make_probe_split, run_probes
from despamo.training.factor_factory import build_factor_system


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, action="append", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("override", nargs="*")
    args = p.parse_args()
    config = load_config(args.config, args.override)
    model, data = build_factor_system(config)
    saved = torch.load(args.checkpoint, map_location="cpu")
    if args.baseline:
        if file_hash(args.checkpoint) != data.train_dataset.provenance["initial_checkpoint_hash"]:
            raise ValueError("baseline probe gate requires the common initial checkpoint")
        shared_keys = {k for k in model.state_dict() if not k.startswith("auxiliary.heads.")}
        if set(saved["state_dict"]) != shared_keys:
            raise ValueError("baseline probe checkpoint is not strict shared-only")
        combined = {**model.state_dict(), **saved["state_dict"]}
        model.load_state_dict(combined, strict=True)
    else:
        if saved.get("factor_provenance") != model.factor_provenance:
            raise ValueError("probe checkpoint/config provenance mismatch")
        model.load_state_dict(saved["state_dict"], strict=True)
    expected_split = make_probe_split(data.train_dataset.source["clips"], seed=0)
    if args.split.exists() and read_json(args.split) != expected_split:
        raise ValueError("probe split changed")
    atomic_json(args.split, expected_split)
    rows = collect_probe_rows(model, data.train_dataset)
    results = run_probes(rows, expected_split, config.seed)
    output = dict(provenance=data.train_dataset.provenance, seed=config.seed,
                  checkpoint_hash=file_hash(args.checkpoint), split_hash=digest(expected_split),
                  gallery_hash=digest(sorted({(r["factor"], r["label"]) for r in rows})),
                  baseline=args.baseline, metrics=results,
                  passed=all(v["available"] for v in results.values()))
    atomic_json(args.output, output)
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Repeat probe unit tests; expect pass. Execute baseline probes only after dataset/DINO gates, write `DESPAMO_PROBE_GATE` to the resulting JSON, and require a complete finite result before non-smoke factor training. Repeat fresh probes per trained checkpoint under identical split/gallery settings.**

## Task 7: Eleven Variants, Paired Seed Reports, And Research Runbook

**Files:** Create `evaluation/factor_experiments.py`, `scripts/factor_experiments.py`, `tests/unit/evaluation/test_factor_experiments.py`.

**Interfaces:** `variants() -> dict[str,list[str]]`; `experiment_matrix() -> list[dict]`; `collect_run(evaluation, probes, variant, seed) -> dict`; `summarize_runs(runs) -> dict`. Each run is `{variant, seed, bleu4, rougeL_f1, probes, comparison_key}` where `probes` is Task 6 output. `comparison_key` is computed from actual artifacts: identical split, features, encoder, decoding, optimizer/budget, and initialization policies, excluding only variant and seed.

- [ ] **Step 1: Add matrix and retention tests.**

```python
# tests/unit/evaluation/test_factor_experiments.py
from copy import deepcopy

import pytest

from despamo.appearance.schema import FACTORS
from despamo.evaluation.factor_experiments import collect_run, experiment_matrix, retain, summarize_runs


def test_matrix_has_eleven_variants_and_three_seeds():
    rows = experiment_matrix()
    assert len(rows) == 33
    assert len({r["variant"] for r in rows}) == 11
    assert {r["seed"] for r in rows} == {0, 1, 2}


def test_retention_uses_factor_role_and_bleu_point_budget():
    assert retain("clothing", -0.5, -0.01, 0.0)
    assert not retain("clothing", -0.51, -0.01, 0.0)
    assert retain("mouth", 0.2, 0.01, -0.01)
    assert not retain("mouth", 0.2, 0.01, 0.01)


def run_fixture(variant, seed, enabled):
    provenance = dict(model_policy={"name": "fixture", "spatial_crop_mode": "full"},
                      initial_checkpoint_hash="initial")
    trained = dict(**provenance, enabled=enabled, weights={f: 1.0 for f in FACTORS}, total_steps=100)
    metrics = {f: dict(available=True, recall1=0.5, recall5=1.0, cosine=0.7) for f in FACTORS}
    metrics["signer"] = dict(available=True, top1=0.5)
    probes = dict(provenance=provenance, seed=seed, checkpoint_hash=f"{variant}-{seed}",
                  passed=True, split_hash="split", gallery_hash="gallery", metrics=metrics)
    evaluation = dict(metrics=dict(bleu4=25.0, rougeL_f1=0.46),
                      items=[dict(clip_id="test-clip", reference="text")],
                       metadata=dict(seed=seed, generation="deterministic", spatial_crop_mode="full",
                         package_lock_sha256="lock",
                         spatial_manifest_sha256="spatial", motion_manifest_sha256="motion",
                         config=dict(model={"name": "fixture", "spatial_crop_mode": "full"},
                                     evaluation={"beam_size": 4}),
                        checkpoint_metadata=dict(source_sha256=f"{variant}-{seed}", factor_provenance=trained)))
    return evaluation, probes


def test_collector_and_report_bind_all_33_runs():
    runs = []
    for item in experiment_matrix():
        evaluation, probes = run_fixture(**item)
        runs.append(collect_run(evaluation, probes, item["variant"], item["seed"]))
    result = summarize_runs(runs)
    assert len(result["variants"]) == 11
    assert result["variants"]["full"]["bleu4"] == dict(mean=25.0, std=0.0)
    assert not any(item["retained"] for item in result["retention"].values())
    with pytest.raises(ValueError, match="33 unique"):
        summarize_runs(runs[:-1])
    changed = deepcopy(runs)
    changed[-1]["comparison_key"] = "different"
    with pytest.raises(ValueError, match="unpaired"):
        summarize_runs(changed)


def test_collector_rejects_different_checkpoint():
    evaluation, probes = run_fixture("full", 0, list(FACTORS))
    probes["checkpoint_hash"] = "another-model"
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        collect_run(evaluation, probes, "full", 0)


@pytest.mark.parametrize("field", ["spatial_crop_mode", "config"])
def test_collector_rejects_cropped_factor_evaluation(field):
    evaluation, probes = run_fixture("full", 0, list(FACTORS))
    if field == "config":
        evaluation["metadata"]["config"]["model"]["spatial_crop_mode"] = "center"
    else:
        evaluation["metadata"]["spatial_crop_mode"] = "center"
    with pytest.raises(ValueError, match="full spatial sequences"):
        collect_run(evaluation, probes, "full", 0)
```

- [ ] **Step 2: Run `uv run pytest tests/unit/evaluation/test_factor_experiments.py -q`; expect missing module.**
- [ ] **Step 3: Implement preparation/reporting without automatic job launch.**

```python
# src/despamo/evaluation/factor_experiments.py
import math
import statistics

from despamo.appearance.provenance import digest
from despamo.appearance.schema import ARTICULATORS, FACTORS, STABLE


def variants():
    return {"dino_baseline": [], **{f: [f] for f in FACTORS},
            "all_nuisances": list(STABLE), "all_articulators": list(ARTICULATORS),
            "full": list(FACTORS)}


def experiment_matrix():
    return [dict(variant=name, seed=seed, enabled=factors)
            for name, factors in variants().items() for seed in (0, 1, 2)]


def retain(factor, bleu_delta, factor_delta, signer_delta):
    if bleu_delta < -0.5:
        return False
    if factor in STABLE:
        return factor_delta < 0 or signer_delta < 0
    return factor_delta > 0 and signer_delta <= 0


def collect_run(evaluation, probes, variant, seed):
    metadata = evaluation["metadata"]
    exported = metadata["checkpoint_metadata"]
    trained = exported["factor_provenance"]
    if (trained["enabled"] != variants()[variant] or metadata["seed"] != seed
            or probes["seed"] != seed or probes["checkpoint_hash"] != exported["source_sha256"]):
        raise ValueError("translation/probe/variant checkpoint mismatch")
    if metadata["generation"] != "deterministic":
        raise ValueError("research comparison requires deterministic decoding")
    if (metadata.get("spatial_crop_mode") != "full"
            or metadata["config"]["model"].get("spatial_crop_mode") != "full"):
        raise ValueError("factor evaluation requires full spatial sequences")
    config_model = {k: v for k, v in metadata["config"]["model"].items() if k != "cache_dir"}
    if config_model != trained["model_policy"]:
        raise ValueError("evaluation changed shared model configuration")
    identities = {k: v for k, v in trained.items() if k not in {"enabled", "weights", "total_steps"}}
    if probes["provenance"] != identities:
        raise ValueError("probe provenance differs from training checkpoint")
    items = evaluation["items"]
    if not items or len({r["clip_id"] for r in items}) != len(items):
        raise ValueError("empty/duplicate translation evaluation")
    comparison = dict(provenance=identities, weights=trained["weights"], total_steps=trained["total_steps"],
                      split_hash=digest(sorted((r["clip_id"], r["reference"]) for r in items)),
                      decoding=metadata["config"]["evaluation"],
                      package_lock=metadata["package_lock_sha256"],
                      spatial=metadata["spatial_manifest_sha256"], motion=metadata["motion_manifest_sha256"])
    return dict(variant=variant, seed=seed, bleu4=evaluation["metrics"]["bleu4"],
                rougeL_f1=evaluation["metrics"]["rougeL_f1"], probes=probes,
                comparison_key=digest(comparison), evaluation_hash=digest(evaluation))


def summarize_runs(runs):
    expected = {(r["variant"], r["seed"]) for r in experiment_matrix()}
    indexed = {(r["variant"], r["seed"]): r for r in runs}
    if len(indexed) != len(runs) or set(indexed) != expected:
        raise ValueError("requires exactly 33 unique completed runs")
    identities = {(r["comparison_key"], r["probes"]["split_hash"], r["probes"]["gallery_hash"])
                  for r in runs}
    if len(identities) != 1:
        raise ValueError("unpaired experiment provenance")
    if any(not r["probes"]["passed"] for r in runs):
        raise ValueError("incomplete probe metrics")
    summary = {}
    for variant in variants():
        rows = [indexed[variant, seed] for seed in (0, 1, 2)]
        metrics = {}
        values = {"bleu4": [r["bleu4"] for r in rows],
                  "rougeL_f1": [r["rougeL_f1"] for r in rows],
                  "signer_top1": [r["probes"]["metrics"]["signer"]["top1"] for r in rows]}
        for factor in FACTORS:
            for metric in ("recall1", "recall5", "cosine"):
                values[f"{factor}/{metric}"] = [r["probes"]["metrics"][factor][metric] for r in rows]
        for key, numbers in values.items():
            if not all(math.isfinite(v) for v in numbers):
                raise ValueError("non-finite experiment metric")
            metrics[key] = dict(mean=statistics.mean(numbers), std=statistics.stdev(numbers))
        summary[variant] = metrics
    retention = {}
    for factor in FACTORS:
        deltas = []
        for seed in (0, 1, 2):
            a, b = indexed[factor, seed], indexed["dino_baseline", seed]
            deltas.append((a["bleu4"] - b["bleu4"],
                           a["probes"]["metrics"][factor]["recall1"] - b["probes"]["metrics"][factor]["recall1"],
                           a["probes"]["metrics"]["signer"]["top1"] - b["probes"]["metrics"]["signer"]["top1"]))
        means = [statistics.mean(d[i] for d in deltas) for i in range(3)]
        retention[factor] = dict(bleu_delta=means[0], factor_delta=means[1], signer_delta=means[2],
                                 retained=retain(factor, *means))
    comparisons = {name: {metric: summary["full"][metric]["mean"] - summary[name][metric]["mean"]
                          for metric in summary["full"]}
                   for name in ("dino_baseline", "all_nuisances")}
    return dict(variants=summary, retention=retention, full_deltas=comparisons,
                unseen_signer_gap=None,
                unseen_signer_status="requires separate signer-disjoint translation artifacts")
```

```python
# scripts/factor_experiments.py
import argparse
from pathlib import Path

from despamo.appearance.provenance import atomic_json, read_json
from despamo.evaluation.factor_experiments import collect_run, experiment_matrix, summarize_runs, variants


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("matrix", "collect", "report"))
    p.add_argument("--input", type=Path)
    p.add_argument("--probes", type=Path)
    p.add_argument("--variant", choices=tuple(variants()))
    p.add_argument("--seed", type=int, choices=(0, 1, 2))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command != "matrix" and args.input is None:
        p.error("collect/report requires --input")
    if args.command == "collect" and (args.probes is None or args.variant is None or args.seed is None):
        p.error("collect requires --probes, --variant and --seed")
    if args.command == "matrix":
        result = experiment_matrix()
    elif args.command == "collect":
        result = collect_run(read_json(args.input), read_json(args.probes), args.variant, args.seed)
    else:
        result = summarize_runs([read_json(p) for p in sorted(args.input.glob("*.json"))])
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Repeat matrix/report tests; expect pass. Generate matrix JSON only.**
- [ ] **Step 5: Run bounded verification before requesting experiment authorization.**

```bash
uv run pytest tests/unit/data/test_factors.py tests/unit/models/test_factor_heads.py tests/unit/training/test_factor_module.py tests/unit/training/test_factor_smoke.py tests/unit/evaluation/test_factor_export.py tests/unit/evaluation/test_factor_probes.py tests/unit/evaluation/test_factor_experiments.py -q
uv run ruff check src scripts tests
git diff --check
git status --short
```

- [ ] **Step 6: Use this command sequence only when each preceding gate is complete.**

```bash
mkdir -p "$DESPAMO_RUN_ROOT"
export DESPAMO_RUN_CONFIG="$DESPAMO_RUN_ROOT/run.yaml"
uv run python -c 'import os
from pathlib import Path
from omegaconf import OmegaConf
from despamo.config import load_config
config = load_config([Path(os.environ["DESPAMO_DINO_CONFIG"]), Path("configs/experiment/seven_factors.yaml")], ["seed=0"])
assert config.model.spatial_crop_mode == "full"
config.trainer.default_root_dir = str(Path(os.environ["DESPAMO_RUN_CONFIG"]).parent)
OmegaConf.save(config, Path(os.environ["DESPAMO_RUN_CONFIG"]))'
uv run python scripts/factor_experiments.py matrix --output "$DESPAMO_RUN_ROOT/matrix.json"
uv run python scripts/probe_factors.py --config "$DESPAMO_RUN_CONFIG" --checkpoint "$DESPAMO_DINO_CHECKPOINT" --baseline --split "$DESPAMO_RUN_ROOT/probe-split.json" --output "$DESPAMO_PROBE_GATE"
uv run python scripts/train_factors.py --config "$DESPAMO_RUN_CONFIG" --smoke
uv run python scripts/train_factors.py --config "$DESPAMO_RUN_CONFIG"
uv run python scripts/export_factors.py --config "$DESPAMO_RUN_CONFIG" --checkpoint "$DESPAMO_TRAINED_CHECKPOINT" --output "$DESPAMO_RUN_ROOT/inference.pt"
uv run python scripts/evaluate.py --config "$DESPAMO_RUN_CONFIG" --checkpoint "$DESPAMO_RUN_ROOT/inference.pt" --generation deterministic --output "$DESPAMO_RUN_ROOT/translation.json"
uv run python -c 'import json, sys
from pathlib import Path
metadata = json.loads(Path(sys.argv[1]).read_text())["metadata"]
assert metadata["spatial_crop_mode"] == metadata["config"]["model"]["spatial_crop_mode"] == "full"' "$DESPAMO_RUN_ROOT/translation.json"
uv run python scripts/probe_factors.py --config "$DESPAMO_RUN_CONFIG" --checkpoint "$DESPAMO_TRAINED_CHECKPOINT" --split "$DESPAMO_RUN_ROOT/probe-split.json" --output "$DESPAMO_RUN_ROOT/probes.json"
uv run python scripts/factor_experiments.py collect --input "$DESPAMO_RUN_ROOT/translation.json" --probes "$DESPAMO_RUN_ROOT/probes.json" --variant full --seed 0 --output "$DESPAMO_RUN_ROOT/completed/full-0.json"
uv run python scripts/factor_experiments.py report --input "$DESPAMO_RUN_ROOT/completed" --output "$DESPAMO_RUN_ROOT/report.json"
```

`DESPAMO_DINO_CONFIG` is a verified merged DINO experiment configuration, containing data/model/optimizer/trainer settings, not a guessed path. Each authorized variant/seed gets a distinct run root and `DESPAMO_RUN_CONFIG` created from the DINO config plus seven-factor overlay with that seed and explicit `factors.enabled` override from matrix; the full/seed-0 example uses the overlay's default enabled list. Do not overwrite full-model checkpoints with single-factor runs. Use that exact saved config for baseline and trained probes, smoke, training, export, and evaluation. Check effective crop mode in evaluation artifact **before** collecting/comparing; collector rejects cropped results even if checkpoint provenance matches. Copy the 33 collected result JSON files into one completed-results directory and run report against that directory. Reuse the same deterministic decoding settings for every comparison.

## Completion Gate

- [ ] Dataset/audit/encoder/frame-map provenance is checked before training and resume.
- [ ] Per-factor shared/head gradient tests pass; disabled heads remain unmodified.
- [ ] Baseline translation tensors/checkpoint keys remain compatible with auxiliary modules removed.
- [ ] Resume retains schedule and prompt RNG; mismatch rejects rather than silently changes experiment.
- [ ] Frozen DINO baseline probes and bounded real-batch smoke pass before large jobs.
- [ ] Eleven variants x three seeds are prepared; report requires exactly 33 actual completed runs.
- [ ] Retention compares paired seed means with prespecified metrics; missing metrics stay unavailable.
- [ ] Signer-disjoint translation evaluation remains an explicit subsequent split milestone, not a claim about the official PHOENIX split.
- [ ] README records observed metrics, gates, manifests, run configuration, hashes, and export command.

Optional task commits require explicit user authorization after status/diff review. Implementing these tools is not permission to launch 33 training runs.
