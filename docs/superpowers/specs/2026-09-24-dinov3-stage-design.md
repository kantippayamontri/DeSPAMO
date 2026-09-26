# DINOv3 Spatial Feature Stage Design

**Date:** 2026-09-24

**Status:** Written spec approved by user

**Depends on:** Accepted 642-clip SpaMo PHOENIX14T baseline (`7e04991`)

## 1. Objective And Boundaries

Replace frozen CLIP ViT-L/14 S2 spatial features with frozen DINOv3 ViT-L/16
S2 features, while retaining SpaMo's VideoMAE motion stream, visual adapter,
Flan-T5-XL, LoRA, translation objective, and VT-Align. Keep both spatial sources
selectable through configuration. Measure the effect of the spatial encoder
alone before enabling seven-factor supervision.

This stage has two independently testable deliverables:

1. Versioned, complete DINOv3 per-frame PHOENIX14T features and a validated
   source-frame-to-feature-row manifest.
2. Controlled CLIP-vs-DINOv3 baseline training/evaluation with otherwise
   identical data, model, optimizer, seed, schedule, and decoding.

No DINOv3 fine-tuning, factor captions, GRLs, or articulator heads occur in this
stage. Frozen DINOv3 and its extractor are absent from translation inference.

## 2. Model Access And Dependency Isolation

- Official encoder: `facebook/dinov3-vitl16-pretrain-lvd1689m`, ViT-L/16 with
  1024-dimensional CLS tokens, four register tokens, and patch size 16.
- This Hugging Face repository has manual license gating. A 2026-09-24
  configuration request returned HTTP 401; no local DINOv3 weights were found.
  User will accept conditions and authenticate locally. No credentials or
  signed URLs enter Git, configs, manifests, or logs.
- User will place `HF_TOKEN` in the ignored `.env` in the original checkout.
  The isolated extractor accepts an explicit path to that file, parses only
  `HF_TOKEN` as data (never executes shell syntax), and sets its value only in
  the current process for gated HF requests. Never copy `.env` into the
  worktree or print/serialize the token.
- Pin the exact resolved model commit SHA; reject mutable `main` as a persisted
  version identifier. Model files live in an external cache. Fail before
  extraction if the selected revision is inaccessible.
- Official HF DINOv3 integration requires Transformers >=4.56. The accepted
  SpaMo baseline pins Transformers 4.32.0 and Torch 2.0.1. Create an independently
  locked extractor environment using Python 3.11, Transformers 4.56.2, a
  compatible newer PyTorch/CUDA build, Pillow, and NumPy. Test the precise
  versions on the available RTX 4080 SUPER before freezing that lock. Keep the
  baseline `pyproject.toml` and `uv.lock` unchanged; translation training and
  evaluation still run in the accepted baseline environment.
- The extractor is a standalone process that writes validated float32 `.npy`
  files and JSON manifests. Training never imports its newer Transformers.

## 3. Input Frames And Preprocessing

Source root on this host:

```text
/mnt/e/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px
```

Use an environment variable for that root in commands. PHOENIX14T annotations
contain 7,096 train, 519 dev, and 642 test clip IDs; source directories match
these IDs. A metadata-only scan found 827,354 train, 55,775 dev, and 64,627
test PNG files: all 8,257 clips had at least five frames and their observed
PNG counts equaled annotation `num_frames`. The existing CLIP spatial manifest
also reports exactly that many feature rows for every annotated clip. The
downloaded frames are 210x260 PNGs, not the historical
`fullFrame-256x256px` path referenced in SpaMo code. The selected five-frame
appearance records later use the same sorted source sequence.

For each clip, enumerate all PNGs in ascending zero-padded filename order;
record every file's zero-based ordinal and source-relative path. Reject a
duplicate ordinal, missing directory, fewer than five images, unreadable PNG,
or disagreement with annotation `num_frames`. No train-only random sampling,
horizontal flip, temporal subsampling, or 512-frame crop happens during
extraction.

For every decoded RGB image, use deterministic square Pillow bicubic resizing
to 224x224 and 448x448; convert to float32 `[0,1]` and normalize with ImageNet
mean `(0.485,0.456,0.406)` and standard deviation `(0.229,0.224,0.225)`.
Both sizes are divisible by the patch size 16. This square-input convention
corresponds to SpaMo's historical square-frame feature source; it does not
claim pixel-identical CLIP preprocessing. Record original resolution, resize
algorithm, normalization constants, color conversion, scale order `(224,448)`,
library versions, and explicit absence of augmentation in the version metadata.

S2 here means *two complete frozen DINOv3 forward passes*, one for each image
scale. Take `last_hidden_state[:,0,:]` (the final-model CLS output)
from each pass; do not include register/patch tokens or use auxiliary heads.
Concatenate scale-224 then scale-448 CLS vectors to `[frames,2048]`, cast to
finite float32, and preserve temporal order. Unit tests use a fake encoder that
returns distinct known 1024-wide CLS vectors per scale; a live gated-model
smoke checks actual output width and finiteness. No silent fallback to a
different encoder, checkpoint, or feature width.

## 4. Feature Storage And Provenance

Store all generated data outside Git on a Linux-native filesystem. Use an
encoder version key derived from canonical JSON containing exact model ID,
immutable model SHA, extraction-library lock hash, ordered preprocessing
settings, and feature format/schema version. Under that key, write one
`train|dev|test/<clip_id>.npy` per clip with shape `[T,2048]`, dtype float32;
use a temporary file and atomic replace only after full-clip validation.

Write the baseline-compatible `FeatureManifest` (schema 1) pointing at those
files. It must contain exactly 7,096 train, 519 dev, and 642 test clips and
reuse current `FeatureManifest.load` and `build_data` validation. Preserve
existing VideoMAE features and manifest. Require each DINO feature row count
to equal the corresponding annotation frame count and CLIP feature row count;
stop rather than comparing differently sampled temporal sequences.

Write a separate immutable row-map JSON matching the dual-path factor plan:

```text
spatial_manifest_hash: SHA-256 of DINOv3 FeatureManifest JSON
encoder_key: version key
clips: keyed by clip_id
  feature_hash: SHA-256 of the corresponding .npy file
  source_indices: ordered zero-based source ordinals, one per feature row
  sampled_images: map of the five 10/30/50/70/90%-sampled source indices
                  to source-image SHA-256
```

Use round-half-up `int((T-1)*position+0.5)` and require distinct sample indices.
The per-clip record also carries source-relative frame paths, full frame count,
and a hash of the ordered source path/size/content list so changed input
frames cannot reuse old features. Never infer row indices from caption frame
position; validate map length equals feature rows and each sampled index occurs
exactly once. Store one manifest for extraction failures with clip ID and
reason. The completion gate requires zero failures; interrupted runs resume
only if output, source hashes, encoder key, and shape/dtype still match.

Writes are single-writer per encoder key. A changed model revision,
preprocessing step, source PNG, or library lock creates a different version
key or invalidates the affected clip; do not overwrite a validated version
silently. Baseline CLIP manifest and all original feature arrays remain intact.

## 5. Controlled Encoder Comparison

Add a DINO experiment config overlay that selects only the DINO spatial
root/manifest, sets corrected `vt_pooling: masked_mean`, and sets
`spatial_crop_mode: full`. Add a matched CLIP control overlay using the same
masked pooling, full sequences, optimizer, Flan-T5-XL/LoRA setup, motion
features, annotation splits, prompt, batch size, scheduler, training-step
budget, seed, and deterministic beam settings. No released CLIP-trained visual
projector is reused as a DINO starting point: initialize trainable adapters
identically for both runs; use the same pretrained frozen/LoRA-enabled
language-model initialization. The only controlled difference is the spatial
feature source.

First run cheap synthetic integration and one bounded real feature batch in
the *baseline* Python environment. Then train bounded matched smoke runs
and verify checkpoints/finite losses. The initial controlled comparison uses
exactly 1,000 optimizer steps per variant with seeds 0, 1, and 2. This
is a fixed-budget encoder comparison, not a claim of 500-epoch convergence.
Do not equate existing released SpaMo metrics with a freshly trained
matched CLIP control. Report every completed run's BLEU-4, ROUGE-L F1,
checkpoint/manifest hashes, decoding mode, seed, and training budget.
Both variants evaluate all 642 test clips in deterministic mode. The
released-SpaMo `--accept-baseline` tolerance applies only to its approved
upstream-mode checkpoint, never to experimental DINO runs.

No improvement threshold is assumed; report the observed paired difference
and variability. If extraction/weight access is unavailable, the comparison
gate remains pending rather than substituting synthetic features.

## 6. Failure Handling And Tests

- Gated/absent model SHA: actionable local-access error before extraction;
  never log a credential or use randomly initialized weights.
- Model incompatibility or CUDA OOM: stop bounded run, keep successful
  resumable clips intact; do not silently change batch size, precision,
  scales, or encoder version.
- Missing/changed/unreadable frame: record clip-specific failure; no shifted
  sampling or claimed complete manifest.
- Wrong width, dtype, non-finite CLS, duplicate clip or row, partial feature
  file, changed source/feature hash: fail closed with clip ID and stage.
- Unsupported platform storage behavior: verify atomic writes and read-back
  before a large extraction; write outputs outside Windows-mounted frame root.
- Translation model/checkpoint mismatch: require strict load; record any
  deliberate initialization or key migration. No partial model load.

Focused tests: filename sorting/ordinal mapping, 5-frame sampling, square
preprocessing at both scales, normalized channel values, CLS selection and
concatenation order, fake-encoder determinism, resume/corruption/version
invalidation, source-hash changes, count and cross-manifest checks, zero-failure
gate, batch/mask parity, same-config CLIP/DINO overlays, and acceptance-flag
separation. Optional live tests: one real gated DINO model and one real
PHOENIX clip; require weights access rather than treating a skip as a pass.

## 7. Implementation Order And Authorization

Write separate implementation plans for (1) versioned offline extraction
and (2) matched encoder comparison. Follow test-driven implementation and
per-task code review. Start with manifest/schema and fake-encoder tests while
HF access is pending. Run a one-clip DINO live smoke only after model access
and pinned extractor environment pass. Ask before any hours-long full-corpus
GPU extraction or full matched training; keep resulting artifacts external.

The subsequent seven-factor dataset and training plans begin only after
DINOv3 features and the controlled comparison are verified.
