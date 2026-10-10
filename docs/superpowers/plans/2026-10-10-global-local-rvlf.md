# Global/Local RVLF-Inspired Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for native implementation/self-review. No subagents unless user explicitly chooses them. Steps use checkbox (`- [ ]`) syntax; do not commit/push unless requested for that execution session.

**Goal:** On another server, implement frozen CLIP global S² plus hand/mouth local features, hierarchical fusion and one fine-grained alignment loss with FLAN-T5-XL.

**Architecture:** Keep published reference global path and connector contracts, add masked local temporal path and provenance-based alignment, then shared2048connector. Build new portable data/runtime/training namespace; historical original-host runners are references, not launch entrypoints.

**Tech Stack:** Existing main uv lock/Python3.11/torch2.0.1/transformers4.32/PEFT0.7.1. Tracker preparation uses separately pinned environment if needed. No Qwen, GRPO or CLIP adaptation here.

**Spec:** [Approved architecture and review defaults](../specs/2026-10-10-global-local-rvlf-design.md).

## Global Constraints

- Frozen CLIP `openai/clip-vit-large-patch14` revision32bd64288804d66eefd0ccbe215aa642df71cc41: global S²224/448→2048; left_hand/right_hand/mouth single224raw hidden CLS→1024each.
- Hierarchical local fusion, then global/local fusion in768channels; one shared768→2048/GELU→2048connector and FLAN-T5-XL revision7d6315df2c2fb742f0f5b556879d730926ca9001.
- One RVLF-inspired fine-grained contrastive loss on fused output, not separate branch losses; raw FLAN target embeddings, masked attention and multi-positive duplicate references.
- Global/local support must derive from original frame/window indices and temporal receptive fields. Existing global concatenation cannot be aligned by row number/equal lengths.
- Missing local evidence yields exact global fallback after all biases/temporal/alignment operations. Left/right anatomical identity, mouth not face; no crop flipping, stale-valid crops or dropped clips.
- Dataset7096/519/642; source frames827354/55775/64627. Portable paths explicit; transferred path-bound receipts require new verified rebinding, not fabricated original extraction.
- Reference recipe8×1/drop-last/887updates per epoch; max200/min40/dev every2/patience10/minDelta0.10; VT4000/LR10000; adapter3e-4→5e-5/FLAN5e-5→1e-5; same ICL/BF16/decoder and best/anchor semantics.
- M0 global-only/fine-grained, M1 local fixed residual, M2 confidence-aware global/local, same local/loss/data/initialization. Seeds0/1; profile before launch; no automatic additional held-out scoring.
- No modification or restart of original machine's active study. No model/data/checkpoints/secrets in Git. Retain native latest2resumes/selected best and verified dev archives; preserve50GiB reserve+actual demand.

## Review Focus

1. Crop/frame/left-right identity errors or stale detections can create false local evidence; reject them and maintain zero invalid features (Task2/3).
2. Global modality-boundary tokens have noncontiguous source support; equal-length interpolation or min/max envelopes must not leak unmatched local evidence (Task5).
3. Padded tokens, duplicate captions and extreme temperatures can break fine-grained contrastive loss or negatives; masks/multi-positive denominators must be correct and finite (Task4).
4. Biases/temporal normalization can produce nonzero missing-local outputs or change global buffers; fallback must be exact and gradient-isolated (Task5).
5. Historical absolute paths, incomplete checkpoints and publication-order crashes must not launch another host's study or alter replay/stopping (Task1/6).

## Execution setup and commands

Review spec first, resolve its marked engineering defaults and detector preparation
input, then execute in branch `feat/global-local-rvlf` on the other server. Keep
per-plan `.superpowers/sdd/2026-10-10-global-local-rvlf/progress.md` ledger, source
snapshot inventory and test output. Preserve copied historical modules unchanged.

CPU test prefix:
`PYTHONPATH=src:. CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
uv run --locked pytest`. Expect failures for newly missing functionality first,
then GREEN after minimal implementation. Do not execute historical same-root
controllers or test scorers just because they are now tracked in this branch.

Create modules with these responsibilities:

| Path | Responsibility |
| --- | --- |
| `tools/global_local/config.py`, `runtime.py` | Strict portable inputs, source/storage/owner guards |
| `tools/global_local/regions.py` | Versioned detector-output admission and deterministic crops/tracks |
| `tools/global_local/extract.py` | CLIP global/local features and verified package publication |
| `src/despamo/data/global_local.py` | Region masks/confidence/timing batches and content provenance |
| `src/despamo/losses/fine_grained_vt.py` | All-pair segment/token matching and one contrastive objective |
| `src/despamo/models/global_local.py` | Hierarchical local branch, support alignment, residual gate/connector |
| `tools/global_local/train.py`, `study.py` | Portable profile/update/resume/dev-only orchestration |
| `tests/unit/{data,losses,models}/test_global_local*.py`, `tools/global_local/tests/` | Corresponding behavioral/recovery tests |
| `configs/experiment/global_local_reference.yaml` | Frozen checked recipe/defaults and explicit local paths |

### Task1: Portable config and owned runtime

**Interfaces:** `load_inputs(path: Path) -> ExperimentInputs` with annotations,
frames/global/local/motion/cache/output/archive roots, seed, fusion/alignment
variant and canonical source/model/package bindings; `preflight(inputs) -> dict`;
`storage_guard(inputs, remaining_native_bytes: int, remaining_archive_bytes: int)`.

- [ ] RED: Test missing paths, unsupported seed/variant, output inside checkout, mixed package identities and configured WSL/non-WSL storage. Assert non-WSL check never invokes PowerShell, source drift blocks resume and native locks reject duplicates.
- [ ] Run new `tools/global_local/tests/test_runtime.py`; expected missing interface failures.
- [ ] Implement typed config, explicit command/state identity and native/backing guards without importing original-host root constants. If same filesystem holds output/archive, account combined demand. No silent batch/recipe overrides or inherited time resets.
- [ ] Run GREEN and import smoke in clean clone with only `PYTHONPATH=src:.`; no original checkout paths may be required.

### Task2: Region provider contract and crop preparation probe

**Interfaces:** `admit_regions(manifest: Path, frame_inventory: dict) -> RegionInventory`;
`crop_regions(rgb_frame, observations) -> dict[str, crop|None]`; region order
left_hand/right_hand/mouth; validity/confidence/frame boxes and tracking provenance.

- [ ] RED: Test out-of-bounds/NaN/degenerate boxes, anatomical swaps, missing mouth landmarks, stale tracks and per-clip state reset. Identical source video processed in different minibatch orders must produce identical crops/masks. Fixture includes moving hands and missing regions.
- [ ] Run `tests/unit/data/test_global_local_regions.py`; expected missing functionality.
- [ ] Implement manifest validation and deterministic crop/track transformations. Conduct train-only preparation probe to choose/pin actual detector/checkpoint and mouth/hand margins/smoothing/thresholds; preserve provider confidence definition. Use a separate locked detector environment if APIs require it; do not install guessed dependencies into translation runtime.
- [ ] Freeze detector/config SHA and parameters before full extraction. Review sample train overlays for identity/crop geometry without fabricating human annotations or changing captions. Missing regions remain invalid, not backfilled as reliable.
- [ ] Run GREEN and record probe/source evidence; provider availability is an execution prerequisite, not an assumed implemented detector.

### Task3: Shared frozen CLIP extraction and aligned batches

**Interfaces:** `extract_global(frames, encoder, processor) -> ndarray[N,2048]`;
`extract_local(frames, regions, encoder, processor) -> (features[N,3,1024], valid[N,3], confidence[N,3])`;
`GlobalLocalBatch` extends existing tensors with local fields and source support;
`admit_packages(inputs) -> dict` and `batch_for(...) -> GlobalLocalBatch`.

- [ ] RED: Test raw hidden CLS versus pooler/projection, global S² aggregation, local single-scale output, left/right/mouth channel order, batch tails and invalid-feature zeros. Distinct sentinel package roots must prove all tensors load from configured roots.
- [ ] Run `tools/global_local/tests/test_extraction.py` and `tests/unit/data/test_global_local_batch.py`; expect missing implementation failures.
- [ ] Implement exact pinned CLIP processing from frozen-CLIP/S² plan; processor metadata at same revision must exist and be hashed. Share one frozen encoder instance. Bind frame/crop/model/precision/batching/provider fingerprints, canonical source indices/content and host-local path mapping separately.
- [ ] Explicitly pin Pillow/image decoding dependencies in the selected extraction environment using existing uv configuration; main cached-feature lock does not include Pillow. Verify processor parity after dependency resolution; never silently import another environment's packages.
- [ ] Publish arrays and receipts atomically; completion only after8257clip/947756frame verification. Preserve ambiguous crash artifacts. Reuse transferred VideoMAE only after array/window/frame provenance checks; rebind paths in new receipts without editing originals.
- [ ] Run GREEN, one-clip real extraction smoke and independent global-S² parity test, then profile and verify complete extraction before training. GPU work occurs only on intended server after ownership/storage checks.

### Task4: Single fine-grained visual/text contrastive loss

**Interfaces:** `FineGrainedVTAlignLoss(visual_dim=2048, text_dim=2048, alignment_dim=256)`;
`forward(visual, visual_mask, text, text_mask, normalized_references) -> scalar`;
`pair_scores(...) -> (visual_to_text[B,B], text_to_visual[B,B])`.

- [ ] RED: Compare toy attention-weighted scores to independent explicit loops for all cross-batch pairs. Assert padding extension/NaN padding invariance, symmetric directions, nonzero visual/projection gradients and frozen FLAN embedding weights.
- [ ] RED: Test duplicate-caption multi-positives, extreme temperatures, masks with no valid token and equivalent one-token examples. Empty visual/text evidence must fail explicitly; no false negatives for exact normalized duplicates.
- [ ] Run `tests/unit/losses/test_global_local_fine_grained_vt.py`; expect failure before loss exists.
- [ ] Implement FP32 normalized projected-token similarities, valid-key softmax and valid-query mean in each direction, then multi-positive contrastive denominators. Avoid padded max/sum contamination; chunk pairs if measured memory needs it. Positive temperatures bounded; one loss only.
- [ ] Run GREEN; on tiny frozen-token-embedding FLAN substitute verify warmup sends gradients to visual/alignment modules but not LoRA, and joint translation adds LoRA gradients. Global-only control uses same loss.

### Task5: Hierarchical local branch and global/local gate

**Interfaces:** `LocalBranch.forward(features, valid, confidence, source_indices) -> LocalSequence`;
`propagate_global_support(frame_indices, motion_windows, temporal_spec) -> TokenSupports`;
`align_local(local_sequence, global_supports) -> (aligned[B,Tg,768], usable[B,Tg], reliability[B,Tg,6])`;
`GlobalLocalAdapter.forward(batch) -> (visual[B,Tg,2048], mask[B,Tg])`.

- [ ] RED: Test region projections and concat2304→768 local fusion, invalid NaN masking before/after bias, temporal support propagation and short clips. Invalid padding must not create observations. Keep local normalization choices identical across fixed/gated comparisons.
- [ ] RED: Use global tokens around spatial/motion concatenation boundary; noncontiguous supports must not become min/max envelopes. Align by true support overlap; uncovered local entries zero, no nearest extrapolation. Tests deliberately use unequal global/local sequence lengths.
- [ ] RED: All local missing yields bitwise baseline equality in train/eval and identical global buffers, with zero local gradients. Partial missing clips never leak another batch item's regions. Common global/FLAN initialization fingerprint is identical across variants.
- [ ] Run `tests/unit/models/test_global_local_adapter.py`; expect missing implementation failures.
- [ ] Implement local modules in forked RNG context, post-temporal/alignment hard masks, support-overlap weighting and shared output connector. Fixed/gated residuals start at same small contribution (reviewed0.1 default); learned gate consumes G/L plus explicit reliability, not confidence alone. No per-region adaptive gate in initial ablation.
- [ ] Run GREEN and real joint/backward smoke, proving local fusion, temporal path, gate and alignment all receive finite gradients when valid. Confirm no visual BatchNorm recomputation under transformer activation checkpointing.

### Task6: Profile, full-state replay and dev-only study

**Interfaces:** `profile(inputs) -> dict`, `run(inputs) -> dict`, `study(inputs) -> dict`;
native resume includes all new local/gate/alignment parameters and persistent
buffers, optimizer/sampler/prompt/global RNG; completion pins actual selected
checkpoint, stopping history and metadata.

- [ ] RED: Test exact schedule and prompt bank, matched model fingerprints, source/data/model/config drift, missing new module tensors, report/pointer publication crash and stop reconstruction. Verify next batch/exemplars/optimizer result equals uninterrupted tiny-model execution after replay.
- [ ] Run `tools/global_local/tests/test_training_recovery.py`; expect missing lifecycle failures.
- [ ] Implement new namespace owned loop using unchanged sampler/checkpoint primitives where contracts fit, not old absolute-root controllers. Preserve elapsed/replayed charges, latest2/best/dev archive publication and minDelta anchor versus actual best. A stop decision prevents further updates on recovery.
- [ ] Run GREEN, then real batch8/long-visual/long-target/full-ICL profile; initialize every trainable AdamW state and verify deterministic generation after fresh-base compact reload. Memory/batch changes need versioned matched recipe, never an implicit fallback.
- [ ] Self-review and operator review before training scope/launch. Freeze M0/M1/M2 seeds0/1 identities, same crop/loss/decoder/recipe, monitor/locks and measured demand. Dev-only orchestration must not call new test scorer or create claims.
- [ ] Report paired dev results, selected/stopped epochs, updates/compute and mask/visibility diagnostics. No novelty/causal-grounding claim from BLEU alone. Qwen/GRPO/local-confidence extensions remain separately planned.

## Final verification

- [ ] Run focused new tests first, then configured main unit suite with GPU disabled. Run motion/DINO tests only in their designated environments when changed.
- [ ] Run Ruff on new modules and targeted typing; inspect scoped diff/status and source inventory. Preserve external datasets/models and secrets outside Git.
- [ ] Run real extractor/model/profile/replay checks on intended GPU only after CPU checks and ownership/storage admission. No full training solely to establish import/shape validity.
- [ ] Native self-review all five Review Focus cases; fix substantive findings with RED→GREEN tests. Update server handoff with actual implemented commands/results, not future assumptions.

## Publication self-review

The baseline snapshot was copied into a separate worktree so current training
source hashes remain unchanged. Required motion provenance helpers accompany
published reference/fusion source; baseline CPU tests must run in this checkout,
not through another checkout's PYTHONPATH. Snapshot includes no model/data blobs.
This plan defines future modules; none are presented as implemented. Detector
choice and marked engineering defaults need implementation review. Explicit
support-set alignment corrects the tempting but invalid equal-length assumption.
Each Review Focus case has tests in owning task. User requested branch publication
and approved architecture/extractors; no automatic local GPU launch is inferred.
