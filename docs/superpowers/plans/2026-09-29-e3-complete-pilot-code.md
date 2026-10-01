# E3 Complete Signer-Pilot Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Project policy forbids subagent dispatch, commits, and long jobs without explicit authorization.

**Goal:** Prepare safe, tested E3 DINO-LoRA adaptation, full adapted-feature extraction, matched SpaMo training, and once-only held-out evaluation commands.

**Architecture:** Reuse reviewed recovered LoRA/head mathematics in a separate pinned DINO environment; bind immutable selected adaptation checkpoint to a complete new spatial feature version. Build matched E3 SpaMo on that verified manifest without weakening the frozen original DINO protocol checks.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu121/Transformers 4.56.2 in `tools/dinov3`; main PyTorch 2.0.1/Lightning 1.9.5 stack; NumPy, pytest, Ruff. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-29-e3-complete-pilot-code-design.md`

## Global Constraints

- Frozen protocol hash `0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7`, dataset key `b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a`, base DINO revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`.
- Logical train 5,746 clips/seven signers; Signer03 dev 582; Signer07 test 768. Physical feature paths always `train/<clip_id>.npy` for all 7,096.
- Partial target masks remain valid; wholly failed clips have all 19 masks false and zero; all 7,096 translation targets remain available.
- E3 labels `qwen-schema98-unreviewed-v1`, `human_review_status=not_assessed`. Do not overwrite original Qwen, DINO, E1/E2, or held-out results.
- No commits, merges, subagents, or full training/extraction runs. Full GPU work requires explicit per-stage cap and run authorization arguments.

## Review Focus

- Partial-valid Qwen rows with null text: retain unmasked targets and reject target/NPZ mismatch (Task 1 test).
- Restart after checkpoint but before ledger update: reject stale or unowned snapshot rather than double-charge/rewind (Task 2 test).
- Interrupted extraction with partial feature file: verify per-clip receipt/hash and replace only incomplete staging (Task 3 test).
- Adapted manifest with correct clip count but different adaptation checkpoint: reject E3 view before model build (Task 4 test).
- Duplicate/partial Signer07 report after interruption: refuse E3 scoring or resume only verified exact same checkpoint/IDs (Task 6 test).

---

### Task 1: Mask-aware train-only E3 dataset

**Files:** Create `tools/dinov3/e3_data.py`; create `tools/dinov3/tests/test_e3_data.py`. Read recovered `tools/dinov3/adapt_data.py`, `adapt_profile.py` without editing archive.

**Interfaces:** `validate_e3_sources(protocol: dict, source: dict, text: dict) -> tuple[str, ...]` returns exact logical-train IDs; `load_e3_batch(ids: tuple[str,...], source: dict, text: dict, rows: dict, *, frames: Path, dino_root: Path, text_manifest: Path) -> dict` returns `images224` `[B,5,3,224,224]`, `images448` `[B,5,3,448,448]`, original teacher `[B,5,2048]`, vectors `[B,19,768]`, boolean masks `[B,19]`, labels `[B,19]`, clip IDs. Separate sample/path verification from image decode so source checks work without GPU.

- [ ] **Step 1: Add failing tests** for 5,641 valid/105 failed train rows (including valid partially masked rows), exact null-text-to-NPZ mask match, zero invalid vectors, train-only selection, duplicate/foreign IDs and PNG/teacher hash drift.
- [ ] **Step 2: Run** `UV_PROJECT_ENVIRONMENT=/tmp/opencode/despamo-e3-smoke-env uv run --offline --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_e3_data.py -q`; expect new tests fail.
- [ ] **Step 3: Implement** functions in `e3_data.py`; match Qwen's five sampled frame indices to `frames.sample_indices(frame_count)` and verify original row map's full source-index order and source receipts. `load_e3_batch` rejects any selection containing Signer03/Signer07.
- [ ] **Step 4: Run** same command; expect pass.

### Task 2: Bounded DINO LoRA trainer with strict resume

**Files:** Create `tools/dinov3/e3_model.py` (reviewed LoRA/head math); create `tools/dinov3/e3_train.py`; create `tools/dinov3/tests/test_e3_train.py`.

**Interfaces:** `step_e3(model, heads, batch, *, step: int, calibration_steps: int, adapt_steps: int) -> dict[str, Tensor]` returns finite combined/factor/reference losses; `validate_e3_resume(path: Path, identity: dict, spent: float) -> dict` checks owned latest snapshot and monotonic spend; CLI `--mode preflight|run --adapt-steps N --calibration-steps M --gpu-cap-seconds S --authorize-e3-run`. Train sampler is deterministic clip-level seed 0, physical batch 4. Adaptation output: final checkpoint and closed ledger with hashes and exact step.

- [ ] **Step 1: Add failing CPU tests** for four negative GRL directions/three positive, detached cosine teachers, all-masked failed sample, joint alpha 0→1 over first 10%, calibration head-only gradients, 8 nonzero Q/V LoRA gradients, exact 4-clip batches, checkpoint identity/step/optimizer/RNG/sampler/spend and stale-resume rejection.
- [ ] **Step 2: Run** `UV_PROJECT_ENVIRONMENT=/tmp/opencode/despamo-e3-smoke-env uv run --offline --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_e3_train.py -q`; expect failures.
- [ ] **Step 3: Implement** reviewed rank-8/alpha-8 last-four Q/V LoRA and seven heads; AdamW 1e-4 LoRA/3e-4 heads, 0.05 nuisance/0.10 articulator/0.10 reference, gradient clip 1.0, explicit budget; no implicit long run. Use existing `tools.dinov3.storage.writer` and atomic snapshot/receipt patterns.
- [ ] **Step 4: Run** same tests; expect pass. Run one bounded real gradient smoke only, not full adaptation.

### Task 3: Resumable full physical-train adapted-feature version

**Files:** Create `tools/dinov3/e3_extract.py`; create `tools/dinov3/tests/test_e3_extract.py`.

**Interfaces:** `adapted_key(identity: dict) -> str` hashes adaptation checkpoint/encoder/preprocess/source identity; `verify_adapted_version(root: Path, identity: dict, expected_ids: tuple[str,...]) -> dict` enforces complete `manifest.json`/`frame_rows.json`/receipts. CLI `--mode preflight|run --checkpoint PATH --gpu-cap-seconds S --authorize-e3-extraction`. Adapted arrays/receipts for 7,096 physical `train/` clips; no official dev/test extraction.

- [ ] **Step 1: Add failing fake-model/PNG tests** for two-scale CLS parity, row order/count, final checkpoint drift, stage/resume after partial file, extra/duplicate IDs, finite arrays, receipt/source hashes, zero-failure atomic complete promotion, output refusal on original encoder root.
- [ ] **Step 2: Run** `UV_PROJECT_ENVIRONMENT=/tmp/opencode/despamo-e3-smoke-env uv run --offline --project tools/dinov3 --locked python -m pytest tools/dinov3/tests/test_e3_extract.py -q`; expect failures.
- [ ] **Step 3: Implement** separate versioned extraction using existing `encoder.extract` and `storage.writer`, restoring only final owned LoRA. Reverify every finished row before reusing and bind all data to the final adaptation checkpoint hash.
- [ ] **Step 4: Run** same tests; expect pass. Do not execute all-frame extraction.

### Task 4: Adapted-feature pilot views without relaxing frozen protocol

**Files:** Create `src/despamo/training/e3_runtime.py`; create `tests/unit/pilot/test_e3_runtime.py`.

**Interfaces:** `load_e3_views(original_args: dict[str, Path], adapted_root: Path, adaptation: dict) -> tuple[dict, dict[str, PhoenixTrainView]]` calls existing `load_pilot_views` for immutable original source, then `verify_adapted_version` metadata using main-stack JSON/manifest-only checks (never import `tools/dinov3`); construct `Phoenix14T` physical `train` with adapted spatial and original motion, `spatial_crop_mode="full"`.

- [ ] **Step 1: Add failing tests** for identical 5,746/582/768 logical IDs, physical `train/` feature lookup, source/adaptation identity mismatch, stale/missing manifest/row receipts, and zero model loading on rejection.
- [ ] **Step 2: Run** `PYTHONPATH=src:. /home/kan/Research/DeSpaMo/.venv/bin/python -m pytest tests/unit/pilot/test_e3_runtime.py -q`; expect failures.
- [ ] **Step 3: Implement** separate adapted-view loader; preserve `load_pilot_views` original DINO hash gate unchanged. Do not import DINO extractor packages in main runtime.
- [ ] **Step 4: Run** same tests; expect pass.

### Task 5: Fresh matched E3 SpaMo training

**Files:** Create `scripts/train_e3_pilot.py`; create `tests/unit/pilot/test_e3_train_cli.py`; update `docs/signer-pilot-protocol.md` with commands after tests pass.

**Interfaces:** CLI `--mode preflight|run --adapted-root PATH --e1-run PATH --gpu-cap-seconds S --authorize-e3-run`; builds `E3_dino_lora` using `make_variant`/`PilotBaselineModule`, `PilotDataModule`, `FullDevCheckpoints`, ledger/snapshot patterns. Runs 4,000 steps, 1,000 warm-up, batch 4, seed 0, dev checks at 1,750/2,800/4,000.

- [ ] **Step 1: Add failing fake-model/CPU tests** for matching fresh shared tensor hash, no E2 factor heads, exact train/dev/signer memberships, 4000-step schedule, strict adapted hash and E1 source config admission, closed budget/resume, highest 582-clip dev BLEU selection and no Signer07 access.
- [ ] **Step 2: Run** `PYTHONPATH=src:. /home/kan/Research/DeSpaMo/.venv/bin/python -m pytest tests/unit/pilot/test_e3_train_cli.py -q`; expect failures.
- [ ] **Step 3: Implement** E3-only runner with distinct run key and immutable E3 adapter provenance; require GPU and explicit cap for real run, no E1/E2 weight reuse. Keep E1/E2 artifacts untouched.
- [ ] **Step 4: Run** same tests; expect pass.

### Task 6: Once-only E3 held-out handoff and final verification

**Files:** Create `scripts/evaluate_e3_pilot.py`; create `tests/unit/evaluation/test_e3_heldout_cli.py`; update `docs/signer-pilot-protocol.md` with handoff commands (no metrics claims).

**Interfaces:** CLI `--mode preflight|run` validates E3 selected checkpoint/dev result/adapted source before calling existing `score_pilot_batches` on exactly 768 Signer07 IDs; immutable output and verified resume; `human_review_status=not_assessed`.

- [ ] **Step 1: Add failing tests** for dev-selected checkpoint binding, reference-free beam-5 generation, ordered 768 IDs, existing/partial output refusal, verified resume and no held-out-driven checkpoint selection.
- [ ] **Step 2: Run** `PYTHONPATH=src:. /home/kan/Research/DeSpaMo/.venv/bin/python -m pytest tests/unit/evaluation/test_e3_heldout_cli.py -q`; expect failures.
- [ ] **Step 3: Implement** E3 held-out CLI, add explicit preflight/run commands and cautions to protocol docs; do not run Signer07 again in this task.
- [ ] **Step 4: Run** targeted tests; expect pass.

### Task 7: Optional matched E3 signer probe and final verification

**Files:** Create `scripts/probe_e3_pilot.py`; create `tests/unit/evaluation/test_e3_probe_cli.py`; update `docs/signer-pilot-protocol.md` with optional diagnostic command.

**Interfaces:** CLI `--mode preflight|run --adapted-root PATH --e3-checkpoint PATH --split-manifest PATH` uses verified E3 views, `pool_spatial`, `split_probe_clips`, and `score_probe` from existing E1/E2 diagnostic on the same train clip IDs, same seven signer classes and fixed partitions; stores an aggregate versioned report only, never individual predictions or a training gate.

- [ ] **Step 1: Add failing tests** for E3 adapted-feature/checkpoint binding, exact E1/E2 probe partition hash and class counts, no Signer03/Signer07 probe samples, and finite aggregate diagnostic without individual predictions.
- [ ] **Step 2: Run** `PYTHONPATH=src:. /home/kan/Research/DeSpaMo/.venv/bin/python -m pytest tests/unit/evaluation/test_e3_probe_cli.py -q`; expect failures.
- [ ] **Step 3: Implement** optional CLI and docs; never treat probe score as E3 acceptance or signer invariance.
- [ ] **Step 4: Run** targeted tests plus `PYTHONPATH=src:. /home/kan/Research/DeSpaMo/.venv/bin/python -m pytest tests/unit -q` and `ruff check scripts src tests tools/dinov3`; expect pass.
- [ ] **Step 5: Inspect** `git diff --check`, status and all new files; document exact bounded smoke and preflight evidence. No commit/merge/training launch.
