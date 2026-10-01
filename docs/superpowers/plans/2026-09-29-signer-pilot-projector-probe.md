# E1/E2 Spatial-Projector Signer Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Project instructions forbid subagent dispatch and commits unless explicitly requested.

**Goal:** Produce diagnostic-only aggregate seven-signer probe comparison from selected E1/E2 spatial-projector outputs.

**Architecture:** Validate frozen inputs and both completed run identities before loading checkpoint data. Pool frozen train-clip DINO features once; apply only two projector affine layers; train fresh, deterministic CPU linear classifiers on identical clip partitions. Write one provenance-bound aggregate report outside frozen runs.

**Tech Stack:** Python 3.11, PyTorch 2.0.1, NumPy 1.26.4, pytest, Ruff; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-29-signer-pilot-projector-probe-design.md`

## Global Constraints

- Protocol hash `0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7`; seven logical train signers / 5,746 clips only.
- Selected E1/E2 step 4000 and hashes `04ce223fd1ae00362a7ceaa1e78c1a814c825a8b9551483d8ac2a332631c7755` / `7d73af1b8c3a190dd6b9f9725f7ebb01e63e3258837bb322e73494fb1b65575e` bound to their respective external runs.
- E2 policy `qwen-schema98-unreviewed-v1`; `human_review_status=not_assessed`. Preserve original artifacts; no E3 run or Signer07 scoring.
- No commits, subagents, or long training. No raw clip-level signer predictions in report.

## Review Focus

- Misordered/missing train clip or duplicate ID: reject rather than silently compare different rows; Task 1 test.
- Empty, nonfinite, or wrong-width feature array: reject before producing a report; Task 1 test.
- Changed/mismatched run checkpoint, selected hash, split hash or variant: reject before feature extraction; Task 3 test.
- Missing signer from fit/validation/test partition: reject before classifier training; Task 1 test.
- Identical or constant feature columns: train-only normalization remains finite; Task 2 test.

---

### Task 1: Deterministic partitions and pooled representations

**Files:** Create `src/despamo/evaluation/pilot_probe.py`; create `tests/unit/evaluation/test_pilot_probe.py`.

**Interfaces:** Produce `split_probe_clips(clips: list[tuple[str, str]]) -> dict[str, tuple[str, ...]]` for `fit|validation|test`, and `pool_spatial(view: PhoenixTrainView, expected_ids: tuple[str, ...]) -> torch.Tensor` returning row-aligned `[N,2048]` float32 data. Use frozen `spatial_manifest.require("train", clip_id)` and `_load_feature` or equivalent verified loader; never load motion data.

- [ ] **Step 1: Write failing tests** for seven per-signer floor(70%)/floor(15%)/remainder counts, deterministic disjoint ID inventory, missing-class rejection; pooled exact small-array means and duplicate/empty/nonfinite/wrong-width/ID-mismatch rejection.
- [ ] **Step 2: Run** `uv run --locked python -m pytest tests/unit/evaluation/test_pilot_probe.py -q`; expect relevant failures.
- [ ] **Step 3: Implement** `split_probe_clips` and `pool_spatial` in `pilot_probe.py`. Seed each sorted signer group from first eight SHA-256 bytes of `"0:" + signer_name` (big-endian); preserve canonical ID order in final partitions and pooled rows. Require exactly seven classes and all classes represented in every partition.
- [ ] **Step 4: Run** same targeted tests; expect pass.

### Task 2: Fresh classifier and paired score

**Files:** Modify `src/despamo/evaluation/pilot_probe.py`; modify `tests/unit/evaluation/test_pilot_probe.py`.

**Interfaces:** Produce `project_pooled(pooled: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor` and `score_probe(features: torch.Tensor, labels: list[str], split: dict[str, tuple[str, ...]], clip_ids: tuple[str, ...]) -> dict`. Caller supplies matched clip IDs and one checkpoint's projector weights; output includes counts, accuracy, balanced accuracy, majority baseline, selected decay and epoch, no per-clip predictions.

- [ ] **Step 1: Add failing tests** for mean-then-project equals project-then-mean, malformed/nonfinite projector rejection, fit-only normalization including constant columns, deterministic best-decay/earliest-epoch choice, score finite and no per-clip predictions.
- [ ] **Step 2: Run** targeted tests; expect failures.
- [ ] **Step 3: Implement** projector extraction math and classifier. Fit new CPU linear head per candidate decay `0`, `0.0001`, `0.01`; AdamW lr `0.01`, 200 epochs, seed 0. Select by validation balanced accuracy (smaller decay then earliest epoch on ties); fit-only normalization replaces zero std with one. Return one untouched test score per variant.
- [ ] **Step 4: Run** targeted tests; expect pass.

### Task 3: Safe CLI, provenance, and report

**Files:** Create `scripts/probe_signer_pilot.py`; create `tests/unit/evaluation/test_pilot_probe_cli.py`; modify `docs/signer-pilot-protocol.md` only to document new optional command after verification.

**Interfaces:** CLI requires `--protocol --dataset --text-manifest --dino-root --motion-root --motion-manifest --annotation --e1-run --e2-run --output`, supports `--mode preflight|run`. `preflight` validates frozen views and run metadata without loading checkpoint tensors or writing; `run` calls Tasks 1–2 and writes one new JSON. Reuse `load_pilot_views`, `file_hash`, `read_json`, `atomic_json`; reject existing output. Bind `run-status.json`, `budget.json` identity, selected checkpoint hash, protocol/split identity and factor policy; hash checkpoint files before loading. Load only local checkpoint state on CPU; ensure `visual_adapter.spatial_projector.weight/bias` shapes and values; release first checkpoint before loading second. Result includes hashes, partition hash/counts, accuracy metrics and E2-minus-E1 differences; no individual predictions.

- [ ] **Step 1: Write failing CLI tests** using tiny fake checkpoint artifacts and monkeypatched heavy reads: reject run-policy/variant/split/checkpoint mismatches, output alias or existing output, and missing clip inventory; accept paired report with fixed provenance and zero writes in preflight.
- [ ] **Step 2: Run** `uv run --locked python -m pytest tests/unit/evaluation/test_pilot_probe_cli.py -q`; expect failures.
- [ ] **Step 3: Implement** CLI and aggregate report. Keep local trusted checkpoint loading consistent with existing pilot runners; do not load FLAN weights into a model. Add brief protocol documentation only after tests pass.
- [ ] **Step 4: Run** both probe test modules and `uv run --locked ruff check scripts/probe_signer_pilot.py src/despamo/evaluation/pilot_probe.py tests/unit/evaluation/test_pilot_probe.py tests/unit/evaluation/test_pilot_probe_cli.py`; expect pass.
- [ ] **Step 5: Check** `git status --short` and `git diff --check`; inspect new files and diff. Run external preflight and live probe only if prerequisites and time permit; report actual runtime and any blocker, never fabricate metrics.
