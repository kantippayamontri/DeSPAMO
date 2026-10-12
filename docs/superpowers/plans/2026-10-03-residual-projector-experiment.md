# Residual Projector Experiment Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. The user requested immediate execution of the approved experiment. No commits or merges are authorized.

**Goal:** Launch residual-projector 8000-step and linear 16000-step downstream runs.
**Architecture:** Opt-in residual wrappers retain base projections and add zero-output
normalized MLP branches. A new versioned runner reuses verified data, initialization,
budgeting and dev evaluation, with two-snapshot retention.
**Tech Stack:** Existing Python 3.11, torch 2.0.1, Lightning 1.9.5, pytest/Ruff/Pyright.
**Spec:** `docs/superpowers/specs/2026-10-03-residual-projector-experiment.md`.

## Global Constraints

- Cached cosine features only; frozen signer split, batch 4, seed 0, no ICL.
- Residual hidden width 1024, branch seed 1, final residual layer zero initialized.
- Horizons/caps: residual 8000/10800 s; linear 16000/14400 s; VT warmup 2000.
- Retain two new-run resume snapshots; preserve all dev checkpoints and old artifacts.
- Run labels remain `qwen-schema98-unreviewed-v1` / `not_assessed`.
- No new packages; no commit/push/merge; implement in the current artifact-bound
  workspace without overwriting unrelated user changes.

## Review Focus

- Default linear model and shared initialization must remain bit-compatible.
- Smoke must restore RNG/buffers and must not consume the production sampler.
- Failed snapshot publication must never trigger retention pruning.
- Source/horizon/architecture/retention changes must create distinct run identities.
- Resume may not silently reset optimizer, scheduler or cumulative budget.

### Task 1: Residual projection wrappers

**Files:** create `src/despamo/models/residual_projector.py` and
`tests/unit/models/test_residual_projector.py`.
**Interface:** `ResidualMLPProjector(base: nn.Linear, hidden_dim: int)`;
`enable_residual_projectors(adapter, hidden_dim=1024, seed=1)`.

- [x] Write tests for baseline-equivalent initialization, RNG isolation, both-branch
  gradients, second-step hidden gradients, masking, and strict round-trip.
- [x] Run tests and observe missing implementation failures.
- [x] Implement wrappers without changing the default visual adapter.
- [x] Run the new tests and existing visual-adapter tests.

### Task 2: Experimental schedules and recovery callbacks

**Files:** create `scripts/train_projector_pilot.py` and
`tests/unit/pilot/test_projector_experiment.py`; modify the
`FullDevCheckpoints` constructor in `scripts/train_signer_pilot.py` only to
accept an optional validated explicit schedule.
**Interfaces:** `projector_checkpoint_steps(steps)`;
`BoundedResumeSnapshots(output, ledger, keep=2)`;
`projector_gradient_smoke(model, batch)`.

- [x] Write failing tests for schedules, legacy tier protection, safe retention,
  failed-save preservation, architecture-sensitive identity, and smoke state restoration.
- [x] Implement the explicit schedule and retention after successful atomic save.
- [x] Implement state-neutral joint gradient smoke.
- [x] Run focused model/pilot tests.

### Task 3: Versioned real-data runner and launch

**Files:** finish `scripts/train_projector_pilot.py`; create a persistent launch
script `scripts/run_projector_experiments.sh`; update experiment documentation.
**Interfaces:** `prepare_projector(args)`, `run_projector(args)`; CLI
`--projector {linear,residual} --steps {8000,16000} --authorize-projector-run`.

- [x] Bind baseline/cached data, complete code hashes, architecture, caps and retention.
- [x] Implement fresh/resumed training with explicit dev selection and durable status.
- [x] Run full tests, Ruff, targeted Pyright and independent review; resolve findings.
- [x] Preflight both real-data configurations and verify complete inventories.
- [x] Launch the two authorized stages sequentially in tmux; inspect smoke receipt
  and first persisted progress rather than infer health from GPU utilization alone.

## Execution ledger

Maintain `docs/superpowers/plans/2026-10-03-residual-projector-progress.md`.
Record RED/GREEN results, rulings and launch identities. User explicitly accepted
deferred Windows compaction; checkpoint retention implements the agreed storage
constraint. No long run starts before its gradient smoke passes.

## Optional later spatial experiment

See [patch-based attention/pooling](2026-10-03-optional-patch-pooling.md) for the
saved follow-up proposal. It is outside the currently authorized projector runs.
