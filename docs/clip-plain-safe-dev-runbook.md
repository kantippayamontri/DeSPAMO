# Plain CLIP / safe dev pipeline — implementation handoff

## Current status

Approved plan implemented in the retained runtime worktree's new
`tools/clip_plain/` namespace. WSL implementation and CPU verification only.
Production features have **not** been derived, the plain model has **not** been
GPU-profiled, and training/validation have **not** been launched. The original
Windows/CUDA crash cause is still unresolved.

Retained runtime worktree:
`/home/kan/Research/DeSpaMo/.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`.

Plan/spec:
- `docs/superpowers/plans/2026-10-11-plain-clip-safe-dev.md`
- `docs/superpowers/specs/2026-10-11-plain-clip-safe-dev-design.md`

Native implementation ledger:
`.superpowers/sdd/2026-10-11-plain-clip-safe-dev/progress.md`.

The existing S² controller remains held. Do not launch it expecting these fixes:
its checkpoint-bound source was intentionally preserved. New tools do not clear
the v3 stop, modify Windows audio, or start a retry/watchdog service.

## Spatial representation and new experiment

Plain means frozen `openai/clip-vit-large-patch14`, revision
`32bd64288804d66eefd0ccbe215aa642df71cc41`, raw final hidden CLS at224, FP32,
1024 dimensions. It is not projected/post-normalized CLIP image embedding.
Original cache layout is `[global224 CLS1024, averaged tile CLS1024]`; CPU
conversion writes an independent contiguous copy of columns0:1024 to a new
provenance-bound package. No image model, Pillow or GPU extraction is needed.

Width1024 changes the spatial MLP first layer to1024→1024→768. Motion1024,
temporal/reference fusion, shared2048 connector and pinned FLAN/LoRA remain as
planned. The plain variant requires **fresh initialization**, not S² epoch2
weights/AdamW sliced or padded into another shape. Old1774updates and
497.5533489369991s costs remain historical lineage, not new optimizer progress.
This is not a strict one-factor S² ablation because projector width/count change.

New native study identifier:
`/home/kan/datasets/despamo/official-phoenix14t-translation-clip-plain-spamo200-v1`.
Matching B: archive suffix. These production output roots are not yet populated.

## Modules and process boundaries

| Module | Responsibility |
| --- | --- |
| `policy.py`, `config.json` | Variant pins, source closure, explicit scope/boot/parent-ticket guards |
| `features.py` | CPU derivation, bitwise verification, parent receipt/source provenance |
| `data.py`, `model.py` | Official1024 feature admission and fresh reference model |
| `state.py` | Compact snapshots, historical lineage, monotonic owned accounting and storage |
| `generation.py` | Explicit cacheTrue, decoder finite checks, token IDs, safe primary-error propagation |
| `evaluate.py` | Fresh optimizer-free dev process, complete/bounded report contracts |
| `train.py` | Train to next dev boundary, full resume/candidate publication, no generation |
| `study.py` | CPU parent, actual child-exit barrier, strict selection and no automatic retry |

Training retains batch8/cacheFalse/gradient checkpointing and approved200-epoch
recipe. It publishes full checkpoints and exits at the next declared dev epoch.
Parent waits for process exit, then launches batch1/beam5/max64/cacheTrue evaluator.
Evaluator does not construct optimizer or restore global/CUDA RNG. New training
process subsequently restores AdamW/sampler/prompt+global RNG from full snapshot.

Generation writes fsynced batch/generation/decoder-start evidence. Raw last-step
logits must be finite; telemetry records dtype/stride/step/length without storing
vocabulary-sized tensors. Per-batch autocast bounds cast-cache lifetime. CPU
hooks/finally restore modes and remove hooks; after failure no deliberate CUDA
RNG, memory, synchronize, empty_cache, tensor transfer or save operation occurs.
An error observer/accounting failure cannot replace the primary exception.

Only ordered complete dev coverage with matching run/checkpoint/data/evaluation
bindings selects weights. Metrics are recomputed from stored predictions during
validation. Bounded probes always have null metrics and cannot select weights,
even when their bound covers the full corpus. Inference exemplar/reference and
token evidence are checked. Post-warmup identical-output guard is fail-stop,
not a trigger for retry.

## CPU preparation commands — require separate execution approval

Run from retained runtime worktree with the existing main interpreter. First,
read-only projection/check (does not convert or allocate package data):

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/kan/Research/DeSpaMo/src:.:/home/kan/Research/DeSpaMo HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/kan/Research/DeSpaMo/.venv/bin/python -B -m tools.clip_plain.features --mode check --config tools/clip_plain/config.json
```

Raw full-split derived payload is3.6154022216796875GiB for8257clips/947756frames,
plus calculated file/receipt/manifest/staging overhead. Native Linux and Windows
C: backing free space must exceed projection plus50GiB reserve. Latest earlier
observation near51GiB C: would be insufficient. Check current space; do not
delete/move old artifacts or silently pick another root to bypass the guard.

Only after explicit CPU preparation approval and a successful storage check:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/kan/Research/DeSpaMo/src:.:/home/kan/Research/DeSpaMo HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/kan/Research/DeSpaMo/.venv/bin/python -B -m tools.clip_plain.features --mode derive --config tools/clip_plain/config.json --authorize-feature-preparation
```

Then `--mode verify` rechecks without rewriting/repairing arrays. Partial,
unreceipted or conflicting files are preserved and block publication. Completed
matching clips can be reused; changed parent arrays/receipts/source frames stop
the converter. Conversion records separate elapsed costs in
`preparation-budget.json`, including failed normal attempts. Unclean termination
leaves running accounting that requires manual reconciliation; do not reset it.

## GPU approval and future rollout

No GPU permission receipt is created by implementation or default preflight.
GPU work requires new reviewed scope matching plain config/source hashes,
experiment, current boot, driver-review receipt and existing v3 hold bytes.
Approval token is one-shot, and child jobs bind exact live parent PID/boot/start
ticks and immutable ticket file. GPU exclusivity query happens only after scope
checks and never after a device fault. A plain-study stop blocks every new launch.

`study --mode preflight --config tools/clip_plain/config.json` checks CPU feature
projection. `--mode run` without approval fails before any model/device work.
Do not manufacture approval JSON from an example or treat a prior S² diagnostic
receipt as permission for the new model.

Before long training, separately review/authorize:
1. Actual CPU feature package and lineage/storage checks.
2. Fixed short/long-clip plain-model cached GPU diagnostics.
3. Complete519-clip dev and bounded train/checkpoint/exit/dev handoff profile.
4. Explicit seed0/1 training scope and new-owned retention permission.

The code provides worker/state primitives; future approvals and any initial
diagnostic job/candidate setup must be bound to the actual prepared plain model.
No production GPU headroom certificate exists yet. Storage projections default
conservatively until compact plain-model sizes are measured.

## Failure and manual recovery

- First worker error stops parent sequencing, writes new-study hold and marks
  scope consumed. No subsequent GPU worker or automated recovery launches.
- Preserve last full snapshot and failed progress/attempt journals. Never save
  tensors or restore CUDA RNG after a device fault.
- Before manual recovery, verify source/data/config/boot/owner and checkpoint
  bindings; reconcile dead-stage elapsed wall time without discarding charges.
  `reconcile_abandoned_stage(..., reviewed=True)` retains attempt high-water
  evidence and records any rollback to durable updates in `replay_history`.
  Backwards wall clock blocks reconciliation with original budget preserved.
- If a worker died after even-epoch resume but before model-only candidate,
  `recover_candidate(..., reviewed=True)` reconstructs candidate on CPU from
  verified new-variant snapshot. It does not replay training or alter old S² state.
- Missing dev report always precedes further training. Existing full reports are
  revalidated rather than regenerated. Evaluation-policy changes require a new
  identity, not rewriting report hashes.
- New-owned resume pruning defaults off. Only later explicit
  `allow_owned_resume_pruning=true` permits removing obsolete new-run full
  resumes after publication/verification, keeping latest2. Archive all model-only
  dev candidates and native best. No old-study artifact is eligible.

## Verification evidence

- Existing focused baseline:23passed.
- New and nearby focused suite:103passed,4 existing dependency warnings.
- Main visual/temporal/FLAN/batch targeted regressions:30passed,1 warning.
- Ruff check/format and installed Pyright with main interpreter:clean.
- Initial broad main unit-suite attempt exceeded165s. A longer CPU-only run
  completed:1625passed,5failed,4skipped,19warnings. No full-suite pass claimed.
  Four failures are existing CUDA-required held-out CLI prerequisites:
  `test_e3_heldout_cli.py::test_e3_heldout_run_writes_once_with_checkpoint_provenance`
  and `test_pilot_heldout_cli.py::{test_run_writes_ordered_paired_reports_once,
  test_pair_rejects_reference_drift_without_summary,
  test_resume_keeps_verified_e1_score_and_only_scores_e2}`. The fifth is unchanged
  `test_new_motion_fusion.py::test_final_summary_revalidates_full_graph_even_with_unchanged_marker`,
  whose fixture lacks the `scripts/run_fusion_study.py` source key. These match
  previously recorded baseline failures; unrelated code/tests were not changed.
- End-to-end file-backed CPU fixture executes train/dev/train/dev in separate
  processes, with actual child-exit barrier and compact AdamW/RNG restore.
- Additional CPU fault tests cover raw NaN/Inf, observer disk-full, forged report/
  job/update bindings, partial checkpoints, backwards clock and idempotent costs.
- Native self-review performed; no independent reviewer/subagent used.

Tests are synthetic/tiny CPU workloads. These results do not establish native
FLAN GPU correctness, throughput, long-run stability or root-cause repair.
All original84source/checkpoint/hold/budget hashes must remain unchanged at final
handoff. No commit, dependency change, Windows device change, real conversion,
official test scoring or training launch is part of this implementation.

## Native execution rulings and costs

1. No commits, subagents or ledger deletion: explicit user/project preferences
   override skill defaults. Cost: this is author self-review, with evidence in
   uncommitted files rather than an independent review or commit history.
2. Bind runtime/config/dependencies and exclude run outputs from source closure.
   Cost: code edits invalidate prepared receipts and require deliberate re-admission;
   output hashes are bound separately to avoid recursive identities.
3. Synthetic counts/paths use monkeypatches in tests only; production keeps full
   official counts. Cost: full native package preparation remains unverified.
4. Direct ledger bookkeeping follows actual tests in the retained runtime tree,
   rather than commit-oriented task scripts against main HEAD. Cost: review the
   ledger/receipt for evidence, not a nonexistent implementation commit range.
5. Tiny real AdamW/T5 and CPU subprocesses test state/generation contracts; they
   do not emulate full FLAN GPU behavior. Cost: staged GPU validation still needed.
6. Existing global Pyright with explicit main interpreter replaces nonexistent
   `.venv/bin/pyright`. Cost: tool path may need adjustment on another machine;
   runtime/dependency bytes remain unchanged.
7. GPU hold remains despite implementation success. Cost: production inference
   and training must wait for reviewed scope and capacity/validation evidence.

No deferred implementation defect was identified by the native final pass.
