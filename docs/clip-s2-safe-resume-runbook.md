# Retained CLIP+S² safe-process runtime — implementation and bounded handoff

## Status

Approved design/plan implemented natively under retained runtime worktree
`tools/clip_s2_safe/`. Separately approved native preparation and one provisional
update1774→1775/checkpoint/actual-exit/one-clip cached dev handoff completed.
After separately approved storage correction/re-admission, all519 production
epoch2 dev clips completed and were admitted at original1774; selected anchor,
verified archive and production dev gate published. Scope consumed, owners exited;
Then the separately authorized1774→3548 train/checkpoint/exit segment completed
after verified snapshot relocation resolved backing storage. Production is now
5322 after separately scoped3548→5322 joint-transition training/checkpoint/exit.
Epoch6 dev519 completed and was admitted: BLEU4=2.0776100886063897, new best5322
over epoch2 .06687137508847464. Significant anchor6/badchecks0/stopfalse; no restart.
Provisional1775 was not adopted. User subsequently authorized continuous Seed0
max200/early-stop continuation and explicitly approved rolling latest2 retention.
Long run launched09:02:17UTC;09:06:31UTC progress5625/durable5322, owned/live.
The Windows/CUDA initiating fault is still unresolved. Current evidence:
`docs/clip-s2-long-run-launch-2026-10-12.md`/JSON and verified DINO200 training parity
receipt; epoch6 dev and previous bounded training,
epoch2 dev admission and previous-binding probe remain historical evidence.

Runtime worktree:
`/home/kan/Research/DeSpaMo/.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`.

Design/plan:
- `docs/superpowers/specs/2026-10-12-clip-s2-safe-resume-design.md`
- `docs/superpowers/plans/2026-10-12-clip-s2-safe-resume.md`

Ledger: `.superpowers/sdd/2026-10-12-clip-s2-safe-resume/progress.md`.

## What is preserved

Same original v3 study and seed0 run. Original2048 S² inputs, motion, model factory,
update kernel, optimizer groups, sampler, recipe and epoch2 checkpoint unchanged.
Production/durable5322updates are authoritative; original1774 anchor is retained.
Long-run admission inherited6430.341189303363s, including prior5140.252766528094s
plus1290.0884227752686s epoch6 dev. Active original budget continues accumulating
costs; all inherited/probe/training/dev charges retained, no zero-cost budget,
new feature extraction, spatial slicing, fresh model initialization or Seed1.

Original anchor SHA256:
`d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9`.

CPU native audit:313 mutable weights/buffers,19 initialized optimizer states,
groups adapter/flan, sampler1774/cursor7096, one816-byte CUDA RNG state. CUDA RNG
was not restored and full FLAN was not constructed for audit. LoRA optimizer states
may be uninitialized during VT-only warmup; this is not missing-state corruption.

All84 original source hashes, epoch1/2 payloads/receipts, legacy hold and complete
diagnostic report remain unchanged. Initial budget bytes are retained separately;
the authoritative budget now includes probe/dev costs and one provisional update.
Plain runtime was not modified by this handoff. The legacy
controller stays blocked by its original manual-stop. Never launch old
`tools.clip_s2.study` expecting cached/process-safe evaluation to be active.

## Runtime layout and identity

New files live under **same original run** `safe-runtime-v1/`, prepared during
separately approved execution. Current binding8298146c binds93 source files;
closure is frozen. Previous396f3b82 source/migration/head preserved under
`readmissions/dev-storage-v1/`, following explicit CPU-tested storage re-admission.
Use full hashes and receipts in the production dev validation report.

| Module | Role |
| --- | --- |
| `policy.py`, `config.json` | Original pins, source closure, external scope/parent-ticket admission |
| `state.py` | Original snapshot overlays plus runtime/CE guard sidecars and migration |
| `accounting.py` | Original-budget intention/head transactions, probe costs and explicit adoption |
| `evaluate.py` | Fresh optimizer-free cached dev and complete/probe report validation |
| `train.py` | Exact original-state restoration and train-to-boundary chunk |
| `study.py` | CPU parent, actual child-exit barrier, safe selection/stopping and holds |

`migration.json` embeds original identity plus independent runtime/config/source/
evaluation binding. Only exact original anchor may lack new sidecars. All later
snapshots need strict original receipt and `.runtime.json`/`.guard.json`; modified
source/policy/checkpoint/guard history fails admission. Safe snapshots remain
outside legacy resume glob. Both1024 feature halves of S² are retained.

Training retains batch8/cacheFalse and original LR/VT positions. Model/device and
optimizer initialize before one-time restoration of AdamW, sampler, prompt/global
RNG. CPU validates moment shapes/dtypes/finiteness and sampler schema before
restoration. One native VT-warmup restore/update/save handoff passed; saved prompt
and CUDA RNG bytes match the anchor. Joint-CE/dropout equivalence remains untested.

Each train child saves healthy full snapshots and model-only candidate, writes
handoff and exits. Parent waits/reaps **actual exit**, then creates fresh dev child.
Dev loads weights only, never optimizer/global CUDA RNG. Dev uses batch1,
beam5/max64/cacheTrue, per-batch BF16 autocast and finite raw-logit/decoder telemetry.
Persistent CE last200 and absolute step survive fresh training workers.

Explicit S² policy is`clip-s2-cached-b1-beam5-v1`. Shared generation core's
`clip-plain` tag is implementation metadata, not feature identity. Plain variant
contracts/runtime are not changed or widened.

Current continuous scope preserves DINO200 training hyperparameters: entire bound
recipe/75 shared source hashes/base model/runtime/ICL/common initialization match.
See `docs/clip-s2-dino200-hyperparameter-parity-2026-10-12.json`.

## Production dev and selection

Initial production epoch2 dev is **admitted at1774**, before production update1775. New
complete report/telemetry was generated; earlier successful519-clip diagnostic
remains prerequisite evidence, not a promoted/relabelled selection report.
Production subsequently reached3548 and epoch4 dev is admitted. Its BLEU4
.06546646339664837 is below epoch2 .06687137508847464; epoch2 remains selected.
Production reached5322 after1322joint updates; epoch6 dev now admitted and5322
selected with BLEU4=2.0776100886063897/ROUGE-LF1=.1306459322071727.

Production reports live under safe `dev/epoch-*.json` and bind original identity,
runtime/evaluation/checkpoint/data/telemetry hashes, exact ordered IDs/references,
original train-only exemplar, token evidence and recomputed metrics. Only full,
complete production reports select. Bounded/probe reports remain diagnostic/null
metrics even when their limit covers the corpus. Missing/changed/duplicate IDs,
wrong binding/policy or stale telemetry rejects the report.

Best is actual maximum BLEU4; earlier ties win. Significant0.10 anchor controls
patience separately. Original minimum40, VT4000 eligibility,200 ceiling and
fail-stop collapse guards preserved. Safe pointers/stopping do not overwrite
legacy reports, controller state or selections. No official test scoring.

## CPU preflight and separate preparation

From retained runtime worktree with existing main interpreter and CPU environment:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/kan/Research/DeSpaMo/src:.:/home/kan/Research/DeSpaMo HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/kan/Research/DeSpaMo/.venv/bin/python -B -m tools.clip_s2_safe.study --mode preflight --config tools/clip_s2_safe/config.json
```

Preflight reads/validates original state and proposes binding; no model/device,
native directory, budget mutation or approved scope. It is not preparation.

After explicit CPU preparation permission and fresh capacity/provenance review:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/kan/Research/DeSpaMo/src:.:/home/kan/Research/DeSpaMo HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/kan/Research/DeSpaMo/.venv/bin/python -B -m tools.clip_s2_safe.state --config tools/clip_s2_safe/config.json --prepare --authorize-preparation
```

This publishes immutable binding and independent copy of initial raw budget,
not a new budget or GPU permission. Changed original files or existing differing
binding blocks preparation. Do not silently revise pins to accept drift.

## Future GPU scopes and bounded adoption

Completed bounded handoff, epoch2/4/6 dev and1774→3548/3548→5322 training receipts are
consumed and cannot admit another run.
New active continuous token is`production-seed0-step5322-to-max200-continuous-v1`,
caps172078 remaining updates/50343dev clips, supervisor/train/dev stages. User
explicitly approved latest2 safe-runtime full-resume retention; all candidates,
selections and original epoch1/2 stay preserved. Permission is boot/source-bound
with90day credential expiry; original training duration limit remainsNone.
Any further external reviewed receipt must
bind runtime/config/anchor/legacy hold/driver-review hashes, current boot, seed0,
mode, stages, expiry, cumulative update/clip caps and unique token. Child ticket
must match exact live parent PID/boot/start ticks and stored job JSON. Scope checks
precede device query/model construction. Original hold remains unchanged; runtime
has its own fail-stop gate. `--mode run` without scope rejects before GPU work.

Normal production respects pending dev then next declared boundary. An intentionally
limited scope stops cleanly as scope_complete when the next stage exceeds caps;
it does not manufacture more permission. No retry/watchdog.

`handoff_probe` mode is a separately reviewed train→checkpoint→actual exit→bounded
dev diagnostic. Probe updates/costs are retained, but production update counter
stays at anchor. A completed handoff evidence file records matching snapshot and
dev hashes, and journal records completed stages. Adoption requires a **new exact
approval**, admitted production anchor dev, unchanged production position and
completed journalled handoff—not merely a saved checkpoint. Copy happens only
after validation; adoption certificate commits with authoritative budget transaction.
No extra elapsed costs on adoption; no automatic promotion. Partial/orphan adoption
copies fail snapshot admission until explicit reconciliation commits proof.

Scope update/clip caps are cumulative across workers. Explicit new-owned resume
pruning permission propagates to storage projections and retention. No original
epoch1/2 or old artifacts pruned. Without permission, conservative remaining-data
projection may block long execution; do not bypass50GiB backing reserve.

## Failure/recovery and accounting

Parent is sole writer of original budget and same-run authority. Intention →
budget/attempt → head/commit journal records new costs and counters. Torn/orphan
transaction blocks normal work until reviewed dead-owner reconciliation. Initial
497.5533489369991s never resets or gets counted twice. Failed progress high-water
and durable replay recorded; backwards clock preserves budget and fails.

First child/device error stops sequencing. CPU-only logging/hold/accounting
failures attach notes rather than replacing primary exception. No post-fault
CUDA RNG/memory/sync/empty_cache/tensor-transfer/checkpoint save. Keep partial
artifacts. Worker failure progress preserves newest actually saved durable step,
not stale entry checkpoint. No next child, automatic recovery or hold clearing.

## Verification and limitations

- Initial68 new safe-runtime CPU tests;168 combined S² safe/helper/plain tests
  passed. Pre-preparation storage-scope regression raised combined total to169.
  Later approved dev-only storage regression raised combined total to170.
- Original main S² runtime/helper baseline50 passed; extractor/package7 passed
  in its separate existing interpreter.
- Ruff check/format clean; Pyright0errors/0warnings with existing main interpreter.
- Main full CPU suite1625passed,5knownfail,4skip. Failures unchanged:
  `test_e3_heldout_cli.py::test_e3_heldout_run_writes_once_with_checkpoint_provenance`;
  `test_pilot_heldout_cli.py::{test_run_writes_ordered_paired_reports_once,
  test_pair_rejects_reference_drift_without_summary,
  test_resume_keeps_verified_e1_score_and_only_scores_e2}` require CUDA;
  `test_new_motion_fusion.py::test_final_summary_revalidates_full_graph_even_with_unchanged_marker`
  has old `scripts/run_fusion_study.py` source-key fixture failure. No full-suite
  pass claimed; no unrelated code/test changes or CUDA enabling.
- Actual CPU subprocess dev2/train/dev4/train/dev6 lifecycle, exact resumed
  weights/AdamW/sampler/prompt RNG/guard history, provisional/adopted accounting,
  fault/observer/partial transaction/owner/scope/report cases verified.
- Native self-review, no independent subagent review; no commits/dependency changes.

CPU tests alone do not validate native GPU behavior. Separately approved native
handoff now validates one restored VT-warmup update and cached dev after actual
exit. Driver/IRQ root cause, joint-CE and long/repeated-epoch stability remain
unproven. Complete production epoch2 safe dev at original1774 is now admitted.
The separately approved1774→3548 VT-only training/checkpoint/exit segment also
passed CPU durable-state audit; epoch4 safe dev519 at3548 also admitted. Next
pending stage is training5322→7096, after epoch6 full safe dev admission and new
best selection5322. This stage now runs under admitted continuous scope with
explicit rolling latest2 retention and storage reserve checks. Future train/dev
stages sequence automatically until configured early stop or200; no duplicate
launches/source edits while live. Earlier diagnostic and provisional1775
remain distinct historical evidence. Continuous Seed0 training is active.

## Native execution rulings and costs

1. No commits/subagents/ledger deletion: user constraints override skill defaults.
   Cost: author self-review and uncommitted audit rather than independent review.
2. Runtime closure freezes on separate explicit migration preparation after code
   verification. Cost: later code changes require deliberate re-admission, not
   silent receipt replacement. Native directory was prepared in the later approved
   bounded handoff, after the storage correction and CPU checks.
3. Intention/head accounting and adoption certificates fail closed on torn writes.
   Cost: partial state may require manual reconciliation; no automatic retry/reset.
4. Test child harness uses real CPU T5/AdamW/sampler/data with a small update kernel.
   Production imports original math unchanged. Cost: native CUDA math/RNG still
   require separately reviewed handoff validation; CPU success is not GPU proof.
5. Existing global Pyright with explicit main interpreter replaces unavailable
   `.venv/bin/pyright`. Cost: another machine may need its installed tool path
   adjusted; runtime/environment bytes are unchanged.

No deferred implementation defect was identified by final native pass.
