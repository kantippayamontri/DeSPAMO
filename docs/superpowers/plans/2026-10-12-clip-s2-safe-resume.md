# Retained CLIP+S² Safe Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan natively, task-by-task. Steps use checkbox (`- [ ]`)
> syntax. User preference is native implementation/self-review; no subagents,
> commits, pushes or workspace deletion without a new explicit request.

**Goal:** Prepare a CPU-verified same-root S² recovery runtime that preserves the
original1774-update checkpoint/accounting and isolates cached dev evaluation.

**Architecture:** Add `tools/clip_s2_safe/` alongside existing runtimes, composing
unchanged original2048-input model/data/update/snapshot primitives with cached
generation and sequential workers. Preserve original snapshot identity; immutable
runtime sidecars bind new source/evaluation policy. Original run budget remains
authoritative; parent waits for actual train-child exit before dev-child creation.

**Tech Stack:** Existing Python3.11/main `.venv`, torch2.0.1+cu117,
Transformers4.32.0, PEFT0.7.1, NumPy, pytest, Ruff and installed Pyright.

**Spec:** `docs/superpowers/specs/2026-10-12-clip-s2-safe-resume-design.md`, approved
by user. User subsequently approved native CPU implementation. Completion evidence
is in `docs/clip-s2-safe-resume-implementation-2026-10-12.json` and this plan's
ledger; original task checklists are retained as executable plan text. Production
preparation and GPU/training scopes remain separately gated.

## Global Constraints

- Original epoch1/2 files and old identity/report/controller artifacts are unchanged.
- The legacy study manual-stop remains present so the old runner remains blocked.
- No production directory is created during CPU implementation/testing.
- Retain exact original checkpoint identity for tensor compatibility.
- Every subsequently produced snapshot needs both original strict receipt verification and the matching runtime sidecar; source/config drift fails.
- Training cacheFalse and gradient checkpointing remain unchanged; dev batch1/cacheTrue/beam5/max64.
- Current integration targets retained seed0 only; do not initialize Seed1 or score test.
- Original budget stays authoritative: initial497.5533489369991s and1774updates are not copied to a new zero-cost budget.
- CPU implementation leaves original budget/hold/checkpoint bytes unchanged; production budget updates happen only inside explicit GPU-execution scope.
- Keep old study stop unchanged and add runtime-local fail-stop gate; no automatic hold release or retry.
- Check native and Windows backing space: projected remaining demand plus50GiB reserve.
- Never delete original epoch1/2 or previous study artifacts; pruning newly owned resumes needs explicit permission.
- No new dependencies, commits, subagents, extraction, Windows settings or GPU work during implementation.

## Review Focus

1. Legacy-looking snapshot from new runtime lacks sidecar or binds another
   checkpoint/source: reject before model mutation; only exact epoch2 anchor is
   exempt (Task2).
2. Logging/accounting/hold write fails after device fault: propagate same primary
   exception; no CUDA cleanup, checkpoint save or next child (Tasks3–6).
3. Train child writes handoff but remains alive, or owner PID/boot is reused:
   actual exit/lock barrier prevents overlapping models and duplicate recovery
   (Tasks1/6/7).
4. CE guard history, sampler or RNG resets across fresh workers: exact CPU
   continuous-vs-resumed comparison and guard-sidecar binding reject drift
   (Tasks2/5/7).
5. Diagnostic/partial/mixed-policy report or validation checkpoint silently
   becomes production progress/selection: explicit mode/coverage/promotion checks
   and authoritative accounting distinguish each (Tasks3/4/6).

---

## Workspace and verification convention

Main project (`PROJECT`): `/home/kan/Research/DeSpaMo`.

Existing isolated runtime worktree (`RUNTIME`):
`/home/kan/Research/DeSpaMo/.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`.

Verify existing isolation with using-git-worktrees. Do not create a bare worktree
from HEAD: required S²/plain/helper files include retained uncommitted work. New
implementation is confined to `RUNTIME/tools/clip_s2_safe/`; docs live in main.
Own plan ledger: `.superpowers/sdd/2026-10-12-clip-s2-safe-resume/progress.md`;
preserve other plans' ledgers/workspaces.

All Python tests below run from `RUNTIME` with:

```text
CUDA_VISIBLE_DEVICES=""
PYTHONPATH=/home/kan/Research/DeSpaMo/src:.:/home/kan/Research/DeSpaMo
HF_HUB_OFFLINE=1
OMP_NUM_THREADS=2
PYTHONDONTWRITEBYTECODE=1
Python=/home/kan/Research/DeSpaMo/.venv/bin/python
```

Tests use temporary file-backed fixtures and tiny real models. Install CUDA-call
sentinels and stub inherited is_available to false in CPU fixtures. Do not run
old controller, instantiate FLAN-XL, mutate native state or restore CUDA RNG to
obtain a green CPU suite. Global Pyright is already installed; use it with
`--pythonpath /home/kan/Research/DeSpaMo/.venv/bin/python`.

### Immutable pins

- Original study/root, run name and namespace: copy exact paths from approved spec.
- Anchor payload SHA: `d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9`.
- Anchor receipt SHA: `b0b3d321a51472d93214c3d7c5ac424063e463d3441b160e1f005b894abcefdf`.
- Initial original budget SHA: `9e2a3bf8d122f1b91947700b3618418b8c8289db6bc91f1ff889e30ff1b5ddf7`.
- Initial original hold SHA: `9b0f3dfebfbb67cbd7e4644ca6d3f67aa6c75dc34214adfc83b209023faeb370`.
- Frozen-base fingerprint: `06298d81fbc2301b0cc53d7211931ab5a7e2685934f8484db45a0312ad567684`.
- Native full-dev diagnostic report SHA:
  `1b29826ade1c066f21ddf7e93b4cba01b78338804f9c6a6159c10933ea7db066`.
- Current original source closure:84files, verified against original receipt.

Capture implementation-entry hashes again. If they differ, investigate current
user work; do not update pins just to make verification succeed.

### New file map (paths relative to RUNTIME)

| File | Responsibility |
| --- | --- |
| `tools/clip_s2_safe/policy.py`, `config.json` | Fixed S²/seed0/anchor/recipe definitions and scoped authorization |
| `tools/clip_s2_safe/state.py` | Strict snapshot overlays, runtime sidecars, immutable migration and CE guard state |
| `tools/clip_s2_safe/accounting.py` | Original-budget writer, append-only attempt costs/replay and diagnostic adoption |
| `tools/clip_s2_safe/evaluate.py` | Optimizer-free cached dev worker and strict report admission |
| `tools/clip_s2_safe/train.py` | Original-state restore/update chunk, checkpoint and handoff |
| `tools/clip_s2_safe/study.py` | GPU-free parent, real exit barrier, safe selection/stopping/holds |
| `tools/clip_s2_safe/tests/` | Temporary snapshots/data, real CPU worker harness and fault cases |
| `tools/clip_s2_safe/pyrightconfig.json` | Target new runtime only with existing import overlay |
| `docs/clip-s2-safe-resume-runbook.md` | Operator handoff, unresolved GPU limits and evidence |

Do not import plain variant's loader/model/state/selection/policy gates: those
require a fresh1024 identity/budget. Read-only reuse of its dimension-agnostic
generation function and memory telemetry is intended. Its `clip-plain` core tag
is implementation metadata; new reports use explicit S² evaluation identity.

### Cross-task schemas

**Binding:** schema1, original identity digest, anchor payload/receipt and initial
budget/hold hashes, original data/base hashes and runtime versions, new source/
config closure, evaluation policy hash, S² variant, schema versions.

**GuardState:** schema1, checkpoint hash, binding hash, last_update, eligible
CE observations (at most200), SHA. Anchor1774 has empty history; future state
must come from matching sidecar, never inferred/reset silently.

**Job:** schema1, mode`production|handoff_probe`, stage`train|dev`, seed0, binding
hash, source/config hashes, original run/runtime roots, input checkpoint+receipt/
sidecar hashes, start_step, target_step, optional epoch/max_batches, parent owner,
claimed scope token, job/result/progress paths. No arbitrary shell-command field.

**Approval:** explicit external reviewed receipt with purpose/mode, binding/
anchor/hold/driver-review hashes, current boot, allowed stages, total update and
clip caps, unique token, expiry, optional owned-pruning/promotion permission.
All caps apply cumulatively to the attempt, not once per newly spawned worker.

**Report:** original identity/checkpoint hashes + binding/evaluation/data/telemetry
hashes, declared mode, exact ordered IDs, references/exemplar, token evidence,
finite metrics only for complete production dev, explicit S² decode policy.

## Task 1: S² contracts, source closure and authorization

**Files:** Create `tools/clip_s2_safe/{__init__,policy}.py`, `config.json`,
`pyrightconfig.json`, `tests/{__init__,conftest,helpers,test_policy}.py`.

**Interfaces:**
- `load_config(path: Path) -> dict`
- `evaluation_policy() -> dict` with ID`clip-s2-cached-b1-beam5-v1`
- `source_hashes() -> dict[str, str]`
- `verify_original(config: dict) -> dict` returns original identity/receipts/budget
- `require_scope(config: dict, stage: str, approval: Path | None, *, job: dict | None = None) -> dict`
- `claim_scope(config: dict, approval: Path) -> dict`
- `owner_alive(owner: dict | None) -> bool`

- [ ] **Step1: Write RED contract/gate tests.** Assert exact2048-input S², seed0,
  original recipe and cached1/5/64 decode. Test wrong policy/width/seed/root,
  source/base/data/runtime/pins or old diagnostic-as-training receipt rejected.
  CUDA/NVML/model-construction sentinels remain untouched for absent/expired/
  consumed/wrong-boot/wrong-hold/mismatched-mode approvals. Parent ticket file,
  PID/startticks/boot and stage must match; same token cannot claim twice.
  Real current process, old-boot PID and same-boot reused PID cases are distinct.
- [ ] **Step2: Run RED:**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_s2_safe/tests/test_policy.py -q`
  Expected: missing new contract functions, with no device/production writes.
- [ ] **Step3: Implement strict config/gates.** Reuse original data/config/recipe
  verification. Bind new own files, original84dependencies and shared-generation
  dependencies; exclude tests/docs/outputs. Original stop remains untouched.
  Runtime-local hold blocks launches; no routine emits an approved GPU receipt.
  Fake paths/counts are fixture overrides, not production alternatives.
- [ ] **Step4: Run GREEN:** same Step2 command; all failure cases pass.
- [ ] **Step5: Record results/baseline hashes in own ledger; inspect diff.** No commit.

## Task 2: Immutable migration, original overlays and guard sidecars

**Files:** Create `tools/clip_s2_safe/state.py`, `tests/test_state.py`.

**Interfaces:** Consumes Task1; produces:
- `make_binding(config: dict, original: dict) -> dict`
- `publish_migration(config: dict, binding: dict) -> dict` (only explicit preparation)
- `load_weights(path: Path, model, binding: dict, *, base_hash: str) -> dict`
- `restore_training(state: dict, model, optimizer, sampler, *, device: torch.device, scope: dict | None = None) -> None`
- `validate_guard_state(state: dict, *, checkpoint_sha256: str, binding: dict, step: int) -> dict`
- `publish_checkpoint(config: dict, snapshot: dict, guard: dict, *, mode: str) -> dict`
- `archive_candidate(config: dict, candidate: Path, binding: dict) -> dict`
- `prune_owned_resumes(config: dict, binding: dict, *, allowed: bool) -> list[str]`

- [ ] **Step1: Add RED snapshot tests.** Only exact allowed anchor loads without
  sidecar. Plain1024, other original checkpoint, malformed tensor shapes/dtypes/
  NaN/buffers, missing/mutated runtime sidecar and wrong binding fail before
  model mutation. Use tiny real AdamW and original sampler for full restoration:
  weights,buffers,groups,moments,order/cursor,RNG,prompt RNG,absolute step all agree.
  Evaluation weights load never calls optimizer/global/CUDA RNG restore. Native
  CUDA RNG bytes are checked as CPU data only. CE history above4000 survives
  checkpoint boundaries; stale guard SHA/checkpoint/last_update fails. Test
  atomic/fsynced snapshot+receipt+binding publication, partial files preserved,
  original epoch1/2 undeletable, retention only on permitted new-owned resumes.
- [ ] **Step2: Run RED:**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_s2_safe/tests/test_state.py -q`
  Expected: missing state adapter, not mocked legacy identity acceptance.
- [ ] **Step3: Implement original-compatible adapters.** Verify original schema1
  payload/receipt and every runtime sidecar before calling `load_snapshot` with
  original identity and replayFalse. Never substitute new binding for original
  tensor identity. Restore full training state only on a clean initialized model;
  synthetic CPU snapshots require empty CUDA RNG list, native audit checks bytes
  without restoring them. New snapshots stay inside safe-runtime namespace,
  still carry original identity, plus separate guard/runtime receipts. Migration
  copies initial raw budget evidence once, not a replacement budget. New-owned
  retention follows explicit permission/publication/archive verification.
- [ ] **Step4: Run GREEN:** same Step2 command; original fixture bytes unchanged.
- [ ] **Step5: Record restoration/sidecar evidence and any ruling.** No native preparation.

## Task 3: Original-budget accounting and bounded-probe adoption

**Files:** Create `tools/clip_s2_safe/accounting.py`, `tests/test_accounting.py`.

**Interfaces:** Consumes Tasks1/2; produces:
- `begin_attempt(config: dict, binding: dict, scope: dict, owner: dict) -> dict`
- `begin_stage(config: dict, attempt: dict, job: dict) -> dict`
- `record_progress(config: dict, attempt: dict, job: dict, progress: dict) -> dict`
- `finish_stage(config: dict, attempt: dict, job: dict, result: dict, *, status: str) -> dict`
- `reconcile_dead_attempt(config: dict, binding: dict, *, reviewed: bool) -> dict`
- `adopt_probe(config: dict, binding: dict, probe: dict, approval: Path | None) -> dict`
- `require_storage(config: dict, binding: dict, job: dict) -> dict`

- [ ] **Step1: Write RED monotonic/idempotent tests.** Begin at1774updates and
 497.5533489369991s; fake10s training +3s dev adds13 exactly once, never a new zero
  budget/profile charge. Repeat reconciliation cannot double charge. Negative
  wall time preserves budget and rejects. Dead-boot gaps charge documented
  unreconciled elapsed time; replay preserves failed update high-water and floor.
  ProbeN updates leave production1774 unchanged, record executed diagnostic
  start/end/N/cost, and cannot be adopted absent new explicit promotion receipt.
  Adoption advances counter once and does not charge probe again; production
  epoch2-dev prerequisite still required. Space below projected demand+50GiB
  blocks before spawn/publication. Parent alone writes authoritative budget.
- [ ] **Step2: Run RED:**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_s2_safe/tests/test_accounting.py -q`
  Expected: missing new accounting functions.
- [ ] **Step3: Implement original ledger adapter.** Preserve budget identity and
  inherited charges; add runtime-binding/journal reference. Parent holds same-run
  writer ownership; workers publish separate progress. Journal intentions before
  work and binding/after-budget hashes after atomic publication. Recover a torn
  journal/budget update idempotently, with no reset. Diagnostic probe updates are
  provisional, separately retained; only explicit adoption enters production
  update counter. Stage caps apply across attempt. Production elapsed charges
  include setup/failed stages; native/B: projection includes retention/staging.
- [ ] **Step4: Run GREEN:** same Step2 command, including observer write failure
  preserving primary error and accounting evidence rather than silent reset.
- [ ] **Step5: Record ledger; actual original budget remains untouched.**

## Task 4: S² optimizer-free dev and strict selection report contract

**Files:** Create `tools/clip_s2_safe/evaluate.py`, `tests/test_evaluate.py`.

**Interfaces:** Consumes Tasks1/2; original loader,PromptPool,shared generation;
produces:
- `score_dev(model, admission: dict, pool, job: dict, binding: dict, *, on_event, scope: dict | None = None) -> dict`
- `validate_report(report: dict, job: dict, binding: dict, admission: dict) -> None`
- `run_dev_job(config: dict, job: dict, approval: Path | None) -> dict`
- CLI `--config <path> --job <path> [--approval <path>] [--check-only]`.

- [ ] **Step1: Add RED tiny-T5/PEFT + file-backed2048 tests.** Actual adapter uses
  both1024 halves; cached decoder sees1 token/5 beam rows, finite logits, training
  cacheFalse preserved. Assert batch1 per-batch autocast, exact reference/order,
  original exemplar and output-token evidence. Full production report may score;
  bounds1/equal-to-corpus/greater-than-corpus and handoff_probe mode are diagnostic
  with null metrics and never select. Reject raw earlier diagnostic, missing/
  reordered/duplicate IDs, changed checkpoint/binding/policy/references/exemplar/
  telemetry/metrics. Disk-full before generation prevents compute; primary CUDA
  fault survives observer failure with zero CUDA cleanup and no retry.
- [ ] **Step2: Run RED:**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_s2_safe/tests/test_evaluate.py -q`
  Expected: missing S² worker/report validator.
- [ ] **Step3: Implement isolated evaluator.** Gate before device/query/model;
  construct original model and strict weights-only overlay, never optimizer/RNG
  restoration. Use original admitted full2048 data, cached shared core, fresh
  per-batch autocast. Preserve core tag separately from explicit S² policy; don't
  mutate plain global constants or weaken plain validator. Fsync batch/decoder
  evidence before compute; final report only after exact coverage and finite
  metrics. Check original post-warmup identical-output guard as fail-stop. All
  reports live in safe namespace; prior519 diagnostic remains evidence only.
- [ ] **Step4: Run GREEN:** same Step2 command and existing57 S² CPU checks with
  their correct main/extractor interpreters; no dependency mixing or GPU tests.
- [ ] **Step5: Record verified inference/selection boundaries in ledger.**

## Task 5: Original training restore, chunks and durable handoff

**Files:** Create `tools/clip_s2_safe/train.py`, `tests/test_train.py`.

**Interfaces:** Consumes Tasks1–3 and original update/sampler; produces:
- `next_boundary(step: int, recipe: dict) -> int`
- `train_chunk(model, optimizer, sampler, admission: dict, pool, binding: dict, guard: dict, job: dict, *, device: torch.device, on_progress) -> dict`
- `run_train_job(config: dict, job: dict, approval: Path | None) -> dict`
- CLI same argument structure as Task4; check-only cannot initialize CUDA.

- [ ] **Step1: Write RED boundary/restore tests.** Anchor1774 schedules dev2 before
  production step1775; admitted dev2 permits training to3548/dev4. Epoch3/2661
  interrupted chunk resumes to3548. No warmup/LR/update reset; compare group_lr at
 1775,4000,4001. Exact actual next sampler batch,prompt,RNG/AdamW and persistent
  buffers match continuous CPU run after fresh worker. Restore CE last200 above
  warmup; corrupted/stale sidecar rejected. No generation inside train worker.
  Inject update failure and ensure no subsequent CUDA calls/capture_state/save;
  healthy declared boundary publishes full snapshot + candidate + handoff.
  Probe respects totalN cap and diagnostic mode; partial/probe resume needs
  explicit adoption instead of legacy modulo887 bypass.
- [ ] **Step2: Run RED:**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_s2_safe/tests/test_train.py -q`
  Expected: missing new train job/chunk.
- [ ] **Step3: Implement new lifecycle around unchanged math.** Use original
  factory,optimizer,param order,update kernel and sampler. Load/validate on CPU;
  transfer model/create optimizer then restore global/prompt/device RNG once.
  For production, require admitted pending dev and next boundary from original
  absolute step. For probe, obey purpose/caps and journal provisional state.
  Carry guard history, reject drift, serialize only on healthy requested stops,
  then exit0 with durable result. No in-process dev, CUDA finally cleanup,
  signal-triggered save-after-fault or old controller call.
- [ ] **Step4: Run GREEN:** same Step2 command. Test-only update kernel is a small
  real differentiable AdamW workload; production imports original kernel without
  test-mode branches or monkeypatches.
- [ ] **Step5: Record handoff state and preserved learning trajectory evidence.**

## Task 6: Same-root supervisor, safe selection and explicit recovery

**Files:** Create `tools/clip_s2_safe/study.py`, `tests/test_study.py`.

**Interfaces:** Consumes Tasks1–5; produces:
- `pending_stage(config: dict, binding: dict, admission: dict) -> dict`
- `execute_job(config: dict, binding: dict, attempt: dict, job: dict, approval: Path) -> dict`
- `select_checkpoint(config: dict, binding: dict, admission: dict) -> dict`
- `run(config: dict, approval: Path | None, *, check_only: bool = False) -> dict`
- CLI `--mode preflight|run --config <path> [--approval <path>]`; no auto watcher.

- [ ] **Step1: Add RED actual CPU subprocess tests.** Train child writes handoff
  and remains alive: evaluator must wait for child exit and locks. Trace pending
  dev2→train→dev4 with separate processes. Death after checkpoint/before handoff,
  report/before parent commit and torn ledger publication recover only under
  explicit review, without duplicate work/charges. Reused PID/old boot/missing
  result/wrong result step all hold. Fake metric/report/probe adoption never
  advances production or selection. Earlier ties, actual-best vs0.10 anchor,
  min40/VT eligibility and epoch-gap guards match original stop_state. Child
  error plus broken log/budget/hold publication preserves primary; no next child.
- [ ] **Step2: Run RED:**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_s2_safe/tests/test_study.py -q`
  Expected: missing supervisor; no GPU/production roots created.
- [ ] **Step3: Implement CPU-only parent.** Verify/claim scope and same-run
  ownership before jobs. Immutable owner-bound tickets use fixed train/evaluate
  entrypoints. Reap actual child then admit durable state/report with caps/hashes;
  files alone never authorize next launch. Parent charges Task3 ledger and
  records safe selections/stopping under runtime only. On first failure consume
  token and runtime-hold, retain legacy hold, stop; no worker restart/watchdog.
  Handle missing safe epoch2 dev first; don't promote previous diagnostics.
  Restrict seed0 continuation and mark single-run result, not whole two-seed
  study completion or paired benchmark. Guarded CPU manual reconciliation is
  explicit and never invokes GPU or clears holds automatically.
- [ ] **Step4: Run GREEN:** same Step2 command and Tasks2–5 suites together.
- [ ] **Step5: Record exact ownership/publication/recovery behavior.** No GPU launch.

## Task 7: End-to-end CPU acceptance, native review and runbook

**Files:** Create `tests/test_lifecycle.py`, test-only CPU child harness,
`docs/clip-s2-safe-resume-runbook.md`, implementation receipt. Update AGENTS only
after verified completion, clearly retaining GPU/training hold.

**Interfaces:** Consumes Tasks1–6; produces evidence and operator handoff, no new
runtime abstraction.

- [ ] **Step1: Write lifecycle tests first.** Synthetic original-like anchor
  and authoritative budget start at retained step/cost,2048 data, real sampler/
  optimizer and tiny T5. Migrate within same root, evaluate in fresh process,
  train chunk and checkpoint, exit, evaluate again, then resume. Exact CPU
  uninterrupted/resumed next batch,prompt,weights,AdamW/RNG/guard-window agree.
  Originals remain byte-identical; no tests/model loader/encoder run. Inject
  checkpoint/report/journal crashes and cumulative-cap boundary cases.
- [ ] **Step2: Run RED:**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_s2_safe/tests/test_lifecycle.py -q`
  Expected: unfinished lifecycle assertion/harness. If already green, record
  verification honestly, not an invented RED cycle; add missing behavior only
  with a meaningful failing regression.
- [ ] **Step3: Run complete focused acceptance.**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_s2_safe/tests tools/clip_s2_inference/tests tools/clip_plain/tests -q`
  Expected: all pass with CUDA-call sentinels. Run original S² runtime and
  extractor tests in correct interpreters, and main targeted visual/temporal/
  FLAN/batch regressions. Configured main `tests/unit -q` is CPU-only; prior
  baseline1625pass/5knownfail/4skip is context, not automatic result. Report exact
  failures; don't enable held-out CUDA scoring or alter unrelated fixtures.
- [ ] **Step4: Static/preservation verification and native review.** Ruff check/
  format new namespace and global Pyright using existing main interpreter.
  Inspect native anchor tensor/AdamW/sampler/RNG schema on CPU only; hash original
  epoch1/2,84sources,budget,hold,diagnostic reports and plain runtime against entry
  baseline. Both worktrees' status/diff checks; verify no native safe-runtime
  directory/approval/GPU process created. Use verification-before-completion and
  native /verify,/review equivalents, fix substantiated regressions and rerun
  affected checks. Self-review is not independent review or GPU stability proof.
- [ ] **Step5: Write handoff and ledger results.** Document migration/sidecar
  namespaces, source/identity boundaries, authoritative/provisional accounting,
  exact commands/results and future execution gates. No commits or GPU work.

## Execution handoff and future scopes

Implementation plan review is next; preserve native execution method. Implement
only code and CPU tests after approval. Existing full-dev approval is consumed,
not permission for model loading, checkpoint adoption or training.

Later execution requires fresh storage/host/owner/source checks and separately
reviewed preparation/GPU receipts. First integration handoff validation declares
exact update/clip bounds and mode; diagnostic results do not select weights or
silently commit progress. Production starts with new-bound epoch2 dev, then
original-step training. Long training or adoption requires its own reviewed scope.

No estimate, unattended retry, driver/audio change, extraction, new experiment,
Seed1 or official test scoring is added by this plan.

## Native plan self-review

- Tasks1/2 cover all pins and dual original/runtime identity, with anchor-only
  sidecar exemption and CE guard restoration; no plain-policy bypass.
- Task3 keeps one authoritative budget and explicitly separates provisional
  updates/promotion, repeated reconciliation and conservative clock recovery.
- Tasks4/6 distinguish complete production reports from diagnostic evidence and
  preserve actual-best/anchor stopping semantics under the new policy.
- Tasks5/6/7 test real optimizer/sampler/process exit, RNG restoration and
  primary-error preservation; no duplicate GPU worker or save after device fault.
- New paths/interfaces and schemas align across tasks; fixture overrides stay
  in tests. Only reviewed CPU verification occurs during implementation.
