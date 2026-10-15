# Original SpaMo Paper-Reproduction Implementation Plan

> **For agentic workers:** Use executing-plans for native implementation. User requested approval and training; preserve ledgers/worktrees, no commits/subagents.

**Goal:** Train original SpaMo under the approved paper recipe and clean dev selection.

**Architecture:** New isolated reproduction namespace composes existing original
linear SpaMo model/data/loss and verified B storage. Native CPU owner/ledger drives
sequential training and fresh cached dev; existing CLIP+S² stays sealed.

**Tech Stack:** Existing main Python3.11/torch2.0.1/Transformers4.32/PEFT0.7.1,
Lightning1.9.5, offline model cache; no installs.

**Spec:** `docs/superpowers/specs/2026-10-13-original-spamo-paper-reproduction-design.md`.

## Global constraints

- Original linear2048→768/1024→768, TCN/connector; fresh FLAN Q/V LoRA16/32/.1.
- Released train/dev arrays7096/519, unchanged annotations; test642unscored in training.
-40epochs/batch8/accum2/seed0, VT4000actual optimizer updates, LRwarm10000,
  peak1e-4/floor5e-5, AdamW.9/.98/eps1e-8/decay.01/clip1.
- Source/asset/config pins, original current ledger preserved, B full-state files,
  C/B50GiB reserves, exact owner/exit barriers, no automatic retry.
- Existing interpreters only; no main/retained-source edits, dependencies, commits,
  extra seeds, old checkpoint adoption or automatic test scoring.

## Review focus

1. Odd last accumulated batch: exactly declared optimizer steps and divisor,
   not an invented effective contrastive negative set.
2. VT update4000→4001: no CE/LoRA update before joint phase, original VT formulation.
3. Restored dropout/prompt/sampler/global RNG: interrupted trajectory matches CPU
   continuous reference; no optimizer survives into dev child.
4. B publication error/low capacity: source and high-water/cost proof retained,
   no next worker or post-device-error CUDA cleanup.
5. Released-feature and reference exposure drift: admission rejects before GPU,
   full dev report metrics/token/clip coverage verified against fixed train-only ICL.

### Task1: Isolated provenance, data and recipe admission

Create `tools/spamo_repro/{policy,data}.py`, `config.json`, tests.
Produces `prepare(config:dict)->dict`, `admit_data(config:dict)->dict`,
`lr_at_update(update:int,total:int)->float`.
- [ ] RED tests for exact recipe, LR0/warm10000/final5e-5, batch/accum distinction,
  feature finiteness/IDs/hashes and forbidden old initialization/test selection.
- [ ] Run targeted main-interpreter CPU tests; expect missing interfaces.
- [ ] Implement minimal pinned source/data/config admission and recorded assumptions.
- [ ] Run tests; require all pass, record provenance/recipe evidence.

### Task2: Original model/update and full B checkpoint lifecycle

Create `tools/spamo_repro/{model,train,state}.py` and focused tests.
Consumes Task1; produces `train_epoch(...)`, `publish_state(...)`, `restore_state(...)`.
- [ ] RED tests for original linear adapter/checkpoint structure, optimizer group LR,
  VT4000/joint4001, accumulation tails, exact sampler/prompt/global RNG restore.
- [ ] Run targeted CPU tests; expect missing interfaces.
- [ ] Compose unchanged original model/loss; implement explicit approved schedule,
  update counters, healthy storage boundaries and immutable B publication.
- [ ] Verify tiny-model continuous/resume equality and primary-error preservation;
  no CUDA during CPU audit, no production update yet.

### Task3: Fresh cached dev, stopping-free40epoch supervisor and accounting

Create `tools/spamo_repro/{evaluate,study,accounting}.py`, tests/runbook.
Consumes Tasks1/2; produces `run(config:dict,approval:Path)->dict` with unique claim.
- [ ] RED real-CPU subprocess tests for train exit before optimizer-free dev, all
  report items/metrics, train-only prompt,20dev boundaries, full-state/cost retention,
  duplicate/unscoped/held/low-space rejection.
- [ ] Run targeted CPU tests; expect missing interfaces.
- [ ] Implement sequential owner/ledger/approval and fresh cached b1/beam5/max64 dev,
  best dev retention; source/data checks before every worker; no time/early stop.
- [ ] Run new plus relevant storage/model/inference CPU regression, Ruff/format and
  scoped typing; native review against all five focus cases and source pins.

### Task4: Bounded real admission then authorized40epoch launch

- [ ] Fresh C/B/host/driver/fault/source/owner checks; current old parent/children
  dead,69186sealed, no other GPU work, demand+50GiB reserves met.
- [ ] Issue unique boot/source/run-bound bounded train/checkpoint/exit/dev scope;
  independently audit state/finite logits/telemetry/actual exits, retain receipts.
- [ ] Admit40epoch production only after successful evidence, preserving any charged
  probe accounting and declared production initialization without silent adoption.
- [ ] Verify actual owned optimizer progress and B checkpoint placement; record
  launch as live, not40epochs complete. End after best dev freezes; test gate separate.

## Self-review and execution

Protocol is user-approved; execute natively with focused TDD. No commit steps
because no new commit authorization. Actual source/parity differences and memory
prerequisites must be established before production claim. No guessed APIs or
silent recipe changes to make launch succeed.
