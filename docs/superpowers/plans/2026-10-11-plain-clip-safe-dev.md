# Single-scale CLIP and Safe Dev Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task, natively in the current session. Steps use
> checkbox (`- [ ]`) syntax. Preserve the user's native implementation/self-review
> preference; do not delegate or commit without a new explicit request.

**Goal:** Implement a provenance-bound single-scale CLIP variant and isolated,
cached, error-safe dev evaluation, verified on CPU before any new GPU execution.

**Architecture:** Derive width1024 global224 CLIP features from the verified first
half of existing S² arrays into a new package. A new `tools/clip_plain/` pipeline
uses a fresh1024-input adapter and alternates training chunks with optimizer-free
evaluation subprocesses, never keeping both on GPU. Source/data/evaluation policy
identities, compact checkpoints and owned accounting survive process boundaries.

**Tech Stack:** Existing Python3.11/main uv environment, torch2.0.1+cu117,
Transformers4.32.0, PEFT0.7.1, NumPy, pytest, Ruff, Pyright; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-11-plain-clip-safe-dev-design.md`.
User approved the design/plan and native CPU implementation. Implementation
verification is recorded in `docs/clip-plain-safe-dev-implementation-2026-10-11.json`
and this plan's ledger. Real package preparation, Windows device changes and GPU
launch remain separately gated. Implementation task steps are retained as the
original executable plan; actual completion evidence lives in the ledger.

## Global Constraints

- Implementation is WSL code plus CPU tests only; production feature conversion, GPU validation and training require separate execution approval.
- Keep v3 manual-stop and automatic-recovery block intact; no Windows device, driver, registry, reboot or package changes.
- Preserve old source, feature, model, checkpoint, budget and test bytes; use only new `tools/clip_plain/` runtime modules and new variant-owned output namespaces.
- Use existing main `.venv`, Python3.11, torch2.0.1+cu117, Transformers4.32.0 and PEFT0.7.1; no dependency/lock changes or environment mixing.
- Frozen CLIP: `openai/clip-vit-large-patch14`, revision `32bd64288804d66eefd0ccbe215aa642df71cc41`; global224 raw final hidden CLS, FP32, width1024, no tiles or normalization.
- Frozen FLAN-T5-XL revision `7d6315df2c2fb742f0f5b556879d730926ca9001`; Q/V LoRA16/32/dropout0.1; prompt limit1024; train-only three-language ICL.
- Official train7096/dev519/test642, original order and source-frame provenance; unchanged motion1024 features; no official test inference/scoring.
- Training batch8/accumulation1/drop-last,887updates/epoch,max200,min40,dev every2epochs,patience10checks,minDelta0.10BLEU4; VT4000/LRwarmup10000,legacy_mean VT.
- Adapter LR3e-4→5e-5; FLAN LR5e-5→1e-5; AdamW betas0.9/0.98,eps1e-8,weight_decay0.01,gradient_clip1; training cacheFalse and gradient checkpointing.
- Dev batch1,beam5,max_length64,do_sampleFalse,use_cacheTrue; fresh optimizer-free inference process, per-batch BF16 autocast and fsynced telemetry.
- Retain old1774updates and497.5533489369991s accounting; plain-CLIP updates start at0 and aggregate historical charges exactly once at study level.
- No automatic GPU retries, automatic hold release, commits, pushes, artifact deletion or expensive production runs during implementation.
- Native and Windows backing storage require projected remaining demand plus50GiB reserve; low space blocks preparation/launch without deleting or relocating old artifacts.

The new run is **not an exact resume of S² epoch2**. This representation-width
change supersedes the earlier same-root S² migration proposal. Preserve old
artifacts; new plain-model initialization and future launch need plan review.

## Review Focus

1. Wrong cache half or source drift silently changes the experiment: pin layout,
   parent identity/receipts and bitwise slice equality (Tasks2/3).
2. GPU failure followed by failing logger/checkpoint cleanup masks the first
   exception or touches CUDA again: inject both faults and prohibit all subsequent
   device operations, with no next worker launched (Tasks5/6/8).
3. Worker writes handoff then remains alive, crashes before final publication,
   or PID is reused: process-exit barrier, atomic state and owner/boot checks
   prevent overlapping GPU models and accidental recovery (Tasks1/4/8).
4. Fresh evaluator perturbs training RNG/optimizer or double-charges repeated
   controller reconciliation: interrupted-versus-continuous tiny training and
   idempotent attempt accounting must agree (Tasks4/7/9).
5. Bounded, reordered, stale-checkpoint or different-policy reports appear
   complete: no selection or subsequent training until exact full-dev admission
   passes (Tasks6/8).

---

## Workspace, commands and file map

Main project: `/home/kan/Research/DeSpaMo` (`PROJECT`).

Existing runtime worktree (`RUNTIME`):
`/home/kan/Research/DeSpaMo/.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`.

Use `using-git-worktrees` at implementation entry to verify this existing isolated
workspace. Its required `tools/clip_s2`, `tools/clip_s2_inference`, benchmark and
followup files include uncommitted work. A bare worktree from HEAD would omit
them. Work in a **new sibling namespace** here; do not edit/move sibling ledgers
or recreate their work. New plan ledger belongs to this plan's separate ignored
SDD workspace and records baseline hashes and task results.

All new runtime/test paths below are relative to `RUNTIME`. `docs/...` paths are
relative to `PROJECT`. Put runbook/spec/plan in main project. No old runtime file
is an intended edit. Baseline main `src/` differences are user work.

Standard CPU environment for **every** Python test command below:

```text
cwd=/home/kan/Research/DeSpaMo/.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree
CUDA_VISIBLE_DEVICES=""
PYTHONPATH=/home/kan/Research/DeSpaMo/src:.:/home/kan/Research/DeSpaMo
HF_HUB_OFFLINE=1
OMP_NUM_THREADS=2
PYTHONDONTWRITEBYTECODE=1
Python=/home/kan/Research/DeSpaMo/.venv/bin/python
```

Commands show the exact interpreter; supply the environment through the tool
invocation, not a persistent environment/lock change. All tests use temporary
synthetic fixtures. No real dataset conversion or FLAN-XL generation in tests.
CPU fixtures stub `torch.cuda.is_available` to false where inherited snapshot
helpers consult it; other CUDA/NVML operations are sentinels. This avoids even
device discovery during fault tests. Do not change global environment files.

| New file | Responsibility |
| --- | --- |
| `tools/clip_plain/policy.py`, `config.json` | Strict variant/config contracts, source hashes and CPU-only authorization/ownership checks |
| `tools/clip_plain/features.py` | Verified CPU global-half derivation, receipts and complete-package publication |
| `tools/clip_plain/data.py` |1024-wide package admission and synchronous batches |
| `tools/clip_plain/model.py` | Fresh plain variant with fixed training recipe |
| `tools/clip_plain/state.py` | Run identity, checkpoint/lineage contracts, archive/retention and stage accounting |
| `tools/clip_plain/generation.py` | Explicit caching, raw-logit checks, decoder telemetry and primary-error preservation |
| `tools/clip_plain/evaluate.py` | Inference-only worker and complete/bounded dev report contract |
| `tools/clip_plain/train.py` | Train-to-next-dev-boundary worker, full checkpoint handoff |
| `tools/clip_plain/study.py` | GPU-free process sequencing, selection, stopping and holds |
| `tools/clip_plain/tests/{helpers,conftest}.py` | Small file-backed data, tiny models, fault/clock/process fixtures |
| `tools/clip_plain/pyrightconfig.json` | New-runtime-only typing with existing source import overlay |
| `docs/clip-plain-safe-dev-runbook.md` | Explicit preparation/validation/recovery handoff |

Reuse read-only pure primitives from existing modules: `FeatureManifest`,
`Phoenix14T`, `collate_phoenix`, temporal/visual classes, `FlanT5Backbone`,
`SpaMoBaselineModule`, `VTAlignLoss`, `PromptPool`, `schedule`, `group_lr`,
`stop_state`, `optimizer_for`, `update`, compact snapshot validation and
atomic/fsynced storage. Never call old controller/run/selection functions with
their absolute roots. Do not reuse old `score_dev` or monkeypatch a loader/gate.
Reuse pure frozen-base/CE/identical-dev guards from `tools/clip_s2/guards.py` as
fail-stop checks only; their exception type must never authorize automatic retry.

Common dict schemas (JSON-safe at durable boundaries):

- **Config:** explicit parent spatial root/hashes, output feature base, motion
  root/hash, annotations, model cache, native study/archive roots, seeds`[0,1]`,
  arm`reference`, recipe, evaluation policy and `test_scoring=false`.
- **Admission:** `roots`, `datasets`, `ids`, `lookup`, `feature_hashes`, `bindings`,
  `identity`, matching the explicit-root CLIP loader's structure, with width1024.
- **Run identity:** policy,seed,arm,recipe,data,code/config/base hashes,
  initialization/frozen fingerprints,runtime,decoder,evaluation-policy hash.
- **Job:** schema_version,stage,seed,epoch,run_root,checkpoint and receipt hashes,
  run/evaluation identity hashes,parent owner/attempt token,optional max_batches.
- **StageResult:** status,stage,owner,completed_updates,durable_updates,
  checkpoint/report paths and hashes; no tensors. A file never proves child exit.
- **Budget:** identity,status,phase,owner,completed_updates,durable_updates,
  charged_seconds,attempt ledger and `duration_limit_seconds=null`.
- **Approval:** explicit reviewed purpose, current boot/driver-review binding,
  existing v3 hold hash, new config/source/experiment identity, authorized seeds/
  stages/limits, unique token and optional owned-resume-pruning permission.

### Task 1: Define variant policies and fail-closed execution boundaries

**Files:** Create `tools/clip_plain/{__init__,policy}.py`, `config.json`,
`tests/{__init__,helpers,conftest,test_policy}.py`, `pyrightconfig.json`.

**Interfaces:** Produces:
- `load_config(path: Path) -> dict`
- `source_hashes() -> dict[str, str]`
- `evaluation_policy() -> dict` (ID `clip-plain-cached-b1-beam5-v1` plus exact decode/telemetry contract)
- `require_gpu_scope(config: dict, stage: str, approval: Path | None, *, job: dict | None = None) -> dict`
- `claim_attempt(config: dict, approval: Path) -> dict`
- `owner_alive(owner: dict | None) -> bool` using exact PID/boot/start-ticks/command.

- [ ] **Step 1: Pin policies and authorization failures with CPU tests.**
  `test_plain_policy_contract` asserts width1024,scales`[224]`,raw CLS,train8,
  dev1/cacheTrue/beam5/max64,seeds`[0,1]`,testFalse and explicit parent hashes from
  spec. `test_gpu_gate_rejects_before_any_device_call` replaces CUDA/NVML hooks
  with throwing sentinels; missing/expired/consumed/wrong-boot/wrong-hold-hash/
  wrong-stage approvals all reject without touching them. Test separate runs
  cannot reuse a claimed token and v3 stop bytes never change. Pin old-boot dead
  owner versus same-boot reused PID behavior. Helper config uses tmp paths via
  explicit monkeypatches of fixed root constants, never production paths.
- [ ] **Step 2: Run RED.**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_plain/tests/test_policy.py -q`
  Expected: missing policy module/functions, not CUDA/model/network access.
- [ ] **Step 3: Implement strict contracts.** Config contains exact native/archive
  identifiers from spec; defaults deny GPU and test scoring. Source closure covers
  new runtime `.py` files plus imported baseline dependencies, main `src/`,
  pyproject/uv lock; exclude docs/tests/outputs. Compare immutable pins, never
  update them to admit drift. Approval is read/claimed, never invented. Add typing
  config with runtime/source import overlay; no environment installation.
- [ ] **Step 4: Run GREEN** using Step2 command; all sentinel/ownership cases pass.
- [ ] **Step 5: Record diff and verification in this plan's ledger.** No commit.

### Task 2: Derive independent global224 cache on CPU

**Files:** Create `tools/clip_plain/features.py`, `tests/test_features.py`.

**Interfaces:** Consumes Task1 config/source policy; produces:
- `global_columns(values: np.ndarray) -> np.ndarray`
- `parent_inventory(config: dict) -> dict` in generic feature-package inventory format
- `feature_identity(config: dict) -> dict`
- `derive_clip(config: dict, split: str, clip_id: str) -> dict`
- `derive_package(config: dict, *, check_only: bool = False) -> dict`
- `verify_derived_package(config: dict) -> dict`
- CLI `--mode check|derive|verify --config <path>`; `derive` additionally requires
  `--authorize-feature-preparation`. `check` creates no production files.

- [ ] **Step 1: Write RED tests for real file-backed conversion.**
  ```python
  def test_global_half_is_exact_and_independent():
      parent = np.arange(3 * 2048, dtype=np.float32).reshape(3, 2048)
      before = parent.copy()
      child = global_columns(parent)
      assert child.shape == (3, 1024) and child.dtype == np.float32
      np.testing.assert_array_equal(child, before[:, :1024])
      assert child.flags.c_contiguous and not np.shares_memory(child, parent)
      parent[:, 1024:] = -999
      np.testing.assert_array_equal(child, before[:, :1024])
  ```
  Add `test_wrong_parent_layout_width_dtype_or_nan_rejected`,
  `test_parent_receipt_or_source_drift_prevents_complete`,
  `test_interrupted_unreceipted_file_preserved`,
  `test_verified_resume_is_idempotent`, `test_complete_inventory_no_extra_ids`,
  and `test_storage_below_projection_plus_50gib_blocks_before_write`.
  CUDA/CLIP model-loader sentinels must never run.
- [ ] **Step 2: Run RED.**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_plain/tests/test_features.py -q`
  Expected: missing conversion module/functions.
- [ ] **Step 3: Implement slice/provenance/publication.** Read parent manifest and
  complete frame rows in official order, verify producer encoder hash/layout and
  pinned model/processor. Use existing generic feature-package width-aware
  validators/writers with child policy; store parent array/receipt hashes in
  child `extra` provenance. Hash-check parent again after reading. Publish child
  complete files and package-result pointer only after full validation. Raw data
  estimate3.6154022216796875GiB plus calculated headers/receipts/staging is a
  preflight input, not permission to allocate. No Pillow/encoder imports.
- [ ] **Step 4: Run GREEN** using Step2 command. Verify parents byte-identical and
  tests wrote exclusively to tmp directories.
- [ ] **Step 5: Record ledger/check diff.** Real8257-clip conversion remains unrun.

### Task 3: Admit1024 features and build a fresh plain model

**Files:** Create `tools/clip_plain/{data,model}.py`, `tests/{test_data,test_model}.py`.

**Interfaces:** Consumes Tasks1/2; produces:
- `admit_data(config: dict, *, verify_arrays: bool = True) -> dict`
- `batch_for(admission: dict, split: str, clip_ids: tuple[str, ...]) -> PhoenixBatch`
- `eval_batches(admission: dict, *, max_batches: int | None = None) -> Iterator[PhoenixBatch]` (dev only, batch1)
- `build_model(seed: int, cache: str) -> tuple[SpaMoBaselineModule, str]`
  returning model and frozen-base fingerprint; builds on CPU.

- [ ] **Step 1: Add tests.** `test_only_plain_complete_package_admitted` rejects
  width2048,wrongpolicy,partial/reordered/extra IDs and hash drift. Small fixture
  patches official count constants while retaining split/order/source joins;
  production counts remain7096/519/642. `test_same_motion_and_frame_order` checks
  cached motion bytes unchanged. `test_bounded_loader_does_not_read_next_clip`
  uses a read sentinel. `test_model_spatial_input_1024_and_language_output_2048`
  asserts projector first weight`(1024,1024)` and actual adapter output`[...,2048]`.
  `test_fresh_model_reproducible_per_seed` compares two fresh initializations;
  no S² warm-start or claimed equality with old initialization. Mock base factory,
  but use actual adapter/temporal code. Test train cacheFalse,checkpointing enabled,
  LoRA16/32/0.1,maxprompt1024,VT4000,legacy_mean; forbid CUDA initialization.
- [ ] **Step 2: Run RED.**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_plain/tests/test_data.py tools/clip_plain/tests/test_model.py -q`
  Expected: missing new loader/factory.
- [ ] **Step 3: Implement explicit-root loader and fresh factory.** Build reference
  visual adapter with spatial1024,motion1024,hidden768,language2048 and plain MLP
  1024→1024→768. Reuse stable FLAN/VT/baseline classes and approved training setup;
  `fork_rng(devices=[])` where needed prevents discovery of CUDA during CPU model
  construction. Do not import old run functions, copy old projector weights or
  duplicate features to keep width2048.
- [ ] **Step 4: Run GREEN** using Step2 command; no pretrained model download.
- [ ] **Step 5: Record initialization/admission decisions in ledger.**

### Task 4: Bind plain checkpoints, lineage, storage and attempt accounting

**Files:** Create `tools/clip_plain/state.py`, `tests/test_state.py`.

**Interfaces:** Consumes Task1 policies and Task3 model/admission; produces:
- `make_identity(config: dict, admission: dict, seed: int, model: nn.Module, base_hash: str) -> dict`
- `ensure_lineage(config: dict) -> dict`
- `load_weights(path: Path, model: nn.Module, identity: dict, base_hash: str) -> dict`
  (CPU snapshot verification/overlay; no optimizer/global CUDA RNG restore)
- `restore_training(state: dict, model: nn.Module, optimizer, sampler) -> None`
  (successful fresh train initialization only)
- `publish_epoch(run_root: Path, state: dict, epoch: int, *, dev_candidate: bool) -> dict`
- `archive_candidate(config: dict, candidate: Path) -> dict`
- `prune_owned_resumes(run_root: Path, identity: dict, *, allowed: bool) -> list[str]`
- `begin_stage(run_root: Path, identity: dict, attempt: dict) -> dict`
- `record_stage(run_root: Path, attempt: dict, progress: dict, *, final_status: str | None = None) -> dict`
- `reconcile_abandoned_stage(run_root: Path, *, reviewed: bool) -> dict`
- `cost_summary(config: dict) -> dict`.

- [ ] **Step 1: Add CPU snapshot/accounting failure tests.**
  `test_s2_epoch2_checkpoint_rejected_without_mutation` uses a synthetic old-policy
  receipt/state, not a production checkpoint write, and asserts old policy/2048
  weight rejected before optimizer/RNG restore. `test_plain_roundtrip_restores_all`
  compares trainable weights,persistent buffers,AdamW moments,param groups,
  sampler,prompt RNG and CPU RNG. Patch CUDA RNG for CPU-only restore tests.
  `test_evaluator_weights_load_never_restores_optimizer_or_cuda_rng` uses sentinels.
  `test_phase_reconciliation_charges_once` injects fake clock durations10+3+7s:
  historical497.5533489369991 plus20 new seconds, even after repeated reconciliation.
  `test_abandoned_stage_needs_review_and_keeps_elapsed_floor` verifies conservative
  wall-time gap accounting and documented durable replay without zeroing charges.
  `test_backwards_wall_clock_preserves_budget_and_blocks_reconciliation` rejects
  negative elapsed time and retains the prior charged floor and evidence.
  Test atomic publication, parent-budget drift, two native resumes, all archived
  dev candidates, and refusal to prune symlinks/foreign identities/old S² paths.
- [ ] **Step 2: Run RED.**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_plain/tests/test_state.py -q`
  Expected: missing new state functions.
- [ ] **Step 3: Implement owned state.** Reuse compact validation with `replay=False`
  for CPU weights loading; validate new policy/schema/shape first. Restore full
  train state only after clean model/device/optimizer construction, once. Parent
  controller exclusively owns budget and append-only attempt journal; workers
  publish separate owned progress/handoff files. Account whole child lifecycle
  including failed stages, and explicit wall-time uncertainty after a dead boot.
  New study lineage references untouched old root/checkpoint/budget hashes and
  cost once. Atomic/fsynced publication precedes retention; pruning defaults off
  until explicitly allowed for new owned resumes. Estimate remaining native/B:
  demand from measured new compact sizes before admitting GPU work.
- [ ] **Step 4: Run GREEN** using Step2 command; foreign/partial artifacts survive.
- [ ] **Step 5: Record ledger and retained-state assertions.**

### Task 5: Implement cacheTrue generation with error-safe decoder telemetry

**Files:** Create `tools/clip_plain/generation.py`, `tests/test_generation.py`.

**Interfaces:** Consumes Task1 checked scope; produces:
- `generate_cached_text(language, visual: Tensor, visual_mask: Tensor, prompts: list[str], *, on_event: Callable[[dict], None], scope: dict | None = None) -> dict`
  returning `texts`, `token_ids`, `output_lengths`, `decoding`.
- `memory_snapshot(device: torch.device) -> dict` (CPU fields null; GPU only on healthy paths).

- [ ] **Step 1: Add real tiny T5 and PEFT regression tests.**
  `test_each_cached_decoder_step_has_one_token` asserts cached lengths all1,
  beam5,max64,no sampling,training cacheFalse/config/mixed modes unchanged.
  Compare cached/uncached IDs for deterministic tiny fixture only.
  `test_nonfinite_raw_logits_abort_with_last_step_record` injects NaN/+Inf.
  `test_negative_infinity_processed_mask_is_not_raw_logit_failure` pins distinction.
  `test_primary_fault_survives_failing_error_observer` asserts same primary exception
  instance with note on logging failure and zero calls to CUDA RNG/memory/sync/
  empty_cache after fault. Assert one generation invocation, hooks removed and no
  retry. Test valid bool masks, masked padding, empty visual rows and mismatched
  prompt counts. Missing scope must fail before joint-input CUDA operations.
- [ ] **Step 2: Run RED.**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_plain/tests/test_generation.py -q`
  Expected: missing new generation function.
- [ ] **Step 3: Adapt proven safe-helper behavior without altering old helper.**
  Explicitly pass `use_cache=True`; fixed beam5/max64,do_sampleFalse. Use installed
  PEFT `get_base_model()` (plain T5 directly for CPU fixture) to install bounded
  pre/post-forward telemetry hooks; capture raw final-position logits before
  log-softmax, record finite/dtype/stride/decoder-length and remove hooks in
  `finally`. Preserve actual output token IDs without monkeypatching tokenizer.
  No modern cache_implementation APIs unsupported by Transformers4.32.0.
  Error handlers perform CPU logging and boolean/hook restoration only.
- [ ] **Step 4: Run GREEN** using Step2 command. GPU sentinels untouched.
- [ ] **Step 5: Record ledger; do not claim CUDA crash fixed from CPU tests.**

### Task 6: Build optimizer-free dev worker and strict report publication

**Files:** Create `tools/clip_plain/evaluate.py`, `tests/test_evaluate.py`.

**Interfaces:** Consumes Tasks1/3/4/5; produces:
- `score_dev(model, admission: dict, pool: PromptPool, job: dict, *, on_event: Callable[[dict], None], scope: dict | None = None) -> dict`
- `run_dev_job(config: dict, job: dict, approval: Path | None) -> dict`
- `validate_report(report: dict, job: dict, admission: dict) -> None`
- CLI `--config <path> --job <path> --approval <path>`; default preflight
  option `--check-only` performs CPU validation only.

- [ ] **Step 1: Add tests with real on-disk1024-feature fixtures.**
  Full3-clip fixture returns complete ordered report and finite metrics;
  max_batches1/3/greater-than-corpus all remain diagnostic,metricsNone.
  Test duplicate/reordered/missing loader output, wrong candidate hash,policy
  mismatch and precommitted reference overlap block publication. Add
  `test_worker_does_not_construct_optimizer` and `test_worker_never_scores_test`.
  `test_identical_post_warmup_output_holds_without_retry` preserves the existing
  collapse guard, while `test_vt_warmup_output_is_not_false_collapse` admits the
  early checkpoint case. Any guard failure blocks selection.
  Observe separate autocast enter/exit for each batch; durable start event precedes
  generation. Inject telemetry disk-full before generation (no model call) and
  after CUDA fault (primary preserved, no complete report).
- [ ] **Step 2: Run RED.**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_plain/tests/test_evaluate.py -q`
  Expected: missing evaluator/validator.
- [ ] **Step 3: Implement worker.** GPU authorization is checked before model
  construction/device queries. Load model-only plain candidate on CPU with exact
  identity/frozen-base binding, then evaluate per-batch under inference_mode/BF16
  autocast. Fsync clip and decoder events; capture no large tensor histories.
  Report binds run/evaluation policy/checkpoint/source/split hashes, IDs/texts,
  token lengths and telemetry hash. Publish final report atomically only after
  strict complete coverage. A bounded report lives under diagnostics and can
  never be passed to selection, even if it covers the entire split.
- [ ] **Step 4: Run GREEN** using Step2 command, then Task5 tests together.
- [ ] **Step 5: Record ledger and supported report schemas.**

### Task 7: Train only to next dev boundary and publish durable handoff

**Files:** Create `tools/clip_plain/train.py`, `tests/test_train_chunk.py`.

**Interfaces:** Consumes Tasks1/3/4 and inherited update/optimizer/recipe;
produces:
- `next_dev_epoch(completed_updates: int, recipe: dict) -> int`
- `run_train_chunk(config: dict, job: dict, approval: Path | None) -> dict`
- CLI `--config <path> --job <path> --approval <path>` with CPU-only
  `--check-only`; no implicit fallback to old CLIP runner.

- [ ] **Step 1: Add boundary and state tests.** Epoch0 targets2;887updates targets2;
 1774updates with missing epoch2 dev refuses further training; completed epoch2
  dev permits target4. Tiny real optimizer updates with fake short recipe exercise
  atomic epoch snapshots and `awaiting_dev` handoff. Verify no call to any
  generation function inside training worker. Inject update failure and prohibit
  capture_state/save_snapshot/CUDA cleanup after fault. Wrong source/data/config,
  old snapshot, invalid sampler step or frozen-base drift stops before next update.
  Existing collapse checks remain explicit failures, never retry triggers.
- [ ] **Step 2: Run RED.**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_plain/tests/test_train_chunk.py -q`
  Expected: missing train chunk functions.
- [ ] **Step 3: Implement new chunk lifecycle.** Start only from fresh plain
  identity or verified new-variant snapshot. Use inherited recipe/update kernel,
  physical8,legacy_mean,4000/10000 warmups, exact optimizer groups and clipping.
  Validate guards and source/data drift; publish full checkpoint each epoch and
  model-only candidate at dev boundary. Include checkpoint step and sampler count
  in handoff, exit0 immediately after handoff. No dev in `finally`; failure writes
  CPU-only progress/error evidence and exits nonzero. Restored global/prompt RNG
  follows initialization, so process restarts do not perturb learning trajectory.
- [ ] **Step 4: Run GREEN** using Step2 command, including interrupted/continuous
  tiny updates with identical next sample,prompt,optimizer and weight values.
- [ ] **Step 5: Record ledger and handoff contract.**

### Task 8: Sequence fresh workers, selection and fail-stop recovery

**Files:** Create `tools/clip_plain/study.py`, `tests/test_study.py`.

**Interfaces:** Consumes Tasks1–7; produces:
- `pending_stage(run_root: Path, identity: dict, admission: dict) -> dict`
- `execute_job(config: dict, job: dict, approval: Path) -> dict`
- `select_checkpoint(config: dict, run_root: Path, identity: dict, admission: dict) -> dict`
- `run_study(config: dict, approval: Path | None, *, check_only: bool = False) -> dict`
- CLI `--mode preflight|run --config <path> [--approval <path>]`.

- [ ] **Step 1: Add actual CPU subprocess lifecycle tests.** Fake train child
  writes handoff while staying alive briefly; evaluator must start only after
  exit and writer-lock release. Trace train0→dev2→train2→dev4 with no overlap.
  Missing/wrong/nonzero child results enter held status and launch nothing else.
  On error, verify new manual-stop, approval invalidation, unchanged v3 hold and
  no watchdog/retry. Check stale old-boot and reused same-boot PID separately.
  Parameterize report rejection for diagnostic,missing clip,reordered IDs,
  wrong source/checkpoint/evaluation policy. Test actual-best selection versus
  patience anchor,early tie,eligible minimum40 and ordered epoch gaps.
- [ ] **Step 2: Run RED.**
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_plain/tests/test_study.py -q`
  Expected: missing orchestration/selection functions.
- [ ] **Step 3: Implement CPU-only parent.** Stage jobs are immutable and tied to
  one claimed parent attempt; verify signed-off source/config/data before spawn.
  Use exact owner identities and nonblocking locks; wait/reap child before any
  next worker. Parent alone charges stage lifecycle and reconciles progress.
  Recovery computes missing dev before training; never bypass a failure by
  creating another root or token. Accept reports only after Task6 validation,
  then use pure `stop_state` semantics and atomic selected pointer/verified
  native best copy. Archive every dev candidate with hashes before optional
  new-owned retention. Support sequential reference seeds0/1, freeze selections,
  publish dev-only summary; no test stage or mixed-policy paired score claim.
- [ ] **Step 4: Run GREEN** using Step2 command and state/evaluate tests together.
- [ ] **Step 5: Record process-state transitions and fault behavior in ledger.**

### Task 9: Verify complete CPU lifecycle and write operator handoff

**Files:** Create `tools/clip_plain/tests/test_lifecycle.py`,
`docs/clip-plain-safe-dev-runbook.md`; update this plan's ledger. Main AGENTS
gets a concise entry only after implementation is actually verified, clearly
stating hold retained and production preparation/validation pending.

**Interfaces:** Consumes complete pipeline. Produces CPU verification evidence,
runbook and ready-but-disabled CLI; no new model API.

- [ ] **Step 1: Add end-to-end CPU tests.** Derive a synthetic parent package,
  admit1024 data, train tiny model to boundary, exit worker, evaluate in another
  process, validate/select, then resume next training chunk. Compare against
  uninterrupted tiny training, including exact RNG/prompt/sampler/AdamW state.
  Inject worker death after checkpoint but before handoff, and death after dev
  report but before parent state update; explicit reconciliation must reuse
  verified artifacts without extra updates or double charges. Assert test set
  never reaches generation and old parent/S² artifacts stay byte-identical.
- [ ] **Step 2: Run RED** with a regression injection or unfinished lifecycle
  boundary proving tests exercise the intended failure; no knowingly broken
  production code or GPU launch to manufacture RED.
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_plain/tests/test_lifecycle.py -q`
  Expected: explicit missing/faulted lifecycle assertion, then repair only the
  new-variant code required to make it pass. If all tests already pass, record
  this as verification of existing tasks; do not claim a RED cycle occurred.
- [ ] **Step 3: Run complete focused verification once.** Standard CPU environment:
  `/home/kan/Research/DeSpaMo/.venv/bin/python -B -m pytest tools/clip_plain/tests tools/clip_s2_inference/tests tools/clip_s2/tests/test_data.py tools/clip_s2/tests/test_owner_identity.py -q`
  Expected: all focused tests pass. Also run main overlay regressions from
  `PROJECT`, `CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:.`:
  `.venv/bin/python -B -m pytest tests/unit/models/test_visual_adapter.py tests/unit/models/test_temporal.py tests/unit/models/test_flan_t5.py tests/unit/data/test_batch.py -q`.
  Expected: pass or explicitly documented pre-existing prerequisite skips;
  investigate failures instead of widening assertions or enabling CUDA.
- [ ] **Step 4: Run static and preservation checks.** From `RUNTIME`:
  `/home/kan/Research/DeSpaMo/.venv/bin/ruff check tools/clip_plain`
  and `ruff format --check` using the same binary path;
  `/home/kan/Research/DeSpaMo/.venv/bin/pyright -p tools/clip_plain/pyrightconfig.json`.
  Expected: no new lint/type errors. Inspect `git status --short` and
  `git diff --check` in both main and retained worktrees; compare every old bound
  source and retained checkpoint/budget/hold hash with implementation-entry
  snapshot. Only planned new files/docs may differ.
- [ ] **Step 5: Write runbook and perform native self-review.** Document1024 raw
  CLS definition, CPU cache derivation/storage demand, fresh-versus-resumed
  identity, authorization/owner barriers, error-safe stop, source/policy hashes,
  actual test commands/results, charges and future stages below. Run
  verification-before-completion, then native `/verify` and `/review` equivalents
  (commands if available; otherwise documented checklist), fix substantiated
  findings and rerun affected checks. Do not claim independent review or root-cause
  resolution. No commit, GPU test or training launch.

## Future execution stages — excluded from implementation authorization

1. Review/approve this plan, especially raw1024 features and fresh initialization.
2. Implement Tasks1–9 with CPU fixtures only and present verification evidence.
3. Separately approve full CPU preparation after storage check; derive/verify8257
   clips and publish new manifest. Low C: space means pause for user storage
   action; do not move/delete artifacts or silently use another filesystem.
4. Review host mitigation and authorize a named bounded plain-model GPU probe.
   Predeclare clip IDs covering short/long inputs, same policy and no retry.
5. If successful, separately authorize full519 dev validation and bounded
   training/update→checkpoint→process-exit→dev handoff/profile for this model.
   Passing old S² smoke or CPU tests cannot satisfy this admission.
6. Only then consider long plain-CLIP seed0/1 study authorization, including
   explicit new-owned retention allowance. No audio/device changes are implied.

Plan does not schedule original batch4/cacheFalse reproduction, an environment
upgrade, a new decoder algorithm, test scoring or an automatic batch-size sweep.

## Native plan self-review record

- Feature conversion is lossless slicing of proven global224 columns, not a
  new encoder run or silent relabeling. Parent/child widths and policies differ.
- Fresh plain-model initialization is explicit; S² epoch2 is preserved, not
  falsely continued under a new projection width.
- Isolated evaluation requires training process exit and full resumable state;
  moving only the model to CPU while retaining GPU optimizer is insufficient.
- New evaluation policy is part of selection identity; bounded probes and old
  reports cannot select weights. No exact BF16 cross-policy equality is promised.
- Failure handlers, logging failures, boot/PID reuse, process ordering, disk
  guards and accounting reconciliation are assigned concrete test owners.
- No unavailable modern Transformers cache APIs, Pillow in main environment,
  new dependencies, old-source edits or unattended GPU launches are required.
- New-owned retention is permission-bound; default preserves artifacts and
  projects demand accordingly. Historical charges are counted once.
- Execution method remains native; plan review is next handoff, not permission
  to run GPU work. All implementation tasks remain unchecked at plan creation.
