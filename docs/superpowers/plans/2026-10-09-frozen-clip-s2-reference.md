# Frozen CLIP + S² Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Native implementation and self-review follow user preference; no delegated agents or commits without explicit request. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Publish pinned frozen-CLIP/S² spatial features and complete two matched reference translation runs, seeds0/1, for a dev-first comparison with adapted DINO.

**Architecture:** New `tools/clip_s2/` namespace in the retained isolated worktree owns extraction, explicit data bindings and two-run orchestration. Reuse unchanged package, FLAN model/update, recipe, sampler and compact-checkpoint helpers. New source/data/output identities preserve all current experiment bytes.

**Tech Stack:** Translation uses existing main Python3.11/torch2.0.1/transformers4.32/PEFT0.7.1. Extraction uses existing main-project tools/motion/.venv with same torch/transformers/NumPy plus Pillow11.3; main lacks Pillow. Existing uv locks/native fsync/flock publication; no dependency upgrade or mixed imports.

**Spec:** [Frozen CLIP + S² reference design](../specs/2026-10-09-frozen-clip-s2-reference-design.md).

## Global Constraints

- Scope: frozen `openai/clip-vit-large-patch14` revision `32bd64288804d66eefd0ccbe215aa642df71cc41`, reference seeds0/1 only; CLIP adaptation/Qwen/fusion changes are later decisions.
- Latest2026-10-10user authorization supersedes finish-six-first: intentionally stop predecessor watchdog/controller/worker, preserve checkpoints and test claims, verify stop receipt/all owners exited, then execute CLIP/S² now. Old source/recipe remains unchanged; no predecessor test scoring.
- Preserve retained worktree `tools/followup/`, `tools/benchmark/`, main `src/`, model snapshots, environments and locks; create new namespace, not refactor live helpers.
- Reuse current main `.venv`; runtime `PYTHONPATH=<main>/src:<worktree>:<main> LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0`.
  Extraction/tests use existing main-project tools/motion/.venv and `PYTHONPATH=<worktree>`; do not import its libraries into main interpreter.
- CLIP encoder FP32/frozen/eval/inference, TF32/autocast off; processor at same pinned revision. Export unnormalized final hidden CLS, not projected/pooler features.
- Scales224/448; upscale already processed224tensor, bicubic `align_corners=False`, `antialias=False`; row-major tiles, mean four CLS vectors, concatenate224then448. Arrays FP32 `[N,2048]`.
- Official clips7096/519/642; frames827354/55775/64627; total8257clips/947756frames. Preserve full source rows and reuse pinned motion in place.
- Physical batch8/no accumulation/drop-last;887updates/epoch; max200/min40; dev every2; patience10/minDelta0.10. Actual best retained, significant anchor separate, earlier ties.
- VTwarmup4000, LRwarmup10000; legacymean VT; adapter3e-4→5e-5 / FLAN5e-5→1e-5; same FLAN LoRA, ICL and transformer checkpointing as current200epoch recipe.
- Native arrays7.23GiB + extraction allowance3GiB + training allowance6GiB, plus50GiB reserve on Linux/C:; archive projection40GiB plus50GiB on B:. Recheck measured demand at execution; shortage blocks launch, no implicit cleanup.
- Two frozen dev selections, dev-only comparison, no automatic new test scoring/claims. No commits, pushes, prior-artifact deletion or caption changes.

## Review Focus

1. Final hidden CLS versus post-normalized/projected output; incomplete cached processor must fail preflight (Task1).
2. Tile-major batching can mix frames; corner/batch tests and SpaMo-wrapper parity must catch this (Task1).
3. Correct admission with old hardcoded `batch_for` can still read DINO; two distinct sentinel feature roots must catch this (Task3).
4. Interrupted dev/checkpoint publication can alter selection, RNG or stopping; durable replay must preserve next update and stop before more updates (Task4).
5. Six-run controller/test defaults can expand scope or restart intentional stops; enforce exact pair, dev-only finish, ownership and storage gates (Task5).

## Paths and verification commands

`M=/home/kan/Research/DeSpaMo`.
`W=$M/.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`.
All implementation paths below are relative to **W**, not main checkout. Verify
existing isolation before implementation; preserve unrelated changes. Store this
plan's ledger under main `.superpowers/sdd/2026-10-09-frozen-clip-s2-reference/`.
Do not delete any ledger/worktree that holds uncommitted or checkpoint-bound work.

From W use this command prefix for tests and CLI:

```bash
PYTHONPATH=/home/kan/Research/DeSpaMo/src:.:/home/kan/Research/DeSpaMo LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/kan/Research/DeSpaMo/.venv/bin/python -B
```

In steps below `PY` denotes that exact command prefix; expand it before execution.
`RUFF=$M/.venv/bin/ruff`. Targeted typing uses installed `pyright` with
`--project $M/.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/pyright-features.json --pythonpath $M/.venv/bin/python`.
No GPU tests are permitted while the active six-run study owns the device.

## File responsibilities

| Create | Responsibility |
| --- | --- |
| `tools/clip_s2/__init__.py` | New experiment namespace |
| `tools/clip_s2/encoder.py` | Pinned processor/CLIP loader and tiled CLS kernel |
| `tools/clip_s2/extract.py` | Input preflight, inference profile, package extraction/verification |
| `tools/clip_s2/data.py` | Explicit spatial/motion admission and batching |
| `tools/clip_s2/train.py` | Separate identities, profile and owned reference training lifecycle |
| `tools/clip_s2/study.py` | Two-run sequential controller, dev freeze, archive and monitor |
| `tools/clip_s2/report.py` | Matched per-seed dev comparison |
| `tools/clip_s2/tests/test_{encoder,package,data,training,study,report}.py` | Corresponding behavioral/recovery tests |
| Main `docs/clip-s2-reference-runbook-2026-10-09.md` | Verified configuration, paths, results and recovery handoff |

Feature base: `/home/kan/datasets/despamo/official-phoenix14t-clip-s2-features-v1`.
Study root: `/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v1`.
Archive root: `/mnt/b/DeSpaMo/archives/official-phoenix14t-translation-clip-s2-spamo200-v1`.
New root creation is a separate experiment identity, never a workaround for prior budgets.

During implementation, write immutable `profile-inputs.json` under new feature
base, with keys `annotation_root`, `source_row_map`, `frames_root`, `cache`,
`processor_file`, `output_base`, `accepted_profile` (null for preflight/profile,
required for smoke/run), and `revision`. Values are the exact spec paths and
revision; strict validation rejects missing/extra keys. The source row-map comes
from completed official spatial package, not a historical partial inventory.
Profile publishes `profiles/accepted.json` with path/SHA and selected frame/tile
batches; write separate immutable `extraction-inputs.json` with that receipt
binding before full inference. Preserve profile inputs; never rewrite their bytes.

Write `study-inputs.json` under new study root with keys `spatial_root`,
`spatial_manifest_sha256`, `feature_identity_sha256`, `predecessor_root`,
`study_root`, `archive_root`, `seeds` (`[0,1]`), `arm` (`reference`), and
`test_scoring` (`false`). Pin predecessor to active200epoch root and roots to
paths above. No configurable schedule/batch overrides; reuse approved recipe.
Bind each input file into its respective immutable identity.

New module CLI contracts: `tools.clip_s2.extract --mode MODE --config PATH`
with `--authorize-extraction` required for profile/smoke/run;
`tools.clip_s2.train --mode profile|run --seed 0|1 --config PATH
--authorize-translation`; and `tools.clip_s2.study --mode preflight|controller|watch
--config PATH` with `--authorize-study` required for controller/watch. Worker
commands preserve the exact config path; changing any bound input is identity
drift, not automatic resume. `tools.clip_s2.report --clip-root PATH --dino-root
PATH --output PATH` emits only dev comparison artifacts.

### Task1: Pin preprocessing and implement S² encoder

**Files:** `encoder.py`, `__init__.py`, `tests/test_encoder.py`.

**Interfaces:**
- `load_encoder(cache: Path, processor_file: Path, device: str) -> tuple[model, processor, dict]` validates fixed revision/width and returns resolved model/processor binding.
- `s2_cls(pixels: Tensor[B,3,224,224], model, tile_batch_size: int) -> Tensor[B,2048]` preserves row/frame order and FP32 inference contract.
- `extract_frames(paths: list[Path], model, processor, frame_batch_size: int, tile_batch_size: int) -> np.ndarray` returns finite FP32 `[len(paths),2048]`.

- [ ] **RED:** Write tests for two-frame corner-coded tensors, exact scale concat, per-frame mean, batch tails, frozen encoder and malformed/nonfinite outputs. The real-behavior expectation is:
  ```python
  features = s2_cls(two_frame_corner_fixture(), cls_probe_model(), tile_batch_size=3)
  assert features.shape == (2, 2048)
  torch.testing.assert_close(features[:, :1024], expected_global_cls, rtol=0, atol=0)
  torch.testing.assert_close(features[:, 1024:], expected_tile_mean, rtol=0, atol=0)
  ```
  Fixture helpers belong in tests; probe output must depend on actual input pixels, not mock call counts.
- [ ] Run `PY -m pytest tools/clip_s2/tests/test_encoder.py -q`; expect missing implementation failure.
- [ ] Implement interfaces above. Hash processor metadata retrieved at pinned revision into new input directory. Missing processor, mismatched revision, width or normalization blocks admission; no shared snapshot edits/default fallback.
- [ ] Run same CPU tests; expect all pass. Run `RUFF check tools/clip_s2` and targeted Pyright.
- [ ] After GPU gate, run opt-in test `test_real_spamo_wrapper_cls_parity` with at least two nonidentical actual frames, same FP32 weights, processor and wrapper settings. Require rtol1e-4/atol1e-5; test raw CLS against wrapper first token, and demonstrate it is not replaced by pooler/projection.
- [ ] Profile frame32/16/8 in descending order with tile microbatch recorded, including longest clip/tail. Persist accepted choice, source/processor hashes, reserve<=21GiB and free>=2GiB; failed attempts remain recorded, no speculative full extraction.

### Task2: Extract, resume and verify complete feature package

**Files:** `extract.py`, `tests/test_package.py`.

**Interfaces:**
- `make_identity(inputs: dict, encoder_binding: dict, profile: dict) -> dict` binds all source/model/processor/runtime/batching decisions.
- `run_extraction(mode: str, config: dict) -> dict`, modes `preflight|profile|smoke|run|verify`, outputs package root/hash/counts/receipts.
- Consumes Task1 encoder and existing `official_inventory`, `source_rows`, `verify_clip`, `write_clip`, `finish_package`, `verify_package`, owned ledger/storage helpers.

- [ ] **RED:** Test source mutation, extra/missing/reordered frames, wrongdtype/width, changed processor/profile, existing receipted clip reuse and mismatched receipt rejection. Test unreceipted array after crash blocks without overwrite; no complete manifest appears before all expected receipts.
- [ ] Run `PY -m pytest tools/clip_s2/tests/test_package.py -q`; expect feature missing failures.
- [ ] Implement preflight from existing full official source receipts/row map and annotation hashes. Validate coverage8257/947756 and each source before/after inference; pass new model/processor profile identity to unchanged publisher.
- [ ] Resume only same identity and verified receipts, retaining accounting. Existing unreceipted files/partial-complete directories require inspection; do not silently erase them. SIGTERM leaves durable interrupted ledger and owned artifacts.
- [ ] Run tests GREEN. Smoke first clip per split and longest train clip, verify shapes/hashes/source rows. Recheck projected remaining bytes+50GiB on native/backing disk.
- [ ] Once profile/source/storage gates pass, execute full extraction and `verify` sequentially. Expect7096/519/642clips and827354/55775/64627frames with finite FP32 arrays, SHA-verified receipts and atomic complete metadata.
- [ ] Freeze root/manifestSHA for consumer config and copy metadata verification receipt to runbook. Test labels are not used in encoder computation.

### Task3: Explicit feature admission and matched batching

**Files:** `data.py`, `tests/test_data.py`.

**Interfaces:**
- `admit_data(spatial_root: Path, spatial_manifest_sha256: str, verify_arrays: bool = True) -> dict` returns existing admission fields plus `roots={"spatial": Path, "motion": Path}`.
- `batch_for(admission: dict, split: str, clip_ids: tuple[str, ...]) -> PhoenixBatch` loads/hashes from those roots, not old globals.
- `eval_batches(admission: dict, split: str, batch_size: int = 4)` preserves official ordered IDs.
- Motion root/manifestSHA fixed in spec. Construct `Phoenix14T` with `spatial_crop_mode="full"`.

- [ ] **RED:** Build two distinguishable synthetic spatial roots and one motion root; prove returned spatial tensors come from admitted CLIP root while motion stays unchanged. Assert source digest/index mismatch, reordered/duplicate IDs and wrong manifestSHA fail.
- [ ] Run `PY -m pytest tools/clip_s2/tests/test_data.py -q`; expect failure before implementation.
- [ ] Implement same coverage/signers/array integrity/timing checks as old admission, parameterizing both root binding and batch loading. Store roots and per-array hashes in admission; canonical identity pins complete metadata, annotation and feature digests.
- [ ] Run tests GREEN; verify full prepared package paired with existing motion without copying motion. Assert ordered train IDs and dev order equal completed DINO identities; test arrays may be validated but never scored in this phase.

### Task4: Matched training lifecycle, GPU profile and replay

**Files:** `train.py`, `tests/test_training.py`.

**Interfaces:**
- `run(seed: int, spatial_root: Path, spatial_manifest_sha256: str, mode: str) -> dict`, modes `profile|run`, reference only.
- `score_dev(...) -> dict` uses unchanged `tools.followup.evaluation.score_batches` with Task3 evaluation batches and existing `PromptPool`.
- `code_hashes() -> dict` pins new namespace, every imported helper, main `src/`, config/locks and parity-reference hashes; source drift blocks recovery.
- Reuse `schedule(8)`, `group_lr`, `stop_state`, `build_model(seed,"reference",cache)`, `optimizer_for`, `update`, sampler/compact state helpers without modifying originals.

- [ ] **RED:** Test recipe equality with completed200epoch reference; no accumulation, VT4000/4001 boundary, group warmup10000/floors at177400, same ICL banks/partner RNG and initial fingerprints.
- [ ] **RED:** Test actual-best vs significant-anchor separation, min40/patience10, finite contiguous reports, and report-publication crash reconciliation. Assert recovered stop prevents another optimizer update.
- [ ] Run `PY -m pytest tools/clip_s2/tests/test_training.py -q`; expect failures before lifecycle implementation.
- [ ] Implement explicit new policy/root/data identity and train source checks. Publish epoch resume before dev, hash/archive model candidate before pruning, derive stop state from reports, reconcile pointer after interruption, publish completion before closing ledger with reconciliation on resume.
- [ ] Run tests GREEN, including real small-model full-state replay: save/restore model buffers+AdamW+sampler+global/prompt RNG; next batch/exemplars and next optimizer result match uninterrupted execution, not just generated text.
- [ ] After GPU gate, profile accepted CLIP features with physical8 ordinary/long-visual/long-target batches and longest eligible full ICL prompt. Require finite joint gradients, all expected AdamW states initialized, free>=2GiB/reserved<=21GiB, native fresh-base compact reload and generation equality. Do not reuse DINO profile as CLIP evidence.
- [ ] Compare initial common/frozen fingerprints for each seed against completed DINO reference. If fingerprints/runtime/recipe drift, block matched launch; do not silently call unmatched runs equivalent.

### Task5: Two-run dev-only controller, archival and monitoring

**Files:** `study.py`, `tests/test_study.py`, main runbook.

**Interfaces:**
- `validate_selections(rows: list[dict]) -> None` accepts exactly reference seeds0/1.
- `preflight_study(config: dict) -> dict` checks verified intentional predecessor stop (or completed study), all predecessor owners exited, idle GPU, new package/profile/source identities and remaining storage.
- `run_study(config: dict) -> dict` trains the pair sequentially, freezes dev selections, archives native final payloads, and completes without test scoring.
- `monitor(config: dict) -> None` owns single lock and1200second observation/recovery loop.

- [ ] **RED:** Assert missing/duplicate/extra arm selections fail; unapproved unfinished predecessor, live/orphan/reused owner, source/profile drift and shortage block launch. Verified user-directed predecessor stop permits this separate study, while a manual stop of the new study prevents its restart. Only latest verified transient failure permits bounded recovery.
- [ ] **RED:** Using a small fixture study, assert two completed selections yield dev-only completion and archive receipts, with no `official-test/claim.json` or test generation calls.
- [ ] Run `PY -m pytest tools/clip_s2/tests/test_study.py -q`; expect failures before implementation.
- [ ] Implement explicit pair commands, ownership publication, locks/SIGTERM/manual stop, verified same-root restart (max3), unknown-fault `needs_agent` and dev-only archive completion. Existing `archive_final` requires a test report, so implement new dev-final payload archival rather than bypassing that old guard.
- [ ] Freeze pair as immutable ordered selection hashes, sort seed order deterministically; verify replayed inventory matches before further publication.
- [ ] Run tests GREEN and final focused suite. Inspect intended diff/status, Ruff and targeted typing; preserve unrelated work. Record measured storage and checkpoint projections; storage blocker requires separate user-authorized action, not reduced reserve or silent deletion.
- [ ] After prerequisite checks, launch detached new controller/monitor. Verify exact owned PIDs/start ticks/commands, finite initial updates and first native resume. Update runbook with real paths/receipts; never invent future progress.

### Task6: Paired dev comparison and follow-up decision

**Files:** `report.py`, `tests/test_report.py`, main runbook.

**Interfaces:** `compare_dev(clip_root: Path, dino_root: Path) -> dict` returns per-seed rows, paired changes, two-seed means, actual selected/stopped exposure and provenance notes.

- [ ] **RED:** Test missing CLIP seed, wrong arm, mismatched ordered data/recipe/initialization, different decoder/metric signatures and inconsistent selection hash rejection. Missing DINOseed1 must be explicitly unpaired, not fabricated; paired mean requires both complete pairs.
- [ ] Run `PY -m pytest tools/clip_s2/tests/test_report.py -q`; expect feature missing failure.
- [ ] Implement comparison against available completed references in `official-phoenix14t-translation-spamo200-v1`, not historical40epoch performance. Predecessor stopped before DINOseed1, so report only existing seed0pair and label seed1unpaired. Report both CLIP results/mean, but no two-seed paired DINO gain until both pairs exist.
- [ ] Run tests GREEN and publish `comparison-dev.json` plus concise markdown summary. State that new CLIP official test is unscored. Recommend adapted CLIP only as separately designed matched follow-up; no test-guided tuning or automatic next experiment.

## Final verification and handoff

- [x] Run extractor/inventory/package tests plus inherited publisher tests with motion interpreter; run data/training/study/report and relevant inherited translation/checkpoint/followup tests with main interpreter/overlay.51passed in correct environments; real GPU parity/profile smoke runs are explicit CLI checks, not silently skipped pytest markers.
- [ ] Run all new GPU-marked parity/profile/replay checks after old-study gate. Save evidence, source/processor/model hashes, exact counts and accepted profiles before freezing launch source.
- [ ] Run `RUFF check tools/clip_s2` and targeted Pyright; expect zero errors. Inspect `git status`, scoped `git diff` and `git diff --check` in main/worktree. No commit.
- [ ] Perform native final self-review against spec and all five Review Focus cases; address correctness findings with RED→GREEN regressions. Preserve plan-scoped ledger and runbook.

## Plan self-review — 2026-10-09

Reviewed against actual SpaMo extractor/wrapper, installed CLIP forward output,
cached model snapshot, hardcoded benchmark data admission/batching, active recipe
and six-selection controller, native publication/storage helpers and live source
closure. Corrected the in-chat draft's gaps:

1. Cached processor is missing: pin its metadata separately, no assumed defaults
   or shared-cache modification (Task1).
2. `pooler_output` applies post-layer normalization: use last hidden CLS and verify
   whole-wrapper parity/tile association before publication (Task1).
3. Old batching still reads hardcoded DINO roots after changing admission: new
   explicit roots govern both admission and loading (Task3).
4. Generation-only reload is insufficient: verify complete next-update replay,
   actual-best/anchor logic and publication crash windows (Task4).
5. Old controller needs six selections and final archive requires test report:
   new exact-two dev-only lifecycle, no automatic new held-out scoring (Task5).
6. Arrays alone are7.23GiB: include staging/training demand and50GiB backing-disk
   reserve, and recheck after current jobs; no cleanup authorization inferred (Task2/5).

Spec coverage: execution gates→Tasks1/5; extraction→Task1;
package/storage→Task2/5; data provenance→Task3; matched training/replay→Task4;
ownership/dev-only freeze/archive→Task5; comparison/claim limits→Task6.
Interfaces are explicit; each Review Focus case has an owning test task. New
functional code remains unwritten. Current source/model/data closure was verified
unchanged during documentation review. User requested save and self-review, not
implementation or a new launch; execution awaits that later instruction.

## Execution amendment — 2026-10-10

User explicitly approved immediate stop/extract/two-run launch in Build mode.
This supersedes prior completion prerequisite. Preserve predecessor epoch96
reference completion (best epoch76) and interrupted fixed-gate resumes. One
DINO200reference seed is available; do not silently create extra DINO runs or
claim a two-seed paired performance comparison. Seed1initialization can be
checked against retained old40epoch common initialization because model weights
and architecture are identical, but old40epoch scores are not comparison data.

Execution record: source/CLI implementations complete;51targeted tests, Ruff,
format and both targeted typing contexts passed. Real CLIP/S² parity,24clip
bounded package and307/307state batch8FLAN joint/reload profile passed. Detached
controller143082/extractor143107/watchdog143108 launched; full8257extraction is
in progress, then independent verification/full-data profile/reference0/1
automatically. At16:13UTC62durable clip receipts; no translation score yet.
See active runbook and switch receipt; old code/artifacts preserved. Do not edit
new runtime source while pipeline is live. Task2full publication and actual
translation completion remain asynchronous pending outcomes, not claimed done.
