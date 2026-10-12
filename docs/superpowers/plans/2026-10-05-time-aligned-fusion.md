# Time-Aligned Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for native execution, or superpowers:subagent-driven-development if the user selects delegation. Steps use checkbox syntax for tracking. No commits or pushes authorized.

**Goal:** Compare original fusion, fixed-gate time alignment and learned-gate time alignment on a matched plain-spatial-MLP backbone using two seeds and Signer03 dev.

**Architecture:** Admit a versioned source-frame/window timing map before training. Add an opt-in visual-adapter subclass that interpolates projected motion and adds zero-initialized gated corrections to spatial tokens, retaining original motion tokens and TCN/output/VT-Align. Independent runner/controller policies reuse existing seeded factories and durable watchdog primitives without editing checkpoint-bound older runners.

**Tech Stack:** Python3.11, existing torch2.0.1+cu117, Lightning1.9.5, transformers4.32, pytest/Ruff, RTX4090/24GiB, WSL.

**Spec:** `docs/superpowers/specs/2026-10-05-time-aligned-fusion-design.md` (approved design and64800-second budget).

## Global Constraints

- Cached cosine DINO key `7122b42e35510611626c2a71d0731b79dd79b20de6e89bd49f62737c5c84a18e`; existing VideoMAE1024 cache; no encoder/caption regeneration.
- Logical train/dev/test5746/582/768; all physical train; only Signer03 dev scored. No Signer07 generation or new selection from earlier test scores.
- Plain spatial MLP2048→1024→768/GELU, motion Linear1024→768, original K5/P2/K5/P2 TCN, output768→2048→2048/GELU, FLAN LoRA, unchanged VT-Align.
- Six fresh runs: seeds0/1 × arms `reference`, `aligned_fixed`, `aligned_learned`; common backbone weights/RNG matched per seed.
- 16000 updates, batch4/no accumulation, VT-only warmup2000, LR warmup1600/cosine-to-zero; adapter/VT LR3e-4, FLAN LR5e-5; AdamW betas0.9/0.98, eps1e-8, decay0.01; BF16/clip1.
- Full-dev checkpoints5500/10400/16000; highest BLEU-4 selects, ties choose earlier step; beam5/max64/noICL.
- Timing:16-frame windows, stride8, centre8j+7.5; within-clip interpolation only; uncovered tails/padding receive no correction.
- Corrector Linear768→768/GELU/Linear768→768 with final layer zero; fixed gate0.5; learned sigmoid Linear1536→1, zero initialized; no new norm/dropout/loss.
- Overall64800-second cap, per-run10800; two seed phases32400 each including setup/smokes; no budget transfer or reset on recovery.
- Latest2 owned resumes every500updates, all new dev checkpoints retained; atomic/fsynced publication before pruning. No deletion of older artifacts.
- At least50GiB reserve on both Linux and Windows before every GPU stage; projected remaining checkpoint demand checked too. Hardware changes on resume require stopping, not silently adopting a new environment.
- New labels retain `qwen-schema98-unreviewed-v1` and `human_review_status=not_assessed` where applicable; no fabricated reviews.
- Preserve user work, source archives and old policy behavior. No commits, pushes or host shutdown/compaction.

## Review Focus

1. Runtime cropping silently renumbers frames: certify full-source row mapping and reject non-full crop modes (Task1/3 tests).
2. Motion cache counts match but extraction chronology is unknown: require provenance admission, label reconstruction, stop unresolved timing (Task1 tests).
3. Zero-init hides disconnected gate gradients: first-step correction-output gradient and post-update gate/earlier-layer gradients checked separately (Task3 tests).
4. Interrupted dev scoring at terminal16000 bypasses Lightning train-start callbacks: evaluation-only completion from exact terminal state (Task3/4 tests).
5. Host disk or controller budget changes between stages/restarts: check both disks, remaining study demand, source identity and uncapped elapsed accounting (Task4 tests).

## File boundaries and existing extension points

Create `src/despamo/data/motion_timing.py` for timing schema/admission; `scripts/prepare_motion_timing.py` for CPU-only provenance mapping. Create `src/despamo/models/aligned_fusion.py` for pure interpolation and fusion adapter; leave base adapter/data batch/training module unchanged.

Create `scripts/train_fusion_study.py` for new run policy `signer-pilot-time-aligned-fusion-v1`, profile/model construction, smoke, owned callbacks and selection. Create `scripts/run_fusion_study.py` for controller policy `time-aligned-fusion-controller-v1`, two budget phases and summaries. Write report `docs/time-aligned-fusion-results-2026-10-05.md` and progress `docs/superpowers/plans/2026-10-05-time-aligned-fusion-progress.md` during execution.

Reuse `scripts.train_projector_pilot._prepare_projector_inputs`, `build_seeded_projector_model`, `build_seeded_projector_data`; `enable_connector_arm(..., 'plain_mlp', seed)`; `PilotDataModule`, `E1Ledger`, `E1BudgetGuard`, `FullDevCheckpoints`, `score_pilot_batches`, `choose_checkpoint`, and `scripts.run_projector_overnight.Study`. Existing connector callbacks are references, not monkey-patched global policies. Baseline E3 selected model and all three dev reports survive cleanup; do not require deleted non-selected old model files.

External study root: `/home/kan/datasets/despamo/signer-pilot/time-aligned-fusion-study-v1`; run outputs remain under existing pilot root with new `fusion-<arm>-seed<seed>-16000-<identity_hash>` keys. Keep generated manifests/profiles/checkpoints external to Git.

### Task1: Timing provenance and admission

**Files:** Create `src/despamo/data/motion_timing.py`, `scripts/prepare_motion_timing.py`, `tests/unit/data/test_motion_timing.py`.

**Interfaces:** Frozen `ClipTiming(clip_id: str, source_indices: tuple[int, ...], frame_count: int, windows: tuple[tuple[int, int], ...])`; `build_motion_timing(frame_rows: dict, motion_manifest: dict, extraction_provenance: dict) -> dict`; `validate_motion_timing(timing: dict, frame_rows: dict, motion_manifest: dict) -> dict[str, ClipTiming]`. CLI outputs versioned `motion-timing.json` with hashes and `provenance_status` (`original_receipt` or `reconstructed_source_convention`). Status is descriptive; unresolved chronology never gets ready status.

- [ ] Write tests `test_window_centres_and_uncovered_tail`: frame_count25 gives windows(0,15),(8,23), centres7.5/15.5, frame24 uncovered; `test_short_clip_real_support`: frame_count10 has one padded extraction window but only0..9 spatial observations.
- [ ] Write `test_reject_count_order_and_provenance_mismatch`: reordered/non-monotonic frame indices, missing source evidence, mismatched motion counts, duplicate IDs or altered source hashes raise ValueError. `test_runtime_requires_full_source_rows` rejects cropped/renumbered runtime contracts, while pure mapping preserves non-contiguous original indices.
- [ ] Run `PYTHONPATH=src:. .venv/bin/pytest tests/unit/data/test_motion_timing.py -q`; confirm failure from missing module/functions before implementation.
- [ ] Implement schema/content-hash validation and read-only source matching. Verify all7096 map/count joins, original numeric frame ordering and annotation clip identity; require `source_indices == tuple(range(frame_count))` for this runtime study.
- [ ] CLI accepts `--frame-rows`, `--motion-manifest`, `--annotation`, `--extractor`, `--helpers`, `--provenance`, `--output`; rejects output aliases/symlinks and writes atomically. Bind exact input/extractor/helper hashes, window/stride/tail rules and motion row hashes.
- [ ] Investigate available historical provenance without importing extractor GPU models. Outer `features.zip` contains nested `mo_feat_p14t.zip`; optionally compare its NPY row bytes to cache for source evidence without extracting/rewriting caches. Archive members do not carry overlap markers; do not equate archive equality or matching counts with proven window timing. If evidence cannot establish chronology, stop before long training and report blocker for user decision.
- [ ] Rerun Task1 tests; expected all pass. Save CPU timing-admission receipt, including coverage fraction and uncovered-frame count; no GPU stage launched here.

### Task2: Pure interpolation and opt-in fusion adapter

**Files:** Create `src/despamo/models/aligned_fusion.py`, `tests/unit/models/test_aligned_fusion.py`.

**Interfaces:** `interpolate_motion(motion: Tensor[M,C], source_indices: Tensor[S], window_starts: Tensor[M], window_ends: Tensor[M]) -> tuple[Tensor[S,C], Tensor[S]]` (values and bool support mask); `enable_aligned_fusion(base: SpaMoVisualAdapter, arm: str, seed: int) -> SpaMoVisualAdapter`; subclass `TimeAlignedFusionAdapter.forward(spatial, spatial_mask, motion, motion_mask) -> tuple[Tensor, Tensor]`, retaining the existing four-argument call used in training and dev scoring. Constants `FUSION_ARMS` and `FUSION_CONFIGS` pin arm dimensions/rules.

- [ ] Write `test_interpolation_at_frame10`: two scalar motion vectors0/8 yield2.5 at frame10 (weights0.6875/0.3125); frame24 invalid for25frames. Check centres, before/after-centre support, one window, non-contiguous indices and no across-clip mixing.
- [ ] Write `test_zero_initialized_forward_matches_plain_mlp`, `test_mask_and_token_count_match_reference`, `test_shared_rng_and_correction_match_fixed_learned`, `test_padding_does_not_change_valid_outputs` (adapter eval mode), `test_invalid_masks_and_empty_sequences_rejected`.
- [ ] Run `PYTHONPATH=src:. .venv/bin/pytest tests/unit/models/test_aligned_fusion.py -q`; confirm missing implementation fails.
- [ ] Implement projected-motion interpolation in FP32 timing arithmetic, then cast weights to feature dtype. Keep autograd to motion projector; invalid rows return zeros, not NaNs. Runtime source positions are0..S-1 only because Task1/3 certify full-source inputs; they are not padded indices or normalized length positions. Reject non-prefix masks and count/window mismatches.
- [ ] Attach correction/gate under `visual_adapter` with forked CPU/CUDA RNG seed200+training_seed after constructing the same plain-MLP backbone in all arms. Reference returns that backbone unchanged. Preserve exact common parameter names, motion stream, TCN, output MLP and masks.
- [ ] Add pure `fusion_frame_stats(...) -> dict` returning supported-frame count, gate count/sum/sumsq/min/max, counts below0.05/above0.95 and correction norm; no hidden mutable inference state. Aggregate only supported real frames; reference reports gate statistics as not applicable.
- [ ] Rerun Task2 tests and existing connector/visual-adapter tests; expected pass. Verify no edits to base adapter, old batch schema or historical runners.

### Task3: New-policy runner, smoke and durable selection

**Files:** Create `scripts/train_fusion_study.py`, `tests/unit/pilot/test_fusion_study.py`.

**Interfaces:** `initialize_fusion(args) -> dict`; `prepare_fusion(args) -> tuple[dict, Path, dict, dict, object]` (identity/output/protocol/views/config); `build_fusion_model(config, arm: str, seed: int, initial_base_hash: str) -> PilotBaselineModule`; `fusion_gradient_smoke(model, batch, arm: str) -> dict`; `validate_fusion_selection(run: Path, protocol: dict, dev_ids: tuple[str, ...], expected_identity: dict | None = None) -> dict`; `run_fusion(args) -> dict`.

- [ ] Write tests for frozen recipe validation, full-frame config requirement, changed timing/hardware/source/profile rejection, both seed attestations, shared plain-MLP parameter parity, correction/gate optimizer coverage at3e-4, and untouched FLAN LR5e-5.
- [ ] Write `test_smoke_zero_init_then_gate_gradients`: positive finite final-corrector gradients initially, finite nonzero earlier-corrector/gate gradients after one disposable AdamW update. `test_smoke_restores_parameters_buffers_rng_and_modes` proves temporary updates do not alter production initialization/sampler.
- [ ] Write tests for report hash/ordered582IDs validation, ties choosing earlier step, all3dev report byte bindings, mismatched run locks refusing status writes, latest2 pruning only after durable publication, checkpoint reload parity and terminal-step evaluation-only recovery. No tests launch real long training.
- [ ] Run `PYTHONPATH=src:. .venv/bin/pytest tests/unit/pilot/test_fusion_study.py -q`; confirm missing implementation fails.
- [ ] Implement independent CLI modes `initialize|preflight|smoke|run`, existing input flags plus `--motion-timing`, `--arm`, `--seed`, `--steps 16000`, `--gpu-cap-seconds 10800`, `--initialization-profile`, `--authorize-fusion-run`, `--resume-checkpoint`. Preflight is read-only and never creates optimizer/model on GPU.
- [ ] New common CPU profile attests both original base and plain-MLP backbone hashes per seed. Identity binds timing artifact, source dependency closure including new controller, source lockfile/schema, protocol/split/data/features/selected baseline, seed/arm/recipe/caps, hardware name/capability/VRAM/driver and torch/CUDA versions. No reuse of earlier initialization profiles with new code/timing.
- [ ] Preserve baseline losses, optimizer group selection and scheduler behavior through existing PilotBaselineModule. New callbacks implement owned durable saves/ledger/report bindings locally; never override imported module globals. Reports include full coverage/gate statistics and checkpoint/source identities, and byte hashes bind all3dev reports at selection.
- [ ] Smoke runs joint CE+VT independent of warmup selection; on RTX4090 require fit within measured VRAM minus2GiB reserve. Measure actual peak for four representative train clips and a long-clip batch. Disposable optimizer steps restore trainable tensors, buffers, modes, global/prompt RNG and temporary optimizer state. Record that these are smoke-only steps, not production updates.
- [ ] Rerun Task3 tests. Check effective optimizer coverage exactly once per trainable parameter and strict state loading including correction/gate; inherited forward path must exercise fusion in both compute_losses and score_pilot_batches.

### Task4: Six-run watchdog controller and paired reporting

**Files:** Create `scripts/run_fusion_study.py`, `tests/unit/pilot/test_fusion_controller.py`.

**Interfaces:** `FusionPhase(root: Path, seed: int)` uses existing Study with32400-second phase ceiling; `stage_command(arm: str, seed: int, profile: Path, timing: Path, mode: str) -> list[str]`; `validate_completion(phase, run: Path, expected: dict) -> bool`; `aggregate_fusion_results(results: list[dict]) -> dict`; `main()` accepts `--study-root`, `--motion-timing`, `--authorize-fusion-study`.

- [ ] Write tests requiring exactly6unique(seed,arm) results; wrong checkpoint/seed/arm/timing fails before child launch. Fixed-minus-reference and learned-minus-fixed deltas use matching ordered dev references; no duplicate row overwrites.
- [ ] Write tests covering cumulative budgets across restart, stages with no elapsed reset, orphan process ownership/boot/PID identity, no overlap, source changes before fit, missing timing provenance, Linux-space and Windows-space failures independently, projected remaining demand and terminal16000 recovery with zero training updates.
- [ ] Run `PYTHONPATH=src:. .venv/bin/pytest tests/unit/pilot/test_fusion_controller.py -q`; confirm missing implementation fails.
- [ ] Implement sequential seed0 reference/fixed/learned then seed1 same. Common CPU initialization is recorded; GPU admission/smoke/profiling and full-dev time charge against phase/overall budgets. Start each child with the lesser remaining phase and10800-second run allowance; global maximum64800 never increases.
- [ ] Reuse Study watchdog primitives rather than create a new detached timer. Require exact validated identities for completed-stage adoption. Prefer highest durable step/resume for incomplete owned runs; ties prefer step checkpoint. Resume validates profile/timing/hardware/elapsed/sampler/RNG before training.
- [ ] Before each GPU stage, require both disks above50GiB and available headroom for remaining study writes plus reserve. Use actual smoke checkpoint size when available; otherwise5.6GiB estimate, with3dev+2resume/run and1atomic temporary. Preserve all newly generated dev models and latest2resumes; no post-completion pruning beyond approved policy.
- [ ] Aggregate BLEU1-4/ROUGE-L, per-seed and mean/std deltas, repetition/unique predictions, all dev trajectories, gate statistics/coverage, params/token counts, actual peak GPU memory and charged elapsed time. Summary says dev-only, signer07_scored=false, two-seed limit, unresolved earlier reference variability and alignment-versus-capacity confounding. No extra shifted/shuffled inference diagnostic in this initial implementation plan.
- [ ] Rerun controller tests; expected pass. Summary publication validates all six selections/reports before declaring complete.

### Task5: Verification, real-data admission and approved execution

**Files:** Update approved spec only for attested implementation facts; create progress/report paths listed above; append final result link to `docs/signer-pilot-protocol.md` after completion.

- [ ] Run focused tests from Tasks1–4; then `PYTHONPATH=src:. .venv/bin/pytest tests/unit -q` and `.venv/bin/ruff check scripts src tests tools/dinov3`. Run targeted Pyright on new Python files with `.venv/bin/python` interpreter; distinguish historical existing diagnostics from introduced issues. Inspect `git diff --check` and user-work boundaries.
- [ ] Self-review code for the five Review Focus conditions. Use independent reviewer only if the chosen execution method authorizes delegation; fix reproduced findings, rerun only affected checks unless changes justify broader tests.
- [ ] Produce CPU timing receipt and both seed initialization attestations under external study root. Stop if timing chronology remains unresolved; no automatic feature regeneration or unapproved assumptions.
- [ ] Under controller accounting, run real RTX4090 BF16 admission smokes for all3arms, including long sequences; record finite losses, two-stage correction/gate gradients, full-state restoration, token parity, optimizer recipe, actual peak memory and profiled step time. Confirm prospective six-run cost fits64800; otherwise report infeasibility without relaxing cap/recipe.
- [ ] Once checks pass and user has approved this plan/execution method, launch the authorized controller from project root with `PYTHONPATH=src:.`, `LD_LIBRARY_PATH=/usr/lib/wsl/lib`, `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=2`; command `.venv/bin/python scripts/run_fusion_study.py --study-root /home/kan/datasets/despamo/signer-pilot/time-aligned-fusion-study-v1 --motion-timing /home/kan/datasets/despamo/signer-pilot/time-aligned-fusion-study-v1/motion-timing.json --authorize-fusion-study`. No new dependency install or batch-size tuning.
- [ ] Track each stage in progress file. On failure/cap/storage event, preserve checkpoint/accounting and report incomplete; recover only through declared controller. On success validate six selections and all18dev-report bytes, exact582reference order, positive/negative per-seed contrasts and charged total<=64800.
- [ ] Save final report and protocol link, verify final source/status/diff and report identities, and summarize both seed results with limitations. Do not automatically score Signer07, commit, push, or promote an architecture from one favorable seed.

## Plan self-review

Spec sections covered: timing Task1/2; architectures Task2; gradients and initialization Task2/3; matched recipe/selection Task3; recovery/budget/storage Task3/4; reporting Task4/5. All five Review Focus conditions have named owning tests. Runtime four-argument adapter is compatible with both training and dev generation because only certified full-source inputs are admitted; generic non-contiguous interpolation remains tested but cropped runtime is rejected. Historical dev-model pruning does not delete reports or the selected E3 baseline used for input admission.

No implementation has begun. Remaining user decision: review this plan and select native or subagent-driven execution. Native is recommended because data admission, initialization, smoke and controller bindings are tightly coupled; delegation is optional and must be explicitly selected.
