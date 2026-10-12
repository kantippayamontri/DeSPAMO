# Documented Motion Extraction and Automatic Fusion Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development, selected by the user. Execute in the explicitly approved current checkout. Task steps use checkboxes. No commits, pushes, merges or unrelated cleanup authorized.

**Goal:** Extract7096documented motion clips under a5hour ceiling, validate their actual frame-window records, and automatically launch the approved six-run fusion study under its separate18hour ceiling.

**Architecture:** New extraction publishes versioned float32 features and contemporaneous source/window receipts as an owned complete package. Explicit opt-in new-motion admission validates the original baseline lineage separately, then constructs matched training views with the new arrays. A durable coordinator chains extraction completion to the existing training controller, with separate capped clocks and no success-by-exit-code shortcut.

**Tech Stack:** Python3.11; dedicated torch2.0.1+cu117/transformers4.32.0/Pillow11.3.0 extraction environment; unchanged main training environment; NumPy/pytest/Ruff/Pyright; RTX4090/WSL; existing neutral watchdog/storage helpers.

**Spec:** `docs/superpowers/specs/2026-10-05-motion-extraction-autotrain-design.md` (user-approved).

## Global Constraints

- Separate extraction18000seconds/5hours and training64800seconds/18hours; combined maximum82800seconds/23hours. No borrowing, reset on resume, or automatic extension.
- Extract all7096physical-train clips/827354frames/93232windows from exact cosine-DINO recorded210x260images; frozen logical groups5746/582/768. No Signer07 translation generation or selection.
- Model `MCG-NJU/videomae-large`, cached revision `12da269a02d3e1fbbb7011e610e591fea8061dca`, actual config/weights hashes; eval/inference FP32, no BF16/TF32 substitution or encoder training.
- Motion1024vector is `outputs.hidden_states[-1][:,0]`; preserve this pre-final-normalization selection rather than normalized last_hidden_state or pooling.
- 16frames/window, stride8, ascending full windows, drop incomplete tails; shortclips repeat last source frame only as model input, real support excludes repeats.
- RGB, bilinear shortestedge224, centrecrop224×224, rescale1/255, mean/std[0.5,0.5,0.5]; bind resolved processor recipe and installed preprocessing source.
- Batch at most16windows, two CPU threads; NumPy→torch input stacking must match standard processor tensors on batch1/8/16 and representative image shapes.
- New cache/records are newly generated original receipts, not recovery of historical downloaded-cache records. Old caches/captions/source frames/models stay intact.
- Six fresh new-motion runs: seeds0/1 × reference/aligned_fixed/aligned_learned, same plain spatialMLP/motion/TCN/output/VTAlign/FLAN recipe; no old-model reuse.
- Training16000updates,batch4,noaccumulation,VTwarmup2000,LRwarmup1600,cosinezero,adapter3e-4/FLAN5e-5,AdamW0.9/0.98eps1e-8decay.01,BF16clip1,dev5500/10400/16000,beam5/max64/noICL,tiesearlier; phase32400/run10800 ceilings retained.
- Retain latest2owned resumes/all new dev checkpoints; no earlier artifact deletion. Require both Linux and Windows projected remaining-storage demand plus50GiB reserve.
- Preserve main/existing environments. Environment/code setup excluded from5hour runtime; extraction model loading, real profiling, IO/inference/writes/validation included. Training initialization/smokes charged to18hour clock.
- Keep existing Qwen policy/human_review_status; no new captions or appearance gate. Bind exact source/lock/hardware/feature/timing identities; changes stop recovery.

## Review Focus

1. Same filenames but changed source bytes: hash the bytes decoded by extractor and compare spatial receipt source_hash; fail before publishing corresponding clip (Task1/2).
2. Correct counts but permuted rows or synthetic frames mislabeled real: validate explicit actual-input indices per row and shortclip support/centre (Task1/2).
3. Old baseline proof accidentally identifies new features: validate old lineage first and bind actual new views/config/package separately under opt-in policies (Task3).
4. Partial/corrupt extraction triggers training or duplicate coordinator launches: complete-marker byte inventory, owned locks and a prelaunch revalidation gate (Task4).
5. Crash/disconnect resets spend or orphan GPU work overlaps next stage: durable uncapped elapsed/PID+boot identity and single owned watchdog per child; independent phase clocks (Task2/4).

## Files and responsibility boundaries

- Create `tools/motion/pyproject.toml`, `tools/motion/uv.lock`, `tools/motion/README.md`, `tools/motion/__init__.py`: isolated dependency recipe and operation docs.
- Create `tools/motion/records.py`: extraction recipe/identity, input frame windows, actual source hashes and per-clip receipt schema.
- Create `tools/motion/package.py`: full complete-package publication/validation and factual timing evidence.
- Create `tools/motion/extractor.py`, `scripts/extract_motion_features.py`: real model/processor IO, owned durable extraction clock and CLI.
- Create `src/despamo/data/new_motion.py`: explicit validated new-package→Phoenix view construction.
- Modify only new fusion `scripts/train_fusion_study.py`, `scripts/run_fusion_study.py`: opt-in package/policy/profile/CLI/source closure, no earlier runner edits.
- Create `scripts/run_motion_fusion_pipeline.py`: automatic coordinator and separate budgets.
- Create focused tests `tests/unit/data/test_new_motion_package.py`, `tools/motion/tests/test_motion_extractor.py`, `tests/unit/pilot/test_new_motion_fusion.py`, `tests/unit/pilot/test_motion_fusion_pipeline.py`; run real PNG/processor extractor tests in dedicated environment, keeping Pillow absent from unchanged main. Extend existing fusion tests only where interface defaults need regression coverage.
- Update `docs/motion-reextraction-estimate-2026-10-05.md` and fusion progress/results documents with actual stage receipts, never fabricate pending results.

Default pipeline root `/home/kan/datasets/despamo/signer-pilot/motion-fusion-pipeline-v1`; cache root under pilot `motion-features-<feature_identity_hash>`; new training controller root `time-aligned-fusion-new-motion-study-v1`. Generated arrays/profiles/checkpoints/records external to Git. Main checkout remains uncommitted; preserve previous plan's SDD ledger and create a new plan-scoped ledger.

### Task1: Provenance, window records and complete-package validator

**Files:** Create `tools/motion/__init__.py`, `tools/motion/records.py`, `tools/motion/package.py`, `tests/unit/data/test_new_motion_package.py`.

**Interfaces:** `window_rows(frame_count: int) -> tuple[dict, ...]`; `source_identity(paths: tuple[Path, ...], frame_sha256: tuple[str, ...]) -> str`; `make_motion_identity(inputs: dict, model: dict, processor: dict, environment: dict, code: dict, hardware: dict) -> dict`; `make_clip_receipt(clip_id: str, spatial_row: dict, source_rows: list[dict], windows: tuple[dict, ...], array_path: Path, identity: dict) -> dict`; `publish_motion_package(root: Path, identity: dict, frame_rows: dict, annotation: Path) -> dict`; `validate_motion_package(root: Path, expected_spatial: dict, *, verify_features: bool = True) -> dict`.

The validated package returned by publisher/validator has exact consumer keys: `package_policy='despamo-motion-features-v1'`, `root` (absolute string), `identity`, `identity_hash` (canonical digest), `manifest_path/manifest_hash`, `timing_path/timing_hash`, `annotation_path/annotation_hash`, `spatial_frame_rows_path/spatial_frame_rows_hash`, `completion/completion_hash`, `clip_ids` (ordered list), and `bindings` (absolute regular-file path→rawSHA256 for every completion/identity/manifest/timing/provenance/source ledger/feature/receipt/input/code/lock evidence file). Paths/hashes must be independently checked, never trusted because this dictionary claims them. Source-row entries contain `index/path/sha256`; window-row entries `row/input_source_indices/real_start/real_end/centre`. Per-clip receipt also binds clip_id, feature key, source_hash, frame_count, full source_indices, actual window rows, array shape/dtype/hash and source rows. This schema travels unchanged to Tasks3/4.

- [ ] Write RED tests `test_windows_preserve_actual_input_order`: N25 gives rows input0..15 and8..23; source24uncovered, centres7.5/15.5; N10 one16input sequence0..9 plus six repeated9, real support0..9 and centre7.5. Swapped rows/input ordinals fail validation.
- [ ] Write `test_source_bytes_match_spatial_receipt`: digest must match existing `tools/dinov3/e3_extract.py` source rule, canonical list of(path.as_posix(),sha256) pairs in recorded order. Changed bytes with unchanged filenames reject. Test per-clip IDs/source/frame counts/hash joins and malformed feature widths/dtypes/non-finite values.
- [ ] Write `test_complete_marker_requires_all_bytes`: missing arrays/receipts/marker, recomputed selfhash with foreign row/identity, aliased paths, incomplete count and changed manifest/timing/feature bytes all reject; synthetic complete fixture passes. Tests accept fixture expected IDs, production fixed7096/93232 remains enforced by real admission.
- [ ] Run `PYTHONPATH=src:. .venv/bin/pytest tests/unit/data/test_new_motion_package.py -q`; confirm missing implementation failures.
- [ ] Implement deterministic recipe/content key excluding elapsed time/cap, exact source/frame metadata and actual model-input row mapping. Save finite float32 `[M,1024]` arrays through neutral atomic helpers; helper never infers row order from count alone.
- [ ] Publish `complete/identity.json`, `manifest.json`, `source-window-records.json`, `extraction-provenance.json`, `motion-timing.json`, then `complete/completion.json` binding every artifact/feature/receipt. Manifest records physical `train/<clip_id>.npy` for all7096; root filenames/time cannot stand in for hashes.
- [ ] Generating evidence origin `new_extraction_v1`, status `original_receipt` refers to this actual run only. Construct existing Task1 timing schema from factual receipts, with exact raw/canonical frame_rows/manifest/annotation/extractor/helper/evidence hashes and all chronology claims. Validate real generator records first; do not promote an arbitrary claimed evidence string to admission.
- [ ] Rerun package tests/Ruff; self-review safe root ownership and no borrowed historical claims. Review before Task2.

### Task2: Isolated extractor, profiling and bounded resumable execution

**Files:** Create `tools/motion/pyproject.toml`, `uv.lock`, `README.md`, `tools/motion/extractor.py`, `scripts/extract_motion_features.py`, `tools/motion/tests/test_motion_extractor.py`.

**Interfaces:** `load_encoder(snapshot: Path) -> tuple[object, object]`; `extract_motion_clip(paths: tuple[Path, ...], model: object, processor: object, *, batch_size: int = 16) -> tuple[np.ndarray, list[dict], tuple[dict, ...]]` (array, actual consumed source rows, input windows); `run_motion_extraction(args) -> dict`. CLI `--mode preflight|profile|run`, `--adapted-root`, `--annotation`, `--model-snapshot`, `--output-base`, `--environment-lock`, `--gpu-cap-seconds 18000`, `--authorize-motion-extraction`, optional `--resume`.

- [ ] Define and create isolated Python3.11 environment using exact torch distribution2.0.1 (assert runtime2.0.1+cu117/CUDA11.7), transformers4.32.0, numpy1.26.4, pillow11.3.0, safetensors0.8.0, huggingface-hub0.36.2, test dependency pytest8.4.2. Resolve full dependencies into dedicated uv.lock, consult current uv docs when issuing setup commands. Reuse package/model caches; no main environment edits or runtime sys.path bridge. Check imports/versions before writing extractor tests; this setup is not a GPU stage.
- [ ] Write RED tests using a recording fake encoder and real small PNGs: capture actual incoming16frames/window/batch order, input hash/decode identity, float32 first-hidden-token rule, bad outputs rejected and source-byte changes before inference stop publication. Keep these dedicated tests outside main testpaths so preserving its Pillow-free environment cannot silently skip or fail extractor verification.
- [ ] Write interruption tests after array-before-receipt and between clip commits: recompute only owned incomplete clip, reuse verified completed clips, corrupt completed evidence rejects. Durable cap identity cannot increase/reset after reboot/resume; cap/exception prevents completion marker.
- [ ] Run `PYTHONPATH=src:. tools/motion/.venv/bin/python -m pytest tools/motion/tests/test_motion_extractor.py -q`; confirm expected RED failures.
- [ ] Implement FP32 eval/inference, window batches<=16, fixed processor parameters, explicit TF32 flags and NumPy stack conversion. Read/hash/decode same per-frame bytes once per clip; compare aggregate source identity with immutable spatial receipt before feature publication.
- [ ] New owned extraction ledger binds feature key/cap/source/lock/boot/PID and accumulates wall time during model load/profile/run/writes/validation. Profile mode shares the same cumulative budget with run; valid profile outputs may be reused, not charged twice or falsely called full completion. Cooperative deadline plus parent watchdog, no independent conflicting timer owner.
- [ ] CLI preflight CPU-only/no publication. Real profile samples deterministic clip-length quantiles plus short/long shape checks, tests standard/optimized tensor parity for batch1/8/16, records actual full IO/write/hash elapsed/memory. Admission projects remaining workload within18000seconds; otherwise stop incomplete without raising cap.
- [ ] Rerun dedicated focused tests/Ruff/types, record actual version/lock provenance and verify the main environment remains unchanged. No full extraction until pipeline integration/final review.

### Task3: Explicit opt-in new-motion fusion admission

**Files:** Create `src/despamo/data/new_motion.py`, `tests/unit/pilot/test_new_motion_fusion.py`; modify `scripts/train_fusion_study.py`, `scripts/run_fusion_study.py` and existing focused tests where needed.

**Interfaces:** `load_new_motion_views(original_views: dict, package: dict, root: Path) -> dict`; add optional `--motion-package PATH` to runner/controller. `prepare_fusion`, `initialize_fusion`, `stage_command`, selection validators retain legacy signatures/default behavior; extend optional keyword/path only when required. New-mode run policy `signer-pilot-time-aligned-fusion-new-motion-v1`, controller `time-aligned-fusion-new-motion-controller-v1`, distinct common-profile policy/key. Never monkeypatch globals for mode changes.

- [ ] Write RED tests `test_legacy_motion_mode_unchanged`, `test_old_lineage_new_actual_views_separate`, `test_reject_foreign_source_package`, `test_new_motion_identity_cannot_adopt_legacy_run`. Original root/manifest validate frozen baseline; new config/view roots match independently admitted package, ordered logical IDs unchanged. Corrupt source hash/window/feature or missing marker rejects before model/optimizer creation.
- [ ] Run `PYTHONPATH=src:. .venv/bin/pytest tests/unit/pilot/test_new_motion_fusion.py -q`; confirm missing opt-in behavior fails.
- [ ] Preserve original `_prepare_projector_inputs`/frozen protocol admission using original input paths. Validate Task1 package independently, construct new Phoenix14T parent/Signer views with same annotations/spatial features and new motion manifest/root; do not alter original shared objects or claim original protocol motion hash identifies new bytes.
- [ ] `load_new_motion_views` obtains annotation from validated `package['annotation_path']` and motion manifest from `package['manifest_path']`; compare annotation bytes and records against original parent before construction. Existing Phoenix14T parent does not retain annotation_path, so do not invent that attribute or silently use an unbound caller path.
- [ ] Resolve new-mode config, identity/profile and output keys explicitly; bind original lineage plus actual new package/timing/arrays/receipts and generator environment/source locks. Actual new timing is authoritative; conflicting `--motion-timing` argument rejects. Both seed common initialization stays matched; all original recipe/optimizer/BF16/mask/callback behavior unchanged.
- [ ] Change policy validation locally to accept explicit selected mode, never broad enum acceptance without matching args/package/identity. Old profile/run policies remain separate; new source closure includes extraction/admission/pipeline files before production profile publication. Existing v1 fixtures continue to pass.
- [ ] Propagate opt-in package through all initialize/preflight/smoke/run/recovery/completion queries and expected identities. Preserve final-summary source revalidation and parameter/mask/peak/stat measurements. Changed completion/package bytes abort later arms or final publication.
- [ ] Rerun new-motion and existing fusion runner/controller tests, Ruff/types. Independent task review before automatic coordinator.

### Task4: Durable automatic extraction-to-training coordinator

**Files:** Create `scripts/run_motion_fusion_pipeline.py`, `tests/unit/pilot/test_motion_fusion_pipeline.py`; minimal new-mode controller invocation plumbing only if required.

**Interfaces:** `pipeline_identity(args) -> dict`; `validate_extraction_handoff(cache_root: Path, expected: dict) -> dict`; `MotionFusionPipeline(root: Path, identity: dict)` with `run() -> dict`; CLI `--pipeline-root`, `--adapted-root`, `--annotation`, `--model-snapshot`, `--extractor-python`, `--environment-lock`, `--authorize-motion-fusion-pipeline`, optional `--resume`. Fixed caps18000/64800, no CLI to exceed them.

- [ ] Write RED tests successful complete package launches exactly one new-mode controller with matching package/timing paths and source identity; extraction exit0 without valid marker fails; extraction failure/cap/corrupt arrays/foreign receipts prevents training dispatch; premature or duplicate launch rejected.
- [ ] Write restart/ownership tests for each transition (launching→running→validated→training→complete), client disconnection, PID reuse/boot change, owned orphan handling, incumbent lock preservation and live source/hardware drift. Training final summary artifact changes reject completion rather than publish fresh unbound hashes.
- [ ] Write budget/storage tests: extraction spend cannot reset or exceed18000, training cumulative spend cannot reset/exceed64800, unused extraction allowance cannot transfer; independent Windows/Linux projected remaining demand plus50GiB reserve; no GPU overlap.
- [ ] Run `PYTHONPATH=src:. .venv/bin/pytest tests/unit/pilot/test_motion_fusion_pipeline.py -q`; confirm expected RED.
- [ ] Implement explicit durable pipeline phases/identities and child process groups using neutral watchdog/PID/stop primitives. Extraction parent watchdog bounds remaining18000 while child ledger persists clip charges. Training delegates run ownership to existing new-mode controller/two32400phases; pipeline tracks whole training-child cumulative elapsed conservatively, using max(outer elapsed, validated inner total), not sum/double count. Its deadline also limits remaining64800 without resetting inner seed/run caps.
- [ ] Reserve30seconds of remaining stage allowance for termination/reconciliation. On timeout/cancellation, stop the coordinator child and every verified owned descendant GPU process group identified by boot/PID/start-tick receipts; controller grandchildren may have separate sessions. Reconcile actual uncapped elapsed before returning incomplete. Add a real small-process regression showing a separately sessioned owned grandchild cannot outlive parent cap handling; never signal unowned groups.
- [ ] Pipeline never waits for the agent to return after extraction: one external supervisor performs validation and launch. Handoff revalidates all source/package/marker bytes under owned locks, records extraction completion and training command/identity before launch, and rechecks source/hardware/clock/disk. Resume adopts existing owned running child or stops/reconciles it before another launch; no duplicate work or transferred cap.
- [ ] Route real extraction profile and fullrun through same extraction cap; then run full package/timing validation before invoking new fusion controller. Training controller runs both-seed profile and ordinary/long gradient smokes under its training clock before any fit. Any failure/cap leaves incomplete state and retained recovery artifacts.
- [ ] Final complete pipeline receipt binds extraction package and final six-run summary/selected reports plus actual separate stage clocks<=18000/64800. Rerun focused suites/Ruff/types and independent review; no launch before Task5 checks.

### Task5: Whole-change verification, accounted smoke and automatic launch

**Files:** Update operation README, estimate, fusion progress/results and create `docs/motion-fusion-pipeline-status-2026-10-05.md`; no historical-result rewrites.

- [ ] Run smallest affected tests, then full `PYTHONPATH=src:. LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/pytest tests/unit -q` and separate `PYTHONPATH=src:. tools/motion/.venv/bin/python -m pytest tools/motion/tests -q`. Run configured Ruff extended to `tools/motion`, targeted Pyright using the appropriate main/extractor interpreter, explicit untracked-file whitespace and Git diff/status. Main baseline most recently1358passed/27existingwarnings; distinguish new failures and report both environments' real test evidence without skipped extractor claims.
- [ ] Verify dedicated extractor versions, offline cached config/weights and processor source hashes, current GPU identity/idleness, original spatial/source integrity and both disks' projected5+18h storage admission. No source changes after freezing production identities.
- [ ] Conduct fresh independent whole-change review after task reviews; combine all findings into one fix wave and scoped re-review. Review actual automatic transitions, new/old identity separation, durable source/budget proofs and factual generating receipts. Preserve plan ledgers/review artifacts because no commits authorized.
- [ ] After user confirms this implementation plan, launch owned detached coordinator from project root: `PYTHONPATH=src:. LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python scripts/run_motion_fusion_pipeline.py --pipeline-root /home/kan/datasets/despamo/signer-pilot/motion-fusion-pipeline-v1 --adapted-root /home/kan/datasets/despamo/signer-pilot/e3-features-7122b42e35510611626c2a71d0731b79dd79b20de6e89bd49f62737c5c84a18e --annotation /home/kan/Research/SpaMo/preprocess/Phoenix14T/train_info_ml.npy --model-snapshot /home/kan/datasets/spamo/hf_cache/models--MCG-NJU--videomae-large/snapshots/12da269a02d3e1fbbb7011e610e591fea8061dca --extractor-python /home/kan/Research/DeSpaMo/tools/motion/.venv/bin/python --environment-lock /home/kan/Research/DeSpaMo/tools/motion/uv.lock --authorize-motion-fusion-pipeline`. Capture log/PID/boot/source receipts through owned supervisor, not an untracked nohup promise.
- [ ] Observe initial real profile/parity/source receipts and stage status; complete successful extraction validation automatically starts training without another user prompt. If infeasible/failed/capped, report exact failure and remaining charges; no hidden rerun or new cap.
- [ ] Verify final7096/93232package and new source-window provenance, actual extraction time/storage, six selected checkpoints/all18dev reports, ordered582IDs and separate spent totals. Report both seeds' reference/fixed/learned metrics/gates/peaks with two-seed/dev-only limits; no Signer07 scoring, commits or cleanup.

## Self-review and handoff

Spec coverage: Task1content/source/window receipts and complete publication; Task2FP32recipe/environment/profiling/recovery; Task3oldproof/newinput separation and matched training; Task4automatic validated handoff/separate budgets/ownership; Task5verification and approved execution. Five Review Focus risks have concrete tests. API names/tuple order and feature-vs-runtime identities are explicit; historical cached receipts are never fabricated. Environment/recipe changes affect only new extraction and opt-in identities; current historical-motion mode remains guarded.

User selected subagent-driven execution and current checkout already. No implementation for this extension has begun. Review this plan and confirm it captures the requested automatic extraction→training chain; then proceed continuously with that preserved execution method, respecting stop-on-failure/caps.
