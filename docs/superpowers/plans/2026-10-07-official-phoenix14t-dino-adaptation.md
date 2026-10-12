# Official PHOENIX14T Merged-Target DINO Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `executing-plans` for native,
> task-by-task execution and self-review. The user previously requested direct
> implementation/self-review; do not delegate. Steps use checkbox syntax.

**Goal:** Prepare approved merged targets, train one shared 40-epoch DINO
adaptation, publish ten-epoch checkpoints, and verify a B: archive.

**Architecture:** Add a separately versioned official-split entrypoint. Reuse
unchanged DINO LoRA/loss helpers and sampler, while adding strict merged target
admission, independent group schedules, owned checkpoints, and cold archival.

**Tech Stack:** Main `.venv` for CLIP text preparation; existing
`tools/dinov3/.venv` for DINO training; JSON/dataclasses, NumPy, PyTorch, pinned
Transformers, existing fsync/flock utilities. No dependency or lock changes.

**Spec:** [Approved design](../specs/2026-10-07-official-phoenix14t-dino-adaptation-design.md).

**Scoped progress, 2026-10-07:** Tasks 1–2 completed in the user-authorized isolated
worktree while fusion training continues. See
[preparation/verification report](../../merged-target-preparation-2026-10-07.md).
Code is uncommitted and not integrated into the active checkout. That initial
batch left Tasks 3–7 pending; no DINO training or archival/deletion was launched.

**Continuation:** Tasks 3–4 are also complete after the user's "do it" instruction;
see [data/schedule verification report](../../official-dino-data-schedules-2026-10-07.md).
Dedicated DINO suite: 231 passed, 1 skipped; real all-clip admission and a physical
16-clip CPU batch passed. Tasks 5–7 remain pending in the same isolated worktree.

**Approved amendment:** batch24 is the production default after a successful real
profile. See [batch24 approval report](../../dino-batch24-approval-2026-10-07.md).
The explicit JSON recipe and new `OfficialBatchSampler` support it without
changing the legacy sampler. Current epoch/update counts below supersede batch16.

**Authorized execution:** user approved proceeding through training without more
permission questions. Tasks5–7 implemented and verified, real GPU smoke passed,
40-epoch adaptation launched from the isolated worktree. See
[active runbook](../../official-dino-adaptation-runbook.md). Source bytes in that
worktree are now frozen; no commits/merges or native-original deletion occurred.

## Global Constraints

- Training IDs: 7,096 official train clips, including all nine signers; dev 519
  and test 642 are disjoint clip inventories and never supply adaptation targets.
- Stable factors come from v1, including exact old vectors/masks; articulators
  come exclusively from v2 with no backfill. Retain every clip, 19 slots, and
  `human_review_status=not_assessed`; publish new merged policy/version.
- DINO: rank/alpha 8/8 Q/V LoRA in blocks 20–23, fresh seed 0, batch 24, five
  frames, scales 224/448; AdamW (0.9,0.999), eps 1e-8, decay 0.05; peak LRs
  LoRA 1e-5 / heads 3e-5, BF16 forward, FP32 losses, norm clipping 1.0.
- Total T=11,800; head calibration C=1,475; LoRA warmup/GRL ramp W=590;
  rates apply before one-based update k. Preserve the spec's ten loss components.
- Keep four compact model exports at epochs 10/20/30/40, plus latest two full
  resumes. Publication is durable before pruning. Epoch 40 is selected by design.
- Working outputs use Linux-native storage; reserve 50 GiB on Linux and Windows
  C:. Completed copies may archive to B: NTFS, with verified manifest-last receipts.
- No writes to legacy E3 entrypoints, old datasets, captions, caches, or active
  fusion sources. No commits, deletion of original artifacts, installs, test scoring,
  or training launch unless separately requested at execution handoff.
- Confirm the live fusion study is fully finalized and GPU ownership clear before
  implementing/running training changes. Do not shut down/compact WSL while active.

## Review Focus

1. Metadata annotation keys or a matching clip count hiding wrong IDs/frame joins:
   reject duplicates, extras, overlap, wrong signer/path/index, and changed hashes (Tasks 1/3).
2. An old invalid clip or v2 missing crop erasing/filling independent slots: preserve
   the exact per-slot mask contract and old vector bytes (Tasks 1/2).
3. LoRA starting at head peak, calibration weight decay, or GRL becoming a 3.5-epoch
   ramp: test applied update boundaries and gradients, not just counters (Tasks 4/6).
4. Resume/smoke/partial checkpoints losing RNG, refunding replay, or being pruned
   before receipts are durable: reject or recover with exact accounting (Tasks 5/6).
5. B: copy corruption, same-size tampering, missing mount, or old NTFS writer fallback:
   verify exact inventory and both working/backing spaces (Tasks 6/7).

---

## File map

| New file | Responsibility |
| --- | --- |
| `src/despamo/appearance/merged_targets.py` | Pure import/join/merge validation and version identity |
| `src/despamo/appearance/merged_text_features.py` | Pinned offline CLIP encoding and atomic vector cache |
| `scripts/prepare_official_factor_targets.py` | Main-runtime preparation/preflight CLI |
| `tools/dinov3/official_data.py` | Official train/teacher/frame/merged-target admission and batch loading |
| `tools/dinov3/official_schedule.py` | Epoch accounting and separate LR/GRL clocks |
| `tools/dinov3/official_sampler.py` | Batch24 drop-last batching and identity/RNG replay |
| `tools/dinov3/official_checkpoints.py` | Owned native checkpoint and replay validation |
| `tools/dinov3/official_adapt.py` | Production/smoke/preflight loop and ledger |
| `tools/dinov3/official_archive.py` | Completed-run copy/verify/archive publication |
| `configs/adaptation/phoenix14t_dino_merged40.json` | Explicit non-secret input paths and approved recipe |

Tests mirror modules: two under `tests/unit/appearance/`, one CLI test under
`tests/unit/pilot/`, and `test_official_{data,schedule,checkpoints,adapt,archive}.py`
under `tools/dinov3/tests/`. Preserve existing E3 tests unchanged.

### Task 1: Import and merge the approved target sources

**Files:** Create `src/despamo/appearance/merged_targets.py`;
`tests/unit/appearance/test_merged_targets.py`.

**Interfaces:**
- `import_crop_export(path: Path, expected_sha256: str) -> dict`.
- `merge_targets(old_manifest: dict, crop_export: dict, source: dict,
  teacher_rows: dict, official_ids: tuple[str, ...], parents: dict) -> dict`.
- `merged_target_key(document: dict) -> str`.
Returns ordered records with 19 normalized `targets`, per-slot provenance, a new
policy, parent digests, and per-factor validity counts. Retain source labels/bboxes
without legacy-schema coercion or unverified generation claims.

- [ ] **Write failing tests:** `test_v1_stable_v2_articulator_sources` asserts
  first four texts equal v1 and all remaining texts/mask reasons equal v2;
  `test_v2_mask_never_backfills` keeps a null v2 hand despite non-null old hand;
  `test_old_failed_clip_keeps_valid_new_articulators` keeps four masked stable slots
  plus independently valid new slots; shuffled inputs still join by keys.
- [ ] **Add admission failures:** duplicates/extras/missing IDs, wrong source-frame
  count/path/index/signer, changed raw digest, records digest or source NPZ reference;
  do not count a string annotation metadata key as a clip.
- [ ] **Run RED:** `PYTHONPATH=src:. .venv/bin/pytest tests/unit/appearance/test_merged_targets.py -q`.
  Expected: new module/API missing, or named assertions fail.
- [ ] **Implement the three interfaces.** Verify the pinned raw export digest from
  spec and its canonical records digest; normalize by `(clip_id,factor,frame_index)`.
  Add new policy `qwen-stable-v1-crop-articulators-v2-unreviewed-v1`; include both
  parents and source-frame contract in key. `parents` maps crop export, old text
  manifest, and previous source document to validated original paths/raw SHA-256s.
  Never modify previous source documents.
- [ ] **Run GREEN and inspect diff.** All merge tests pass, no legacy input writes.

### Task 2: Publish the pinned merged text-vector cache

**Files:** Create `src/despamo/appearance/merged_text_features.py`,
`scripts/prepare_official_factor_targets.py`,
`tests/unit/appearance/test_merged_text_features.py`, and
`tests/unit/pilot/test_prepare_official_factor_targets.py`.

**Interfaces:**
- `load_pinned_encoder(cache_root: Path) -> TextEncoder` in main runtime only.
- `encode_merged_targets(document: dict, old_text_root: Path, encoder: TextEncoder,
  output_base: Path) -> Path` returns completed manifest path.
- Preparation CLI: `--mode preflight|encode`, `--crop-export`, `--old-manifest`,
  `--source`, `--teacher-root`, `--train-annotation`, `--clip-cache`, `--output-base`.
Output manifest `metadata/records` is consumable by Task 3 and explicitly records
the previous stable encoder metadata, merge key, parents and encoding policy.

- [ ] **Write failing tests:** copied rows `vectors[:4]` are byte-equal to verified
  old rows; masks match old stable/new articulator texts; output shape/dtype are
  `(19,768)/float32` and `(19,)/bool`; every masked row is exactly zero.
- [ ] **Add edge tests:** changed old NPZ hash, wrong width/revision, overlong
  tokenizer sequence, same encoder but changed v2 source, existing different cache,
  failed publication and offline network access. A source changing during snapshot
  copy fails; a changed sealed snapshot fails admission; a later workspace export
  edit cannot retarget an already completed version. No legacy `Record` parsing of v2.
- [ ] **Run RED:** `PYTHONPATH=src:. .venv/bin/pytest tests/unit/appearance/test_merged_text_features.py tests/unit/pilot/test_prepare_official_factor_targets.py -q`.
- [ ] **Implement offline load and cache writer.** Use CLIP/tokenizer revision
  `32bd64288804d66eefd0ccbe215aa642df71cc41`, width 768, `local_files_only=True`.
  Call `TextEncoder.encode()` on sorted unique non-null articulator strings in
  CPU/FP32 batches of 64. Validate model token limit with no truncation. Copy
  stable vectors; atomic/fsynced NPZ publication precedes manifest/completion.
  Key includes parents, frames, model/tokenizer, width/dtype/device and batching.
  Snapshot the three parent JSON inputs under `inputs/` with verified byte hashes;
  bind original and snapshot-relative paths. Do not duplicate old videos/NPZ caches.
- [ ] **Run GREEN and inspect diff.** Idempotent admission verifies file hashes
  and content; prep preflight never generates vectors or overwrites inputs.

### Task 3: Admit official-split adaptation data

**Files:** Create `tools/dinov3/official_data.py`,
`tools/dinov3/tests/test_official_data.py`.

**Interfaces:**
- `admit_official_data(train_annotation: Path, dev_annotation: Path,
  test_annotation: Path, source_path: Path, merged_manifest: Path,
  teacher_root: Path, frames_root: Path) -> dict`.
- `load_official_batch(ids: tuple[str, ...], admission: dict) -> dict`.
Admission binds hashes and paths, exact ordered train IDs, and source/teacher/text
metadata. Batches retain E3 keys: `images224/images448`, `teacher`, `vectors`,
`masks`, `labels`, `ids`; widths/shapes are unchanged.

- [ ] **Write failing tests:** integer-key rows yield 7,096/519/642 counts; train
  includes all nine signers; no train/dev/test ID overlap; training batch accepts
  official-train Signer03 and Signer07 but rejects a dev/test clip of any signer.
- [ ] **Add failure tests:** source/teacher/target mismatch, path escape, incorrect
  sample indices, invalid zero masks, changed PNG or NPZ bytes, malformed shape or
  dtype. A masked articulator cannot discard its clip or independent stable targets.
- [ ] **Run RED:** `PYTHONPATH=src:. tools/dinov3/.venv/bin/python -m pytest tools/dinov3/tests/test_official_data.py -q`.
- [ ] **Implement admission and loading.** Reuse `frames.preprocess/sample_indices`
  and hash utilities. Match full teacher rows `0..N-1`, then gather the same five
  source frames at both resolutions. Validate local PNG/teacher/NPZ hashes and
  masks. Do not call the legacy signer-pilot `_inputs` or `load_e3_batch`.
  DINO-side module uses stdlib/NumPy/Torch; it must not import the main-runtime
  appearance/Pydantic stack just to read a completed manifest.
- [ ] **Run GREEN and inspect diff.** Legacy E3 source tests still retain signer
  exclusion; the new path admits only official train IDs.

### Task 4: Implement independent head, LoRA and GRL schedules

**Files:** Create `tools/dinov3/official_schedule.py`,
`tools/dinov3/tests/test_official_schedule.py`.

**Interfaces:** Frozen dataclass `AdaptSchedule(train_clips=7096,batch_size=24,
epochs=40,calibration_epochs=5,lora_warmup_epochs=2,grl_ramp_epochs=2,
checkpoint_epochs=10)` with derived `steps_per_epoch,total_steps,calibration_steps,
lora_warmup_steps,grl_ramp_steps,milestone_steps`. Functions
`head_factor(k: int, schedule: AdaptSchedule) -> float`,
`lora_factor(k: int, schedule: AdaptSchedule) -> float`,
`grl_factor(k: int, schedule: AdaptSchedule) -> float`.

- [ ] **Write failing boundary tests:** derived values are 295/11,800/1,475/590/590;
  milestones `(2950,5900,8850,11800)`; at k=1,475 head=1, LoRA=GRL=0;
  at k=1,476 LoRA=GRL=`1/590`; at k=2,065 LoRA=GRL=1;
  at k=11,800 head=LoRA=0 and GRL=1. Reject k outside `1..T`.
- [ ] **Add tests:** both post-peak LR sequences decrease; 40 includes calibration;
  epoch-boundary batching drops the final16 clips then reshuffles; serialized
  completed count restores next k+1 without applying the old shared warmup or 10% ramp.
- [ ] **Run RED:** `PYTHONPATH=src:. tools/dinov3/.venv/bin/python -m pytest tools/dinov3/tests/test_official_schedule.py -q`.
- [ ] **Implement spec formulas.** LR functions return multipliers; driver applies
  them before update k. Validate recipe arithmetic and serialize schedule parameters
  with completed/applied update counters in snapshot state.
- [ ] **Run GREEN and inspect diff.** Existing `test_e3_schedule.py` remains unchanged.

Approved sampler addition: `OfficialBatchSampler(ids: tuple[str,...], *, seed=0,
batch_size=24)` in `official_sampler.py` exposes `next_ids/state_dict/load_state_dict`.
It preserves seeded reshuffling/drop-last, binds ordered ID hash and batch size,
and rejects incompatible cursor/update/RNG state. Extend Task4 tests for24 and
config consumption; keep explicit batch16 helper regressions as history/parity.

### Task 5: Publish owned milestone weights and exact resumes

**Files:** Create `tools/dinov3/official_checkpoints.py`,
`tools/dinov3/tests/test_official_checkpoints.py`.

**Interfaces:**
- `publish_milestone(run: Path, state: dict, epoch: int) -> dict` returns model
  and resume receipts after durable publication.
- `validate_official_resume(path: Path, identity: dict, ledger: dict) -> dict`.
- `prune_owned_resumes(run: Path, keep: int = 2) -> list[Path]`.
`state` fields: identity, epoch, completed/applied update, model LoRA dictionary,
heads, optimizer, schedule, sampler, RNG (Python/NumPy/Torch/CUDA), and charged time.

- [ ] **Write failing tests:** each approved milestone emits compact model and
  full-resume files/receipts with exact epoch/step; base frozen parameters are absent;
  model receipt hash identifies the retained 40-epoch export.
- [ ] **Add tests:** duplicate/changed publication, wrong owner/identity/step,
  missing RNG or optimizer, bad sampler count, NaN tensors, smoke checkpoint,
  partial file, missing/corrupt receipt, stale resume, or rewound time fails.
  Inject failure before receipt fsync: old resumes must survive. Epoch-30 publication
  retains full resumes 20/30 but all model exports 10/20/30.
- [ ] **Run RED:** `PYTHONPATH=src:. tools/dinov3/.venv/bin/python -m pytest tools/dinov3/tests/test_official_checkpoints.py -q`.
- [ ] **Implement native writer publication.** Reuse `storage.writer`/durability
  and unchanged LoRA state helpers. Validate the canonical owned run/path and
  latest complete snapshot before loading. Publish data and receipts durably before
  pruning only earlier owned full resumes. Keep every model export/receipt.
- [ ] **Run GREEN and inspect diff.** Caller cannot signal completion from a
  marker without validating actual snapshot bytes/state.

### Task 6: Integrate the official adaptation loop, ownership and accounting

**Files:** Create `tools/dinov3/official_adapt.py`,
`configs/adaptation/phoenix14t_dino_merged40.json`,
`tools/dinov3/tests/test_official_adapt.py`.

**Interfaces:** `compute_official_losses(model, heads, batch: dict, k: int,
schedule: AdaptSchedule, relational_weight: float = 0.10) -> dict`;
`train_official_update(model, heads, optimizer, batch: dict, k: int,
schedule: AdaptSchedule, relational_weight: float = 0.10) -> dict`;
`run_official_adaptation(config: dict, mode: str, resume: Path | None = None) -> dict`.
`require_official_storage(root: Path, remaining_bytes: int,
reserve_bytes: int = 50*1024**3, windows_drive: str = "C") -> dict`
checks native output admission and both Linux/Windows projected demand.
CLI: `--config`, `--mode preflight|smoke|run`, `--resume-checkpoint` optional,
`--authorize-adaptation` required for GPU modes. Config contains the spec's source
paths, resolved existing cache roots, output base, seed/recipe/losses/schedules,
`duration_limit_seconds: null`, reserve bytes, and archive base. No secret keys.

- [ ] **Write failing loss/gradient tests:** calibration includes only seven
  weighted factors and leaves LoRA/base gradients absent and parameters unchanged,
  including AdamW decay. Joint loss equals `0.05*sum(nuisances)+0.10*sum(articulators)
  +0.10*reference+0.10*pattern+0.10*spread`. Teacher gradients remain absent;
  GRL reverses nuisance encoder gradients but not head or articulator gradients.
- [ ] **Add integration tests:** same seed/data produces same initial LoRA/head
  state; interrupted tiny training restored from a milestone matches uninterrupted
  next batches/LRs/GRL/parameters/RNG. Invoke the same `train_official_update` kernel
  with smaller CPU schedules; production CLI rejects changed approved recipe and
  any fixture/smoke resume. No public test-bypass flag. Replay uses old charged floor; no checkpoint
  before epoch 10 restarts initialization within the same identity and retained ledger.
- [ ] **Add ownership/resource tests:** live/mismatched boot/start ticks or locked
  writer, non-native working destination, either disk below projected demand plus 50 GiB,
  changed source after preflight, non-finite losses/gradients, or corrupt resume
  cannot launch/adopt/complete. No model download or dependency update is attempted.
- [ ] **Run RED:** `PYTHONPATH=src:. tools/dinov3/.venv/bin/python -m pytest tools/dinov3/tests/test_official_adapt.py -q`.
- [ ] **Implement integration.** Use Task 3 admission, `OfficialBatchSampler`, Task 4
  pre-update clocks, unchanged model/loss/state helpers, and Task 5 publication.
  Fresh pretrained DINO and seed 0 initialize production. Apply both group rates
  in `train_official_update` before k, encoder grad only after C, FP32 losses,
  gradient clipping and finite checks, then optimizer step. Name the two optimizer
  groups `lora` and `heads` so rate assignment is explicit. Record heartbeat at least every 60 seconds, owned
  process identity, actual steps, individual losses, LRs, GRL and elapsed time.
  Preserve stopped/replayed charges and validate inputs during load/use and final
  publication. Stop at T; explicitly select epoch 40 after all model receipts verify.
  Finalize/close the ledger, then seal `completion.json.payload_files` with run-relative
  sizes/hashes of immutable exports/resumes/receipts, identity/input/runtime records,
  closed budget and selection. Exclude the manifest itself, writer locks, scratch,
  and future `archive-receipts/`. No payload writes occur after completion publication.
- [ ] **Implement separate preflight/smoke paths.** Preflight is read-only. Smoke
  uses disposable real-batch models at calibration/joint indices, has a distinct
  purpose/identity, reports finite losses/gradients and measured snapshot demand,
  and writes no production resumes. Carry its charged time once into production.
- [ ] **Run GREEN and static checks.** Then, only after the fusion study is
  finalized and hardware idle, run actual-data preflight and smoke for plan review.
  Do not launch the production 40-epoch job as part of implementing this task.

### Task 7: Verify B: archive and document execution handoff

**Files:** Create `tools/dinov3/official_archive.py`,
`tools/dinov3/tests/test_official_archive.py`,
`docs/official-dino-adaptation-runbook.md`.

**Interfaces:** `archive_completed_run(run: Path, archive_base: Path) -> Path`
returns verified archive manifest path;
`verify_official_archive(archive: Path, source_identity: dict) -> dict`.
`require_archive_storage(archive_base: Path, needed_bytes: int,
reserve_bytes: int = 50*1024**3) -> dict` verifies an actual B: mount and
Windows B: free space before staging.
CLI: `--run`, `--archive-base`, `--mode copy|verify`. No delete/move flag.

- [ ] **Write failing tests:** incomplete adaptation cannot archive; full inventory
  copies with matching sizes/hashes; marker appears last; same-size tampering,
  unexpected/missing files, symlinks/path escapes, interrupted copy or missing B:
  mount fail. Unsupported directory fsync records a verified-copy durability level,
  not crash durability; native archive receipt is fsynced, and every later archive
  admission rehashes bytes. Native bookkeeping added after archival cannot alter
  retry inventory. Retry verifies exact content or fails without overwriting.
- [ ] **Run RED:** `PYTHONPATH=src:. tools/dinov3/.venv/bin/python -m pytest tools/dinov3/tests/test_official_archive.py -q`.
- [ ] **Implement cold copy/verify.** Check B: mount and space before staging.
  Copy only completed run artifacts; preserve source identities as historical
  bytes, with separate original/archive-relative path mapping and selected hash.
  Use Task 6's sealed payload plus `completion.json` (bind its raw hash separately),
  not a recursive glob that later absorbs mutable archive bookkeeping.
  Verify all staged bytes, file-fsync, publish manifest/completion last, and freshly
  reopen/recheck finished inventory. Record directory-fsync support and persist
  the archive receipt under native `archive-receipts/` after verification. Do not route B:
  publication through `storage.writer`, which correctly rejects NTFS. No marker-only
  archive admission. Active output writers still reject NTFS.
- [ ] **Run GREEN.** Originals remain usable; no registry edits or removal.
- [ ] **Write runbook:** separate main/DINO runtimes, target-prep CLI, native
  working and B: archive paths, exact recipe/milestones, ownership/resume procedure,
  diagnostic interpretation, and verified archive restoration for future training.
  Note that freeing Linux space may require later offline VHDX compaction to
  return Windows C: space; never perform it during active training.

## Verification and execution handoff

Run focused RED/GREEN checks in each task before broader checks. After all tasks:

```bash
PYTHONPATH=src:. .venv/bin/pytest tests/unit/appearance/test_merged_targets.py tests/unit/appearance/test_merged_text_features.py tests/unit/pilot/test_prepare_official_factor_targets.py -q
PYTHONPATH=src:. tools/dinov3/.venv/bin/python -m pytest tools/dinov3/tests -q
.venv/bin/ruff check src/despamo/appearance/merged_targets.py src/despamo/appearance/merged_text_features.py scripts/prepare_official_factor_targets.py tools/dinov3/official_*.py tests/unit/appearance/test_merged*.py tests/unit/pilot/test_prepare_official_factor_targets.py tools/dinov3/tests/test_official_*.py
pyright --pythonpath .venv/bin/python src/despamo/appearance/merged_targets.py src/despamo/appearance/merged_text_features.py scripts/prepare_official_factor_targets.py
pyright --pythonpath tools/dinov3/.venv/bin/python tools/dinov3/official_*.py
git diff --check
git status --short
```

Use `verification-before-completion`, native self-review, and final intended-file
diff inspection. Do not rerun expensive checks without new failures/changes.
Record actual results; no passing claim is implied by this plan. Production
launch, full feature extraction, six translation runs, and deletion of native
artifacts require their respective execution/cleanup handoffs. No commits are
included in this plan because none were requested.
