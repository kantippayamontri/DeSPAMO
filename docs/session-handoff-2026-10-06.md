# DeSpaMo Session Handoff — 2026-10-06

**Superseding recovery:** [October 7 time-waiver record](fusion-time-waiver-2026-10-07.md).
Seed0 reference subsequently stopped at 13,000 steps; latest durable resume was
12,500. User waived training duration stops and requested immediate launch.
The active study root is unchanged. Recheck live state before acting.

Detailed historical snapshot moved from project `AGENTS.md`. For current operating
instructions, read [AGENTS.md](../AGENTS.md). All progress, process identities,
disk observations and verification counts below describe the recorded snapshot;
live ledgers take precedence. Earlier [project handoff](project-handoff.md) remains
preserved as history.

## Latest Fresh-Session Handoff — 2026-10-06

Read this section first when resuming. This is a **snapshot**, not live status:
recheck the JSON ledgers and processes below before acting. Historical notes in
`docs/project-handoff.md` predate later E3/held-out work; current receipts and this
handoff take precedence over their old progress claims.

### Current work and live snapshot

- Goal: test **time-aligned spatial/motion fusion** using newly extracted,
  explicitly documented VideoMAE features and frozen cosine-adapted DINO features.
- Extraction **complete: 7,096 clips, 827,354 source frames, 93,232 motion windows**.
  Complete package, actual generating receipts and frame-window timing published.
- **Actual training is running.** At `2026-10-06T13:32:20Z`, Seed0 `reference`
  ledger recorded **3,488 / 16,000 optimizer steps**, phase `train`, no error.
  GPU: RTX4090, driver610.62, approximately12.5GiB resident at observation.
  Seed1 not started; no dev report yet (first full-dev check at step5,500).
- Current run key:
  `fusion-new-motion-reference-seed0-16000-767d35b9821c629afcadda90f7a34e35781d93c193fef81f9c0e19f8a05ea0ac`.
- User approved current checkout and initially subagent-driven implementation;
  later explicitly requested direct implementation/self-review **without time
  estimates**. Do not add forecast-based vetoes or repeated launch approvals for
  this already authorized study. Existing actual hard caps still apply.
- All implementation is uncommitted on `main`; HEAD at handoff
  `ebb0f6bb853eee3d3ec410dddcb386777b7eff2d`. Preserve unrelated user work.

### Authoritative paths: inspect these, not the retired pipeline

All experiment data are external to Git under `/home/kan/datasets/despamo/signer-pilot/`.

| Artifact | Path relative to that pilot root |
|---|---|
| **Active training controller** | `time-aligned-fusion-new-motion-validated-study-v1/` |
| Controller state | Active root `state.json` |
| Per-seed accounting/process records | Active root `seed0/study-state.json`, `seed1/study-state.json` |
| Model/run ledgers and checkpoints | Active root `runs/<run-key>/budget.json`, `run-status.json`, `resume-*.ckpt`, `step-*.ckpt` |
| Shared initialization proof | Active root `common-initialization.json` |
| **Completed new motion cache** | `motion-features-fb35379b1e2ea0c4d19a186cd290952b512eff270f5394540c902f344eae842c/` |
| Complete new-data proof | Cache `complete/identity.json`, `manifest.json`, `completion.json`, `motion-timing.json`, `extraction-provenance.json`, `source-window-records.json` |
| Per-clip generation evidence | Cache `receipts/train/<clip-id>.json`, arrays `train/<clip-id>.npy`, root `generating-run.json` |
| Completed extraction bootstrap | `motion-extraction-bootstrap-measured-profile-v1/` |
| Extraction restart lineage | `motion-fusion-pipeline-v1/measured-profile-recovery.json` |
| Training restart lineage | `time-aligned-fusion-new-motion-study-v1/validation-recovery.json` |

**Retired/interrupted histories:** `motion-fusion-pipeline-v1/` stopped at the
forecast gate; `motion-fusion-pipeline-measured-profile-v1/` finished extraction
but stopped during its original training preflight; `time-aligned-fusion-new-motion-study-v1/`
is that interrupted controller. Their `failed`/`incomplete` states do **not** mean
the active `...validated-study-v1` recovery is stopped. Keep all histories intact.

Status/report entrypoints:
- `docs/motion-fusion-pipeline-status-2026-10-05.md` — overall history and current root.
- `docs/validation-recovery-2026-10-06.md` — validation optimization and actual training restart.
- `docs/measured-profile-self-review-2026-10-06.md` — forecast removal/extraction restart.
- `docs/signer-pilot-protocol.md` — cumulative older experiments; newest progress
  may only be in live ledgers/status documents until a run completes.

### How to check progress or resume safely

1. Read active controller `state.json`, both seed states, and each run's
   `budget.json`: `identity.seed`, `identity.arm`, `optimizer_step`, `phase`,
   `status`, cumulative charges. Confirm process identity and `nvidia-smi`.
2. If running, **observe only**. Do not start another controller/extractor or
   modify source/locks/environment: active profiles/checkpoints bind source bytes.
   Documentation-only updates to project `AGENTS.md`/`docs/` do not change the
   frozen Python/dependency source closure.
3. If interrupted, inspect exact log/owned PID, boot ID, start ticks, locks and
   durable checkpoint before recovery. Never delete status, edit elapsed counters,
   substitute another root, or reuse a completed-looking marker without validation.
4. The active controller command is `scripts/run_fusion_study.py` with
   `--study-root <active-root> --motion-package <completed-cache>` and
   `--motion-timing <cache>/complete/motion-timing.json`,
   `--recovery-record <pilot>/time-aligned-fusion-new-motion-study-v1/validation-recovery.json`,
   `--authorize-fusion-study`. It automatically reconciles owned controller/run
   state; it has **no `--resume` CLI flag**. Individual runner has checkpoint resume.
   Do not run this while current controller is alive.
5. Initial outer supervisor was GNUtimeout61905s with INT/30s kill grace; observed
   PIDs360600(timeout),360601(controller),365601(runner) are **snapshot identities**,
   not permission to signal those numbers later. Recompute remaining actual global
   allowance/original deadline before any relaunch, never restart the full duration.
6. Runtime environment: project `.venv`, `PYTHONPATH=src:.`,
   `LD_LIBRARY_PATH=/usr/lib/wsl/lib`, `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=2`.
   Extractor has separate `tools/motion/.venv`; main intentionally lacks Pillow.

### Approved experiment and budget contract

- Logical split: train5,746 across seven signers; dev582 **Signer03**;
  test768 **Signer07**. All physically PHOENIX `train/`.
- Protocol hash `0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7`;
  split hash `a95fa8f2710ebca33d6565a26f0f076817d839ecc611bc43648a48761fd6527b`.
- Six fresh sequential runs: Seed0 reference/fixed/learned, then Seed1 same.
  Reference is **plain spatial MLP backbone + original sequential fusion**, not
  the earlier linear-projector arm. New motion data are identical across all arms.
- Spatial2048→1024/GELU→768; motion1024→768; original K5/P2/K5/P2 TCN and
  output768→2048/GELU→2048; FLAN-T5-XL LoRA and VT-Align retained.
- Interpolate projected motion by actual window centres onto spatial source
  frames. Add `valid * gate * corrector(motion)` to spatial tokens; append original
  motion tokens afterward. Corrector768→768/GELU→768, final layer zero;
  fixed gate0.5 or learned sigmoid Linear1536→1 initialized0.5. Uncovered tails
  receive no correction. Common weights/RNG matched across arms within each seed.
- Recipe:16,000updates, batch4/no accumulation, VT-only warmup2,000, LR warmup1,600
  then cosine-to-zero; adapter/VT LR3e-4, FLAN LoRA LR5e-5; AdamW betas.9/.98,
  eps1e-8, decay.01, BF16, clip1. Do not change batch because4090 has more VRAM.
- Full582clip dev checks at5,500/10,400/16,000; highest BLEU-4 selects, ties earlier.
  Decoder beam5/max64/noICL. No automatic Signer07 translation evaluation or tuning.
- Extraction cap18,000s/5h; training64,800s/18h; training phase32,400s/seed and
 10,800s/run. Separate clocks, no borrowing/reset. Prior extraction216.610646s
  carried into new extraction; prior training setup **2864.189098s** retained
  against Seed0/reference and global training. Parent/child overlapping charges
  use conservative maximum, not double sum.
- Latest2 owned resume checkpoints every500steps, all dev-selection models kept;
  atomic/fsynced publication before pruning. Do not delete older artifacts again
  without explicit authorization. Check both Linux and Windows free space and
  projected remaining demand plus50GiB reserve. Last free411.69/243.71GiB snapshot.
- No commits, pushes, merges, extra seeds/ablations, new captions or test scoring
  authorized by the ongoing training request.

### New motion provenance and fixes already made

- Original downloaded SpaMo cache had no recovered historical window/row/source
  receipts. Local/recovery/archive searches established code conventions and byte
  equality, not actual historical frame mapping. User chose new documented extraction.
- New cache uses exact recorded210x260 images used by spatial features. Frozen
  `MCG-NJU/videomae-large` revision `12da269a02d3e1fbbb7011e610e591fea8061dca`,
  eval/inference FP32; vector `outputs.hidden_states[-1][:,0]`, not normalized
  `last_hidden_state` or mean pooling. Window16/stride8/full chronological windows;
  repeat-last only for short synthetic inputs, drop incomplete tails.
- RGB/bilinear shortestedge224/centercrop224/rescale1/255/mean/std.5. NumPy stacking
  then torch.from_numpy verified bit-identical at batch1/8/16. Existing main
  torch2.0.1+cu117/transformers4.32; dedicated Pillow11.3/extractor lock/private
  copied packages. Old feature/cache files were never relabeled as new generation.
- Original forecast veto treated first cold one-window cost as every window and
  charged another inference-rate scan for CPU validation. User requested no time
  estimates: profile schema3 `measured-smoke-hard-cap-v1` now checks actual smoke
  evidence while actual deadlines still stop overruns.
- Original training validation serialized5MB identity for every clip, reread PNGs
  twice, retraversed common ancestors and repeated full scans in one transaction.
  Read-only consumer `src/despamo/data/motion_validation.py` reuses identity/ancestor
  work, verifies source/array bytes once, then rechecks FD/file/ancestor signatures.
  In-process admission token may be reused only after mapping/descriptor/metadata
  checks; plain JSON descriptors always scan. Generator source evidence unchanged.
- Runtime robustness fixes: early durable extraction clock before preparation;
  candidate validation before completion marker; caller-admitted terminal proof
  distinct from raw `terminal_pending`; owned orphan/descendant cleanup; watchdog
  termination independent of receipt filesystem IO; length-framed stream IPC;
  controller/seed leases through final publication; completed resume read-only;
  final summary revalidates every source; huge identity CPU queries use owned
  hash-bound request files (not argv; real production JSON exceeded Linux limits).
- Implementation files: `tools/motion/{records,package,extractor,recovery}.py`,
  `scripts/{extract_motion_features,run_motion_fusion_pipeline,train_fusion_study,run_fusion_study}.py`,
  `src/despamo/data/{motion_timing,new_motion,motion_validation}.py`,
  `src/despamo/models/aligned_fusion.py`, `src/despamo/training/validation_recovery.py`.
- Most recent full main verification1624passed/27pre-existing warnings; affected
  recovery suites301passed/4warnings; configured Ruff and targeted Pyright clean.
  Extractor132passed/1known upstream parity warning at last full extraction-fix
  check. Do not rerun broad tests during live training unless a new change needs it.

### Earlier completed research: do not lose these conclusions

- E3 cosine adaptation/extraction finished. Spatial feature key
  `7122b42e35510611626c2a71d0731b79dd79b20de6e89bd49f62737c5c84a18e`;
  adaptation key `e3-adapt-71a3c8c0805813a73c445ca2141b5e8b702933a809c641f483cb5d7e61c3ad8a`,
  encoder checkpoint SHA `5c4d2f62491998512e596ac2cb5f59efc4fab4f3de1054deaa5e04a158f87e30`.
  Frozen E1 dev/test4.0381/3.4238; constant-LR E3 4.4447/3.8554;
  cosine E3 linear8k 4.7099/4.3618. Constant-vs-cosine also changed LRs/decay;
  not a cosine-only ablation. See training audit/protocol, not stale docstrings.
- Probe repaired: matched five frames, cached CPU inputs, explicit PASS/FAIL/
  INCOMPLETE. Both tested adaptations failed heuristic hand/signer gates;
  free-text hand labels/tiny tests make these unvalidated BLEU predictors.
- Residual16k study did not replicate across seeds: held-out linear4.7848/4.9912,
  residual5.9151/3.8499, means4.8880/4.8825. No consistent residual improvement.
  `docs/projector-heldout-robustness-2026-10-04.md` is authoritative summary.
- Dev grounding128clips/64pairs: spatial shuffle strongly hurt BLEU; correct
  motion pairing showed no clear BLEU benefit on that restricted subset, despite
  changed predictions. This motivated fusion; no proven motion-causal claim.
  `docs/dev-visual-grounding-2026-10-04.md`.
- Completed three-arm connector study (linear reference / plain spatial MLP /
  small from-scratch Q-Former replacing output MLP), two seeds/16k each:
  held-out Signer07 reference4.3600/4.5133, plainMLP4.6373/5.6984,
  Q-Former5.1502/4.6551. Means4.4366/5.1678/4.9026. Both candidate point estimates
  improved both seeds, but limited two-seed/one-signer evidence; arm ranks reversed.
  Q-Former Seed1 dev-selected10,400; others16,000. Reference rerun variability
  remains unexplained. Do not compare new-motion recovery directly as fusion-only gain.
  `docs/three-arm-connector-{results,heldout}-2026-10-05.md`.
- Shared `projector-heldout-registry-v1.json` has10complete once-only claims
  (4residual/linear +6connectors),768items each. Preserve reports/journals/claims;
  dev selects, held-out scores must not be regenerated or used for reselection.
- Current time-aligned study has **no selected dev score or Signer07 result yet**
  at the snapshot. Do not fabricate winner, completion, robustness or invariance.
- Other-server closed-vocabulary/cropped hand-mouth descriptions are separate,
  not inputs to this study; new schema/source/text/encoder provenance needed later.
- Optional compact patch pooling is saved but not authorized/launched. Dense
  patch caches would be huge; do not allocate them as an automatic follow-up.

### Storage and durable implementation records

- WSL crashes previously came from Windows VHDX backing-store exhaustion despite
  Linux headroom; inspect `/mnt/c` as well as Linux. User compacted disk and
  installed RTX4090/24GiB. Do not shut down or compact WSL during active training.
- Approved older cleanup removed107.39GiB, then crash recovery removed113.508GiB
  duplicate resumes/pending artifact. Latest fusion audit/explicit cleanup removed
 35listed files/194.42GiB:4redundant completed resumes +31non-selected/non-terminal
  dev models, retaining selected and terminal models and all reports.
  `docs/storage-{cleanup*,recovery*,audit-fusion*}` contain inventories/receipts.
- Source archives, completion/data manifests, budget ledgers and selected models
  are identities, not disposable duplicates. Project symlinks remain in place.
- Specs/plans: `docs/superpowers/{specs,plans}/2026-10-05-time-aligned-fusion*`,
  `2026-10-05-motion-extraction-autotrain*`. Plan-owned implementation/review ledgers
  at `.superpowers/sdd/2026-10-05-time-aligned-fusion/` and
  `.superpowers/sdd/2026-10-05-motion-extraction-autotrain/` are retained and ignored.
  They record completed tasks/fix rounds/rulings and lossless reviewed sources;
  consult before redispatching work, never overwrite sibling/old flat ledgers.

## Qwen Appearance Supervision (2026-09-28 Decision)

- The user does **not require human verification** of LLM-generated appearance descriptions before exploratory DeSpaMo training. Use the versioned `qwen-schema98-unreviewed-v1` policy instead of the older mandatory-human training gate.
- Current Qwen train dataset key: `b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a`. It has 6,977 schema-valid and 119 failed clips of 7,096 (98.323%). The original 99% rule failed; the newer schema floor is 98%.
- Unreviewed training verifies complete dataset/source/model/encoder provenance and the 98% schema floor. Preserve all 7,096 translation clips; all 19 appearance targets for each failed clip remain masked and zero-valued. Schema validity does not prove visual correctness.
- Label such runs, checkpoints, and results `qwen-schema98-unreviewed-v1` with `human_review_status=not_assessed`. The existing 700-row optional `audit/review.csv` is blank: do not fabricate ratings or write an audit pass.
- Keep historical original-99%-failure artifacts unchanged. The audited route remains available if a human later reviews it; it is not a prerequisite for the unreviewed route.
- Start with the cheapest meaningful data-alignment/gradient smoke. Long training or ablations require separate explicit authorization. Do not commit, push, merge, delete prior artifacts, or change frozen caption records without explicit request.

## Workspace Handoff

- OpenCode project root: `/home/kan/Research/DeSpaMo`. Read [project handoff](docs/project-handoff.md) before continuing signer-pilot or recovery work.
- `external-data` and `recovery-worktree` are ignored local links, not copies. Keep their targets in place: published artifact paths and old checkpoint identities depend on them.
- Earlier handoff notes predate E3/probe/Signer07 follow-ups. See the latest handoff above and actual receipts for current status. Do not claim signer invariance from dev BLEU or limited held-out point estimates alone.
