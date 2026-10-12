# Official PHOENIX14T merged-target DINO adaptation design

Date: 2026-10-07. Status: approved conversational requirements, documented for
implementation-plan review. This document does not start implementation or training.

Batch amendment: user approved physical batch **24** after requested profiling.
Its real profile passed at20.002GiB peak reservation, 2.357GiB free memory and
16.885clips/s. This supersedes original batch16 arithmetic; earlier reports retain
their historical measurements. All other approved hyperparameters are unchanged.

## 1. Goal and scope

Train one fresh, seed-0 DIFFER-inspired DINOv3 adaptation on the 7,096 official
PHOENIX14T training clips. Use the resulting epoch-40 encoder as the shared,
frozen spatial backbone for the separately planned six conventional-benchmark
translation runs. The two downstream seeds therefore measure translation-stage
variability conditional on this single encoder, not end-to-end adaptation variance.

This project includes merged target preparation, adaptation, milestone exports,
owned resume state, and verified archival to B:. Full-video feature extraction,
translation training, official-test scoring, migration of old artifacts, VHDX
creation, and WSL compaction are separate work.

## 2. Approved inputs and immutable merge policy

Paths below are existing inputs, not destinations to overwrite:

- Crop export: `artifacts/pheonix_new_version/phoenix14t_qwen_text_targets_v2.json`.
  The spelling `pheonix_new_version` is intentional because it is the actual path.
- Export SHA-256: `73948b9784e23acac61f51cf46d0aec4259f8402faf10be97df40aa8a7e606dc`.
- Export records digest: `8db97d81133a3163bdcc2e2efdb434cd006693f8b2f9dc63c074fac7e3e9b24c`.
- Previous dataset root:
  `/home/kan/datasets/despamo/appearance/datasets/b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a`.
- Previous text manifest under that root:
  `text/24ad915ab5bde47335bfc8839f654b0a54ad66ac1253fc4218e1c415951fa571/manifest.json`.
- Official annotations:
  `/home/kan/Research/SpaMo/preprocess/Phoenix14T/{train,dev,test}_info_ml.npy`.
- Source frames: `/home/kan/datasets/despamo/phoenix-frames-210x260px`.
- Frozen, unadapted DINO teacher root:
  `/home/kan/datasets/despamo/features/1e84e43d142abc6242cce75c3e65abe9e1e03a6c61e3770181d7be381f3bc31d`.

For each clip, the merged target order is four clip-level slots followed by three
articulator slots at each of five sampled frames:

| Factors | Source and masking rule |
| --- | --- |
| biometric, clothing, hair, background | Preserve previous text, vector rows, and masks exactly. |
| left_hand, right_hand, mouth | Use v2 text and masks exclusively. No old articulator backfill. |

Keep every training clip. Valid new articulators remain usable when the previous
clip failed; unavailable stable factors remain masked. All masked vectors are zero.
Never apply the old whole-clip 19-slot failure mask to independently valid new slots.

Join by `(clip_id, factor, frame_index)`, not record position. Require identical
clip membership, signer, frame count, sampled indices, and source-frame paths
against official annotations, previous source records, and teacher row maps.
Read only integer annotation keys: each annotation dictionary also has metadata.
Counts are train 7,096, dev 519, test 642; clip IDs are disjoint, while all nine
signers occur in each official split. Only train images/targets enter adaptation.

The crop export has 7,096 unique records and 19 slots per clip. Verified valid
articulator slots are left 24,305, right 27,618, mouth 34,732; each factor has
35,480 possible slots. The four placeholder stable slots in v2 are always masked.
The exporter-declared source/journal hashes and crop boxes must be retained.
They are asserted parent provenance unless the referenced producer files are also
available and independently verified. Export digest verification and local frame
joins do not establish that descriptions are visually correct.

Publish a new content-addressed merged dataset with policy
`qwen-stable-v1-crop-articulators-v2-unreviewed-v1` and
`human_review_status=not_assessed`. Keep the previous dataset and its policy intact;
its historical 98% whole-record schema criterion is not a merged slot-coverage
threshold. Report source validity and per-factor availability separately.

## 3. Text encoding contract

Use `openai/clip-vit-large-patch14`, model and tokenizer revision
`32bd64288804d66eefd0ccbe215aa642df71cc41`. Width is 768. Run text preparation
in the main `.venv`, separately from the DINO runtime. Load pinned local snapshots
offline; do not call the legacy `TextEncoder.load()` network revision resolver.

Copy the first four verified old NPZ vector rows without re-encoding them. Encode
only non-null v2 articulator strings. Preserve raw strings and exact-string
multi-positive labels. Do not force the new closed-vocabulary label dictionaries
through the incompatible legacy `Record` schema or silently truncate text.
Deduplicate exact strings, sort them, and encode deterministic CPU/FP32 batches of
64 using the existing `TextEncoder.encode()` inference behavior.

Output per clip: float32 `vectors[19,768]`, boolean `valid[19]`, and target metadata.
Cache identity includes both source digests, merge policy, source-frame contract,
pinned encoder/tokenizer revisions, CPU/FP32 encoding and batch policy. It must not
reuse the old cache directory keyed only by encoder metadata. Publication is
atomic and manifest-last; reruns validate existing content instead of overwriting.
Snapshot the raw crop export, old text manifest, and previous source document
under the new version's `inputs/`, verifying byte hashes before and after copying.
Record imported path and snapshot-relative path separately. Train against these
sealed inputs and completed merged vectors; a later edit of the workspace export
must not silently change an existing dataset version. Changed snapshots or actual
frame/teacher/vector files still fail admission. Do not copy full videos or old
NPZ caches merely to preserve the input JSON provenance.

## 4. Adaptation recipe

- Backbone: `facebook/dinov3-vitl16-pretrain-lvd1689m`, revision
  `ea8dc2863c51be0a264bab82070e3e8836b02d51`.
- Fresh adaptation seed 0. Base weights remain frozen; train rank-8/alpha-8 Q/V
  LoRA in zero-based blocks 20–23 and seven factor heads. Keep base evaluation
  behavior; evaluation mode does not disable LoRA autograd.
- Batch 24 clips, five deterministic frames, scales 224/448, concatenated CLS
  width 2,048. Drop the incomplete batch and reshuffle once per epoch.
- AdamW betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.05`.
- Peak head LR `3e-5`; peak LoRA LR `1e-5`.
- BF16 forward; contrastive/geometry calculations FP32; clip head and LoRA
  gradient norms separately at 1.0. Check finite losses and trainable gradients.
- Runtime: existing `tools/dinov3/.venv`/`uv.lock`, Python 3.11,
  torch `2.5.1+cu121`, transformers `4.56.2`, numpy `1.26.4`, pillow `11.3.0`.
- Environment: `PYTHONPATH=src:.`, `LD_LIBRARY_PATH=/usr/lib/wsl/lib`,
  `HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=2`. No installs or lock changes.

Reusable schedule/update/checkpoint helpers may accept smaller CPU fixture
configurations for meaningful replay tests. The production CLI validates this
exact approved recipe and official inventory; fixture/smoke state cannot be
admitted as a production run or resume. Do not add a public bypass flag.

### Exact schedules

40 **total** epochs include five head-only calibration epochs and 35 joint epochs.
With 295 complete batches per epoch: total T=11,800, calibration C=1,475, and
post-unfreeze warmup/ramp W=590. Drop the final16 clips then reshuffle; every clip
remains in the training inventory. Set learning rates before optimizer update k,
where k is one-based. Persist completed updates; resume next at k+1.

- Head multiplier: `k/C` for `1 <= k <= C`; thereafter
  `0.5*(1+cos(pi*(k-C)/(T-C)))`.
- LoRA multiplier: zero through C; with j=k-C, `j/W` for `1 <= j <= W`;
  thereafter `0.5*(1+cos(pi*(j-W)/(T-C-W)))`.
- GRL: zero through C; thereafter `min(j/W,1)`.

Thus head LR reaches peak at 1,475; first joint update is 1,476; LoRA LR and GRL
reach peak/one at 2,065; both LR multipliers reach zero at 11,800. During
calibration, encoder gradients must be absent, so AdamW cannot decay LoRA weights.
This deliberate convention includes a zero-LR final update. Do not substitute
the old shared LambdaLR or its 10%-of-joint-duration GRL ramp.

### All ten loss components

| Joint component | Weight |
| --- | --- |
| biometric, clothing, hair, background (each) | 0.05 |
| left_hand, right_hand, mouth (each) | 0.10 |
| per-scale teacher cosine reference | 0.10 |
| relational pattern | 0.10 |
| relational spread | 0.10 |

Reuse unchanged helpers in `tools/dinov3/e3_model.py`: factor temperature 0.07,
exact-string multi-positive contrast, nuisance-only GRL, detached teachers, and
per-scale direction/pattern/spread preservation. Calibration includes only the
seven factor losses; joint adaptation adds the three teacher terms. Reference
and relational diagnostics may be computed during calibration but are not added.
Missing/one-label factor batches return the existing zero contrast loss. Translation
CE, VT-Align, ReID identity/triplet objectives, and a mouth video encoder are not
part of this adaptation objective.

## 5. Ownership, accounting, and checkpoint contract

Use new policy `official-phoenix14t-merged-dino-adaptation-v1` and run key
`official-dino-adapt-<identity-digest>`. Identity binds input hashes, target policy,
model revisions, recipe, source closure, lock/runtime, and canonical working root.
Old pilot hashes, signer exclusions, budgets, and time-waiver records are not reused.
Termination is the approved 11,800-update horizon plus integrity/resource failures;
record elapsed work without introducing the pilot's wall-clock training stop.

Acquire a Linux writer lock before mutation. Persist owned PID, boot ID, start
ticks, and a heartbeat/accounting receipt at least every 60 seconds. Reject live
or mismatched owners. Resume only the newest complete owned receipt-bound snapshot;
validate schema, identity, tensors, optimizer, schedules, sampler and all RNG states.
Charge unrecorded elapsed time conservatively after unclean stops; never refund
replayed or earlier work. A rollback to a durable sampler/update is explicit and
does not reduce elapsed accounting. Non-finite step failures leave no success receipt.

| Epoch | Completed updates | Publications |
| --- | ---: | --- |
| 10 | 2,950 | `model-epoch-10.pt`, `resume-epoch-10.pt`, receipts |
| 20 | 5,900 | `model-epoch-20.pt`, `resume-epoch-20.pt`, receipts |
| 30 | 8,850 | `model-epoch-30.pt`, `resume-epoch-30.pt`, receipts |
| 40 | 11,800 | `model-epoch-40.pt`, `resume-epoch-40.pt`, receipts |

Compact model exports contain LoRA and factor heads, epoch/update, identity and
pinned base revision/hash, not duplicated frozen DINO parameters. Full resume
snapshots additionally contain optimizer/schedule state, sampler, Python/NumPy/
Torch/CUDA RNG, and charged time. Preserve all four model exports and their
receipts; retain latest two full resumes. File fsync, atomic rename, parent fsync,
and durable receipts precede pruning only older owned resumes.

There are no extra periodic or emergency checkpoints. An interruption before
epoch 10 restarts from initialization; later interruptions can replay up to ten
epochs. Restarting from initialization after such failure still requires retaining
ledger charges. Smoke snapshots can never serve as production resumes.

Publish completion/selection only after epoch 40 and all four milestone model
receipts verify. Epoch 40 is the predeclared shared encoder; neither adaptation
loss, diagnostic probes, official-dev BLEU nor official test select another epoch.
Receipt hashes, not low-precision feature probes, establish artifact validity.
Close/finalize the training ledger before sealing `completion.json`. Its
`payload_files` maps required immutable, run-relative files to sizes/SHA-256s;
include model exports/receipts, latest two resumes/receipts, identity/input/runtime
records, closed ledger, and selection. It excludes itself, writer locks, scratch,
and subsequent `archive-receipts/` bookkeeping. After publication no payload
file may change. The archive includes this payload plus `completion.json`, whose
raw hash is bound separately to avoid a self-referential digest.

## 6. Working storage, smoke, and archive

Working root: `/home/kan/datasets/despamo/official-phoenix14t-dino-adaptation-v1`;
merged targets live beneath `targets/<merged-key>/`, runs beneath `runs/<run-key>/`.
Active outputs use existing Linux-native filesystem. Preflight checks projected
publication demand and a 50 GiB reserve against both Linux free space and Windows
C: backing-disk free space. Use measured serialization sizes from the smoke to
include retained exports/resumes and in-flight scratch. Existing verified input
caches are reused read-only. B: archival does not immediately reclaim C: VHDX bytes.

After focused tests and source validation, use a disposable real-batch GPU smoke
that exercises calibration and joint gradients, BF16, schedule application, and
save/load. Production starts fresh at seed 0. Smoke has a distinct identity and
receipts, no adoptable production checkpoint, and its measured charges are
carried into production accounting once. No additional GPU job runs alongside
the existing fusion study. Read current state/process ownership before execution;
docs alone are not proof of idle hardware or completed final publication.

Archive destination:
`/mnt/b/DeSpaMo/archives/official-phoenix14t-dino-adaptation-v1/<run-key>`.
Copy a completed run to a temporary B: directory, verify every byte size/hash,
and publish archive manifest/completion marker last. Bind source identity, source
paths and hashes, archive-relative paths, and the selected model hash. Retries
validate exact existing content; corrupt, extra, or incomplete files fail admission.
The copied inventory is exactly the sealed payload plus `completion.json`, not
every mutable file discovered by a fresh recursive glob. Native archive receipts
live under `archive-receipts/` and cannot change the sealed payload or retry inventory.
Check that `/mnt/b` is actually mounted to B: before creating staging directories;
an ordinary directory on the C:-backed root is not an acceptable fallback. NTFS
archival does not use the Linux-native training writer. File fsync and a fresh
post-publication read verify the copy. Record whether archive directory fsync is
supported; unsupported directory fsync must not be presented as crash-durable
publication. Write the verified archive receipt durably under the native run as
well. Future archive admission rehashes files; a completion marker alone is not
evidence of preserved bytes after reboot or power loss.
Archival never edits historical identities or deletes native originals. Any removal
of originals requires a separate exact cleanup inventory and user authorization.
Existing training resume requires restoration to its owned native paths. A future
extractor may consume verified archived weights under its own new input contract.

## 7. Acceptance and review focus

Success requires all tests/static checks below, a completed 11,800-update run,
four verified milestone model exports, latest two verified full resumes, an exact
epoch-40 selected receipt, and a verified B: archive. No translation score or
signer-invariance claim is implied by completion.

Tests must cover five cross-cutting hazards: mixed-source/frame misjoins; v2 masks
accidentally replaced by old data or whole-clip masking; schedule/gradient boundary
errors; lost state/accounting or pruning before durable publication; and corrupt
archive/storage fallback. Preserve old source bytes and never weaken native writer
guards to permit NTFS execution.
