# Official full-split feature preparation — 2026-10-08

**Completed update:** both8257clip packages finalized; controller and both workers
exited successfully. The [official translation benchmark](official-translation-benchmark-runbook-2026-10-08.md)
has now launched with profiled batch8/40epochs. Pending/running statements below
describe earlier extraction snapshots, not current study state.

## Active preparation

User requested next steps after verified adaptation completion. Sequential
feature preparation launched from retained isolated worktree:
`.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`.
New code is under `tools/benchmark/`; no old generators/dependencies changed.

Output root: `/home/kan/datasets/despamo/official-phoenix14t-features-v1`.
Controller state: `controller/state.json`; logs `controller/{spatial,motion}-run.log`;
launcher log: `preparation-controller.log`.

1. Epoch40 DINO full-video spatial extraction for all splits.
2. After spatial completion, automatically prepare matching motion: copy the
   byte-verified train arrays from documented generating package, extract official
   dev/test with unchanged VideoMAE generator in its dedicated environment.

No simultaneous GPU workers. Published feature roots use separate identities:

- Spatial: `spatial-a866a2a0bfcc45d7e6ff5555ee22bb928fd86f1f166af7382d1d05b233cac90b`.
- Motion: `motion-1123ff445677bf310ac0b8c9b45cac4f690b9102f5dad1960aa2fef04937e82d`.

At `2026-10-08T03:46:47Z`, controller PID673090 and spatial child673093 were
live with owned command/boot/start-tick receipts. Spatial ledger recorded25clips,
26per-clip receipts existed, stage`extract`, status`running`; GPU3953MiB/57%.
This is a snapshot; recheck before action. Production extraction is not complete.

Follow-up at `2026-10-08T03:49:01Z`: spatial ledger60clips,61published receipts;
controller/worker running, ownedPID/start ticks/command verified, source hashes
still exact. GPU3963MiB/36%. Motion remains queued behind spatial publication.

## Exact inventories and feature contracts

| Official split | Clips | Frames | Motion windows |
| --- | ---: | ---: | ---: |
| Train | 7,096 | 827,354 | 93,232 |
| Dev | 519 | 55,775 | 6,230 |
| Test | 642 | 64,627 | 7,172 |
| Total | 8,257 | 947,756 | 106,634 |

DINO features are FP32 `[full_frames,2048]`, concatenated224/448 CLS, using
predeclared epoch40 LoRA on pinned ViT-L/16. Inference batch32 is **frames**,
not the adaptation batch24clips/five frames. No DINO training gradients are enabled.

Motion is FP32 `[windows,1024]`, window16/stride8/full-tail/drop/repeat-last-short
policy, first token of final hidden state from frozen VideoMAE revision
`12da269a02d3e1fbbb7011e610e591fea8061dca`. Dev/test extraction uses8windows per
batch with exact RGB/bilinear224/center-crop/normalization from existing producer.
Train origin is explicitly `verified_parent_copy`, not falsely "newly generated".
Regular copies preserve byte hashes without hardlinks/symlinks to old artifacts.

Per-clip receipts bind actual ordered source PNG bytes, array hashes/shapes,
identity and source indices/windows. Final manifest is standard
`FeatureManifest` schema with exact split-relative array paths; complete publication
requires full inventory/receipt verification. Resume reuses only matching complete
clips; changed sources or unreceipted arrays fail closed.

## Verification and storage

Plan: [full-split preparation](superpowers/plans/2026-10-08-official-phoenix14t-feature-preparation.md).
Tests/static checks: **286passed/1skipped**, Ruff and both runtime-specific Pyright
checks passed. Pure merge/package/epoch40/motion-parent/window tests exercised
source drift, feature corruption, dtype/width errors and exact final manifest.
Dedicated motion imports authoritative unchanged producer from main checkout;
DINO uses its2.5.1/4.56.2 environment, motion its2.0.1/4.32 environment.

Actual spatial smoke: four clips spanning train/dev/test and longest train475frames;
complete, peak reserved3.397GiB. Actual motion smoke: train copy plus freshdev/test
clips after authoritative parent validation; complete, peak reserved7.193GiB.
GPU idle before production launch. All epoch40 adaptation checkpoint and verified
B: archive originals remain intact.

Raw projected arrays: spatial7,764,017,152bytes, motion436,772,864bytes. Spatial
admission includes3GiB receipt/metadata/scratch:10,985,242,624bytes. Linux and
Windows C: free space plus50GiB reserve checked before/through execution; initial
Linux255GiB/C:75GiB/B:663GiB free. These are observations, not current readings.

## Translation handoff

After **both** complete manifests verify, the approved conventional benchmark is
reference/fixed/learned x seeds0/1 with official519dev checkpoint selection and
642test scoring only after selections freeze. Completed signer-held-out results
are separate and remain untouched.

Translation training has **not** launched yet. Its checkpoint writer must store
trainable adapter/VT/FLAN-LoRA tensors plus optimizer/sampler/RNG state while
binding frozen base-model bytes independently. Six old-format full-model runs
would exceed C: free-space reserve. Prove compact reload/resume/generation
equivalence and profile physical training batch before enabling that stage.
Do not infer feature extraction success from controller liveness, score held-out
clips during preparation, delete old artifacts, or weaken native-storage guards.
