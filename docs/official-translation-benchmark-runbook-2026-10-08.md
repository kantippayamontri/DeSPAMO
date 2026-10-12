# Official mixed-signer translation benchmark runbook — 2026-10-08

**Unattended monitoring:** [20-minute watchdog](benchmark-watchdog-2026-10-08.md)
is active outside the frozen source closure. Read study `watchdog/status.json`
and `history.jsonl` before recovery. Verified stopped transient jobs can resume
automatically; novel/integrity faults are flagged for active agent diagnosis.

## Active study

User approved execution through profiling/recovery checks and launch. Controller
root `/home/kan/datasets/despamo/official-phoenix14t-translation-v1/controller`;
state `state.json`, per-stage logs `seed{0,1}-{reference,aligned_fixed,aligned_learned}.log`.
Launcher log: parent `study-controller.log`. Runs beneath parent `runs/`.

Source worktree:
`/home/kan/Research/DeSpaMo/.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`.
New code under `tools/benchmark/`, uncommitted and not merged. Uses main source
overlay `/home/kan/Research/DeSpaMo/src` for already validated fusion modules.
**Keep worktree translation/compact modules, main `src/`, locks/environments,
cached models and published feature bytes unchanged while active.** Docs are safe.

Main interpreter `/home/kan/Research/DeSpaMo/.venv/bin/python`, torch2.0.1,
transformers4.32/PEFT, offline pinned FLAN revision
`7d6315df2c2fb742f0f5b556879d730926ca9001`. Required environment:
`PYTHONPATH=<main>/src:<worktree>:<main>`, `LD_LIBRARY_PATH=/usr/lib/wsl/lib`,
`HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=2`, `PYTHONDONTWRITEBYTECODE=1`,
`CUDA_VISIBLE_DEVICES=0`. No dependency changes or duplicate GPU workers.

## Approved matched experiment

Sequential seed0 reference/fixed/learned, then seed1 same. Plain spatial MLP
2048→1024/GELU→768, motion1024→768, original K5/P2/K5/P2 TCN, output MLP
768→2048/GELU→2048, FLAN-T5-XL rank16/alpha32 Q/V LoRA/dropout.1. Fixed gate.5
or learned sigmoid gate changes only approved aligned motion correction.
Q-Former remains excluded. Seed-specific common mutable initialization hashes
and frozen base fingerprints must match across the three arms.

| Setting | Value |
| --- | --- |
| Official splits | 7,096train / 519dev / 642test, nine overlapping signers |
| Physical batch | **8**, no accumulation; exact887complete batches/epoch |
| Epoch horizon | **40 per run / 35,480 optimizer updates** |
| Full dev | Every2epochs,20reports; first at update1,774 |
| Selection | Highest official-devBLEU4; earlier epoch on ties |
| Early/wall-time stops | Disabled; integrity/resource failures remain real stops |
| Initial VT phase | First2,000updates VT-only, then translationCE+VT(weight1) |
| LRs | Adapter/VT3e-4, FLAN-LoRA5e-5;10%warmup3,548updates then cosinezero |
| AdamW | betas.9/.98,eps1e-8,decay.01; norm clipping1.0 |
| Precision/pooling | BF16, mask-aware VT mean pooling |
| Generation | German task prompt, beam5,max64,noICL; eval batch4 |
| Final official test | Once logically after all six dev selections freeze |

Epoch2 may still be VT-only because its1,774updates precede update2,001. This
retains the prior matched VT warmup policy; it is not an early-stop or omitted
evaluation. Batch8/40epochs schedule is different from the earlier signer-held-out
16,000-update protocol. Standard benchmark does not establish unseen-signer
generalization or significance. Report literature comparisons with exact settings.

## Feature admission and metrics

Both packages finalized before launch, matching actual ordered source hashes:

- Spatial `official-phoenix14t-features-v1/spatial-a866a2a0bfcc45d7e6ff5555ee22bb928fd86f1f166af7382d1d05b233cac90b`;
  manifest SHA `5769cbe92343356e0ffaf87f7cb3fda305b4422fdf597dbf82ad856f1265ae5d`.
- Motion `official-phoenix14t-features-v1/motion-1123ff445677bf310ac0b8c9b45cac4f690b9102f5dad1960aa2fef04937e82d`;
  manifest SHA `023a054fe0d3b9e04b5c0472ef77ebb8d031130e604841d67acfe7d81f66fd03`.

Every array hash/shape/float32/finiteness checked at admission; per-loaded arrays
rehash and full-source/window counts match. Annotation/manifest identities recheck
each epoch. Only official training data supplies optimizer batches. Caption
normalization/metrics retain existing lowercasing/trailing-period and SacreBLEU13a
plus ROUGE-L implementation; ROUGE-L is stored as0–1fractions, not percentages.
No pilot Signer07 registry/scoring/tuning is involved.

## Compact checkpoint and B: retention

Compact snapshots include every trainable parameter and persistent buffer,
including TCN BatchNorm statistics/counters; frozen base tensors are omitted and
bound by raw model-file hashes plus loaded frozen-tensor fingerprint. Full resumes
also save AdamW, sampler, global Python/NumPy/Torch/CUDA and prompt RNG state.
Strict reload rejects wrong frozen base, missing/extra mutable keys or tensor drift.

Every epoch publishes an atomic full resume **before** potential dev generation;
keep latest2owned resumes. Every2epoch model candidate serializes natively, copies
to `/mnt/b/DeSpaMo/archives/official-phoenix14t-translation-v1/<run-key>/model-epoch-N.pt`,
verifies hash, then discards only owned native temporary candidate. **All20dev
weight files and receipts remain on B:.** Native selected best is retained, with
reconciliation for interruption between report and pointer publication. Existing
B: candidate without receipt can recover only from matching owned native bytes.

After six training selections freeze, official test journaling saves each batch's
predictions/references durably, checks IDs/references on resume, and never reselects
using test. Completed claim/report hashes prevent logical re-scoring with another
checkpoint. Final summary aggregates two-seed test means. Completed run payloads,
including full resumes, also copy/verify to B: `final-payload/`; native originals
remain. Historical pilot artifacts and once-only registry are unchanged.

Native admission requires a conservative6GiB pending-demand allowance plus50GiB
reserve on Linux and Windows C:. B: must be actually mounted and maintain its own
50GiB reserve. Initially Linux245GiB/C:67GiB/B:663GiB; no cleanup/compaction/migration
was performed. Do not delete original artifacts or compact/shutdown active WSL.

## Physical batch/reload measurements

Profiles use fresh processes/real ordinary and longest training clips, four joint
updates per case (first warmup), all313AdamW states initialized. No production
weights come from profiles. Their measured setup charges are retained and inherited
once into seed0 reference accounting.

| Batch | Long peak reserved GiB | Long clips/s | Full compact bytes |
| --- | ---: | ---: | ---: |
| 4 | 11.727 | 12.575 | 255,400,941 |
| **8** | **17.090** | **24.049** | **255,400,941** |

Mutable model bytes85,004,312. A fresh pinned frozen base/model was rebuilt after
native save; compact optimizer/model reload produced identical deterministic
beam5 predictions on four longest-dev clips in both profiles. Timing excludes
data loading and does not prove final BLEU gains. Profiles live in study `profiles/`.

## Verification/self-review

- Compact/model/controller initial8tests watched RED→GREEN. Later interruption
  regressions for dev report/pointer and B: receipt gaps also watched RED→GREEN.
- Correct-context isolated unit suite: **672passed**.
- Configured benchmark/fusion target suite: **89passed**, including both CUDA
  tests. Ruff and Pyright clean; intended source diffs inspected.
- Mixed-checkout broad invocation failed45tests (missing worktree merged modules
  and old comparison fixture code hashes); rerunning its proper source context
  passed672. No unrelated source/test edits hid those failures.
- An initial unconfigured GPU-visible combined invocation aborted natively;
  GPU tests passed standalone and full89passed with required WSL library/device
  environment, which production uses. Both full real model profiles also passed.
- Post-profile changes concern selection/archive/recovery only; model, data and
  compact-forward/reload contracts stayed as measured. Native self-review, no
  subagents/commits. No known important open findings in this execution scope.

## Fresh launch observation and recovery

At `2026-10-08T14:20:29Z`, controllerPID755979 and owned seed0-reference child
PID756000 were running. Run key:
`translation-reference-seed0-d3de376c7f49bb9575993098ecd4a7e04a61969d536fced5e0984ee9c2808eaa`.
Ledger **198/35,480updates**, phase`train`, charged234.837950s including retained
profiles/setup, aboutepoch0.22/40. Step200log finiteVTloss withCE`null` as approved
warmup; GPU7045MiB/19%. These are snapshots; recheck current receipts/processes.

Follow-up at `2026-10-08T14:24:51Z`: seed0 reference reached **epoch2/40,
1,774/35,480updates**, phase`dev_generation`. Native full resumes for epochs1/2
published. First epoch2 compact candidate on B: independently SHA-256 verified,
80,408,271bytes. Full519dev generation is in progress; no score published yet.
Source hashes, actual worker command/boot/start ticks verified at14:23:56Z;
GPU8507MiB/32%. Five other runs remain queued; official test has not started.

Monitor controller/stage logs and `runs/*/budget.json`, verify PID/boot/start ticks
and command against actual `/proc/` before action. Recovery reuses same command,
root/identity and newest owned compact resume; charges never reset or refunded.
Dev RNG/train flags restore before next training batch. Do not change batch, source,
features or recipe to bypass recovery admission. No automatic walltime cutoff.
