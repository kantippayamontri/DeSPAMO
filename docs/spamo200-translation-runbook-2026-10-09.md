# Active SpaMo-style 200-epoch translation study

## User-directed switch

User explicitly requested stopping the old40epoch study and starting the approved
200epoch follow-up now. This supersedes the earlier finish-six-and-test prerequisite.
Old root `official-phoenix14t-translation-v1/manual-stop.json` records the stop:
threeSeed0 runs complete; Seed1reference interrupted at2123updates, latest native
resume at epoch2. Watchdog763322/controller755979/worker850974 exited after exact
PID/boot/start-tick/command verification. Controller ignored inherited SIGINT;
SIGTERM exited it; worker handled SIGTERM and closed its ledger interrupted.
All prior source, checkpoints, reports and B archives remain intact. Old controller
state can still say running because its TERM handler was absent; stop receipt is
authoritative. No old official test was scored.

## New immutable recipe

- Root `/home/kan/datasets/despamo/official-phoenix14t-translation-spamo200-v1`.
- Execution source: retained isolated adaptation worktree `tools/followup/`.
- Main interpreter/environment and frozen main `src/` overlay unchanged.
- Sequential reference/fixed/learned arms × seeds0/1; fresh initialization.
- Existing DINO epoch40 whole-image224/448 spatial features and VideoMAE motion.
- Physical batch8, accumulation1,887updates/epoch; max200epochs/177400updates.
- VT-only4000updates, ordinary padded mean (`legacy_mean`), then CE+VT(weight1).
- Linear LRwarmup10000updates; cosine over declared200epoch horizon:
  adapter/VT3e-4→5e-5; FLAN-LoRA5e-5→1e-5. No schedule reset/compression on stop.
- Dev519clips every2epochs; min40; patience10eligible checks; significant
  gain0.10absolute BLEU4. Exclude VT-only checks from patience. Track actual best
  separately from significant anchor; smaller gains accumulate against anchor.
  Earlier epoch wins ties. Early stop is reconstructed from durable dev reports.
- Three English/French/Spanish=German pairs from one official-train partner.
  Training partners exclude every current-batch German reference including text
  duplicates. Partner RNG is checkpointed. Inference partner is precommitted by
  seed without consulting held-out inputs; overlap guard rejects, never reselects.
  Seed0 partner `12February_2010_Friday_tagesschau-122`; Seed1 partner
  `02December_2010_Thursday_tagesschau-3640`. Both have zero dev/test reference
  overlaps. No held-out translation enters any prompt.
- Prompt token limit1024; measured maximum complete prompt569tokens, mean179.206.
  Old64token limit is not reused. Decoder remains beam5/max64.
- Transformer-only activation checkpointing, embedding-output gradient hook and
  training `use_cache=False`; no checkpointed visual BatchNorm. Allows full ICL
  while retaining physical contrastive batch8. No gradient accumulation.
- Native full resume every epoch before dev; latest2retained. Selected best stays
  native; all actual dev candidates hash-archived to new B root. No old deletions.
- Freeze all six selections before once-only official642test; final verified
  archive/native retention. No wall-time stop, extra seeds, CLIP, S² or Q-Former.

## Profile and checks

Accepted profile at new root `profiles/batch8-3a96da7b3437/profile.json`, referenced
by hash in `profiles/accepted.json`. Ordinary/long-visual/long-target batches each
used the longest eligible full multilingual prompt and four real joint updates:

| Case | Peak reserved GiB | Clips/s | Initialized AdamW tensors |
| --- | ---: | ---: | ---: |
| Ordinary | 9.6055 | 6.0246 | 313/313 |
| Long visual | 11.4727 | 5.4778 | 313/313 |
| Long target | 11.4727 | 5.4542 | 313/313 |

Native compact checkpoint plus fresh frozen-base reconstruction produced identical
beam5generation; GPU free13.7598GiB after reload. This is a worst-prompt microprofile,
not a duration forecast. Accepted profile binds complete recipe/source/data.
26focused follow-up/legacy-contract/checkpoint tests passed; Ruff and targeted
Pyright clean. Main full unit-suite attempt exceeded120second tool timeout and
was terminated; no full-suite pass claimed. No test process remained afterward.

## Runtime entrypoints

Use retained worktree cwd and:
`PYTHONPATH=<main>/src:<worktree>:<main> LD_LIBRARY_PATH=/usr/lib/wsl/lib
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0`.

- Controller: `<main>/.venv/bin/python -B -m tools.followup.study --mode controller --authorize-study`
- Monitor: `<main>/.venv/bin/python -B -m tools.followup.study --mode watch --authorize-study`
- Controller state/logs: `controller/state.json`, `controller/seed*-*.log`.
- Launcher logs: `study-controller.log`, `watchdog-launcher.log`.
- Run state: `runs/*/{identity,budget,stopping,selected-checkpoint,complete}.json`.
- Monitor: `watchdog/status.json`, `history.jsonl`, `recovery-controller.log`.
- Summary appears after all frozen once-only test reports and final archival.

Monitor interval1200seconds, unique lock, max3verified same-root recoveries.
Live/orphan/reused PIDs prohibit duplicate launch. Unknown failures flag
`needs_agent`; historical transient traces do not authorize unrelated faults.
Source/base/data/resume/storage checks precede recovery. Manual-stop receipt
disables restart. Controller SIGTERM stops its verified active child and records
manual stop. Monitor is not a boot-persistent service or autonomous coding AI.

Check native/C:/B: free space before additional actions; preserve50GiB reserve.
Initial snapshot: native243GiB/C:63GiB/B:658GiB; require native/C:6GiB projected
demand and B:60GiB projected candidate/final-archive demand in addition to reserve.
Recheck live owners/space and source closure before claiming progress or recovering.

## Verified launch snapshot

At2026-10-09T03:08:46UTC: controller859275, Seed0reference worker859294 and
monitor859295 running; exact owner identities verified. Worker native budget
recorded157updates/epoch0.177, phase train; finite VT-only loss/gradient logs through
update150. New run key
`translation-reference-seed0-90105640d6d36be42fa5f5a3742b296b1fcb8b75d519aaee47b6833a63404cf5`.
Monitor first check observe_only, no recoveries. GPU7047MiB/16% during VT-only
warmup. First full dev at epoch2; no new test scoring. These are launch observations,
not live progress guarantees; reread current state on return.

Subsequent check: native epoch1/2full resumes published at887/1774updates,
203551321bytes each; both SHA/size receipts independently verified. Active full
source closure still matches stored identity.

## Storage interruption and checkpoint relocation — 2026-10-10

Fresh01:18UTC check: Seed0reference completed at96epochs/85152updates after
10eligible checks without a0.10BLEU4 gain. Best actual selection is epoch76,
devBLEU4=21.735491222433016. Seed0fixed stopped at3548updates/epoch4 when the
native/backing storage guard failed, before epoch4resume/dev publication; latest
native resume is epoch3, with epoch2also retained. The worker interruption was
recorded2026-10-09T21:07:35UTC. Controller failed, GPU idle; monitor alive and
needs_agent, zero recoveries. Test unscored; remaining four runs not started.

User then requested moving a large checkpoint toB:. Completed historical
QFormerSeed1 terminal `step-16000.ckpt` (6102377277bytes/5.6833GiB) was copied,
flushed and independently hash-verified against its dev report before native
removal under the old run's writer lock. This is not a current-study checkpoint.
The historical selected `step-10400.ckpt` remains native, with selected pointer
and dev reports unchanged. Receipt:
[checkpoint-relocation-2026-10-10.json](checkpoint-relocation-2026-10-10.json).
Archive contains exact file and durable relocation metadata. Restore terminal
checkpoint to original regular native path before historical full-run validation;
old validators reject symlink substitution. No source code, current-study resume,
selection, test claim or historical report was modified.

At01:28UTC after move: Linux free247.15GiB/B:648.57GiB, WindowsC:55.70GiB.
Native blocks freed but host VHDX did not shrink; current study still requires
56GiB onC: (6GiBprojected +50GiBreserve). No resume, reserve reduction, WSL shutdown
or compaction performed. Additional Windows-side storage action is prerequisite;
offline VHDX compaction needs separate explicit authorization and stops WSL.
See [existing offline compaction instructions](wsl-compaction-fusion-2026-10-05.md),
reconfirm current VHDX path and all processes before applying. After headroom is
restored, use existing verified same-root recovery with source/data/base/resume
and ownership checks; epoch4will replay from epoch3resume, not a new run identity.

User subsequently requested a broader search and move of large project files.
Two inactive original SpaMo ZIP archives were moved toB: after source CRC checks,
source/destination size/SHA verification and durable copy publication:
`/home/kan/datasets/spamo/features.zip` (12.6021GiB) and
`/home/kan/datasets/spamo/features/spa_feat_p14t.zip` (6.7028GiB). Original paths
are now symlinks to the exact B: files; extracted8257spatial arrays and all motion,
model/checkpoint bytes remain native and unchanged. Current training identities
do not bind these ZIPs. For any historical tool that requires a regular native
archive, restore verified bytes over only the relocation symlink before use.
Full mapping/hashes: [project-file-relocation-2026-10-10.json](project-file-relocation-2026-10-10.json).

At02:27UTC: additional19.3049GiB freed native, Linux266.45GiB/B:629.26GiB free;
C:55.68GiB still below56GiB. Source closures match current run identities.
No Windows-native project file of comparable size was established in the checked
locations. No host cleanup, VHDX compaction, WSL shutdown or training resume
performed. Space inside WSL remains distinct from free space on hostC:.

## Same-root recovery after user storage action — 2026-10-10

User reported completing Windows storage-reclamation action. At02:56UTC observed
new boot `f6daa9c3-d5b0-4347-953c-02073ed495f9`, C:104.52GiB free,
Linux266.66GiB/B:629.26GiB free. All previous controller/worker/monitor owners
had exited and GPU was idle. Existing `verify_recovery()` passed accepted-profile,
source/base/data/recipe/native-resume and native/C:/B: storage checks. Both native
controller/monitor locks were available; no manual-stop receipt existed.

Restarted exact same controller and20minute monitor using unchanged source/runtime:
controller3427, Seed0fixed worker3448, watchdog3449, all exact owners verified.
Seed0reference completion/selection were skipped as already complete, not retrained.
Fixed restored epoch3resume at2661updates and replayed887uncheckpointed updates
from interrupted epoch4; prior elapsed charges remain. At03:01:46UTC epoch4native
resume published at3548updates,217730361bytes,
SHA`b5e6d16c55cb7d15efdc6dcde0f5a4883d8655b79218c03ed3abc3f95eac8ce1`,
independently verified. Phase dev_generation; this crosses the prior storage-guard
failure point. Reference best21.735491at76/stopped96 retained; test unscored.
Current remaining recipe/budgets/identities are unchanged, no source/environment
patches and no agent-performed WSL shutdown/compaction. Prior failure text in
controller state can remain historical; use status plus exact live owners/budgets
and latest log rather than treating retained error text as a new failure.

Recovery receipt: [spamo200-storage-recovery-2026-10-10.json](spamo200-storage-recovery-2026-10-10.json).
Archive relocation receipts and restoration requirements above remain applicable.
Reread live state/space on return; these observations are snapshots.
