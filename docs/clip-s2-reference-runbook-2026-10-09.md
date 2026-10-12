# Frozen CLIP + S² reference pipeline

## Current hold — Windows GPU-timeout crashes, 2026-10-11

All training/GPU diagnostics stopped. v3 manual-stop receipt blocks recovery;
epoch1/2resumes and complete features retained, test unscored. Windows logs confirm
VIDEO_TDR_FAILURE0x116, related NVIDIA driver nvlddmkm.sys, after dev generation.
GPU unknown exception surfaced in beam search and later CUDA RNG restoration;
those synchronization sites do not identify a faulting kernel. AutoReboot=1
explains unexpected restart behavior. No more GPU reproduction/automatic retries
or new roots until mitigation is reviewed.

[Host-reboot investigation](host-reboot-diagnosis-2026-10-11.md) and
[evidence JSON](host-reboot-diagnosis-2026-10-11.json) contain Windows timeline,
affected stack, protected dump paths and CPU-only code audit. Training cache
disable is inherited by inference, causing redundant decoder-prefix recomputation
at batch4/beam5. This is confirmed inefficiency, not a proven cause of the BSOD.
No production generation/cache/precision changes, driver updates or TDR registry
edits were applied. Earlier corruption attribution was an inference; precise
driver/kernel/hardware cause still requires dump analysis.

User-approved CPU mitigation is prepared in new sibling `tools/clip_s2_inference/`:
explicit cached deterministic generation, batch1dev, bounded diagnostic-only
probes and timing/memory/error telemetry. Old source identities/checkpoints are
unchanged; this is not activated in the legacy runner. GPU hold remains active.
[Mitigation report](clip-s2-inference-mitigation-2026-10-11.md) and
[receipt](clip-s2-inference-mitigation-2026-10-11.json) record17new/34focused CPU
checks, broad-suite limitations and pending Admin dump review/compatibility bridge.

## Earlier status — v3 relaunch, 2026-10-11

- v1 collapsed from in-process corruption (diagnosis:
  [clip-s2-collapse-diagnosis-2026-10-11.json](clip-s2-collapse-diagnosis-2026-10-11.json)).
- v2 interrupted at epoch2 dev by unexpected host reboot. Both preserved untouched.
- Reboot-safe owner check applied to `tools/clip_s2/study.py` (PIDs from an earlier
  boot cannot collide with live owners).
- Guards active: frozen-base fingerprint check per epoch, CE-collapse window (last200
  mean ≥8.0 after VT warmup), identical-dev-output rejection, raising
  `RecoverableFault` for bounded watchdog replay.
- Active study root:
  `/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v3`,
  archive `/mnt/b/DeSpaMo/archives/official-phoenix14t-translation-clip-s2-spamo200-v3`.
- Reuses verified feature package and accepted training profile.
- Controller5742 / watchdog5765. Seed0 reference running, Seed1 queued. Dev-only.

## Authorized switch — 2026-10-10

User approved immediate stop of predecessor SpaMo200 and implementation/extraction/
two-reference launch. This supersedes finish-six-first. Watchdog3449/controller3427/
fixed worker3448 exited after exact ownership SIGTERM checks. Controller published
intentional manual stop. Predecessor root
`/home/kan/datasets/despamo/official-phoenix14t-translation-spamo200-v1` retains
reference completion at96epochs/best21.735491at76; fixed interrupted59071updates
with epoch65/66resumes. All old code, selected/candidate weights, reports and test
claims preserved. Old monitor must not restart it. Official test remains unscored.

## Runtime and source

- New modules in retained isolated worktree `tools/clip_s2/`, branch
  `work/official-merged-targets`; no commits/pushes/merges.
- CLIP extraction: existing **main-project `tools/motion/.venv` interpreter**,
  torch2.0.1/transformers4.32/NumPy1.26.4/Pillow11.3.0. Main runtime lacks Pillow;
  no dependency changes or cross-environment imports were made.
- Translation/controller/monitor: existing main `.venv`, FLAN-T5-XL/PEFT.
- Worktree cwd; extraction `PYTHONPATH=<worktree>`; training
  `PYTHONPATH=<main>/src:<worktree>:<main>`. Both use
  `LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0`.
- Inherited `tools/followup/`, benchmark helpers and main `src/` remain unchanged.
  New namespace and imported helpers are checkpoint-bound after launch; do not edit.

## Features and parity

Frozen `openai/clip-vit-large-patch14`, revision
`32bd64288804d66eefd0ccbe215aa642df71cc41`. Processor metadata fetched at exact
revision and stored under new feature base inputs, not shared HF snapshot.
CLIP resize/center-crop/normalization to224, bicubic upscale normalized tensor448,
four224tiles, final raw hidden CLS per tile averaged per frame, concatenate global
CLS and averaged tile CLS to2048. FP32/no autocast/no TF32; output `[N,2048]` FP32.
No post-normalized pooler/projected image embeddings or regenerated motion features.

Real two-frame parity against original SpaMo S² function bodies passed
rtol1e-4/atol1e-5. Existing motion environment lacks optional einops; parity-only
grid permutations use tested equivalent reshape/permute operations while executing
unchanged original function ASTs, with reference source hash bound. Production
extractor is CLS-only and does not use those patch-grid operations.
Frame/tile batch32 accepted: peak reserved2300575744bytes (~2.14GiB), measured
free21711814656bytes. Profile native `profiles/accepted.json` binds source/model/
processor and selected batching; no fallback under same package identity.

Feature base:
`/home/kan/datasets/despamo/official-phoenix14t-clip-s2-features-v1`.
Immutable `profile-inputs.json`, `extraction-inputs.json`, processor inputs and
accepted profile. Source row map is completed official DINO metadata with
**absolute paths**; new inventory validates exact native root/order without
changing inherited relative-path helper. Full expected8257clips/947756frames:
train7096/827354; dev519/55775; test642/64627. Original frame hashes are checked
before/after extraction and against DINO receipts. Extracting test images is
label-free preparation; no held-out scoring or target-conditioned inference.

## Verification before launch

- 14extraction/publisher tests and37training/controller/report/inherited tests
  passed (51total), including source identity, tile/frame association, explicit
  root loading, candidate copy verification, full optimizer/BN/RNG next-update
  replay, exact two selections, immutable freeze and dev-only archive.
- Ruff/format clean; extractor typing uses motion interpreter, training typing
  uses main interpreter/overlay; both targeted Pyright passes.
- Full real bounded smoke package24clips published/verified, including ordinary,
  longest train visual, longest train target and fourlongdev clips.
  `spatial-smoke-96b48a23fd58f8403d34090d285503f3a19c6ad5167276b68aea4c391f8d9472`;
  manifestSHA`dc21daf4149fb66284541165d0a9d14ca59519e1d4c568c1084ae07eb200e4f7`.
- Real main-runtime batch8 CE+VT profile with full ICL: all307/307AdamW tensors
  initialized, longvisual/longtarget peak11.4648GiB; fresh-base compact reload
  generated identical beam5translations. Bounded profile is not admitted for
  training: full package receives its own profile after publication.
- Seed0common/frozen initialization matches predecessor200epoch receipt; seed1
  matches retained old40epoch initialization (architecture/weights identical).
  Old40epoch performance is not a paired baseline. DINO200seed1 never ran; report
  explicitly labels it unpaired and does not fabricate two-seed paired gain.
- Native self-review fixed completion-pointer crash recovery and strict profile/
  official data distinction. No full unrelated unit-suite pass claimed.

## Automated stage order and training recipe

Study root:
`/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v1`.
Archive root with same suffix under `/mnt/b/DeSpaMo/archives/`.

Detached controller runs:
1. Full extraction, immutable package publication, independent complete verify.
2. Bind full package in immutable `study-inputs.json` (explicit spatial root/SHA,
   existing motion root, seeds0/1/reference/test_scoringfalse).
3. Real full-data batch8 training profile; require accepted source/data/config,
   generation equality and GPU margin before training admission.
4. Fresh reference seed0, then seed1. No DINO resumes or profiled model weights
   initialize training; both use same seeded original model initialization.
5. Freeze two dev selections, compare only dev, verify cold-copy final payloads,
   publish summary. No test runner/claim is launched by this controller.

Same downstream200epoch recipe: physical8/accumulation1/drop-last,
887updates/epoch/177400ceiling; VT-only4000, LRwarmup10000; adapter3e-4→5e-5 /
FLAN5e-5→1e-5; ordinarymean VT + CE thereafter, BF16/FP32VT/clip1/AdamW0.9,0.98/
decay0.01. Plain spatial MLP2048→1024→768, same temporal/output adapter and FLAN
Q/V LoRA16/32/0.1; transformer activation checkpointing only. Three train-only
multilingual pairs; prompt limit1024; same fixed per-seed inference bank. Full519dev
every2epochs, max200/min40/patience10/minDelta0.10; actual highest dev selected,
earlier ties; retain native latest2resumes/all B:dev candidates. No wall-time stop.

## Entry points and monitoring

Main interpreter from worktree with training environment above:

```text
python -B -m tools.clip_s2.study --mode preflight
python -B -m tools.clip_s2.study --mode controller --authorize-study
python -B -m tools.clip_s2.study --mode watch --authorize-study
```

Controller state/logs: study `controller/state.json`, `controller/extract-run.log`,
`controller/extract-verify.log`, `controller/training-profile.log`,
`controller/seed0-reference.log`, `controller/seed1-reference.log`.
Launcher logs `study-controller.log`, `watchdog-launcher.log`.
Monitor `watchdog/status.json/history.jsonl`, interval1200seconds, unique lock,
max3verified transient same-source/same-root recoveries. Unknown faults flag
needs_agent; live/orphan/reused owners never cause duplicate launch. Manual stop
prevents restart; controller SIGTERM stops only its verified child. No boot service
or autonomous coding AI is promised.

Feature progress: feature-base `spatial-*/budget.json` completed_updates counts
**clips**, not optimizer updates; total8257. Training progress is study
`runs/*/budget.json` completed_updates in optimizer units. Smoke directories are
explicitly marked and are not official extraction completion.

Check original manual stop, exact owned processes, boot/start ticks, source/data/
profile/input bindings and current Linux/C:/B: headroom before manual recovery.
Native arrays7.23GiB +9GiBallowance +50GiBreserve; archive40GiB+50GiBreserve.
No automatic prior-artifact cleanup/relocation, reserve bypass or WSL compaction.
Reread live state for progress; expected full package and two training results are
future pipeline outcomes, not completion claims at launch.

## Verified pipeline launch — 2026-10-10

At16:09:32UTC controller143082, extractor143107 and watchdog143108 were owned/live
on boot`f6daa9c3-d5b0-4347-953c-02073ed495f9`; stage extract-run, monitor observe_only.
Official feature root
`spatial-f0ea5e0b10af4a796375bcfd935f69c9988caec9e7ddb6c27de0e0c35a098488`
had10durable train receipts, budget still0because updates publish every25clips.
Do not mistake coarse budget for no progress; count verified receipts. GPU2605MiB
and9%utilization at snapshot. Full extraction is running, translation seeds are
queued automatically behind full verification/profile; no translation score exists
yet. Smoke counts/results are not official full completion.

[Switch receipt](clip-s2-switch-2026-10-10.json) records approval, preserved old
state, verified initializers/tests/profiles and new owners. Recheck live state
before new claims; inherited monitor/controller are intentionally stopped.
