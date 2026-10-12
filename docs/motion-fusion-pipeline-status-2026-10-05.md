# Motion extraction → automatic fusion status — 2026-10-05

**Superseding recovery:** [October 7 time waiver](fusion-time-waiver-2026-10-07.md).
Reference subsequently stopped at 13,000 steps under the historical duration
cap. User authorized duration-unbounded completion of the same six 16,000-step
runs; restart uses the owned 12,500-step checkpoint and existing study root.
The running claim below describes the earlier October 6 snapshot.

Updated by combined final-fix verification on 2026-10-06 in `/home/kan/Research/DeSpaMo`.

**TRAINING RUNNING — RECOVERED SEED0 REFERENCE; OPTIMIZER UPDATES VERIFIED.**
Extraction completed all7096clips and the package remains unchanged. The initial
training controller timed out during its300second preflight. A read-only consumer
validator now eliminates repeated large-identity serialization, duplicate source
reads and duplicate full scans within the same admission transaction. All source,
array, receipt, timing, directory, alias and metadata-drift checks remain.

Active recovered training controller:
`/home/kan/datasets/despamo/signer-pilot/time-aligned-fusion-new-motion-validated-study-v1`.
Seed0reference passed common initialization, preflight and GPU gradient smoke;
at the latest observation actualoptimizerstep1000of16000 was saved in its ledger.
The remaining five runs follow sequentially under the same matched recipe.

Training recovery record:
`/home/kan/datasets/despamo/signer-pilot/time-aligned-fusion-new-motion-study-v1/validation-recovery.json`.
All7oldartifact hashes remain unchanged. Previously charged2864.1890981140023s
is retained against seed0/reference and the18hour global ceiling. An outer
GNUtimeout also bounds the restarted controller to its remaining allowance,
including startup and closing margin. No extraction repeats or budget resets.

Current run:
`fusion-new-motion-reference-seed0-16000-767d35b9821c629afcadda90f7a34e35781d93c193fef81f9c0e19f8a05ea0ac`.
See [validation recovery/self-review](validation-recovery-2026-10-06.md).

## Retained extraction restart history

The following snapshot describes the earlier extraction restart, now complete.
Latest user instruction requested direct implementation/self-review without time
estimates. Duration forecasts no longer gate extraction. Actual source/feature
checks, tensor parity, process ownership, disk checks and existing hard caps stay.
At the last verified observation,123of7096clips had actual output receipts.
Training had not started at that extraction snapshot and was queued only after complete
new-package/timing validation and training smokes pass.

## Measured-profile restart and self-review

- Active pipeline root:
  `/home/kan/datasets/despamo/signer-pilot/motion-fusion-pipeline-measured-profile-v1`.
- Active bootstrap:
  `/home/kan/datasets/despamo/signer-pilot/motion-extraction-bootstrap-measured-profile-v1`.
- Active feature cache:
  `/home/kan/datasets/despamo/signer-pilot/motion-features-fb35379b1e2ea0c4d19a186cd290952b512eff270f5394540c902f344eae842c`.
- Profile schema3/admission policy `measured-smoke-hard-cap-v1`: actual samples,
  observation hashes and input bit parity at batch1/8/16; no duration projection.
- Recovery record:
  `/home/kan/datasets/despamo/signer-pilot/motion-fusion-pipeline-v1/measured-profile-recovery.json`.
  It binds11unchanged prior artifact hashes and carries216.61064642599376seconds
  extraction charge into both new producer/pipeline ledgers. Training charge0.
  Old failed pipeline/bootstrap and7partial clip records are not rewritten.
- Existing5hour extraction/18hour training hard caps remain; no reset, transfer,
  new allowance or ETA. Inherited-charge regressions reject attempts to refund it.
- Direct self-review checked recovery arguments through preparation child,
  extraction/clock queries, handoff leases, fixed new roots, old record immutability,
  counter floors, profile byte bindings, actual deadline and automatic training path.
- Fresh main suite **1616passed/27existing warnings**, dedicated suite
  **132passed/1known upstream parity warning**, Ruff and both targeted Pyright
  checks clean; Git whitespace check clean. Tests include RED→GREEN forecast-veto
  removal and inherited recovery credit, plus preparation-child propagation.

No source edits after this restart. Documentation/status updates do not alter
the frozen source closure. Below is retained history of the first failed startup.

## Historical first launch outcome and estimator issue

- Pipeline identity `1723ffd316e2871476488f78076bd2429e5b0f920178fb9a0bdcbe011f150944`.
- Actualnewcache partialroot
  `/home/kan/datasets/despamo/signer-pilot/motion-features-73adfb8cd6df37d233eed29608234c8a0ac2299d8ef503021be780ece05c54b2`.
- 7real clips/124windows generated with source/window receipts. No complete package
  or fusion checkpoints/results. GPU peak approximately11.334GiB, now idle.
- Error `TimeoutError: profile projects beyond remaining extraction cap`;
  inner producer spent127.51seconds, outer pipeline216.61seconds. These overlap:
  preserve conservative max, do not sum them or reset either record.
- Profile pooled6.881windows/s; excluding the single cold1window clip,
  observed7.273windows/s gives3.561hours processing extrapolation for7096clips,
  before full validation/extra overhead. This is not a new admission pass.
- Current estimator takes worst clip seconds/window×1.25: the first16frame clip
  took1.1093s, including0.9187s first inference. It applies1.3867s/window to all
  remaining93108windows. It then reserves another93232windows at the same
  inference-inclusive rate for CPU/source validation. Formula
  `rate*(remaining_windows+93232)+120` produces258509.65s/**71.808hours**.
- That forecast is not observed fullrun duration. It extrapolates cold startup
  across the corpus and prices CPU validation as another full inference pass.
  A calibrated workload-aware forecast and separately measured validation cost
  are needed before a5hour fullrun can be admitted confidently.

No cap increased or guard bypassed. Source code stayed frozen after launch.
Correcting the estimator changes bound source bytes; a restart must explicitly
preserve this failed attempt's histories, partial receipts and216.61s outer charge,
with audited old/new identity lineage. Do not delete state, rewrite clocks, or
create an untracked fresh allowance. Training's18hour authorization remains unspent.
Profile evidence file rawSHA256
`87cc573835dfc04385aa035573e2fcdea9e57d1ce051bdae2b840b114380bf0e`.

## Fresh verification

| Check | Result |
|---|---|
| Final affected package/new-motion/pipeline/legacy fusion tests | 463 passed, 4 warnings, 97.45s |
| Fresh full main unit suite after final review | **1614 passed**, 27 warnings, 208.78s, exit0; final lookup change included |
| Fresh dedicated extractor full suite | **126 passed**, 1 expected upstream warning, 30.00s, exit0 |
| Ruff: scripts/src/tests/tools/dinov3/tools/motion | Pass |
| Targeted Pyright, main and dedicated interpreters | Both 0 errors, 0 warnings, 0 informations |
| Both UV locks, offline check | Retained Task5 pass; 84/51 resolved packages; locks unchanged |
| Both installed dependency checks | Retained Task5 pass; 84/49 compatible installed packages; environments unchanged |
| Explicit scoped whitespace/final newline | Final source/test/docs and new review artifacts checked |
| Tracked `git diff --check` | Pass |
| Exact final-fix overlay | Separate before→after package; historical full packages retained |

Main warnings remain the 27 dependency/Lightning/trainer configuration warnings;
dedicated warning is upstream slow list-of-NumPy-arrays tensor construction during
standard-path parity. Neither warning was suppressed. Pyright emits an optional
version-update notice, not a diagnostic. No environment installation/change occurred.

### Combined named fixes and review boundary

- F1: full production-sized expected identities cross completion/recovery CPU
  boundaries through owned immutable nonce/raw-hash-bound files. Legacy and new
  mode regressions use real subprocesses; full bound-file provenance is retained.
- F2: admitted completed public/CLI resume branches before interval/fork. Bounded
  read-only package, summary and source validation holds producer/controller leases;
  successful resume returns existing receipt, with no stage/model dispatch or
  closed-clock/candidate/admission/seal rewrite. Tampering and timeout reject.
- F3: debugger established fixture custody-release race. Fixture waits on actual
  resume exclusive acquisition under one fixed monotonic deadline, catching only
  acquisition `BlockingIOError`; production nonblocking guard remains intact.
- Final audit added a RED regression for retained earlier package attempts, then
  one-line latest-attempt selection fix. Final 463-test/static checks cover frozen
  bytes. Independent final re-review subsequently accepted all fixes; controller
  then repeated full main/extractor suites on final frozen bytes before launch.

Exact evidence, operator contracts and preservation hashes:
[final-fix report](../.superpowers/sdd/2026-10-05-motion-extraction-autotrain/final-fix-report.md).

## Historical Task5 actual CPU admission (before this source change)

Actual dedicated extractor CLI `--mode preflight` returned exit0, `status=ready`,
in **74.85s**. Its 22-field envelope has `package=null`, `profile=null`,
`terminal_admission=null`, `spent_seconds=0`, `remaining_seconds=18000`, and no
bootstrap/budget/binding/status paths. This uncharged read-only readiness call
does not prove production extraction runtime.

- Actual producer identity/feature key:
  `73adfb8cd6df37d233eed29608234c8a0ac2299d8ef503021be780ece05c54b2`.
- Prospective root:
  `/home/kan/datasets/despamo/signer-pilot/motion-features-73adfb8cd6df37d233eed29608234c8a0ac2299d8ef503021be780ece05c54b2`.
   This root now exists with7partial profile clips. Identity changes must reject
   after production freeze unless explicitly audited without budget reset.
- Actual frame map:7096 clips,827354 recorded paths/frames,93232 windows;
  lexicographic unique paths and exact production source-root/ordinal checks pass.
  Actual annotation passes5746 train/582 Signer03 dev/768 Signer07 test plus seven
  training signers.
- Main-interpreter Task3 `validate_new_motion_production` passes using this actual
  producer identity and actual frame/annotation inventory. No live actor/schema
  incompatibility reproduced. The isolated handoff-shaped argument is not a
  completed package; no arrays/receipts/completion were claimed.
- Cached config/weights and actual copied processor sources are byte-bound in
  [Task5 report](../.superpowers/sdd/2026-10-05-motion-extraction-autotrain/task-5-report.md).
  Config24 layers/width1024/16 frames/FP32/use_mean_pooling=false. Planned output
  `hidden_states[-1][:,0]`; RGB/bilinear shortestedge224/centercrop224/rescale1/255/
  mean/std0.5; windows16/stride8/full windows/drop tails/repeat-last short padding.
  CPU schema agreement does not establish real GPU tensor/output parity.

Historical Task5 main pipeline identity (superseded by this fix wave):
`4f1e07b6c8af89dcdff56b6c62449dd47856d6f53e8de0e24c42321ef018bf60`,
binding95 source/lock/interpreter paths. Main retains `.venv/bin/python`, dedicated
extractor retains `tools/motion/.venv/bin/python` as invocation paths, although
both resolve to the same Python3.11.15 binary. Main has no Pillow and does not
import the GPU extractor; no site-packages bridge is used.

Current read-only pipeline identity:
`1723ffd316e2871476488f78076bd2429e5b0f920178fb9a0bdcbe011f150944`,
still binding95 source/lock/interpreter paths. Earlier identities must reject
changed controller/pipeline source bytes; no new real extractor preflight run is
claimed in this wave. Source code frozen after final focused/static checks.

## Hardware and both disks

RTX4090 UUID `GPU-758013eb-2b8c-0588-a783-71a7f0a625e1`, driver610.62,
capability8.9,24564MiB. Observation:0% utilization,15MiB used, no compute apps.
Startup subsequently loaded the actual frozen VideoMAE encoder for the7clip
profile; no trainable encoder update or fusion training occurred.

| Capacity/demand snapshot | Bytes | GiB |
|---|---:|---:|
| Linux available | 456465592320 | 425.117 |
| Windows C: free | 292343570432 | 272.266 |
| Fresh feature payload | 381878272 | 0.356 |
| Estimated training checkpoint growth,31×5.6GiB | 186401580646.4 | 173.6 |
| Exact pipeline helper required free, including2GiB overhead+50GiB reserve | 242618033766.4 | 225.956 |

Read-only `storage_admission` with these observed free-space values passes.
Dedicated extraction additionally budgets two payload copies+2GiB overhead:
2911240192 bytes before its50GiB reserve. Current capacity is not runtime storage
admission: both disks/GPU must be checked again immediately before real stages;
observed checkpoint sizes and owned allocated bytes supersede fallback estimates.

Pipeline root `motion-fusion-pipeline-v1` and bootstrap
`motion-extraction-bootstrap-v1` now contain failed startup/profiling evidence.
New training root `time-aligned-fusion-new-motion-study-v1` remains absent.
Old downloaded motion, frozen DINO/adapted spatial inputs, captions/text, selected
baseline checkpoints and reports remain available.23 protected checkpoints,
18 selected paths,161 monitored metadata files and10 registry report bindings
are preserved; full old checkpoints/feature arrays were not rehashed/deserialized.

## Operator command after final review

Run from `/home/kan/Research/DeSpaMo`. This command **was executed** and reached
the stopped profile outcome above. Do not blindly repeat it against owned state:

```sh
PYTHONPATH=src:. LD_LIBRARY_PATH=/usr/lib/wsl/lib HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  .venv/bin/python scripts/run_motion_fusion_pipeline.py \
  --pipeline-root /home/kan/datasets/despamo/signer-pilot/motion-fusion-pipeline-v1 \
  --adapted-root /home/kan/datasets/despamo/signer-pilot/e3-features-7122b42e35510611626c2a71d0731b79dd79b20de6e89bd49f62737c5c84a18e \
  --annotation /home/kan/Research/SpaMo/preprocess/Phoenix14T/train_info_ml.npy \
  --model-snapshot /home/kan/datasets/spamo/hf_cache/models--MCG-NJU--videomae-large/snapshots/12da269a02d3e1fbbb7011e610e591fea8061dca \
  --extractor-python /home/kan/Research/DeSpaMo/tools/motion/.venv/bin/python \
  --environment-lock /home/kan/Research/DeSpaMo/tools/motion/uv.lock \
  --authorize-motion-fusion-pipeline
```

Actual generated extractor command preserves dedicated venv invocation, exact
inputs/output-base/lock, cap18000 and extraction authorization. Existing pipeline
state requires `--resume`; recovery retains clocks/identity, no fresh allowance.
Final controller owns detached launch and must verify actual spawn before marking
RUNNING. Documentation of this command is not a spawn receipt.

Implemented chain: charged CPU preflight → real profile/parity/storage admission →
extraction → independently closed package/clock handoff → automatic fresh two-seed,
three-arm fusion controller. Extraction18000/training64800, no cap transfer;
max(outer interval, inner bound) avoids double counting. Training32400 per seed,
10800 per run; source/hardware/storage/owned-process/clock failures stop the chain.
Independent guardians cover consumer/delivery tails; leases and immutable final
summary closure remain held inside the5s final closing window.

Pending: scoped final re-review; real B1/8/16 source/tensor parity;
GPU profile/peak/throughput/whole-validation feasibility; actual storage admission;
new complete package; both real initializations/smokes; six training runs and
validated dev reports. No new BLEU, model peak, throughput, signer-invariance or
full-cache generating-proof claim is available. Historical downloaded-cache
chronology remains unresolved and is not relabelled as new generating evidence.
