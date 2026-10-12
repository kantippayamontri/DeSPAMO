# Fusion training time-waiver recovery — 2026-10-07

## Authorization and scope

User directed: **"just train and do not care the time"**, followed by
**"Start the training without asking me any question."** This supersedes training
duration stops for the existing six matched runs: seeds 0/1, reference,
aligned_fixed, aligned_learned, each 16,000 optimizer steps. The frozen split,
recipe, feature package, dev-selection rule, and once-only held-out registry stay
in force. No additional seeds, steps, ablations, or Signer07 scoring are included.

Study root:
`/home/kan/datasets/despamo/signer-pilot/time-aligned-fusion-new-motion-validated-study-v1`.
Authorization record: `time-waiver.json` under that root. Controller log:
`time-waiver-controller.log`. Live JSON/process evidence takes precedence over
this snapshot and the October 6 handoff.

## Previous stop and exact recovery

Seed0 reference stopped at 13,000 optimizer steps when its controller's 10,800s
allowance exhausted. Latest complete durable checkpoint is `resume-12500.ckpt`;
resume restores step 12,500, optimizer/scheduler, sampler, model, and RNG state.
The closed runner ledger retains 6,228.953107277004s, and reference controller
stages retain 10,789.065188278983s, including inherited setup. These clocks are
different accounting scopes and must not be added as independent GPU demand.

Existing full Signer03 dev BLEU4: 4.471882883497552 at step 5,500;
5.788668148392375 at step 10,400. These are intermediate observations, not final
six-run results or test scores. Five other runs remain pending at recovery.

## Implementation and verification

`scripts/fusion_time_waiver.py` validates the fixed authorization scope, binds
three upgraded time-control source hashes, and checks all other original source
hashes. Historical source/cap fields remain exact checkpoint identities; explicit
execution metadata records the effective no-duration-limit policy. GNU timeout's
duration `0` disables its deadline while preserving owned process-group recovery.
All elapsed work remains monotonically charged. Legacy E1 and pilot budget
policies remain unchanged for other experiments.

Focused verification: **224 tests passed**, Ruff passed, Pyright reported zero
errors. Regression coverage includes owned subprocess recovery above the old
limit, preserved elapsed charges, source identity mapping, and checkpoints above
the inherited 24-hour limit with sampler/accounting checks. Actual owned step
12,500 checkpoint, sampler position, original runner/controller source identities,
and original controller identity hash were verified before launch.

Storage admission before launch: remaining projected artifacts 160,639,141,379
bytes, reserve 53,687,091,200 bytes, required free 214,326,232,579 bytes. Linux free
429,471,481,856 bytes; Windows backing disk free 247,671,255,040 bytes. Keep the
existing storage checks and checkpoint retention policy.

Controller launched as PID 402856 using the existing motion package, timing,
validation recovery record, and new `--time-waiver-record`; no outer training
timeout. Startup validates the full motion package before publishing running
state. Recovery and preflight both completed. At the fresh observation
`2026-10-06T18:31:27Z`, the reference runner was **running at optimizer step
12,504**, phase `train`, policy `fusion-time-waiver-v1`. GPU used 9,247 MiB with
32% utilization. Owned reference-stage supervisor PID was 409000. Its GNU timeout
duration is `0`, and effective duration limit is null. All five remaining runs
follow automatically after reference completion. Recheck live JSON/process
identities before acting; these are observations, not completion claims.

Follow-up at `2026-10-06T18:35:04Z`: **13,018/16,000 optimizer steps**, phase
`train`, status `running`, effective duration limit null. Training has passed the
previous 13,000-step stop. Runner cumulative charge is 6,523.507791286s; latest
two complete resumes are `resume-13000.ckpt` and `resume-12500.ckpt`, confirming
durable publication and retention. GPU used 10,705 MiB with 20% utilization.
Supervisor PID 409000 matched its recorded boot/start-tick/process-group identity;
runner PID 409001 was live. All three upgraded source hashes still matched the
authorization record, and `git diff --check` passed.
