# CLIP+S² continuous Seed0 long-run launch — 2026-10-12

## Running

Continuous safe supervisor launched from existing production checkpoint5322.
Fresh09:06:31UTC observation: train progress5625, durable5322, parent193877 and
train193953 owned/live with matching boot/start ticks. This records a verified
launch and ongoing progress, not completion or long-stability proof.

Attempt started09:02:17.003823UTC. Token:
`production-seed0-step5322-to-max200-continuous-v1`.
Remaining authorized caps172078 training updates and50343 dev clips/97 full checks.
The original recipe stops on configured dev early stopping or the200-epoch ceiling.
No per-segment approval is needed within this admitted scope. Errors fail-stop;
no automatic retry, new seed, held-out test or restart from zero.

## DINO200 training parity verified

Compared native completed DINO200 reference Seed0 identity against checkpoint-bound
CLIP+S² identity: entire training recipe equal;75 shared source files hash-equal
to both identities and current disk; FLAN base files, runtime, common initialization
and beam/prompt/ICL parameters equal. See
`clip-s2-dino200-hyperparameter-parity-2026-10-12.json`.

- Batch8/accumulation1,887 updates/epoch; max200/min40, dev every2 epochs,
  patience10 eligible checks/minDelta0.10 BLEU4.
- VT warmup4000, LR warmup10000; original CE+VT1/legacy_mean.
- AdamW betas0.9/0.98, epsilon1e-8, weight decay0.01, gradient clip1.0.
- Adapter LR3e-4→5e-5, FLAN5e-5→1e-5, linear warmup then cosine decay.
- FLAN-T5-XL same revision; Q/V LoRA16/32/dropout0.1, BF16 autocast,
  transformer activation checkpointing, training cacheFalse.
- Train-only3 multilingual ICL pairs, prompt limit1024, beam5/max64.

Spatial representation remains requested frozen CLIP+S². Safe dev execution is the
approved fresh optimizer-free batch1/cacheTrue process, replacing DINO's legacy
batch4/cacheFalse execution. Training hyperparameters were not altered.

## Retention and launch admission

User explicitly selected **Approve latest2 retention**. Existing runtime policy
keeps latest2 safe-runtime full resumes, deleting older safe-runtime full resumes
only after healthy checkpoint publication and verified candidate archival. All dev
candidates/selections and original epoch1/2 remain protected. Added real retention
fixtures verified protected artifacts and archive failure before any deletion;
16 retention/scope/accounting checks passed, new test Ruff/format clean.

Fresh driver/host/storage/checkpoint/report/accounting admission passed. Original
budget6430.341189303363s and all optimizer/sampler/RNG/CE-guard state inherited;
no budget/reset or model reinitialization. Same frozen8298146c/93-source binding.
Scope authorizes existing rolling retention; no production source changes.
Original training-duration limit remainsNone. Scope credential expires after90days
and remains boot/source-bound; this is not a training-duration budget change.

## Native authority and monitoring

Runtime under original Seed0 run:
`safe-runtime-v1/attempts/production-seed0-step5322-to-max200-continuous-v1/`.
Inspect `attempt.json`, job `progress.json`/`owner.json`/`worker.log`, native
`safe-runtime-v1/state.json`, original `budget.json` and committed accounting head.
The supervisor sequentially waits for actual train child exit, runs full cached
dev, verifies/archives/selects, then restores the next training child. No parallel
training/evaluation workers.

Durable supervisor stdout/stderr:
`.superpowers/sdd/2026-10-12-clip-s2-safe-resume/long-run-supervisor.log`.
External boot/source-bound approval:
`.superpowers/sdd/2026-10-12-clip-s2-safe-resume/production-long-run-approval.json`.

Latest observation: train5322→7096 currently5625/durable5322, pruning permission
propagated in exact child ticket. Next scheduled resume6209, then7096 and epoch8
dev. Current selected epoch6/5322 BLEU4=2.0776100886063897 until later admission.
Preserve bound code/environment/data while live; do not launch duplicate workers
or the held legacy controller/watchdog. Check native owners/state before action.
