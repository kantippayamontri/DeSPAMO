# SDD ledger — plan: docs/superpowers/plans/2026-10-03-residual-projector-experiment.md

Execution: inline, following the user's explicit request to launch the approved
experiment now. No commits/merges/pushes are authorized.

Ruling: use 16000 steps for the longer control — twice the 8000-step reference;
keep VT warmup 2000 and the existing 10% LR-warmup convention — cost if wrong:
about one longer downstream run, whose declared recipe remains reproducible.

Ruling: preserve the current artifact-bound workspace and add opt-in experimental
modules/runner; default model behavior is regression tested — avoids duplicating
environments or losing temporary work during the planned WSL compaction.

Pre-flight interfaces: residual wrapper retains base parameters; runner constructs
the seeded original model before attachment. Explicit dev schedules feed the
existing evaluator. New-run-only retention feeds the existing resume validator.
No conflicting signatures identified.

Task 1: residual import RED; baseline/gradient/masking/round-trip tests GREEN.
Ruling: fork CPU and CUDA RNGs — torch.manual_seed also seeds CUDA; an actual GPU
regression reproduced dropout RNG drift with a CPU-only fork. Restoring both
retains the intended matched initialization and dropout stream.

Task 1: complete — residual + existing adapter tests: 11 passed with CUDA RNG test.
Task 2: runner import RED; focused tests: 32 passed after implementing explicit
schedules, policy-owned retention and state-neutral joint smoke. Fixtures now
carry the declared new run policy, and a historical-policy refusal test protects
old outputs. CPU/CUDA RNG, buffers, parameters and prompt RNG are preserved by smoke.

Independent review: no Critical findings, three Important recovery/provenance
findings accepted. Regression tests reproduced all three before fixes:
checkpoint-file fsync before pruning; refusal paths must not overwrite run status;
source manifest must include storage/identity modules. Targeted fix checks: 30 passed.
No Minor findings. Reviewer set aside user-accepted Windows compaction timing,
unknown future BLEU/significance, live GPU resource validation and unrelated edits;
those remain respectively accepted scope, future measurement, launch checks,
and protected user work. No extra implementation scope was introduced.

Task 3: complete — post-review full suite 663 passed, 4 skipped; full Ruff clean;
targeted Pyright 0 errors; shell syntax/whitespace clean. Both real-data preflights
verified 5746 train / 582 dev / 768 test-unscored. All three review findings have
RED-to-GREEN regressions; no re-review or speculative extra refactor was needed.

Launch: tmux `projector-experiment-v1`, persistent log
`/home/kan/datasets/despamo/signer-pilot/projector-experiment-v1-launch.log`.
Residual run key:
`projector-residual-8000-def98a1f1fd934780ba7e18fe39e221c4d0a839e669cfa072d43931729dce325`.
Queued linear control:
`projector-linear-16000-eb0a5bef8210afc3d894f4ac57338f8cd5e0f01b6b4d74a8a38cb285b4f5f313`.

Observed residual progress: step 2356/8000, status running, 509.98 charged seconds,
28,200,449 trainable parameters. Real joint gradient smoke passed: loss 6.088694;
spatial/motion/language gradient norms 7.346725/1.681340/4.907654, no optimizer
updates, and initialization hash unchanged. Snapshot retention verified in the
live run: resume-1500 and resume-2000 only, resume-500 pruned; about 5.536 GiB each.
Pipeline starts the 16000-step linear control after successful residual completion.
No translation performance claim is made before full-dev checkpoint selection.

Latest launch verification: residual step 3459/8000, 14.39 charged minutes,
status running, tmux pane alive. Resumes 2500/3000 retained, all 57 current source
hashes match the active run identity. First dev evaluation is scheduled at 3500.
