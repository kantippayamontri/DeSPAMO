# Matched residual-MLP 16000-step comparison

**Status:** User approved adding this arm after the original linear 16000 control.
It starts fresh; the completed 8000-step residual checkpoint is not extended.

**Goal:** Complete the 2x2 architecture/horizon comparison. The existing matched
8000 comparison is linear 4.709851 versus residual 4.470941 dev BLEU-4. The new
comparison is original linear 16000 versus residual 16000.

## Declared matched recipe

- Fixed cosine DINO cache `7122b42e...`, original translation annotations and
  logical signer split: 5746 train, 582 Signer03 dev, 768 test unscored.
- Fresh common initialization hash `9e3a0220...`, seed 0, physical batch 4,
  no accumulation, no in-context examples, masked VT pooling, VT weight 1.
- Same optimizer groups: adapter 3e-4, FLAN LoRA 5e-5, AdamW (0.9, 0.98),
  weight decay 0.01, gradient clipping 1, BF16.
- Total steps 16000; VT warmup 2000; LR warmup 1600; cosine horizon 16000.
- Full-dev checkpoints 5500 / 10400 / 16000; deterministic beam 5, length 64.
- Residual branches match the 8000-step architecture: hidden 1024, branch seed 1,
  zero final residual layers on both spatial and motion input projectors.
- Cap 14400 seconds; retain latest two durable resume snapshots and all dev models.
- Supervision labels remain `qwen-schema98-unreviewed-v1` / `not_assessed`.

## Provenance and execution

- [x] Preserve the original 57 bound source files before extending the runner.
  Archive: `/home/kan/datasets/despamo/signer-pilot/projector-original-source-19f42b5328e513fea1e0c2276325acc46f00a61b88b6579e078ca1fc76aee954.tar.gz`.
  Archive SHA-256: `2c199e762b7f5f77e1a55ca197889f73a59049f8c033dc9600a0f887a94f5ff2`.
- [x] Wait for linear control completion and closed budget before editing its
  bound runner code or launching another GPU stage.
- [x] Add residual 16000 to the explicit allowed architecture/horizon pairs.
- [x] Validate matched control identity/configuration, complete full-dev schedule,
  deterministic decoder and selected checkpoint hash; bind that control into the
  new residual run identity.
- [x] Add RED-to-GREEN tests for the new arm and refusal of incomplete/mismatched
  controls. Run full tests/lint and targeted types.
- [x] Preflight the real completed control and current immutable feature inputs.
- [x] Launch the fresh residual 16000 run and verify real joint gradient smoke,
  state-neutral initialization and persisted live progress.

Compare dev-selected results at equal budgets, plus repetition and resource use.
The residual intervention changes both spatial and motion projectors; inference
about the spatial branch alone requires a separate ablation. No automatic
Signer07 scoring or new-description extraction is part of this launch.

## Execution record

Linear control completed all 16000 steps and selected step 16000, dev BLEU-4
**5.489715**; selected checkpoint SHA-256
`8cde697bf31ba67f0f24ddcb229d7f39829eb3d3cafc9c3f03b8171cf90fef4a`.
The matcher verifies equal resolved configs, feature/split/baseline identities,
warmup/horizon/schedule, retention/caps and all shared source hashes. Only the
approved entrypoint extension differs. Archived versus current training-loop,
smoke, retention, ledger and schedule ASTs were verified identical.

New run key:
`projector-residual-16000-f8deb3d4b807d59f3e0df2e6e6253ec069fb01d1a3365c36d4c257422d87526d`.
Session: tmux `projector-residual-16000`. Persistent log:
`/home/kan/datasets/despamo/signer-pilot/projector-residual-16000-launch.log`.
Launch script: `scripts/run_residual_16000.sh`.

Real-data joint gradient smoke passed with no optimizer update. Initial variant
hash matches the residual 8000 arm exactly:
`705932e1b28f41be6b4aa6cdc4fe2b9bbe0c98466eea8fddb279986d19296fb2`.
Observed live progress: 1257/16000, 4.39 charged minutes, resumes 500/1000 present.
No dev outcome is claimed until the declared evaluations run.

Verification: **672 passed, 4 skipped**, full Ruff clean, targeted Pyright clean,
shell/whitespace checks clean, real-data matched-control preflight passed. Focused
independent review found no Critical/Important blockers. Minor deferred: direct
preparation-level guard/identity-binding coverage could be expanded; current
behavior was validated by real preflight and launch. Unknown final quality and
future runtime/memory needs remain measurements, not assertions.
