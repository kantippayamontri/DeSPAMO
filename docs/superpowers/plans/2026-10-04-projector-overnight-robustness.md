# Projector held-out comparison and paired-seed robustness study

## Authorization and fixed scope

The user authorized all four tasks and selected full robustness: evaluate the two
dev-selected seed-0 16000-step projectors, analyze errors, prepare a conditional
future-description baseline, and train/evaluate a matched seed-1 pair. Overall
additional GPU-stage wall budget is **36000 seconds (10 hours)**. Proceed without
further questions; no commits, pushes, prior-artifact deletion or new-description
generation is authorized.

## Predeclared experimental rules

- Use the fixed cached cosine DINO features and frozen logical signer split.
- Both architectures: 16000 steps, VT warmup 2000, LR warmup 1600, identical
  optimizer groups and cosine horizon, physical batch 4, no accumulation.
- Seed 1 changes common initialization, data sampling, dropout and prompt RNG;
  residual branch seed is `training_seed + 1` (seed-0 branch stays 1).
- Attest seed-1 common initialization before launching training, and verify both
  architectures against the same attestation. Preserve original seed-0 source archives.
- Select every checkpoint exclusively on complete ordered Signer03 dev reports.
- Signer07: exact ordered 768 clips, deterministic beam 5, length 64, no ICL.
- One logical held-out evaluation per selected checkpoint. Completed outputs are
  reused; interrupted evaluation resumes its durable journal under the same
  checkpoint/decoder/split identity. Test scores never tune recipes or choose seeds.
- GPU stages run sequentially. Training caps 14400 s each; scoring caps 1800 s
  each, further limited by remaining overall budget. Record actual child-stage
  elapsed time conservatively, including model setup.
- Retain two durable resumes per new training run and all dev-selection models.
- Appearance policy remains `qwen-schema98-unreviewed-v1` / `not_assessed`.

## Tasks

- [x] Preserve exact residual seed-0 source closure before seed support changes.
- [x] Implement and test dev-selection/architecture-aware held-out evaluator,
  checkpoint registry, deterministic batch journal, and no-repeat completed scoring.
- [x] Implement/test seed-1 model/data initialization and a bound common-initialization
  attestation; keep the seed-0 path and all shared training hyperparameters unchanged.
- [x] Freeze CPU analysis rules using training/dev vocabulary. Report BLEU/ROUGE,
  repetition, lengths, and rule-based German date/location/temperature diagnostics.
  Explicitly count unavailable parses; do not fabricate semantic/human ratings.
- [x] Implement and test an overall-budgeted sequential controller with durable
  state/logs and conservative resume behavior. Stop and report incomplete work if
  budget/integrity constraints prevent completion; do not silently exceed the cap.
- [x] Run focused/full tests, lint, targeted types and independent review.
- [x] Score existing seed-0 linear/residual 16000 checkpoints once each.
- [x] Train fresh seed-1 linear 16000, select on dev, then train matched residual
  seed-1 16000. Score both selected models once each.
- [x] Produce per-seed paired comparisons and exploratory broadcast-cluster
  bootstrap intervals; separately report between-seed variation. Two seeds are
  insufficient for a broad statistical-significance or signer-invariance claim.
- [x] Write final report and conditional baseline recipe, with artifacts and provenance.

## Conditional future-description baseline

Residual 16000 is the prespecified candidate based on seed-0 dev superiority.
If its seed-0 held-out BLEU-4 advantage is positive, record it as the future-description
baseline recipe. Qualify confidence with the seed-1 matched result: both positive
is a two-seed replication; mixed signs require a provisional/mixed-robustness label.
If primary gain does not transfer, retain linear 16000 as the baseline and document
the failed transfer. Keep seed 0 as the default reproducibility seed; do not choose
a favorable training seed from test scores. Report dev-selected checkpoints as they are.

New descriptions remain on the other server; the baseline recipe is prepared for
that future data version, not a claim that the new descriptions were trained or
evaluated. A new-data run must bind its actual schema/source/encoder manifests and
preserve all translation clips with unknown/failed appearance targets masked.

## Budget and recovery record

Controller state, per-stage logs, reports, evaluation journals and training budgets
live under `/home/kan/datasets/despamo/signer-pilot/`. A persistent progress ledger
records implementation, verification, model identities, spent budget and outcomes.
Source/model integrity failures halt the affected stage rather than overwrite a
completed result. Existing checkpoint/decoder identities remain frozen.

## Source archives

- Original residual 8000 / linear 16000 source closure:
  `projector-original-source-19f42b5328e513fea1e0c2276325acc46f00a61b88b6579e078ca1fc76aee954.tar.gz`.
- Residual seed-0 16000 source closure:
  `projector-original-source-0c1cd6f5cf2f516e2d52314377843b05bc70fef7b086bc583d192b2339237a15.tar.gz`;
  verified archive SHA-256 `452221bdf5088372486d68430805f961aa7d62fcd21268c547b2429678bfb0cc`.

## Completion

All tasks completed under 13936.07 seconds (3.8711 GPU-stage wall-hours).
Final report: [held-out robustness](../../projector-heldout-robustness-2026-10-04.md).
Residual-minus-linear BLEU-4 deltas: seed0 +1.130312, seed1 -1.141264;
mean -0.005476. Baseline status is `provisional_mixed_robustness`; keep the linear
16000 comparator. No robust improvement or signer-invariance claim established.
