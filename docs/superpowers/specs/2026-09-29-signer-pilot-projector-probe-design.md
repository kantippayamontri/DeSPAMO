# E1/E2 Spatial-Projector Signer Probe

## Purpose and scope

Run a diagnostic-only comparison of signer information accessible from the
selected E1 and E2 spatial-projector outputs. E1 is the frozen-DINO baseline;
E2 shares the frozen DINO inputs but learned its own projector under factor
supervision. This probe neither gates E3 nor establishes signer invariance or
unseen-signer translation performance. No E3 training or Signer07 scoring is
part of this work.

## Inputs and identity

Read the frozen signer-pilot protocol and the two completed external run roots.
Verify protocol hash, split hash, variant, complete run status, selected step
4000, and each `selected-checkpoint.json` hash against its `step-4000.ckpt`.
Require the expected E1/E2 policies and common initial shared tensor hash.
Reuse `load_pilot_views` to validate physical-train annotation and feature
provenance; only its logical `train` view is used. Reject unexpected signer or
clip inventories. Do not modify any run/checkpoint/source artifact.

## Representation and partitions

For each of the 5,746 logical train clips, load its complete frozen DINO
spatial feature array, validate shape and nonempty finite frames, and compute
one mean-pooled 2048-dimensional vector. Read only each checkpoint's
`visual_adapter.spatial_projector.weight` and `.bias` from `state_dict` on
CPU. Validate both tensors' shapes and finiteness against the 2048-dimensional
source. Apply the checkpoint-specific linear projector to the pooled vector.
Linearity makes this exactly equal (within floating-point rounding) to
mean-pooling the projector output across all spatial frames. No FLAN model,
motion branch, factor head, or GPU training is needed.

Freeze one deterministic, signer-stratified, clip-disjoint probe partition of
the logical train IDs (70% fit / 15% validation / 15% test per signer; fixed
seed 0). Sort IDs within each signer, shuffle using the integer represented by
the first eight bytes of SHA-256(`"0:" + signer_name`) as the signer-specific
seed, assign floor(0.70 × count) to fit, floor(0.15 × count) to validation,
and remainder to test. Require all seven train signers in all three partitions
and identical clip IDs/order for E1 and E2. Signer03 and
Signer07 never enter probe fitting, tuning, or scoring. Persist the partition
identity/hash in the result. Probe-test clips were seen by the original E1/E2
translation training; the probe test measures linear accessibility on those
trained clips, not generalization to unseen clips or unseen signers.

## Classifier and report

Train one fresh seven-way linear classifier per checkpoint on the fit subset;
never reuse E2 adversarial/factor heads. Compute feature normalization using
fit-subset statistics only. Select weight decay from {0, 0.0001, 0.01} on
validation balanced accuracy, breaking ties toward smaller decay; use fresh
CPU PyTorch linear classifiers, seed 0, AdamW learning rate 0.01, and 200
epochs per candidate with finite-loss checks. Select best validation epoch
for each candidate, breaking epoch ties toward earliest epoch.
Score the untouched probe-test subset once per checkpoint. Report top-1 and
balanced accuracy, per-signer counts, majority-class baseline, and E2-minus-E1
differences, along with classifier/normalization/partition settings and all
checkpoint, protocol, source, and feature hashes. Aggregate results only; do
not emit clip-level signer predictions. Label the report `diagnostic_only` and
`human_review_status=not_assessed`; do not turn a lower accuracy into an E3
admission or signer-invariance claim.

## Implementation boundary and verification

Add a focused probe implementation under `src/despamo/evaluation/`, a separate
CLI under `scripts/`, and targeted unit tests. CLI defaults to no overwrites;
output is a new aggregate JSON outside frozen runs. Test stratification and
nonoverlap, pooling/projector equivalence on synthetic tensors, checkpoint
identity and shape rejection, train-only normalization, deterministic tuning,
and matched E1/E2 report assembly without loading full pretrained models.
Run targeted unit tests and Ruff. Live probe on existing external checkpoints
starts only after implementation and preflight pass; record actual runtime,
not a guessed duration.
