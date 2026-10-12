# Dev-only spatial/motion grounding diagnostic

**Authorization:** User approved Step 2, inference only, capped at **3600 seconds
of GPU-stage wall time**. No new training, vision extraction or Signer07 scoring.

## Question

How much do the two dev-selected linear 16000 models depend on spatial and motion
content, and how does each stream affect translation under a controlled mismatch?
Dependence does not by itself locate a projector bottleneck or prove semantic accuracy.

## Models and fixed conditions

- Linear seed0: `projector-linear-16000-eb0a5bef8210afc3d894f4ac57338f8cd5e0f01b6b4d74a8a38cb285b4f5f313`.
- Linear seed1: `projector-linear-16000-0319401a7579e5fc6ee3a37748c01a441b2e0b75cf97ce9778f6fc60e1f457da`.
- Both selected step16000 on complete full dev; validate selection/bytes/metadata.
- Fixed Signer03 dev subset, deterministic beam5 / max length64 / no ICL, batch4.
- Existing cached spatial `[T,2048]` and motion `[M,1024]` features.

## Donor selection

The 582 dev clips contain 196 exact `(spatial_length,motion_length)` groups.
135 groups have at least two clips with distinct normalized reference texts,
covering 521 clips. Therefore exact length matching is feasible and stronger than
approximate buckets; it avoids a content-versus-padding/prefix-length confound.

Using selection seed0, shuffle sorted eligible groups and choose64 groups. In
each choose two distinct-reference clips deterministically, giving128 recipients.
Donors swap within each pair. Record eligible/excluded counts and selected IDs,
pairs, references/length fingerprints and donor-map digest. This is a restricted
diagnostic subset, not a representative full-dev score.

## Four conditions, same recipients/batch order

1. `normal`: recipient spatial + recipient motion.
2. `shuffle_spatial`: donor spatial + recipient motion.
3. `shuffle_motion`: recipient spatial + donor motion.
4. `shuffle_both`: donor spatial + donor motion, using the **same donor** for both.

Every condition keeps recipient IDs, references, prompts and exact input lengths/
masks. No reference/gloss/multilingual field enters generation. Cache all selected
original arrays read-only; never mutate or overwrite feature files. Generate normal
fresh with the same subset batching as perturbations; do not mix cached full-dev
predictions into a numerically different batching comparison.

## Measurements and safeguards

- BLEU1-4, ROUGE-L, repetition/uniqueness/whitespace lengths per condition.
- Paired deltas relative to normal and exact prediction-change rates.
- Exact match to each donor's fresh normal prediction, including pairs where normal
  recipient/donor predictions differ. Generic identical predictions cannot alone
  establish modality independence.
- `shuffle_both` donor consistency is a sanity check: identical donor inputs should
  mostly reproduce donor normal output. Document possible BF16/batching differences.
- Optional exploratory donor-pair bootstrap: resample64 pair units, not128 independent
  recipients; conditional on this subset, not a broad significance claim.
- Preserve completed condition outputs and progress; refuse inconsistent source,
  checkpoint, decoder or donor map on reuse. Model loads sequentially, no overlapping GPUs.
- Hard child watchdog deadline and graceful shutdown reserve inside3600seconds.
- Use dev only; no change to canonical test registry or existing evaluation reports.

## Execution and interpretation

- [x] Add meaningful CPU regression tests for exact matched donors, no self/same-text
  donors, deterministic selection, equal masks, immutable originals and both-stream
  donor coherence; normal-condition output and metric/delta checks.
- [x] Implement isolated diagnostic runner, preflight and versioned artifacts.
- [x] Verify tests/lint/types, inspect resource/selection preflight, run under watchdog.
- [x] Compare both seeds and explain stream dependence versus redundancy, mismatched
  stream distribution effects and poor overall semantic grounding.
- [x] Recommend the next bounded intervention from evidence; do not automatically
  launch another training or architecture change.

## Completion

Primary inference completed in816.29seconds; supplementary batch1 donor sanity
in226.81seconds, combined1043.10seconds (17.39minutes) within3600.
Spatial-shuffle BLEU-4 deltas: seed0 -5.0895, seed1 -4.5876.
Motion-shuffle deltas: +0.0671/+0.3778, despite prediction changes75.78%/67.97%.
No absence of spatial conditioning found; motion influences decoding without a
demonstrated correct-pairing BLEU benefit on this subset.
See [full results and interpretation](../../dev-visual-grounding-2026-10-04.md).
