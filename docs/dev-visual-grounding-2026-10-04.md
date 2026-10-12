# Dev-only visual grounding diagnostic

## Result

Approved Step 2 completed on both dev-selected linear 16000 checkpoints.
Primary four-condition inference took **816.29 seconds (13.60 minutes)**.
A small donor-batching sanity replay took **226.81 seconds (3.78 minutes)**;
combined **1043.10 seconds (17.39 minutes)**, within the one-hour authorization.
No training, feature extraction, or Signer07 generation occurred.

**Spatial content is strongly used.** Shuffling it reduced subset BLEU-4 by
5.09 points at seed0 and4.59 at seed1. Motion also affects predictions, but
correct motion pairing did not show a corpus-BLEU advantage over shuffled motion
on this restricted subset. This does not mean motion is ignored or useless.

## Fixed models, data and donor selection

Both selected linear checkpoints are step16000, seeds0/1, reconstructed with
strict full-checkpoint metadata/state validation. Decoder is deterministic beam5,
maximum length64, no ICL, inference/eval BF16 and batch4.

The 582 Signer03 dev clips contain196 exact `(spatial_length,motion_length)`
groups.135 groups with distinct normalized references contain521 eligible clips;
61 singleton clips/groups were excluded. Selection seed0 chose64 groups and
two different-reference clips per group, giving128 sorted recipients and a
reciprocal donor map. These subset scores are not canonical full-dev results.

Donors have exactly the same spatial and motion lengths as recipients. Recipient
IDs, references, prompts, batch order, lengths and masks remain fixed. Arrays
are cloned rather than modified. Both-stream shuffling uses one coherent donor.
Normal predictions were freshly generated with the same batching as perturbations.

## Four-condition results

ROUGE-L F1 is on its native0-1 scale. Changed predictions are exact string changes
relative to the recipient's fresh normal output. Repetition flags duplicate
overlapping casefolded whitespace4grams, not every repeated word.

| Seed | Condition | BLEU-4 | Delta vs normal | ROUGE-L F1 | Changed predictions | Repeated-4gram outputs /128 |
|---|---|---:|---:|---:|---:|---:|
| 0 | Normal | 6.9564 | 0 | 0.2321 | 0% | 43 |
| 0 | Shuffle spatial | 1.8668 | **-5.0895** | 0.1077 | 100% | 44 |
| 0 | Shuffle motion | 7.0234 | +0.0671 | 0.2231 | 75.78% | 47 |
| 0 | Shuffle both | 0.8709 | **-6.0855** | 0.0921 | 100% | 40 |
| 1 | Normal | 6.7571 | 0 | 0.2309 | 0% | 28 |
| 1 | Shuffle spatial | 2.1694 | **-4.5876** | 0.1093 | 100% | 22 |
| 1 | Shuffle motion | 7.1348 | +0.3778 | 0.2320 | 67.97% | 24 |
| 1 | Shuffle both | 1.3657 | **-5.3913** | 0.0991 | 100% | 31 |

### Interpretation

- Spatial shuffling produces a large and directionally consistent loss in both
  seeds. The hypothesis that this linear model barely conditions on spatial
  features is not supported by this diagnostic.
- Motion shuffling changes97/128 predictions at seed0 and87/128 at seed1.
  Motion therefore affects decoding. Its correct correspondence was not better
  by aggregate BLEU here; seed0 ROUGE nevertheless fell slightly.
- Small positive motion-shuffle BLEU deltas are not evidence that random motion
  should be deployed. Perturbation creates a cross-modal mismatch; correlated
  spatial cues, redundancy, aggregation and decoder response can explain the result.
- Both-stream shuffling yields much worse scores against recipient references,
  consistent with predictions following donor visual content.
- Fewer repeated outputs under a corrupted condition do not establish better
  translation: seed1 spatial shuffle reduces repetition while BLEU collapses.

## Donor consistency and batching check

In the primary batch4 study, shuffled-both output exactly matched donor-normal
output for98/128 clips at seed0 (76.56%) and92/128 at seed1 (71.88%). All128
recipients had different normal predictions from their donors, so the rates are
not inflated by generic identical-output pairs.

These matches were lower than an ideal equivalence check. A supplementary replay
used the first eight selected donor pairs (16 clips), batch size1, normal and
shuffled-both only, for both models. It reproduced donor-normal output for
**16/16 at both seeds**. The same clips had11/16 and14/16 donor matches in the
primary batch4 outputs.

This supports batching/numerical sensitivity as an explanation and confirms the
small replay's donor construction. It does not identify the exact BF16/kernel or
generation mechanism, nor resolve every primary discrepancy. Do not claim
bitwise batch-invariant generation from this study.

## Recommended next step

Retain linear16000 as the stable primary reference. Do not treat deeper spatial
MLP capacity or an absence of visual dependence as the diagnosed bottleneck.

The next architectural candidate is **time-aligned spatial/motion fusion**:
verify actual VideoMAE window timing, align corresponding appearance/movement
internally, and test one fusion change at equal16000 budgets across matched seeds.
It can retain the existing two input streams and feature dimensions. No such
training is launched by this diagnostic.

When crop-derived closed-vocabulary descriptions arrive, separately compare
their encoder supervision/features using a fixed downstream recipe. Spatial
dependence is not proof that current global descriptors preserve fine articulator
details or ground dates/temperatures correctly. Patch pooling remains optional.

## Limits

- 128 selected clips, 64 length groups, one dev signer; singleton lengths excluded
  and groups sampled uniformly. Do not generalize subset BLEU to all582 clips.
- One fixed donor map; no perturbation-randomness sweep or bootstrap implemented.
- Two trained seeds only; no broad significance or signer-invariance claim.
- Shuffling measures dependence under mismatch, not causal identification of one
  faulty layer. A single-stream test cannot separate redundancy from weak fusion.
- Caption differences ensure distinct utterances, not equal semantic distances.
- Source/model identities, layouts, feature bytes and code are bound; source drift
  refuses partial/completed reuse. Timers have a parent hard watchdog.

## Artifacts and verification

Runtime directory:
`/home/kan/datasets/despamo/signer-pilot/projector-dev-grounding-v1/`.

- `summary.json`, `selection.json`, `provenance.json`, `state.json`.
- `seed{0,1}-{normal,shuffle_spatial,shuffle_motion,shuffle_both}.json`.
- `batch1-donor-sanity.json`: supplementary16-clip results and combined elapsed time.
- Launch log: `../projector-dev-grounding-launch.log`.
- Runner `scripts/diagnose_projector_grounding.py`; launcher
  `scripts/run_projector_grounding.sh`; plan in `docs/superpowers/plans/2026-10-04-dev-visual-grounding.md`.

Code verification: **908 tests passed,4 skipped**; full Ruff, targeted Pyright,
whitespace and shell syntax clean. Independent review's inference-source binding
finding was regression-fixed before inference. Deferred minor coverage:
exhausted-budget preflight readiness and real mid-stage alarm/crash accounting.
Current fresh run completed with all eight condition reports, both checkpoint
identities and no test-set registry change.
