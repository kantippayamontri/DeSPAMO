# Time-aligned spatial/motion fusion experiment

## Intent and approval status

User approved an experiment using temporal interpolation and gated residual
motion correction. Goal: test whether attaching nearby movement to spatial
frames improves translation over the current sequential fusion. This is an
architecture experiment, not an encoder or caption-data experiment.

User approved this written design and its **18 GPU-stage hour / 64800 second**
ceiling on 2026-10-05 after storage cleanup and hardware verification. Backbone
is the recommended plain spatial MLP. The ceiling covers six training/dev-selection
runs, admission and GPU smokes. Implementation follows implementation-plan review
and execution-method selection.

Study hardware is now RTX4090 (24564MiB), driver610.62. Existing torch2.0.1+cu117
passed CUDA/BF16 forward/backward smoke. Measured Linux/Windows free space is
435.07/285.82GiB after user compaction, above the approximately224GiB estimate.
Record fresh hardware/software identities and profile runtime before training;
retain batch4 and the matched recipe rather than spend extra VRAM on a new recipe.

## Fixed inputs and backbone

Use existing cosine DINO spatial cache `7122b42e...` and existing 1024-wide
VideoMAE motion cache. Frozen logical train/dev/test counts are 5746/582/768.
Train on seven signers; choose checkpoints only on Signer03 dev. Signer07 is
excluded from this study's scoring and selection. Its earlier inspected scores
mean it cannot be presented as a never-inspected test in future studies.

Backbone: spatial Linear2048→1024/GELU/Linear1024→768; motion Linear1024→768;
original K5/P2/K5/P2 TCN; output Linear768→2048/GELU/Linear2048→2048;
FLAN-T5-XL LoRA and existing VT-Align. Retain original motion tokens after the
enriched spatial sequence. Input/output token counts and masks stay matched.
Start all runs fresh, not from selected earlier checkpoints.

## Timing contract

User selected authentic original extraction records after discussing assumed
alignment. This study will use the original-record route; assumed timing is not
authorized by that choice. A focused follow-up found no such records in surveyed
project, recovery, SpaMo or dataset metadata; see
[records check](../../original-motion-records-check-2026-10-05.md). Obtain actual
historical run/source-order/packaging records before admitting the existing cache.
New extraction with new records would be a separate data-generation decision;
it would not recover the original released cache's historical records.

Local SpaMo extractor uses 16-frame windows and default overlap8 (stride8),
with full windows only; shorter clips repeat the last frame to reach16. The
existing motion manifest stores lengths but not explicit window positions.
All7096 physical-train clip lengths matched this rule during read-only checking,
but length agreement alone does not prove extraction order or historical flags.

Before training, bind extractor/helper bytes, recover available extraction
provenance and compare source frame ordering with spatial mapping receipts.
Write a separate versioned timing-map artifact; never rewrite frozen caches.
If stride or row order cannot be established, stop and report the missing
provenance rather than claim verified time alignment from length ratios alone.
If reconstructed from source conventions rather than an original extraction
receipt, label that provenance explicitly as reconstructed.

For verified window j, start=8j, end=8j+15 and centre=8j+7.5 in zero-based
source frame indices. For a source frame t between centres c_j and c_(j+1):

    a(t) = (1-w) * m_j + w * m_(j+1)
    w = (t-c_j)/(c_(j+1)-c_j)

Interpolate projected 768-wide motion vectors from the same clip only. Before
the first centre or after the last centre, clamp to that boundary vector only
within the recorded real-frame support of a window. Disable correction on
uncovered trailing frames, padding, or invalid spatial rows. No extrapolation
across uncovered tails. A single-window clip uses that vector on its support.
Short-clip support excludes synthetic repeated frames as spatial observations.
Use original source indices, not padded positions or normalized length ratios.
Reject count/map mismatches and invalid/empty feature sequences during admission.

## Three matched arms, two seeds

Seed0 and seed1 each train these three arms:

1. **Reference:** original concatenation, using the plain spatial MLP backbone.
2. **Aligned fixed gate:** s'(t)=s(t)+v(t)*0.5*R(a(t)).
3. **Aligned learned gate:** s'(t)=s(t)+v(t)*g(t)*R(a(t)).

Here v(t) is the timing-valid mask. Correction R is Linear768→768/GELU/
Linear768→768, with final weight and bias zero. Learned scalar gate is
sigmoid(Linear1536→1(concat(s(t),a(t)))); zero weights/bias initialize it to0.5.
No extra normalization, dropout or auxiliary loss in the fusion branch.
Scalar gate per frame limits capacity and simplifies gate diagnostics.

After enrichment, concatenate valid spatial tokens with original valid motion
tokens and run the existing TCN/output path. Fixed/learned arms share identical
correction initialization. Fork RNG for new modules so shared backbone weights
and RNG are unchanged across matched arms. Attest hashes and exact output parity
at initialization; account for added parameters explicitly.

At zero correction initialization, gate and first correction layer may have
zero gradients on the first update. Require finite nonzero correction-output
gradients initially, then gate/earlier-layer gradients after correction begins
learning. Do not misclassify intentional zero-init behavior as disconnected code.

## Matched recipe and selection

16000 updates, batch4, no accumulation, VT-only warmup2000, LR warmup1600,
cosine to zero; adapter/VT LR3e-4, FLAN LoRA LR5e-5; AdamW betas0.9/0.98,
eps1e-8, decay0.01; BF16, gradient clip1. Beam5/max64/noICL dev decoding.
Evaluate full582-clip dev at5500/10400/16000; select highest BLEU-4, ties choose
earlier step. Preserve existing Qwen policy and human_review_status=not_assessed
where those provenance fields apply; new descriptions are not introduced.

Use fresh within-study references: earlier reference variability is unexplained.
Compare aligned-fixed versus reference for the combined alignment/correction
effect, and learned versus fixed for gating's incremental effect. The first
contrast does not isolate timing from extra capacity. A shifted-timing control
would be a separately approved follow-up, not an automatic extra training arm.

## Verification, budget and recovery

Tests cover interpolation weights, first/last supports, uncovered tails,
single/short windows, non-contiguous source indices, batching/padding isolation,
initial backbone parity, matched common initialization, parameter optimizer
coverage, finite joint CE+VT gradients, checkpoint restore and timing admission.
Run cheapest real-data smoke before admitting long training.

Approved overall cap64800 seconds, per-run cap10800 seconds. Count GPU-stage
initialization/smokes/dev scoring conservatively. Hard watchdog, sequential GPU
jobs, durable elapsed accounting and incomplete status on cap/integrity failure.
Check both Linux and Windows-host free space with50GiB reserve before each run.
Retain all dev-selection models and latest2 atomic/fsynced resume snapshots.
No automatic deletion or host compaction. Resumes validate immutable source,
feature, timing, initialization and recipe hashes.

## Reporting and limits

Report both seeds' complete dev trajectories, paired BLEU1-4/ROUGE-L deltas,
repetition/uniqueness, gate distributions and timing coverage, added parameters,
token counts, peak memory and charged runtime. Show per-seed effects rather than
claiming a winner from one favorable run. Optional frozen-model dev motion
shuffle/timing perturbation diagnostic must fit the approved budget and be
predeclared in the implementation plan before training. No automatic Signer07
generation, additional seeds, new encoders, patch caches or caption changes.

Positive dev gains justify further evaluation, not broad signer invariance or
proof of causal motion grounding. Implementation remains opt-in under a new
policy; previous checkpoint-bound runners and reports retain their identities.
