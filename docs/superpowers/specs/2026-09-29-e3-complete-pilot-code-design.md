# E3 Complete Signer-Pilot Code Design

## Outcome and scope

Prepare runnable, tested code for E3: adapt the pinned pretrained DINOv3
ViT-L with final-four-layer Q/V LoRA, extract adapted spatial features, then
train a fresh SpaMo/FLAN LoRA translation model on the frozen signer pilot.
This is code preparation, not permission to run the full adaptation, extraction,
or translation training. Keep recovered E3 files and old artifacts untouched;
integrate only reviewed behavior into the isolated signer-pilot worktree.

The diagnostic objective is reduced *nuisance* appearance information, not
removal of all appearance information. Preserve hand/mouth information,
measure translation quality separately, and make no invariance claim from the
one-step smoke or dev BLEU alone.

## Frozen inputs and admission

Require protocol hash
`0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7`,
dataset key `b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a`,
source/annotation/text/DINO/motion hashes, and immutable DINO base revision
`ea8dc2863c51be0a264bab82070e3e8836b02d51`. Physical PHOENIX `train/`
holds all 7,096 clips; logical train is seven signers/5,746, Signer03 dev 582,
Signer07 held-out test 768. Only logical train enters adaptation or translation
training. Failed Qwen clips retain translation targets but have all 19
appearance targets masked/zero. Schema-valid clips may have some individual
targets masked: 161 logical train clips do. Require target text-nullness to
agree with CLIP NPZ valid masks and masked vectors to be zero; do not treat
partially masked clips as wholly failed.

Training commands require explicit positive `--adapt-steps`,
`--calibration-steps` (strictly less than adaptation steps), and a separate
positive GPU-time cap for each GPU stage, including extraction. Fresh commands
refuse an existing output; resume accepts only its own incomplete output and
latest verified snapshot. Refuse insufficient storage, missing CUDA, or
unmatched source/checkpoint identity before GPU work. No
implicit full run, default run authorization, or new 20-hour feasibility claim.

## Stage 1: DINO adaptation

Use the separate locked `tools/dinov3` environment (transformers 4.56.2,
torch 2.5.1+cu121), CPU-pinned provenance preflight, and local pinned base
weights. Freeze base DINO weights; attach rank-8/alpha-8 LoRA to Q and V in
layers 20–23 only. For each sampled train clip, load five source PNGs at the
original frozen frame indices at 224 and 448 resolution; concatenate CLS to
2048-wide per-frame features. Fit seven fresh factor heads against cached CLIP
text vectors: four nuisance heads via gradient reversal, three hand/mouth heads
with positive gradients. Per-target masks control every contribution. Detach
original frozen DINO features as 224/448 cosine teachers, with reference weight
0.10; nuisance weights 0.05, articulator weights 0.10. Calibrate heads with
frozen LoRA for the declared calibration steps, then fit heads and LoRA jointly
with AdamW LR 3e-4 and 1e-4 respectively, gradient clipping, finite checks,
seed 0, physical batch 4, and `num_workers=0`. Joint GRL coefficient ramps
from 0 to 1 across the first 10% of joint steps; no held-out clip or reference
enters losses. Save durable bounded snapshots containing optimizer, sampler,
RNG, completed optimizer step, elapsed GPU spend, frozen identities, code and
lock hashes. Resume only from the latest owned complete snapshot without
rewinding the timer/sampler. Final adaptation checkpoint is the specified last
step; no dev/test selection in this stage.

## Stage 2: Adapted spatial feature version

Restore and verify the selected final DINO LoRA weights exactly; switch model
to eval and re-extract **all frames for all 7,096 physical `train/` clips** at
the same two scales, emitting `train/<clip_id>.npy` with 2048 columns and
unchanged row order. Dev/test logical groups refer to this physical `train/`
folder; official PHOENIX dev/test are untouched. Produce a new immutable
schema-1 manifest, complete frame-row map and per-clip receipts in a separate
versioned root bound to DINO checkpoint, source PNGs, base revision,
preprocessing, code/lock hashes. Verify zero missing/extra clips, expected
frame counts, finite values, and source hashes before atomic completion.
Interruptions resume only verified clips; never overwrite original DINO
features or present partial output as complete.

## Stage 3: Fresh matched E3 translation

Validate the **original** frozen protocol against its original DINO source,
then separately validate complete adapted spatial manifest/row map against
the selected E3 adaptation checkpoint. Construct logical train/dev/test views
using adapted spatial root plus unchanged motion root and unchanged physical
`train/` annotations. Do not make `load_pilot_views` accept an arbitrary
spatial manifest without checking its E3 binding.

Fresh seed-0 `E3_dino_lora` SpaMo/FLAN initialization must match E1/E2 shared
tensor hash `9e3a02201ca44b7ee1a1d62da3eba1308a91ee914cab983036dc6505aa8a4d88`.
E3 has no E2 projector factor heads: adaptation already supplied the factor
objective. Train 4,000 optimizer steps, physical batch 4, no accumulation,
1,000 VT-only warm-up, `num_workers=0`. Save complete checkpoints and score
all 582 Signer03 dev clips at steps 1,750, 2,800 and 4,000 using reference-free
German prompts and deterministic beam-5 decoding, selecting maximum corpus
BLEU-4 (earliest step on ties). Store protocol, adapted feature, initialization,
environment, and selected-checkpoint hashes with bounded spend and safe resume.

Only after dev selection, the separate held-out evaluator may score the
selected E3 checkpoint once on all 768 Signer07 clips under the frozen
decoder. It must verify IDs/references and all adapted-feature identities;
never select a checkpoint using Signer07. Include optional fresh signer probe
on the same fixed closed-set training-signer partitions as E1/E2; no probe
threshold gates training or justifies an invariance claim.

## Verification and handoff

Place DINO-environment adaptation/extraction code in `tools/dinov3` and
pilot-only E3 runtime/CLI in the main Python package/scripts. Test mask joins,
GRL/LoRA and teacher gradients, ownership/budget/resume, feature row/hash
integrity, adapted-view validation, shared initialization, dev selection and
held-out boundaries with small offline fixtures. Run focused tests, full unit
suite and Ruff, plus cheapest real data-alignment/gradient smoke. Provide
preflight and explicit training/extraction commands; do not execute full jobs
as part of code preparation. Label every E3 run, checkpoint and result
`qwen-schema98-unreviewed-v1` and `human_review_status=not_assessed`.
