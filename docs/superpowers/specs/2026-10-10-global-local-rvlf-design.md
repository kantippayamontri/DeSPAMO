# SigLIP 2 Global/Local RVLF-Inspired Translation Design

## Approved objective and publication scope

User approved a two-branch sign translation architecture, later replaced face
with mouth, and originally approved frozen CLIP ViT-L/14 for global/local features.
Latest instruction replaces that backbone with frozen
`google/siglip2-so400m-patch14-384` for both branches. The
[SigLIP 2 amendment](2026-10-10-siglip2-global-local-amendment.md) records verified
checkpoint metadata and changed extraction/projection contracts; it supersedes
CLIP-specific readout/resolution/dimensions for this separate-server experiment.
User additionally approved native attention pooling,384/768global and384local
extraction,2304global/1152local widths, retained768/2048fusion/connector, isolated
supported extraction runtime and publication of the amended branch documents.
User also approved FLAN-T5-XL, a2048channel connector, and one fine-grained
RVLF-inspired VT-Align loss on fused visual representation. User requested a
separate Git branch pushed for another server to develop this project while the
original machine continues its six-run200epoch study.

This publication contains baseline code, this design, a task-by-task plan and
server handoff. New model/extractor/loss implementation and training occur on
the other server. No local GPU experiment, source edit in the live study,
automatic test scoring or decoder replacement accompanies publication.

## Architecture

```text
GLOBAL BRANCH
Original RGB frames -> Frozen SigLIP 2 + tiled scales -> [N,2304] ----+
                                                                    |
Existing VideoMAE features [M,1024] -> Global fusion + temporal path --+
                                         |
                                 Global features G [Tg,768]
                                         |
                                         +---------------------------+
                                                                     |
LOCAL BRANCH                                                         |
Left-hand crop  -> Same frozen SigLIP 2 -> [N,1152] -> Project -+       |
Right-hand crop -> Same frozen SigLIP 2 -> [N,1152] -> Project -+       |
Mouth crop      -> Same frozen SigLIP 2 -> [N,1152] -> Project -+       |
                                                             |       |
                                    Masks -> Local fusion MLP |       |
                                                             v       |
                                          Local temporal path        |
                                                  |                  |
                                         Local features L [Tl,768]   |
                                                  |                  |
                                     Source-support alignment        |
                                                  |                  |
                                                  +------------------+
                                                                     |
                                                       Global/local residual fusion
                                                                 [Tg,768]
                                                                     |
                                                         Shared MLP connector
                                                                 [Tg,2048]
                                                                     |
                                          +--------------------------+---------------+
                                          |                                          |
                                    Fine-grained VT-Align                       FLAN-T5-XL
                                          ^                                          |
                                          |                                   Translation CE
                            German training token embeddings [Lt,2048]
```

There is one language model and one shared output connector. VT-Align is a loss
branch, not a processing stage before the LLM. Inference has no target reference,
reward computation or text-alignment loss. All three local regions combine into
one local representation before global/local fusion.

## Models and extraction

- Spatial/local encoder: `google/siglip2-so400m-patch14-384`, revision
  `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`, frozen/eval, vision width1152.
  Official config declares fixed-resolution `siglip`/`siglip_vision_model`, not
  NaFlex. Load according to pinned config; validate vision-only pooling output.
- Global: pinned SigLIP processor once at384, interpolate normalized tensor to768,
  four384tiles, attention-pool each view with pretrained native vision head;
  concatenate original1152vector then mean of four tile1152vectors for2304.
  This is S²-style pooled-view extraction, not exact SpaMo CLS wrapping. Do not
  apply CLIP center-crop/normalization or treat first patch as a CLS token.
- Local: one384processed RGB crop per region/frame; native attention-pooled
  vector1152. No local S², first-patch/mean-patch substitution, independently
  trained regional encoders or appearance adaptation. Pooling head remains frozen.
- Use one encoder instance/microbatched inference for all regions; store finite
  FP32 arrays with model/processor/runtime/precision and batching identities.
  Missing observations produce zero stored features and `valid=false`.
- Motion: unchanged VideoMAE-L/16 recipe,16frame window/stride8, width1024;
  reuse validated arrays with original generation provenance or prepare exact
  matching cache. Never infer window chronology from lengths alone.
- Decoder: `google/flan-t5-xl`, revision
  `7d6315df2c2fb742f0f5b556879d730926ca9001`; Q/V LoRA rank16/alpha32/dropout0.1.
  Base token embeddings stay frozen. LoRA receives translation CE gradients;
  raw target embedding lookup does not traverse LoRA transformer blocks.

## Region observations and tracking boundary

Input is a versioned region-observation manifest for every original frame and
region (`left_hand`, `right_hand`, `mouth`). Its provider must bind detector/model
configuration and checkpoint SHA, keypoint/box naming convention, coordinate
system, confidence definition, tracking/smoothing/crop policy and frame hashes.
No specific detector checkpoint is assumed available by this publication. A
detector preparation probe is a prerequisite to local feature extraction; it
must satisfy this contract before weights/thresholds are frozen. MMPose-style
whole-body landmarks are a candidate, not an unverified installed dependency.

Each observation carries original source index, pixel-coordinate box before and
after clipping, confidence in[0,1], hard validity, tracking identifier and quality
flags. Require anatomical left/right identity rather than screen-left ordering;
do not flip crops. Mouth crop includes lips plus surrounding jaw/margin, not
full-face appearance. Crop/tracking parameters are explicit identity fields,
frozen after the train-only preparation probe and never tuned on test labels.
Reject NaN/out-of-bounds/degenerate boxes. Occlusion/missing landmarks must not
silently copy an old crop as valid. Smoothing must remain within the current
clip; batch order must not affect boxes. Face/eye/eyebrow detail has no dedicated
stream in this design, although global view remains available.

## Data and portable provenance

Official PHOENIX14T split:7096train/519dev/642test, clip IDs disjoint; all nine
signers appear in each split. Full source frames, ordered per clip; expected
frame totals827354/55775/64627 (947756total). No crop/caption changes in the
existing original-host artifacts. Prepared regional features may cover test
frames without using translations; that is label-free preparation, not scoring.

Store global `[N,2304]`, local `[N,3,1152]`, local validity `[N,3]`, confidence
`[N,3]` and explicit source indices/boxes. Separate canonical frame/content
identity from machine-local path bindings. Same bytes/indices can transfer
across servers; path-bound old receipts cannot silently be relabeled unchanged.
Preserve original package receipts and publish separately versioned rebinding
receipts that verify actual transferred arrays/source hashes. Runtime loading
uses explicit package roots; no source globals/manifest paths from another host.

New portable config contains annotation/frames/global/local/motion/cache/output/
archive roots, seed, fusion variant and alignment variant. Resolve paths from
config/environment, not `/home/kan` or `/mnt/b`. Normal Linux storage checking
uses actual native filesystems; Windows backing-drive check applies only when
configured for WSL. Preserve50GiB reserve plus measured remaining demand.
Global+local FP32 arrays alone need about20.34GiB before motion, boxes/receipts,
checkpoints and archives. Compute actual candidate/checkpoint demand in preflight.

## Global path and temporal alignment

New spatial projector2304→1024/GELU→768 accommodates SigLIP global width; keep
motion1024→768 and existing global temporal construction. Match fresh global
weights across the new SigLIP variants, not against the old2048input layer, which
has a different shape. Do not pad/truncate features to fake old-checkpoint
compatibility. The published baseline
`SpaMoVisualAdapter` concatenates spatial and motion tokens before K5/P2/K5/P2.
The aligned-fusion extension also retains motion tokens. Thus its output is not
automatically a uniform frame-grid sequence; do not interpolate local features
by output row number or equal lengths and claim correct temporal alignment.

Assign each global input token its actual source-frame support: spatial token
maps to its original frame; motion token maps to real16frame window support
(short-window policy from provenance). Propagate support sets through conv/pool
receptive fields. K5/P2/K5/P2 has nominal16input-token support with output stride4,
but source-frame support can be noncontiguous at the modality boundary.

Local temporal outputs carry their own actual frame support and reliability.
Align local to each global output using normalized overlap of those support
sets, without extrapolating into uncovered frames or replacing noncontiguous
sets with broad min/max envelopes. Empty/unsupported local contribution is zero.
Implementation must preserve global ordering and tensor values exactly when
local branch is disabled, including existing BatchNorm buffers. This adds
provenance-aware local alignment without silently redefining global baseline.

## Local and global/local fusion

Each local region projects1152→768. Mask before projection to prevent invalid
NaN values poisoning gradients, and after projection to remove bias-only output.
Concatenate regions in fixed anatomical order, producing2304channels; local
MLP2304→768/GELU→768 produces one frame-level local representation. No local
attention or per-region confidence gate in the first controlled comparison.
Local temporal module preserves declared source support and never treats padded
frames as observations; a learned bias cannot create valid local evidence.
Temporal normalization/short-sequence policy must be pinned before model identity:
prefer per-token LayerNorm in the local path so missing regions do not contaminate
batch statistics. For short clips, zero-pad to required temporal support, mask
padding and carry only real-frame support; do not drop translation clips.

Fixed residual: add fixed-weight local contribution to G. Confidence-aware
residual: gate sees G, aligned L, three validity/coverage summaries, and three
observation-confidence summaries; dynamically weights the same local feature.
Hard usable-local mask applies after local fusion, temporal modeling and alignment.
All local evidence missing must produce bitwise-equal global-path output and no
local gradient contribution. Reliability is not synonymous with semantic value.

Initial engineering defaults for implementation review: fixed contribution0.1;
learned gate `0.2*sigmoid(logit)` with zero final weights/bias so initial weight
also0.1. This avoids conflating different starting weights with learned gating.
Use768-wide common G/L, one shared output MLP768→2048/GELU→2048. Create local
modules in separate seeded RNG context; common global/FLAN parameters must match
across compared runs. These numeric gate/local-normalization choices are proposed
implementation defaults, not empirical optima or additional completed approvals.

## One fine-grained VT-Align loss

Consume fused post-connector visual tokens `[B,T,2048]` and frozen FLAN target
token embeddings `[B,L,2048]`. RVLF uses a separate contextual text encoder;
this uses the RVLF matching objective with raw FLAN embeddings and must be labeled
RVLF-inspired, not full RVLF reproduction. No auxiliary global/local contrastive
losses or GRPO are included in initial model.

Project visual/text tokens to a shared alignment space (proposed width256),
L2-normalize tokens, compute similarities for all B×B video/sentence pairs.
For each visual segment, softmax over valid text tokens and aggregate similarities;
compute reverse text→visual aggregation over valid segments. Average only over
real valid query positions. Temperatures positive/bounded and computations FP32.
Padded/invalid positions never enter attention denominators or mean counts.
Handle extreme temperatures and short/empty masks explicitly without NaN.

Apply bidirectional batch contrastive objective with duplicate normalized German
references treated as multi-positive, avoiding false-negative pressure. One
loss on fused representation updates connector, local/global modules and alignment
projections/temperature. Candidate selection is dev BLEU4, not alignment scores.
Initial objective: CE+alignment(weight1) after VT-only warmup; global-only control
uses identical fine-grained loss so local-feature gains are not confounded by a
simultaneous objective change. Pooled-vs-fine-grained loss is a separate paired
global-only experiment in the approved roadmap.

## Training, comparison and recovery

Reference recipe: physical8/no accumulation/drop-last,887updates/epoch,
max200/min40; VT-only4000, LRwarmup10000, adapter3e-4→5e-5 and FLAN5e-5→1e-5
cosine over177400updates. AdamW0.9/0.98, eps1e-8, decay0.01, clip1; BF16 network
and FP32 alignment; transformer-only activation checkpointing, no visual-BN
recomputation. Three train-only English/French/Spanish=German pairs, same exclusion
policy and fixed per-seed inference bank, prompt limit1024, deterministic beam5/
output64. Profile actual global/local memory before choosing deviations; changing
physical batch changes contrastive negatives and is a versioned experiment.

Dev519 every2epochs; best actual retained/earlier ties; patience10 against last
significant0.10gain, ignore VT-only checks, stopping allowed at40. Save complete
mutable model/buffers/optimizer/sampler/global RNG/prompt RNG and expanded modules.
Resume exact identity/recipe/source/model/data only, retain elapsed charges and
replayed updates. Publish checkpoint before dev, archive candidate before pruning,
reconcile reports/stopping/pointers after interruption; latest2native resumes plus
selected best, all actual dev candidates retained. Owned locks/PID/boot/start
identity, explicit manual stop and bounded verified recovery; no duration stops.

Matched local comparisons (two seeds0/1 each) after prerequisite SigLIP global baseline:
M0 global-only/fine-grained; M1 same + local fixed residual;
M2 same local branch + confidence-aware global/local gate. Hold crop/cache/loss/
initial global weights/decoder/schedule/stopping fixed. Do not choose global fusion
arm based on test; initial global arm is reference. Additional local confidence
weighting, decoder Qwen, appearance adaptation and grounding rewards are later
separately designed comparisons.

This branch provides development scope, not an unattended launch of six more
runs. Another-server operator reviews spec/defaults/detector choice, implements
and verifies plan, then launches within explicit agreed training scope. Use train
references for optimization, dev for selection. No new automatic official-test
scoring; a later frozen evaluation needs its own exact selection/claim protocol.

## Evidence and claim limits

RVLF paper: Rao et al., "RVLF: A Reinforcing Vision–Language Framework for
Gloss-Free Sign Language Translation", CVPR Findings2026, sections3.1/3.2;
segment-token matching Eq3–9 and hand/face addition Eq12. That paper uses skeleton
pretraining, DINOv2 hand/face, Qwen3 and GRPO; this design does not reproduce those
components. Mouth-only high-resolution stream also differs from its full face.
Confidence gating/two branches are hypotheses, not established novelty or causal
grounding. Test low-visibility/missing-region behavior and gradient flow; higher
BLEU alone does not establish calibrated reliability or signer invariance.

Before implementation, review this written design and the task plan, particularly
detector preparation, local temporal policy, gate/projection defaults and temporal
support mapping. Plain-text diagrams are used per user preference.
