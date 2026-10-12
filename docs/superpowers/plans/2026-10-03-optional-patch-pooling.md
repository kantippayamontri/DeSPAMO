# Optional research plan: patch-based spatial attention/pooling

**Status:** Optional follow-up, saved at the user's request. No implementation,
GPU experiment, feature extraction, or training is authorized by saving this plan.
This is an experiment outline; establish the concrete design and bounded budget
when the user chooses to activate it.

**Goal:** Test whether selecting information from DINO patch features improves
translation compared with its current global CLS summaries, while retaining
SpaMo's existing spatial input shape.

**Context:** Finish evaluating the current residual-MLP / longer-linear control
and use its dev-selected downstream recipe as the comparison baseline. The other
server's closed-vocabulary, crop-derived descriptions may provide improved
train-only supervision when their versioned artifacts are available.

## Input contract and proposed approach

- Current spatial input: one 2048-dimensional vector per frame, obtained by
  concatenating 1024-dimensional DINO CLS features at resolutions 224 and 448.
- Proposed approach: score patch features, pool them into one 1024-dimensional
  vector per resolution, and concatenate the two outputs into the same 2048 width.
- A residual fusion with CLS is a useful candidate: preserve the global summary
  initially and learn a patch-derived refinement. Specify a non-dead initialization
  before implementation; zeroing both a gate and its residual output can block learning.
- The first learned-pooling prototype freezes the DINO encoder and trains a small
  pooler. Fine-tuning the encoder is a separate, more expensive future intervention.
- The downstream model continues to receive a `[T, 2048]` spatial array; separate
  hand/mouth input streams or new translation-time crop detectors are not required
  by patch pooling itself.
- Current cached CLS arrays cannot recover the discarded patch features. DINO
  must be run again, or a bounded patch cache created, to test this method.

This changes how the spatial summary is built. It does not guarantee attention
will focus on hands/mouth; usefulness must be measured against a matched control.

## Stage A: small forward and memory prototype

- [ ] Choose a deterministic sample of 100-500 training frames spanning the
  seven training signers; preserve source frame indices and image hashes.
- [ ] Inspect the pinned DINO configuration and output layout. Exclude CLS and
  register tokens from the patch grid; do not assume every token after CLS is a patch.
- [ ] Compare CLS with uniform patch pooling as a cheap implementation control,
  then exercise the proposed attention pooling output.
- [ ] Check token counts, finite values, output width 2048, and correct handling of
  both resolutions. If crop metadata supervises attention, verify its coordinates
  against the actual resize transformation and anatomical-side convention.
- [ ] Benchmark elapsed time and peak VRAM at representative batch sizes.
  Schedule GPU work after the current sequential experiment pipeline.
- [ ] Keep patch tensors transient or in a bounded diagnostic cache. Save compact
  pooled vectors and provenance for any persistent output.

**Pass condition:** Input/output alignment and numerics are correct, peak memory
fits the 16-GiB GPU, and the measured cost is acceptable. This establishes
feasibility, not translation improvement.

## Stage B: bounded learned-pooling feasibility study

- [ ] Select a small, clip-disjoint train/validation study inside the logical
  training inventory; frames from one clip must stay in one partition.
- [ ] Choose and document the pooler's learning objective and initialization.
  Candidate supervision is the new versioned articulator descriptions, with
  unknown/failed targets masked; never silently substitute a different text version.
- [ ] Validate compatibility of the new schema, masks, clip/frame join and encoder
  provenance. New annotation identities require an appropriate versioned experiment
  binding while preserving the logical signer split and translation sources.
- [ ] Learn only the small pooler first, keeping encoder weights fixed. Retain
  the CLS/uniform-pooling controls to distinguish a learned effect from an output change.
- [ ] Inspect gradients and variability/collapse diagnostics. Use visual attention
  inspection as a diagnostic, not proof of semantic correctness or signer invariance.
- [ ] Measure wall time, peak VRAM, cache size and useful learning before proposing
  a full comparison. Set a strict cache budget, such as at most 10 GiB for this study.

**Planning estimate:** Roughly 1-3 GPU-hours depending on objective, frame count,
cache strategy and optimization budget. This is not a measured benchmark or
automatic spending authorization.

## Stage C: full matched translation comparison, only if justified

- [ ] Freeze/version the selected pooler and extraction policy.
- [ ] Re-extract the full frame inventory, pool immediately on GPU, and persist
  only compact `[T, 2048]` arrays. Record pooler checkpoint, DINO revision/adaptation,
  token policy, frame/image hashes, code hashes and library versions.
- [ ] Train the same dev-selected downstream architecture and declared horizon
  against CLS and patch-pooled features. Hold split, initialization, decoder,
  batch size, optimization settings and checkpoint-selection policy matched.
- [ ] Compare full Signer03 dev BLEU-4, ROUGE-L, repetition rate, output lengths,
  and prediction diversity. Report corrected signer/articulator diagnostics
  separately; their heuristics are not validated BLEU gates.
- [ ] Select on dev before any authorized once-only Signer07 evaluation.

**Planning estimate:** Approximately 10-12 GPU-hours for full extraction plus
downstream comparison work with a fixed pooler, with additional pooler-learning
cost depending on the chosen recipe. Benchmark Stage A/B before setting caps.

## Resource evidence and storage constraints

Measured reference from the completed cosine feature run:

| Quantity | Value |
|---|---:|
| Clips | 7096 |
| Frames | 827354 |
| Two-resolution extraction time | approximately 9.37 GPU-hours |
| Compact float32 feature payload | approximately 6.31 GiB |
| Completed cache including manifests/receipts | approximately 6.58 GiB |

For patch size 16, the 224/448 grids contain 196 + 784 = 980 patches per frame.
At width 1024, storing all full-inventory patch features would require about
**3.02 TiB in float32** or **1.51 TiB in float16**, excluding metadata and extra
tokens. This is not an acceptable cache strategy for the current storage setup.

Prefer immediate compact pooling for extraction and a bounded subset/RAM cache
for pooler training. Joint online DINO + SpaMo training repeats encoder work at
every downstream step and can be much more expensive than the cached workflow;
its cost is outside the first prototype estimate.

Before activation, recheck Linux and Windows host free space. Use atomic, fsynced
checkpoint publication and bounded resume retention for any new training run;
preserve dev-selection models and historical artifacts. Keep supervision labeled
`qwen-schema98-unreviewed-v1` / `human_review_status=not_assessed` when applicable.

## Activation decisions

The user decides when to activate this optional work. Before implementation,
record the exact pooler design/objective, new-description manifest if used,
dataset subset, comparison recipe and GPU/storage caps. Start with Stage A;
advance to learned pooling and full extraction only on measured evidence.
