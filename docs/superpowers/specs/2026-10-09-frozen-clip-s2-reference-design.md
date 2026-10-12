# Frozen CLIP + S² Reference Comparison Design

## Decision and purpose

User approved trying frozen CLIP + S² with two reference translation runs, seeds
0/1, after the active six-run SpaMo-style200 study and its planned frozen test
evaluation finish. Adapted CLIP is a later decision based on dev evidence. This
document records the accepted experiment design; implementation and launch are
not part of the current documentation request.

**2026-10-10amendment:** user subsequently approved immediate intentional stop of
current watchdog/controller/worker, implementation/extraction and two-run CLIP
launch in Build mode. This supersedes finish-six-and-test prerequisite below.
Use verified predecessor manual stop/all owners exited as execution gate. All
old weights/receipts/source/test claims remain intact. No predecessor test is
scored. Only DINOseed0reference is complete; report seed1as unpaired rather than
inventing baseline or adding unapproved DINO runs.

Compare the spatial feature pipeline against matching adapted-DINO reference
runs. The contrast changes encoder family, adaptation status, preprocessing and
spatial extraction together. It cannot isolate adaptation or S² alone, prove
signer invariance, or establish exact reproduction of published SpaMo.

## Execution boundary

- Keep current six-run jobs, code, locks, model/feature bytes and environments
  unchanged. No extraction/profile worker competes for the GPU while live.
- Before GPU work, require the active controller to report complete, all six
  training completions and frozen selections to verify, six existing once-only
  test reports and archival receipts to verify, and all former GPU owners to exit.
- Implement in the retained isolated worktree under new `tools/clip_s2/` namespace.
  Preserve existing `tools/followup/`, `tools/benchmark/` and main `src/` bytes.
  Bind every imported helper, source, lock and model file to the new run identity.
- Use existing main `.venv` and uv locks. Do not mix extractor environments or
  upgrade dependencies. Runtime:
  `PYTHONPATH=<main>/src:<worktree>:<main> LD_LIBRARY_PATH=/usr/lib/wsl/lib
  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0`.
- No commits, pushes, prior-artifact deletion, caption changes, extra seeds,
  CLIP adaptation, Qwen, Q-Former or additional fusion arms in this experiment.

## Spatial feature contract

Model: `openai/clip-vit-large-patch14`, immutable revision
`32bd64288804d66eefd0ccbe215aa642df71cc41`. Existing cache:
`/home/kan/datasets/spamo/hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41`.
Verified configuration: vision width1024, image size224, patch size14,24layers.
Load `CLIPVisionModel`, freeze parameters and use FP32 evaluation/inference with
autocast and TF32 disabled. Store finite FP32 `[frame_count,2048]` arrays.

The cache contains weights/config but lacks `preprocessor_config.json`. Fetch
only processor metadata at the pinned revision during later implementation;
store/hash it in the new experiment's native input directory, not in the shared
frozen snapshot. Verify its CLIP resize, center crop, RGB rescaling and
normalization configuration before inference. No default-only fallback. Record
processor class, resolved settings, metadata bytes and runtime versions.

Match `/home/kan/Research/SpaMo/scripts/vit_extract_feature.py:46-59` and
`/home/kan/Research/SpaMo/utils/s2wrapper.py:63-95`:

1. Decode original frame to RGB. Apply pinned CLIP processor once, producing
   normalized224×224 tensor. This includes its center crop; do not substitute
   DINO whole-frame preprocessing or per-tile processing.
2. Process scale1 through the model and take final hidden-state CLS of width1024.
3. Bicubic-interpolate the preprocessed tensor to448×448 using the wrapper's
   `align_corners=False`, `antialias=False` behavior. Split into four contiguous
   224×224 tiles in row-major tile order, retaining frame association.
4. Encode each tile at224. Mean the four final hidden-state CLS vectors per frame.
5. Concatenate scale1 CLS followed by scale2 mean CLS, obtaining2048dimensions.

Use `hidden_states[-1][:,0]` or equivalently verified `last_hidden_state[:,0]`.
Do not use `pooler_output` (post-layer normalization), projected image features,
text embeddings, patch pooling or L2-normalized CLS. SpaMo computes patch grids
inside its wrapper but discards them in the exported first-token result. A
CLS-only implementation is valid only with whole-wrapper output parity tested.

Test tolerance is fixed before results: FP32 CUDA parity
`torch.testing.assert_close(rtol=1e-4, atol=1e-5)` with identical weights/runtime
and processor. CPU deterministic toy aggregation tests require exact equality.
Test at least two frames simultaneously so tile/frame mixing is detectable;
include one-frame and incomplete frame-batch cases. Tolerance failures block
publication rather than permitting unrecorded precision changes.

Source comparison scripts are read-only parity references, not runtime imports
from the mutable SpaMo checkout. Their hashes join the verification receipt.
Frame inference batch is selected from32/16/8 by a real profile, largest passing
first; microbatching tiles may preserve frame batch. Require at least2GiB measured
GPU free headroom and peak reserved<=21GiB. The accepted batching and precision
are part of feature identity; no automatic batch change under the same key.

## Data, provenance and storage

Annotations: `/home/kan/Research/SpaMo/preprocess/Phoenix14T/{train,dev,test}_info_ml.npy`.
Frames: `/home/kan/datasets/despamo/phoenix-frames-210x260px`.
Use the completed official feature package's row map and ordered source receipts,
not guessed filename sampling. Expected coverage:

Source row-map:
`/home/kan/datasets/despamo/official-phoenix14t-features-v1/spatial-a866a2a0bfcc45d7e6ff5555ee22bb928fd86f1f166af7382d1d05b233cac90b/complete/frame_rows.json`.
Bind its bytes and resolve recorded paths against the frames root without changing
frame order or silently joining a different clip/split.

| Split | Clips | Frames |
| --- | ---: | ---: |
| train | 7096 | 827354 |
| dev | 519 | 55775 |
| test | 642 | 64627 |
| total | 8257 | 947756 |

New feature base:
`/home/kan/datasets/despamo/official-phoenix14t-clip-s2-features-v1/spatial-<identity-digest>`.
Reuse the existing manifest schema, per-clip receipts, `source_indices` and atomic
complete publication. Every source frame row and hash must match the current
DINO/motion source inventory. Test-frame extraction is label-free preparation,
not test scoring; test references never enter extractor computation or prompts.

Reuse motion package in place:
`/home/kan/datasets/despamo/official-phoenix14t-features-v1/motion-1123ff445677bf310ac0b8c9b45cac4f690b9102f5dad1960aa2fef04937e82d`.
Pin manifestSHA
`023a054fe0d3b9e04b5c0472ef77ebb8d031130e604841d67acfe7d81f66fd03`.
For each clip verify equal ordered source indices/content hashes and motion length
`max(1,(frame_count-16)//8+1)`. Preserve all frames, no spatial cropping/subsampling.

Native arrays alone occupy `947756*2048*4 = 7764017152` bytes, about7.23GiB.
Require remaining arrays plus3GiB extraction metadata/staging allowance and6GiB
translation allowance, plus50GiB reserve on both Linux and Windows backing disk.
For B: budget40GiB plus50GiB reserve for up to200model-only dev candidates across
two runs and final payloads; verify projection against measured checkpoint sizes.
Recalculate remaining demand as artifacts publish, without charging completed
arrays twice. A2026-10-09 snapshot showed C:about58GiB free; current headroom does
not satisfy the initial combined native projection. Recheck after current study.
If still insufficient, report storage blocker and request explicit storage action;
no automatic deletion, relocation, reserve bypass or active WSL compaction.

## Matched translation contract

New root: `/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v1`.
Archive: `/mnt/b/DeSpaMo/archives/official-phoenix14t-translation-clip-s2-spamo200-v1`.
Reference arm only, seeds0/1 sequentially, fresh initialized model/optimizer.

- Same2048→1024/GELU→768 spatial MLP, existing motion path, K5/P2/K5/P2 temporal
  connector and2048-dimensional FLAN output connector. FLAN-T5-XL revision
  `7d6315df2c2fb742f0f5b556879d730926ca9001`; Q/V LoRA16/32/dropout0.1.
- Physical batch8, accumulation1, drop-last;887updates/epoch,177400update ceiling.
- VT-only first4000optimizer updates, then CE+VT(weight1), ordinary padded mean.
- Linear LRwarmup10000updates; cosine adapter/VT3e-4→5e-5 and FLAN5e-5→1e-5
  over declared200epoch ceiling. AdamW betas0.9/0.98, eps1e-8, decay0.01, clip1.0.
- Same BF16 training, FP32 VT, transformer-only activation checkpointing,
  embedding-output grad hook, training cache disabled; no visual-BatchNorm replay.
- Same three train-only English/French/Spanish=German pairs, training reference
  exclusions, checkpointed prompt RNG,1024prompt limit and per-seed inference bank.
- Dev519 every2epochs; max200/min40; patience10eligible checks; minDelta0.10
  against significant anchor. Exclude VT-only checks; select actual best even
  below threshold, earlier ties; stopping reconstructed from ordered reports.
- Generation deterministic beam5/output64, evaluation batch4; same SacreBLEU13a
  and ROUGE-L fractions. No test-driven prompt or schedule tuning.
- Verify common mutable/frozen initial fingerprints, ordered train IDs and prompt
  partner IDs against completed200epoch DINO reference identity for each seed.
  Full run identity intentionally differs in spatial pipeline, source and roots.

`tools.benchmark.translation_data` hardcodes DINO roots in both admission and
batch loading; do not reuse its `batch_for` unmodified. New admission returns
explicit feature roots and validated bindings; new batching uses those roots.
Reuse unchanged model/update/recipe/sampler/checkpoint helpers by explicit imports.
Lifecycle orchestration belongs to new namespace; do not mutate old globals or
reuse six-selection validation for a two-run study.

## Recovery, publication and evaluation boundary

Native full resume before every epoch's dev evaluation; retain latest2 and selected
best. Archive every actual dev candidate with verified bytes before native pruning;
retain native originals required by policy. Checkpoint state includes persistent
buffers, AdamW, sampler, global RNG and prompt RNG. Recovery derives patience from
durable reports, reconciles selected pointer and closes publication crash windows
without advancing after a stopping decision. Detect extra/missing/reordered reports.

Unique extraction/controller/worker/monitor locks; exact PID/boot/start ticks and
command ownership. SIGTERM stops only owned work and leaves durable stop/resume
state. Monitor interval1200seconds, maximum3verified same-root transient recoveries;
unknown failures flag `needs_agent`, manual stop prevents restart. No duration stops.

Freeze exactly two reference dev selections and publish dev-only comparison plus
verified archive. This first phase does not launch new official-test scoring or
create test claims. Existing six-run test evaluation remains governed by its
original protocol. New CLIP test scoring requires explicit later authorization
against the frozen pair; no new test runner is included in this phase.

Report each seed's selected/stopped epochs, updates, charged compute, BLEU1-4 and
ROUGE-L; paired dev differences versus matching200epoch DINO reference, two-seed
means and recipe/initialization checks. Do not compare against stopped40epoch
study as primary baseline or infer significance from two seeds. A later adapted
CLIP comparison should hold S² and downstream recipe fixed and be separately planned.

## Self-review disposition

The in-chat draft was corrected to (1) pin missing processor metadata without
editing shared cache, (2) distinguish raw hidden CLS from normalized pooler,
(3) replace both hardcoded admission and batch access, (4) make two-run selection
and dev-first scope explicit, (5) include real storage demand, and (6) require
resume/update equivalence rather than generation-only reload. All requirements
are mapped to the implementation plan tasks. Documentation-only changes are safe
while the current source closure remains frozen.
