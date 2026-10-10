# SigLIP 2 Encoder Amendment for Global/Local Project

## User-directed change

User requested replacing CLIP with `google/siglip2-so400m-patch14-384` for both
spatial/global and local hand/mouth feature extraction in the separate-server
project. Model selection supersedes CLIP-specific portions of global/local design,
plan, README and handoff. Historical CLIP/S² comparison documents remain reference
material; active original-host DINO six-run study is not modified.

The global/local implementation is still pending on the other server. This
amendment changes design and implementation contracts, not a completed extractor,
cache, adapter checkpoint or training result. No weights/GPU work or environment
upgrade is performed by this documentation change. User reviewed and approved
native attention pooling,384/768global scales,384local crops,2304/1152feature
widths, retained768/2048fusion/connector and isolated extraction runtime, and
explicitly authorized committing/pushing these branch documents.

## Verified checkpoint facts

- Repository: `google/siglip2-so400m-patch14-384`.
- Pinned revision: `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`, verified against
  public repository metadata and config/processor files at that revision.
- Vision width1152,27layers,16heads, patch14, native image size384.
- Root `model_type` is `siglip`; vision type is `siglip_vision_model`.
- Processor declares `SiglipImageProcessor`, fixed height/width384, resize enabled,
  rescale1/255, normalize enabled, mean/std[0.5,0.5,0.5], resample2(bilinear).
- This is a fixed-resolution SigLIP-compatible checkpoint, not a NaFlex input
  pipeline. Dispatch from pinned config; a generic Siglip2/NaFlex snippet is not
  sufficient evidence of the correct class/processor for these exact assets.
- Vision sequence has patch tokens, not CLIP's CLS token. Use pretrained native
  multihead attention-pooling head output1152. First patch or ad hoc patch mean
  is not a substitute for native pooled image representation.

Transformer4.56.2 SigLIP source was inspected: vision states receive post-layer
normalization, then `pooler_output = self.head(last_hidden_state)`; its learned
probe performs multihead pooling and residual MLP and returns one vector per
image. Keep backbone/head frozen/eval and do not L2-normalize stored vectors
implicitly. Full zero-shot model outputs may normalize embeddings for similarity;
extraction parity must compare native vision pooling, not image/text logits.
Metadata/static API inspection is not a claim that actual checkpoint loading or
GPU parity has been tested. That belongs to the destination extraction smoke.

## Revised feature pipeline

Preserve two-scale global and single-scale local structure, adjusted to native
checkpoint resolution:

```text
GLOBAL
RGB frame -> pinned SigLIP processor ->384x384 normalized tensor
        +-> native vision attention pooling ------------------>1152
        |
        +-> bicubic upscale to768x768
             -> four384x384tiles
             -> same vision attention pooling on each tile
             -> mean four tile vectors ----------------------->1152

Concatenate base vector + mean tile vector -------------------->2304

LOCAL
Left-hand RGB crop  -> pinned processor384 -> native pooling --->1152
Right-hand RGB crop -> pinned processor384 -> native pooling --->1152
Mouth RGB crop      -> pinned processor384 -> native pooling --->1152
```

Convert inputs to RGB explicitly because processor metadata does not declare a
true RGB conversion flag. Native processor resizes directly to384×384; do not
reuse CLIP center crop, normalization or224pixel assumptions. Upscale normalized
base tensor with bicubic `align_corners=False`, `antialias=False`, then split
row-major tiles. Each model call stays at native384, requiring no768positional
embedding interpolation. Model's declared384/patch14 grid is valid even though
384is not divisible by14; never invent a divisibility rejection or resize to392.

This is S²-style tiled pooled-view aggregation. Original SpaMo exports averaged
tile CLS prefixes; SigLIP exports native attention-pool vectors. Label this
adaptation correctly rather than claiming exact original-SpaMo CLS parity.
Missing regional observations stay zero-valued with hard validity=false.
No local multiscale, full-face stream or vision fine-tuning is introduced.

## Dimensions and model construction

| Tensor/module | Old CLIP design | Selected SigLIP design |
| --- | --- | --- |
| Native pooled image vector |1024raw CLS |1152native attention pool |
| Global two-scale feature |2048 |2304 |
| Left/right/mouth feature each |1024 |1152 |
| Global spatial MLP |2048→1024→768 |2304→1024→768 |
| Region projection each |1024→768 |1152→768 |
| Three projected regions concatenated |2304 |2304 |
| Combined local temporal width |768 |768 |
| Global/local fusion width |768 |768 |
| Shared FLAN connector output |2048 |2048 |
| German target token embeddings |2048FLAN |2048FLAN |

Keep VideoMAE width1024 and existing source-window timing. Same shared connector,
FLAN model/revision, LoRA recipe and one fused RVLF-inspired fine-grained loss.
VT supervision uses FLAN German token embeddings, not SigLIP's text tower, and
remains contrastive matching; SigLIP's own sigmoid pretraining loss is not imported
into downstream objective automatically.

Published baseline factories/backbone checks still hardcode2048spatial input and
are historical references. New global/local builder must consume2304manifest
width and create1152regional projections. Do not silently zero-pad/truncate to
old width or load old incompatible first-layer weights. Train fresh SigLIP M0/M1/
M2 controls with identical common initialization across those variants. Comparing
against old CLIP/DINO also changes preprocessing/pooling/input-layer width; report
those differences instead of attributing all gains solely to encoder identity.

## Runtime and cache identity

Main training stack stays Torch2.0.1/Transformers4.32/PEFT0.7.1. It cannot load
selected SigLIP assets. Create separate destination-server vision-preparation uv
project/lock under `tools/global_local/vision/`; starting candidate Torch2.5.1,
Transformers4.56.2, Pillow11.3.0, NumPy1.26.4. Validate exact loading, native pooling,
RGB processing and device compatibility before freezing that stack. Do not modify
original-host main/DINO/motion environments or import across venvs.

Extraction environment should use config-dispatched supported SigLIP vision
implementation and `AutoImageProcessor`; a verified vision-only loader avoids
running/retaining unused text tower on GPU. Require pooled shape[B,1152] and
processed pixel shape[B,3,384,384]. Do not force NaFlex flattened-patch inputs or
pixel masks/spatial-shapes onto this fixed-resolution processor. Tests compare
single-view extraction with pretrained native head, then independently validate
five-view global aggregation and regional ordering/tail batches.

Publish new encoder/package keys binding revision, weight/processor hashes,
native-pool head,384/768scales, normalization, resize/interpolation, crop/provider,
precision, runtime and accepted inference batching. Preserve older caches.
Model/feature profile must be rerun; a CLIP profile does not establish memory
headroom at384with1152channels. No extraction frame batch is assumed accepted.

For947756frames, FP32 global `[N,2304]` uses8.1347GiB and local `[N,3,1152]`
uses12.2020GiB: total20.3366GiB before motion, masks/boxes/receipts, models and
checkpoints. Keep50GiB reserve plus measured remaining demand. Retain one pooled
vector per view/region, not all729patch embeddings as a permanent feature cache.

## Acceptance and review

1. Verify pinned config/processor dispatch and actual native-pool forward output.
2. Validate independent global384/768tile aggregation and local384pooling.
3. Reject wrong widths, first-patch readout, old normalization or old cache identity.
4. Verify new2304/1152adapter construction and unchanged768/2048fusion/decoder.
5. Verify masks, time alignment, exact missing-local fallback, full-state replay
   and one fused fine-grained loss using FLAN text embeddings.
6. Implement and verify approved extraction configuration on destination before
   publishing method results. Existing recipe, current six-run study, test claims
   and historical caches stay unchanged. Documentation approval/publication does
   not imply trained SigLIP results.

Sources: [pinned configuration](https://huggingface.co/google/siglip2-so400m-patch14-384/blob/e8e487298228002f3d8a82e0cd5c8ea9c567f57f/config.json),
[pinned processor](https://huggingface.co/google/siglip2-so400m-patch14-384/blob/e8e487298228002f3d8a82e0cd5c8ea9c567f57f/preprocessor_config.json),
[model card](https://huggingface.co/google/siglip2-so400m-patch14-384), and
Transformers4.56.2 `models/siglip/modeling_siglip.py` vision/attention-pooling implementation.
