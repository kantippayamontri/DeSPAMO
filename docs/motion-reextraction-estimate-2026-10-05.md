# Motion re-extraction time estimate — RTX4090

## Recommendation and scope

Plan approximately **4–5 hours** to re-extract all7096physical-train clips with
explicit source-frame/window records. Extractor and automatic handoff are now
implemented; real profile/admission remains pending. This remains a historical
small real-video benchmark estimate, not completed extraction or a guaranteed
deadline. User already approved separate5-hour extraction and18-hour automatic
fusion training ceilings; no further approval is required within those caps.
Production clocks include preparation/IO/admission overhead; prior implementation
and environment setup are not extraction runtime observations.

Combined final-fix verification2026-10-06: affected463passed/4warnings;
single main full run1614passed/27warnings. Final one-line latest package-attempt
lookup and fixture extension followed full run; final focused/static checks cover
frozen bytes. Dedicated126passed/1warning evidence retained with unchanged sources.
F1/F2/F3 addressed; **NOT RUN, pending scoped final review**. Historical Task5 CPU
preflight74.85s/7096clips/827354frames/93232windows and disk observations remain
snapshots from before source changes. No new real profile/source-output parity,
full-cache throughput/peak or extraction runtime was measured. See
[current status](motion-fusion-pipeline-status-2026-10-05.md).

New extraction creates **new features and new contemporaneous records**. It
does not recover original historical records for the downloaded motion cache.
Use the same recorded210x260 source frames as the spatial cache to establish
shared frame ordinals. All three fusion arms would use this new cache with fresh
references; results cannot be attributed to fusion alone by comparing directly
to older runs using the downloaded motion cache.

## Exact workload counted from current frame receipts

Window16, stride8, full windows only. Current clips all have at least16frames.

| Logical split | Clips | Real source frames | Motion windows | Float32 feature payload GiB |
|---|---:|---:|---:|---:|
| Train | 5746 | 662357 | 74553 | 0.284 |
| Signer03 dev | 582 | 68841 | 7767 | 0.030 |
| Signer07 test | 768 | 96156 | 10912 | 0.042 |
| All physical train | 7096 | 827354 | 93232 | approximately0.356 |

Train+dev alone:6328clips/82320windows. Full7096cache simplifies complete frozen
input admission. Test feature extraction is unsupervised inference, not test
translation generation, selection or model training.

Allow approximately1–2GiB for feature arrays plus timing/source/hash records;
confirm actual record representation during smoke. Existing model weights/raw
frames are reused, not duplicated. No training checkpoint is required for frozen
encoder extraction. Output record size and hashing/write time were not measured.

## Measured setup

- RTX4090, existing torch2.0.1+cu117/transformers4.32.0; WSL CUDA library path.
- Cached `MCG-NJU/videomae-large` revision
  `12da269a02d3e1fbbb7011e610e591fea8061dca`:24layers,width1024,16frames,
  use_mean_pooling=false. Offline loading from cached snapshot only.
- Inference-only FP32, window batches up to16, two CPU threads. Output selected
  from `hidden_states[-1][:,0]`, matching the local extractor's selection rule;
  do not silently substitute normalized `last_hidden_state` for that rule.
- Three real clips at frame-count quantiles20/50/80%, respectively75/112/154frames
  and8/13/18windows. Two repeated optimized FP32 passes:78window observations.
- Includes PNG decode, RGB conversion, preprocessing, CPU→GPU transfer,
  inference, GPU→CPU transfer and finite/shape checks. Does not include writing
  feature files, durable records, fullsource hashing or fullcache validation.
- Main environment lacks Pillow. Benchmark process appended the existing
  extractor environment's site-packages after main paths, loading Pillow11.3.0
  while retaining main torch/transformers. No install or environment-file edits.
  A production extractor should use a clean, pinned environment rather than rely
  on this process-local bridge.
- Cached snapshot lacks preprocessor_config.json. Speed benchmark used documented
  VideoMAEImageProcessor defaults:shortestedge224/centercrop224, mean/std0.5.
  This is **not** evidence of the released cache's original preprocessing.
  Final preprocessing must be explicitly selected/versioned before extraction.

## Results

Naive processor `return_tensors='pt'` emits a slow list-of-NumPy-arrays conversion
warning. Its pooled FP32 speed was39windows/11.4948s = approximately3.393windows/s,
projecting7.63hours for allclips before additional overhead.

Benchmark-only optimization: request NumPy batch output and construct input with
`torch.from_numpy(...)`, then transfer to GPU. One real16-frame window's input
tensor was bit-identical to the naive path. No saved features were changed.

| Optimized FP32 observation | Windows | Total seconds | Windows/s |
|---|---:|---:|---:|
| 75frames, pass1 | 8 | 1.1435 | 6.996 |
| 112frames, pass1 | 13 | 1.7554 | 7.406 |
| 154frames, pass1 | 18 | 2.5089 | 7.174 |
| 75frames, pass2 | 8 | 0.9942 | 8.047 |
| 112frames, pass2 | 13 | 1.5949 | 8.151 |
| 154frames, pass2 | 18 | 2.3271 | 7.735 |
| Pooled | 78 | 10.3240 | **7.5552** |

Measured peak allocated GPU memory approximately8.846GiB. Cached-model load in
the first successful benchmark took6.852seconds. These are small-sample facts,
not full-workload maxima or full cold-start IO behavior.

Extrapolated decode/preprocess/inference times:

- Train+dev:82320/7.5552/3600 = approximately**3.03hours**.
- All7096:93232/7.5552/3600 = approximately**3.43hours**.
- Practical allowances:approximately**3.5–4.5hours train+dev**, **4–5hours all7096**.

The allowances cover expected record/hash/write/validation work and some sample/
IO variation, but have not been directly measured. Benchmark reads were warm
after repeated passes; full corpus has many additional files. CPU preprocessing
remains a substantial bottleneck. BF16 alone did not remove the naive bottleneck;
FP32 optimized conversion is the more conservative baseline for this estimate.

## Required before a full extraction

1. Scope/preprocessing/separate caps approved; extractor/receipts/new-mode handoff
   implemented. Final whole review and reproduced test finding remain pending.
2. Clean pinned dedicated environment and actual CPU identity/count/actor gates
   verified. Production B1/8/16 tensor/output parity and write overhead remain pending.
3. Run real small-subset profile under cumulative extraction cap; admit full run
   only if measured throughput/storage/validation demand fits remaining allowance.
4. Publish and independently close complete new cache/timing/generating graph;
   preserve downloaded cache and frozen spatial/caption records.
5. Automatic controller then consumes distinct new-motion identities and fresh
   initialization/profile evidence within separate18-hour training ceiling.

No full extraction, new cache, original-record claim, fusion run or test scoring
was performed for this estimate. Existing18-hour training authorization remains
separate and does not authorize unbounded data regeneration.
