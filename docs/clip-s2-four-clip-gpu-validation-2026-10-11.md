# Four CLIP+S² cached GPU generations completed

**All four approved cases completed without observed CUDA/Windows fault.**
This validates selected short/median/long clips in optimizer-free cached
inference, not full dev, training or the original batch4/cacheFalse path.

## Scope and prerequisites

User explicitly approved four predeclared dev clips, batch1/cacheTrue/beam5/max64,
fresh inference process, no retries/training/Windows-device changes. Initial
preflight was blocked at47.54GiB C: backing free space; subsequent recheck at
19:05:36UTC showed52.45GiB, above50GiB reserve plus64MiB diagnostic margin.

User's intervening storage action changed WSL boot identity to
`a25da066-fad8-4652-92f2-4d30dd670f2c`; Windows boot remained08:25:48.5000000UTC,
driver617.42. A fresh read-only host/source/checkpoint review preceded launch.
GPU audio was enabled, ProblemCode0, and never changed for this diagnostic.

Both global224 and averaged tiled448 S² feature halves were used: the original
2048-input adapter and retained epoch2/1774-update checkpoint were loaded.
Checkpoint SHA256:
`d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9`.

All84 original source files, pinned FLAN bytes, study configuration and admitted
feature/data identity matched. Selected feature-array hashes and declared lengths
were validated on CPU. No optimizer/global CUDA RNG state restored for inference.
Fixed train-only inference exemplar was unchanged.

## Results

| Dev clip | Frames | Batch seconds | Output tokens | Joint tokens | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
|11August_2010_Wednesday_tagesschau-2 |42 |6.13 |48 |154 |5.99 /6.15 |
|11December_2009_Friday_tagesschau-3509 |16 |2.90 |48 |147 |5.89 /6.15 |
|30January_2011_Sunday_tagesschau-6345 |105 |3.36 |47 |172 |6.00 /6.15 |
|03November_2010_Wednesday_tagesschau-6506 |272 |2.58 |52 |219 |6.03 /6.60 |

All decoder pre-forward events reported5 beam rows and input length1. All raw
last-step decoder logits were finite. Decoder forward counts were63/63/63/53;
selected output token counts can be smaller because completed beam hypotheses
are retained while other candidates continue. Output counts include special
tokens; no translation-quality scores were computed.

Time includes adapter/prompt and instrumented generation, with per-step numerical
checks and durable logging. First-case warmup and varying beam termination mean
these timings are not controlled throughput comparisons. Memory values are
PyTorch allocator peaks, not total Windows GPU-memory usage. Maximum measured
allocated6471566848bytes, reserved7090470912bytes.

Four successive clips ran in one fresh inference-only process, using separate
BF16 autocast scopes per batch. Host/storage/event guards ran before/after each
batch. Final event query at19:10:34.7859606UTC found0 new NVIDIA13/14/153,
critical/error Display/WER crash events or Kernel-Power41 records since19:05UTC.
Windows and WSL boot identities stayed unchanged during execution. No CUDA
exception, retry or additional generation occurred.

## Interpretation and next review

CLIP+S² cached batch1 inference succeeded on this selection, including the longest
official dev clip. This is stronger coverage than the earlier single42-frame
probe and weakens a universal "S² spatial features break GPU inference" claim.

It does not identify the initial fault, establish an audio fix, measure original
training-resident peak memory, or validate519-clip/long-run stability. The old
batch4/cacheFalse runner is still held and has not been patched or resumed.
Checkpoint1774updates remains in VT-only warmup, so generated texts are not an
evaluation of trained translation quality.

Diagnostic reused the dimension-agnostic `tools/clip_plain/generation.py` safe
core; telemetry's helper policy tag contains`clip-plain`. That tag describes the
shared cached/telemetry settings, not feature provenance: this run used original
S²2048 features/checkpoint, not the plain1024 model. It is diagnostic-only and
cannot be admitted as a plain-model or production selection report.

Next review: separately scoped complete cached batch1 dev validation and an
explicit checkpoint-compatible S² train/dev process bridge before any recovery.
No automatic full-dev/training launch is authorized by four successful cases.

## Artifacts and final containment

Native same-study namespace:
`diagnostics/clip-s2-cached-four-dev-v1/` contains one-shot `started.json`, fsynced
decoder/batch `events.jsonl`, four case receipts and diagnostic `report.json`.

- Report SHA256:`0de8f20a4d0239d6c90bebaa53022af6f6ef9ec2d4786879eb0f16d93194869d`.
- Telemetry SHA256:`29347a2e6d58989adc02d97fd3e08fba5de54f811f7f8819e8105772352f19c4`.
- Invocation:`docs/diagnostics/clip-s2-four-clip-cached-probe-2026-10-11.py`, SHA256
  `9d4b24ff3cdb3a56fde82d8ee4ad3292cc7122e2f261559dbaf271891f8fc3de`.

Temporary scoped permission was revoked and marked consumed after completion;
manual-stop remains`stopped_by_user`, automatic recovery disabled. Training budget
and checkpoint/source bytes unchanged:1774updates/497.5533489369991s retained.
No native full-dev reports/selections, new training updates, Seed1 or test scoring.
No driver/device/runtime/dependency changes, archive relocation or old deletion.
