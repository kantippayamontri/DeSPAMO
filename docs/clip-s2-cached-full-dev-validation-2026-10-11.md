# Full519-clip CLIP+S² cached dev diagnostic completed

**All519 official dev clips completed in one fresh inference-only process without
observed CUDA error, host restart or new matching Windows fault.** The process
exited; scoped GPU permission was revoked. Training and automatic recovery remain
held. This is diagnostic evidence for the cached batch1 path, not a root-cause
finding or authorization to resume the old controller.

## Scope and controls

User approved exactly one complete519-clip diagnostic after the four selected
clips passed. Same original official order, original CLIP+S²2048 spatial features,
unchanged motion, train-only three-language ICL and fixed seed0 exemplar.

- Native epoch2/1774-update checkpoint SHA256:
  `d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9`.
- Dev ID/order digest:
  `0147dcfd2438ee98cf2d485d9a994264794d0e4bb7b3dee9dc03193b10d15a44`.
- Batch1,beam5,max64,deterministic,explicit cacheTrue; per-batch BF16 autocast.
- Fresh model process, optimizer absent; original training cacheFalse unchanged.
- Original84source hashes, pinned base files and data/config identities verified.
  Each clip's actual spatial/motion array hash checked when loaded.
- Driver617.42, Windows boot08:25:48.5000000UTC, WSL boot
  `a25da066-fad8-4652-92f2-4d30dd670f2c`, unchanged throughout test.
- GPU audio enabled throughout. No Windows device/driver/registry changes.
- C:52.45GiB at preflight satisfied50GiB reserve plus64MiB diagnostic margin.
- Host/boot/storage/event guards before and after every clip; durable batch and
  decoder-step events. First fault would abort with no retry or CUDA RNG cleanup.

The shared safe-core telemetry policy tag contains`clip-plain`; this describes
the helper implementation, not the representation. Actual model/input/checkpoint
are original S²2048, and diagnostic outputs are not plain-model selection reports.

## Verified result

| Measurement | Observation |
| --- | --- |
| Coverage |519/519, exact official order, no duplicates or retries |
| Model-load/diagnostic start |2026-10-11T19:20:46.629630Z |
| Report finish |2026-10-11T19:58:59.694640Z |
| Wall duration |2293.06501s, including model load and host/logging checks |
| Sum of instrumented batch durations |1479.483407607s |
| Peak allocated |6473623040bytes =6.029031GiB |
| Peak reserved |7090470912bytes =6.603516GiB |
| Spatial frame range |16–272 |
| Selected output-token count range |47–64, including special tokens |
| Decoder forwards |31887 |
| Decoder input length / beam rows |1 /5 for every recorded forward |
| Raw last-step decoder logits |Finite on all31887 recorded forwards |
| CUDA errors or failure reports |None |
| Final Windows fault check |19:59:49.3389614UTC;0 new matching events since19:17UTC |

Memory is PyTorch allocator measurement, not total Windows GPU memory. Per-step
numerical checks/fsync and two host checks per clip contribute overhead; wall
duration is not an optimized throughput benchmark. Beam hypotheses may continue
after the eventually selected hypothesis completes, so decoder step count and
selected output length need not match.

Final event query covered NVIDIA13/14/153, critical/error Display/WER crash
events and Kernel-Power41 in the stated window. Boot identities and enabled GPU
audio state also matched. No claim is made about every possible Windows event or
future workload stability.

## Diagnostic metrics and training-stage caveat

Complete diagnostic corpus BLEU1/2/3/4:
8.90547264 /1.36085667 /0.20328787 /**0.06687138**.
ROUGE-L F1/precision/recall:
0.05784573 /0.04742629 /0.08172001.

This retained checkpoint is at1774updates, before the4000-update VT-only warmup
ends; translation CE has not yet begun. These very low scores are warmup
diagnostic results, not final translation quality or a reason to select/tune on
this run. No official test scored. No legacy `dev-epoch-2.json`, selection pointer,
early-stopping state or budget was created/rewritten from these metrics.

## Interpretation and next step

Original CLIP+S² features support complete cached batch1 dev generation under
this fresh, optimizer-free configuration. This weakens the hypothesis that S²
features necessarily break GPU inference. It does not explain the original
batch4/cacheFalse illegal-instruction/Windows failures: batch size, caching and
optimizer/process residency differ, and no original-path reproduction was run.
It does not prove an audio fix or stability after joint CE training/repeated
epochs, and driver/kernel/hardware root cause remains unresolved.

**Next: CPU-tested checkpoint-compatible S² integration** that restores original
epoch2 weights,AdamW,sampler/global+prompt RNG and charges, while alternating
train-to-dev-boundary workers with fresh cached batch1 evaluator workers. Bind
new evaluator policy separately; do not rewrite old source/checkpoint identity.
Then separately review a training/checkpoint/process-exit/dev handoff validation
before long training. The old controller still uses the unsafe batch4/cacheFalse
path and must not be restarted merely because this diagnostic passed.

## Artifacts and final containment

Same-study namespace:`diagnostics/clip-s2-cached-full-dev-v1/` contains
authorization, owned startup receipt, per-clip atomic receipts, progress,
fsynced decoder/batch telemetry, worker log and final report.

- Report SHA256:`1b29826ade1c066f21ddf7e93b4cba01b78338804f9c6a6159c10933ea7db066`.
- Telemetry SHA256:`0c8b2c8d96fee75ed099b51cdc53b3c2b09f9e1ed3679618381e513768e8ddc7`,17185635bytes.
- Invocation:`docs/diagnostics/clip-s2-full-dev-cached-probe-2026-10-11.py`, SHA256
  `b6752192a0f32fdab857df925625dba37b7ec520bbb82c5a2daa96c612a21b49`.

Started owner PID8612/startticks176041/WSL boot above exited. Full scope consumed
at19:59:00.510884UTC, `gpu_validation_approved=false`, original manual-stop remains
`stopped_by_user` with automatic recovery disabled. No remaining CLIP process
found. Original84source/checkpoint/budget hashes unchanged, training still
1774updates/497.5533489369991s. No new seed, production selection, test scoring,
device change, source change, artifact deletion or training attempt.
