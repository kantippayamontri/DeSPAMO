# DINO joint-training batch16/32 profiles — 2026-10-07

**Superseding decision:** [approved batch24](dino-batch24-approval-2026-10-07.md)
passed subsequent real profiling. The batch16 recommendation below describes
the initial16/32 comparison, not the updated production default.

## Conclusion

**Keep physical adaptation batch16 for the current configuration.** Batch32
completed without a CUDA OOM, but exhausted available GPU memory, reported peak
allocator reservation above the physical device capacity, and was approximately
6.82x worse in clips/second. It is not a practical batch32 production profile.

Measured device: RTX 4090, UUID `GPU-758013eb-2b8c-0588-a783-71a7f0a625e1`,
driver 610.62, dedicated DINO runtime torch 2.5.1+cu121.

| Batch (clips) | Peak allocated GiB | Peak reserved GiB | Mean GPU update seconds | Clips/second | Initialized AdamW states |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 11.238 | 14.104 | 0.967561 | 16.536420 | 30 / 30 |
| 32 | 20.980 | 26.330 | 13.205495 | 2.423234 | 30 / 30 |

CUDA reported device capacity 23.988 GiB. End-of-probe free memory was 8.256 GiB for
batch16 and0 for batch32. Allocator reservation is not a dedicated-residency
measurement: the large reservation, zero free memory and stable slowdown are
consistent with severe memory pressure/oversubscription and possible Windows GPU
system-memory fallback. Dedicated/shared resident counters were not collected,
so this report does not claim the exact operating-system mechanism was proven.

Batch32 delivered 14.65% of batch16's compute throughput. Its three measured steps
were13.233457/13.192304/13.190723s; batch16 steps were
0.971589/0.966420/0.964675s. These are measured GPU updates, not training-duration
forecasts or end-to-end data-loader throughput.

## Exact workload

- User explicitly authorized both real profiles after asking for training summary.
- Fresh isolated process and seed0 for each batch, sequential execution; no other
  GPU compute process before launch or after CPU data admission.
- Real official training data and prepared merged v1/v2 vectors; strict admission
  hashes/frames/masks checked by dedicated DINO runtime.
- Nested signer-diverse clip batches: batch16 is the first16 of batch32.
- Five sampled frames per clip, both224 and448 resolutions. Batch32 means160
  source-frame samples at each resolution, not32 frames total.
- Pinned DINOv3 ViT-L/16 revision
  `ea8dc2863c51be0a264bab82070e3e8836b02d51`; base weights frozen/eval, rank8/alpha8
  Q/V LoRA in blocks20–23 and seven factor heads.
- BF16 forward; FP32 contrastive/teacher terms; SDPA; **no gradient checkpointing**.
- Full joint backward, all ten loss components, relational weight0.10, GRL1,
  separate gradient clipping1.0 for LoRA/heads.
- AdamW peak rates LoRA1e-5/heads3e-5, decay0.05. Existing loss helper's late-joint
  index forces GRL1; it does not apply epoch scheduling. Both groups use constant
  conservative peak rates for this memory probe, not the zero-LR horizon endpoint.
- Two warmup updates allocate all AdamW states; three further measured updates
  include forward, loss, backward, clipping and optimizer step. CUDA synchronized
  before/after timers; data loading and initial transfer excluded from step timing.
- Expected initialized parameter tensors30; observed30/30 in both. Optimizer
  tensor state size89,172,088bytes; no optimizer CPU offload.
- Fresh process exits release all GPU resources. No production model checkpoint,
  resume, epoch completion or adaptation selection was published.

After both probes:15MiB/0% GPU, no compute processes. Existing fusion controller
continued CPU final validation with all six per-run training results complete.

## Durable raw reports

Working root:
`/home/kan/datasets/despamo/official-phoenix14t-dino-adaptation-v1/profiles`.

- Batch16:
  `batch16-867bb1a1069291cc75e5db1ace8eedb1fc22c877bd870e346de6ec118c0032e8/profile.json`.
- Batch32:
  `batch32-fc825b2b33cf4bb0f14d07c09b2129fce7da56fa339c5976c08630bf70b4e6ef/profile.json`.

Both raw content hashes were independently recomputed and matched; statuses were
`complete`, all loss/gradient scalars finite, and nested clip IDs verified. Reports
bind dataset identity, model/recipe, source hashes and owned worker PID/boot/start
tick receipts. Hardware name and measured device capacity are in each report;
physical UUID above was witnessed by `nvidia-smi` before/after profiling.

The first attempt failed native JSON identity serialization **before GPU work**
because clip IDs were a tuple. This was fixed by a JSON-list identity constructor
with a RED-to-GREEN native atomic-JSON regression. Its empty failed profile root
was retained; no historical state or charges were erased to mask the attempt.

## Code, checks and next step

New profiler/tests remain uncommitted in the authorized isolated worktree:
`.superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree`,
`tools/dinov3/profile_official_adapt.py` and
`tools/dinov3/tests/test_profile_official_adapt.py`.

Dedicated suite: **235 passed, 1 skipped**. Ruff/Pyright passed. Native self-review
covered joint loss/gradient behavior, full optimizer initialization, frozen base,
real/nested batches, process isolation, JSON receipts and memory/timing scope.
No source files were integrated into the active checkout; no dependency or
production hyperparameter changes were made.

The approved batch16 production schedule therefore remains40 total epochs,
17,720updates, five calibration epochs, two joint warmup/ramp epochs, and model
milestones4,430/8,860/13,290/17,720. Task5 checkpoint/replay and Task6 owned runner
still need implementation before production adaptation. An intermediate batch
or activation-checkpointing variant would need a separately requested measured
profile; do not silently adopt batch32 based only on absence of OOM.
