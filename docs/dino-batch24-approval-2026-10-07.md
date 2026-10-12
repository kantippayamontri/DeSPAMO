# Approved DINO adaptation batch24 — 2026-10-07

## Decision and measured acceptance

User approved physical batch **24**. A fresh, authorized real GPU profile used
two warmup joint updates and five measured joint updates, fully initialized AdamW
state, all ten loss terms, BF16/FP32 losses, GRL1 and conservative peak group LRs.
Batch24 met the proposed acceptance criteria: peak reserved at most21GiB, end
free GPU memory at least2GiB, and clips/second at least the measured batch16 value.

| Batch | Peak allocated GiB | Peak reserved GiB | End free GiB | Seconds/update | Clips/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 11.238 | 14.104 | 8.256 | 0.967561 | 16.536420 |
| **24** | **16.108** | **20.002** | **2.357** | **1.421400** | **16.884758** |
| 32 | 20.980 | 26.330 | 0 | 13.205495 | 2.423234 |

Batch24 compute throughput was **2.106% higher than16** in this short probe.
It offers a larger physical contrastive batch without batch32's measured memory
pressure/slowdown. These small timing samples do not establish statistical speed
significance, end-to-end loader throughput, or improved translation accuracy.
GPU memory margin is measured for this configuration; model/capacity changes or
another GPU workload require fresh profiling.

All **30/30** AdamW parameter states initialized,89,172,088 tensor-state bytes.
All recorded losses and gradient norms finite. Five measured durations:
`1.419721, 1.419026, 1.422743, 1.422711, 1.422801` seconds.
Nested signer-diverse clip IDs matched first24 of the original batch32 profile.
No model checkpoints or production epoch/resume artifacts were created.

Raw report:
`/home/kan/datasets/despamo/official-phoenix14t-dino-adaptation-v1/profiles/batch24-0d03e2a667f77438de631cd70afee24d78c95140076de43a6048f05dbc2163dd/profile.json`.
Its content digest, complete status, nested IDs, state initialization and acceptance
metrics were independently verified. GPU returned to15MiB/0%, no compute process.

## Approved production arithmetic

40 total epochs still include five head calibration and35 joint epochs. Learning
rates, losses, resolutions, sampled frames, LoRA rank and checkpoint cadence stay
as approved. Only physical batch and derived update counters change:

| Setting | Batch24 value |
| --- | ---: |
| Complete batches per epoch | 295 |
| Clips per complete epoch | 7,080 |
| Dropped tail, reshuffled next epoch | 16 |
| Total optimizer updates | **11,800** |
| Head-only calibration updates | **1,475** |
| First joint update | **1,476** |
| LoRA warmup / GRL ramp updates | **590** |
| LoRA peak / GRL1 update | **2,065** |
| Model exports at epochs10/20/30/40 | **2,950 / 5,900 / 8,850 / 11,800** |

Every one of the7,096 clips remains in the shuffled inventory. Dropping an
incomplete tail does not statically remove16 selected clips from the dataset.

## Implemented changes and handoff

Changes remain uncommitted in isolated `work/official-merged-targets` worktree:

- `official_schedule.py`: production default24; generic batch16 regression
  fixtures still exercise their historical clock arithmetic explicitly.
- `official_sampler.py`: new batch24-capable seeded/drop-last sampler with exact
  ID-hash/batch/cursor/RNG replay; old E3 sampler remains unchanged.
- `configs/adaptation/phoenix14t_dino_merged40.json`: explicit24/40-epoch recipe,
  pinned input/model hashes and working/archive locations, same optimizer/losses.
- Profiler accepts16/24/32; only24 uses the newly requested five measured steps.
- Authoritative design/plan updated to24 and use `OfficialBatchSampler` in Task6.

The new sampler was necessary because the old sampler rejects24. CPU replay
tests verify dropped-tail behavior, identical next batch, rejection of a24-state
restore into16, and rejection of malformed counters/schema/RNG. The config test
loads actual JSON and verifies consumer schedule arithmetic rather than text lines.

Verification: dedicated DINO suite **243 passed, 1 skipped**; Ruff and Pyright
passed. Self-review fixed bool-schema acceptance and malformed RNG exceptions
with RED-to-GREEN replay regressions. Old E3 sampler/schedule tests still pass.

Batch24 adoption does not launch the40-epoch job. Task5 checkpoint publication
and Task6 owned runner/preflight/GPU smoke still need implementation. The new
config is a prepared recipe, not an already executable training runner.

## Prior study finalization

The previous fusion controller is now **complete**. Its final `summary.json`
was present with matching content digest and `signer07_scored=false` at
`2026-10-07T15:25:56Z`.
Raw summary SHA-256:
`aecdb6b85a00791b1b6eb1f383793fb45d039de036187ae748c037d2064c1f00`.
The earlier training-summary publication-pending observations are historical.
No source/checkpoint migration, deletion, commits or integration were performed.
