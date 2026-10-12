# Two-seed time-aligned fusion training summary — 2026-10-07

**Finalization update:** controller now reports `complete`; final `summary.json`
content digest verified at `2026-10-07T15:25:56Z`, with `signer07_scored=false`.
Raw summary SHA-256:
`aecdb6b85a00791b1b6eb1f383793fb45d039de036187ae748c037d2064c1f00`.
Publication-pending statements later in this document describe earlier snapshots.

## Training result

All six runs reached 16,000 optimizer steps, closed their runner ledgers, and
published selected-checkpoint receipts. Selection uses the same 582-clip Signer03
dev set at steps 5,500/10,400/16,000, highest BLEU4 with earlier-step tie breaking.
Signer07 held-out translation remains unscored for this study.

| Model | Seed0 selected step | Seed0 dev BLEU4 | Seed1 selected step | Seed1 dev BLEU4 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 10,400 | 5.788668 | 16,000 | 5.841412 | 5.815040 |
| Aligned fixed gate | 16,000 | 6.222396 | 16,000 | 5.776657 | 5.999527 |
| Aligned learned gate | 16,000 | 5.955299 | 16,000 | 6.137285 | 6.046292 |

| Contrast vs reference | Seed0 delta | Seed1 delta | Mean delta |
| --- | ---: | ---: | ---: |
| Fixed gate | +0.433728 | -0.064755 | +0.184486 |
| Learned gate | +0.166630 | +0.295872 | +0.231251 |

Learned gating has positive point gains in both seeds and the higher two-seed
mean. Fixed gating improves the mean but not both seeds. These are two seeds and
one dev signer; this is not a significance, signer-invariance, causal-motion, or
conventional-benchmark result. The new official-split study is separate.

## Matched recipe and provenance

All arms use the same adapted DINO features, documented new VideoMAE motion
package, plain spatial MLP backbone, temporal encoder, and FLAN-T5-XL LoRA.
Logical split is 5,746 training/seven signers, 582 dev/Signer03, 768 test/Signer07,
all from the physical PHOENIX training partition. Batch 4, BF16, 16,000-step
horizon and deterministic beam5/max64/noICL are matched. The reference arm means
original sequential fusion on this same backbone, not the earliest linear model.

New motion cache contains 7,096 clips, 827,354 source frames and 93,232 windows.
Appearance supervision is unreviewed; `human_review_status=not_assessed`.
Seed0 reference recovered from its owned 12,500-step checkpoint after the old
duration stop; all elapsed/replayed charges were retained under the explicit
time-waiver policy. No original captions or held-out registries were changed.

Study root:
`/home/kan/datasets/despamo/signer-pilot/time-aligned-fusion-new-motion-validated-study-v1`.

## Publication state

At `2026-10-07T12:31:09Z`, all six trainer PIDs had exited and all per-run selected
receipts were present, but the controller was still performing CPU final
validation and had not published `summary.json`. A later check after DINO profiles
still showed controller `running` and no summary. Thus **training is complete,
aggregate final publication is pending**. This document derives the table from
per-run selected receipts and does not pretend to be the controller's final
validated aggregate artifact. Recheck state before moving source/checkpoint files.

GPU was RTX4090, 24,564MiB reported VRAM, UUID
`GPU-758013eb-2b8c-0588-a783-71a7f0a625e1`; all six runs used that hardware.
Before authorized DINO profiling it was idle at15MiB/0%, with no compute process.
Only isolated profiling used it; the active controller's source/data paths remain
intact during final validation.
