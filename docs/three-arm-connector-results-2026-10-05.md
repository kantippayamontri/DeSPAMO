# Three-arm spatial projector / Q-Former study

## Completion and primary result

All six fresh 16000-step runs completed after the storage interruption was
recovered. This is a **dev-only comparison**, on the same ordered 582 Signer03
clips, using seeds 0 and 1. No Signer07 generation was performed.

| Arm | Seed 0 dev BLEU-4 | Seed 1 dev BLEU-4 | Mean | Sample standard deviation |
|---|---:|---:|---:|---:|
| Reference linear spatial / original output MLP | 4.541238 | 4.571809 | 4.556524 | 0.021617 |
| Plain spatial MLP 2048→1024→768 | 5.197683 | **7.090514** | **6.144099** | 1.338434 |
| Small Q-Former output connector | **5.695729** | 5.901238 | 5.798483 | 0.145317 |

| Candidate | Seed 0 BLEU-4 delta vs fresh reference | Seed 1 delta | Mean paired delta |
|---|---:|---:|---:|
| Plain spatial MLP | +0.656445 | +2.518705 | +1.587575 |
| Small Q-Former | +1.154491 | +1.329428 | +1.241960 |

Both alternatives beat this study's fresh reference in both seeds. Plain MLP has
the higher mean, but much greater observed between-seed spread. Q-Former has the
more even BLEU gain, but does not consistently beat plain MLP: it wins at seed0
and loses at seed1. Two seeds do not establish general robustness or significance.

## What changed and what stayed fixed

| Arm | Spatial input projector | Motion projector | Output connector after TCN |
|---|---|---|---|
| Reference | Linear2048→768 | Linear1024→768 | Linear768→2048 / GELU / Linear2048→2048 |
| Plain MLP | Linear2048→1024 / GELU / Linear1024→768 | Same linear | Same output MLP |
| Q-Former | Same linear | Same linear | Learned-query attention, then Linear768→2048 |

Plain MLP has no residual bypass or added normalization/dropout. It changes only
the spatial projector, unlike the earlier residual experiment that changed both
spatial and motion projectors.

Q-Former is a **small from-scratch Q-Former-style connector**, not pretrained
BLIP-2. It has32 learned queries,2 layers, width768,12 attention heads, FFN3072,
dropout0.1, pre-normalized query self-attention / visual cross-attention / FFN,
memory normalization, fixed sinusoidal temporal positions and final normalization.
It attends to the original TCN's temporal frame summaries, not image patches.
All32 projected query tokens are valid visual inputs to FLAN. This changes token
count, capacity, normalization and attention together; it is not a pure parameter-
matched attention-versus-MLP experiment.

All arms retain:

- The same cached cosine DINO spatial features `7122b42e...` and VideoMAE motion.
- Frozen logical split: train5746, dev582 Signer03, test768 Signer07 unscored here.
- Original fusion order and K5/P2/K5/P2 temporal module.
- **VT-Align:** first2000 steps VT-only, then translation CE plus VT with weight1.
- 16000 total steps, batch4, no accumulation, LR warmup1600 and cosine to zero.
- AdamW adapter LR3e-4 / FLAN LoRA LR5e-5, betas0.9/0.98, eps1e-8, decay0.01,
  gradient clip1, BF16 and deterministic beam5/length64/noICL.
- Common fresh weights attested within each seed, with forked seed100+seed for
  new modules, preserving common weights and CPU/CUDA RNG at attachment.
- Complete dev evaluations at5500/10400/16000 and dev-only checkpoint selection.
- Appearance labels `qwen-schema98-unreviewed-v1` / `human_review_status=not_assessed`.

## Selection and dev trajectories

| Seed | Arm | BLEU-4 at5500 | At10400 | At16000 | Selected step |
|---|---|---:|---:|---:|---:|
| 0 | Reference | 3.952006 | 4.182323 | 4.541238 | 16000 |
| 0 | Plain MLP | 4.451559 | 4.940878 | 5.197683 | 16000 |
| 0 | Q-Former | 4.955839 | 5.102174 | 5.695729 | 16000 |
| 1 | Reference | 4.045201 | 4.360368 | 4.571809 | 16000 |
| 1 | Plain MLP | 5.763263 | 6.261023 | 7.090514 | 16000 |
| 1 | Q-Former | 4.538131 | **5.901238** | 5.794842 | **10400** |

The seed1 Q-Former completed all16000 steps but its dev-selected checkpoint is
step10400. Do not confuse its final checkpoint score with its selected-model score.

## ROUGE and repetition diagnostics

ROUGE-L F1 uses the native0-1 scale. Repetition counts duplicate overlapping
casefolded whitespace4grams and is not a human correctness annotation.

| Seed | Arm | ROUGE-L F1 | Repeated-4gram outputs /582 | Unique predictions |
|---|---|---:|---:|---:|
| 0 | Reference | 0.165967 | 204 | 349 |
| 0 | Plain MLP | 0.187911 | 203 | 404 |
| 0 | Q-Former | **0.210121** | **158** | 379 |
| 1 | Reference | 0.175777 | 218 | 348 |
| 1 | Plain MLP | **0.234529** | **98** | **526** |
| 1 | Q-Former, selected10400 | 0.207400 | 231 | 301 |

ROUGE improved versus reference for both candidate arms in both seeds. Repetition
did not improve consistently for Q-Former: seed0 improved, seed1 worsened despite
better corpus BLEU. Plain MLP's largest repetition/diversity improvement occurred
at seed1. Neither an average BLEU gain nor an attention mechanism proves fine
handshape, date or temperature grounding.

## Resource measurements

| Arm | Trainable parameters | Visual-adapter parameters | FLAN LoRA parameters | Four-clip admission smoke peak |
|---|---:|---:|---:|---:|
| Reference | 23,472,129 | 14,034,944 | 9,437,184 | 7.18 GiB |
| Plain MLP | 24,783,873 | 15,346,688 | 9,437,184 | 7.19 GiB |
| Q-Former | 38,206,977 | 28,769,792 | 9,437,184 | 7.14 GiB |

One additional trainable VT temperature parameter accounts for the remainder.
Smokes checked joint CE+VT, all trainable groups, Q-Former learned-query/attention
gradients and state restoration without an optimizer update. These are smoke
allocated-memory peaks, **not measured full-training VRAM maxima**.

| Seed | Arm | Recorded run wall/GPU-stage hours |
|---|---|---:|
| 0 | Reference | 1.6140 |
| 0 | Plain MLP | 1.6837 |
| 0 | Q-Former | 1.8635 |
| 1 | Reference | 1.8654 |
| 1 | Plain MLP, crash/recovery included | 2.0913 |
| 1 | Q-Former | 1.7922 |

Overall controller charge **12.1521 hours**, within18 authorized hours. The
controller conservatively charged interrupted-child allowance across reboot,
plus setup/admission/validation work. Its charge is not GPU utilization time and
is not directly comparable to uninterrupted per-run throughput. Budgets were
not reset after the storage crash. Each run kept its immutable10800-second cap.

## Important reference-rerun caveat

These new reference scores differ from the earlier nominally matched linear16k
runs (5.489715 /5.715049). A read-only audit verified equal common trainable hashes,
resolved configs, feature/split identities, optimizer/scheduler, smoke input/loss,
sampler and seed0 path/default adapter semantics. No recipe discrepancy explaining
the difference was established. Deterministic decoding does not guarantee
bitwise-deterministic training kernels; backend nondeterminism is possible but
was **not proven** as the cause. Common hashes cover trainable shared parameters,
not every frozen-weight/buffer/backend state.

Therefore report this study's paired fresh references, preserve historical
results, and qualify absolute improvement claims. A controlled reproducibility
check is useful before a strong architecture claim. Do not tune the new reference
until it reproduces a favorable historical score or silently substitute the
historical values into this study's paired comparisons.

## Storage interruption and recovery

Windows C reached about3GiB available while Linux still had156GiB. User authorized
cleanup.20 duplicate resume checkpoints in completed runs plus one failed pending
write were removed, recovering113.51GiB inside Linux. All step/dev checkpoints,
reports, caches and incomplete recovery points were retained and verified.
The seed1 MLP already had all full-dev reports and valid terminal checkpoint;
selection was finalized without training or generation, then controller adoption
was reconciled conservatively. User compacted the WSL disk, restoring about120GiB
Windows headroom. Q-Former seed1 resumed as a fresh remaining arm after storage
admission and source-hash validation. [Recovery receipt](storage-recovery-2026-10-05.json).

No experiment source/profile/model settings changed during recovery. Earlier
completed runs were reused and verified; no old runs or canonical test evaluations
were regenerated. Snapshot retention remains latest2 for incomplete/new training
plus all dev-selected artifacts.

## Next decision

Plain spatial MLP is the best mean dev candidate and much smaller than Q-Former.
Q-Former gives the more even BLEU gains across these two seeds, but repetition
and the unexplained reference rerun variability require caution. Neither should
be declared the best unseen-signer model from dev results alone.

Recommended follow-up: verify reproducibility, then a separately authorized
matched held-out comparison of dev-selected reference/plain-MLP/Q-Former models,
with the same decoder and no test-based tuning. More training seeds or a new
signer-disjoint fold would strengthen evidence. New crop-derived descriptions
remain a separate future data intervention. The current study does not activate
optional patch pooling or change the previous provisional residual baseline.

## Artifacts and verification

Study directory: `/home/kan/datasets/despamo/signer-pilot/three-arm-connector-study-v1`.
`summary.json` contains all six run roots, selected checkpoint hashes, complete
metrics and paired deltas, trajectories, repetition, initialization and smoke
receipts, and resource accounting. `common-initialization.json` profile hash:
`9eaad3e46d27779925605d119bea1eaafcb97878c08790ad9dbb7eeca74ef54d`.
Each `seed*/study-state.json` binds guarded child stages/recovery and budget.

Final Q-Former seed1 run:
`connector-qformer-seed1-16000-c099b17106f2a4d14724e4ecf5969826aed573e77d7bf12775242533d87e901b`.
Selected step10400 SHA256:
`8c91ceb7e8158718d9f4d24e2131516f1820281c286282d43db0d6323c5afae0`.

Code verification before launch: **1002 tests passed,4 skipped**, full Ruff and
targeted Pyright clean, meaningful baseline-parity/masking/gradient/RNG/round-trip
tests. Independent review findings fixed before launch: terminal-step recovery,
exact report byte binding and full controller/watchdog source closure. Real-data
input validation, common attestations and all three admission smokes passed.
No commits, pushes or Signer07 inference were performed.

## Separately authorized held-out follow-up

The subsequent user-requested comparison scored each of these six dev-selected
models once logically on the same768Signer07 clips, without changing selections
or recipes. Both candidates' positive point estimates transferred in both seeds:
mean testBLEU-4 reference4.436637, plainMLP5.167847, QFormer4.902613.
Read [the held-out report](three-arm-connector-heldout-2026-10-05.md) for uncertainty,
repetition, registry provenance and limits. Earlier sections remain the original
dev-only training-study record, not a claim that test was used for selection.
