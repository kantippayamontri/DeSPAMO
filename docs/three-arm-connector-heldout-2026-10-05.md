# Three-arm connector comparison on unseen Signer07

## Result

All six fixed dev-selected checkpoints were evaluated once logically on the
same ordered **768 Signer07 clips**, with identical decoding. No additional
training, checkpoint reselection, architecture change or test-based tuning.

| Arm | Seed0 test BLEU-4 | Seed1 test BLEU-4 | Two-seed mean | Sample standard deviation |
|---|---:|---:|---:|---:|
| Fresh reference | 4.360015 | 4.513259 | 4.436637 | 0.108360 |
| Plain spatial MLP | 4.637257 | **5.698437** | **5.167847** | 0.750368 |
| Small Q-Former connector | **5.150172** | 4.655053 | 4.902613 | 0.350102 |

| Candidate | Seed0 BLEU-4 delta vs reference | Seed1 delta | Mean paired delta |
|---|---:|---:|---:|
| Plain MLP | +0.277242 | +1.185178 | **+0.731210** |
| Small Q-Former | **+0.790157** | +0.141794 | +0.465976 |

Both alternatives' positive dev gains transferred to this unseen signer as
positive held-out point estimates in both seeds. Plain MLP has the best mean
among these three arms, but its gain magnitude remains seed-dependent. Q-Former
wins against MLP in seed0 and loses in seed1; smaller observed spread across two
seeds is not proof of universal stability.

## Frozen setup and selection

Original three-arm training study:
[dev study and architecture details](three-arm-connector-results-2026-10-05.md).
Same cached cosine DINO spatial features and VideoMAE motion; all arms trained
for16000 steps, batch4, VT warmup2000, LR warmup1600, matching optimizer groups
and schedule. Motion and TCN unchanged. Plain MLP modifies only spatial input
2048→1024→768; Q-Former replaces the output MLP with a from-scratch32-query,
two-layer temporal-summary attention connector. VT-Align retained.

The logical training split excludes Signer03 dev and Signer07 test. Evaluation
is deterministic beam5, maximum length64, no in-context examples, BF16 and
physical batch4. Prompts contain only language instructions, never test references.
The test clips physically originate from PHOENIX train under the frozen logical
signer split; released SpaMo results on the official split are not comparable.

| Seed | Arm | Dev-selected checkpoint | Dev BLEU-4 | Test BLEU-4 | Test ROUGE-L F1 |
|---|---|---:|---:|---:|---:|
| 0 | Reference | 16000 | 4.541238 | 4.360015 | 0.175385 |
| 0 | Plain MLP | 16000 | 5.197683 | 4.637257 | 0.186974 |
| 0 | Q-Former | 16000 | 5.695729 | 5.150172 | 0.198186 |
| 1 | Reference | 16000 | 4.571809 | 4.513259 | 0.181687 |
| 1 | Plain MLP | 16000 | 7.090514 | 5.698437 | 0.214316 |
| 1 | Q-Former | **10400** | 5.901238 | 4.655053 | 0.188804 |

Seed1 Q-Former selection was fixed at10400 by dev before admission. Its16000
checkpoint was not scored or substituted using test performance. Every arm
received the same declared training budget and selection schedule.

## Other metrics

Residual direction here means candidate minus fresh reference; neither candidate
is the earlier residual-MLP architecture. ROUGE uses native0-1 scale.

| Seed | Candidate | BLEU-1 delta | BLEU-2 delta | BLEU-3 delta | ROUGE-L F1 delta |
|---|---|---:|---:|---:|---:|
| 0 | Plain MLP | +1.461442 | +0.818249 | +0.431735 | +0.011589 |
| 1 | Plain MLP | +3.540287 | +2.348933 | +1.664092 | +0.032629 |
| 0 | Q-Former | +1.901647 | +1.457310 | +1.058445 | +0.022801 |
| 1 | Q-Former | -0.345234 | +0.173327 | +0.211052 | +0.007117 |

The Q-Former seed1 gain is modest and not uniform across BLEU orders. Neither
attention nor corpus-score improvement establishes handshape/semantic correctness.

## Repetition and diversity

| Seed | Arm | Outputs with repeated4grams /768 | Unique predictions | Mean whitespace words |
|---|---|---:|---:|---:|
| 0 | Reference | 292 | 493 | 14.1784 |
| 0 | Plain MLP | 295 | 542 | 14.7682 |
| 0 | Q-Former | **241** | 452 | 13.4180 |
| 1 | Reference | 289 | 477 | 14.2839 |
| 1 | Plain MLP | **164** | **665** | 13.8190 |
| 1 | Q-Former | 248 | 382 | 13.0547 |

Q-Former reduced duplicate4gram outputs in both held-out seeds, but generated
fewer distinct complete predictions and shorter outputs. Plain MLP's large
repetition reduction appears in seed1; seed0 repetition increased slightly.
These diagnostics do not prove truncation or human semantic quality. Word counts
are whitespace words, not decoder tokens. Repetition/diversity should be judged
alongside translation metrics rather than optimized as isolated targets.

## Paired uncertainty and limits

Exploratory paired broadcast-cluster bootstrap:1000 resamples, fixed bootstrap
seed0,69 inferred broadcast groups. Intervals condition on the already frozen
reports/checkpoints and do not account for training-seed uncertainty.

| Seed | Candidate vs reference BLEU-4 delta | Conditional95% interval |
|---|---:|---|
| 0 | Plain MLP: +0.277242 | [-0.0970, +0.6413] |
| 1 | Plain MLP: +1.185178 | [+0.8061, +1.5982] |
| 0 | Q-Former: +0.790157 | [+0.3947, +1.1689] |
| 1 | Q-Former: +0.141794 | [-0.2332, +0.5409] |

Plain MLP seed0 and Q-Former seed1 intervals include zero. Positive point
estimates in two seeds therefore support promising transfer, not a strong
across-seed significance or signer-invariance claim. Two seeds and one held-out
signer are insufficient for broad generalization assertions.

The fresh reference reruns differ from earlier nominally matched linear16k
results. As documented in the dev report, audits found no explaining recipe
discrepancy; backend nondeterminism is possible but not established as the cause.
Use the current matched references for paired conclusions, retain historical
results, and address reproducibility before a strong architecture claim.

The study does not show that a larger projector always helps, that Q-Former
beats plain MLP universally, or that either solves numeric/location grounding.
Archived automated lexical analysis includes supported/unparseable/unsupported
cases; these are not human/visual appearance audit ratings.

## Recommendation

Plain spatial MLP is the leading mean-score candidate within this comparison
and costs substantially fewer trainable parameters than Q-Former (24.78M versus
38.21M total). Keep Q-Former as a useful alternative with stronger seed0 transfer
and lower observed two-seed spread. Do not select a favorable seed using test
scores or alter checkpoints after seeing these outcomes.

Next research evidence: a controlled fixed-seed reproducibility check, more
matched training seeds, or an additional genuinely signer-disjoint fold. New
closed-vocabulary/crop-derived descriptions are a separate future data-version
intervention and were not used here. No new training is launched by this report.

## Once-only artifacts and accounting

New study root:
`/home/kan/datasets/despamo/signer-pilot/three-arm-connector-heldout-v1/controller/`.

- `seed{0,1}-{reference,plain_mlp,qformer}-signer07.json`.
- `seed{0,1}-{plain_mlp,qformer}-vs-reference-comparison.json`.
- `summary.json`, `study-state.json`, per-child preflight/scoring logs.
- Shared authoritative registry:
  `/home/kan/datasets/despamo/signer-pilot/projector-heldout-registry-v1.json`.

Registry now has **ten complete claims**: six new connector checkpoints plus
four preserved earlier projector claims. Each new claim has exactly768 committed
items, a durable checkpoint/decoder/source-bound journal and report byte hash.
Completed scoring can only be reused after validation; interrupted work resumes
durably committed batches under the same claim. Changing/deleting authoritative
registry/journals defeats once-only history and is not permitted.

Scoring controller charged **1.1045 GPU-stage wall-hours**, below10800seconds
(3hours). CPU setup/admission/revalidation counted conservatively; no additional
checkpoint files or extracted features created. No canonical reports overwritten.

## Verification

Connector evaluator uses a private instance of the existing journal engine;
the historical evaluator file and globals are unchanged. Strict connector
selection validates all three dev-report bytes, full checkpoint metadata/state,
architecture/common initialization, source/schema and fixed cache/decoder lineage.
Controller validates exact six seed/arm pairs and expected selection before
launch; the review finding was reproduced with failing tests and fixed.

Final code checks: **1048 tests passed,4 skipped**, full Ruff and targeted Pyright
clean. All six real-data preflights validated the exact intended checkpoints and
768 test clips before scoring. Final registry/report/paired references validated
again without generation. No commits, pushes or test-based selection performed.
