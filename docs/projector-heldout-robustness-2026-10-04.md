# Projector held-out comparison and robustness results

## Executive result

All requested tasks completed: both original dev-selected 16000-step models were
scored once on Signer07, errors were compared, a matched second training seed was
run and scored, and a conditional downstream recipe was saved for future descriptions.
Workflow used **3.8711 conservative GPU-stage wall-hours**, below the authorized
10-hour cap. All four registry entries are complete with 768 committed clips.

The residual MLP's primary gain transferred to the unseen signer, but **did not
replicate at seed 1**. Its two-seed average BLEU-4 is essentially the same as linear,
with much greater observed variation. It is therefore a **provisional seed-sensitive
candidate**, not a demonstrated robust architectural improvement.

## Matched experiment

Both architectures used the same cached cosine DINO features, logical split,
16000-step horizon, 2000 VT warmup, 1600 LR warmup, optimizer groups, physical
batch 4, no accumulation, deterministic beam 5 / length 64 and no in-context
examples. Spatial and motion residual branches both changed; this is not a
spatial-only ablation.

Seed 1 varied common initialization, data order, dropout/prompt RNG, and residual
branch initialization (branch seed 2). Within each pair, common initial weights
were identical and attested before residual attachment. Every selected checkpoint
was chosen using only the complete ordered Signer03 dev schedule. The test set is
the exact ordered 768 Signer07 clips, physically from PHOENIX train.

| Training seed | Architecture | Selected step | Dev BLEU-4 | Signer07 BLEU-4 | Signer07 ROUGE-L F1 |
|---|---|---:|---:|---:|---:|
| 0 | Linear | 16000 | 5.489715 | 4.784803 | 0.197896 |
| 0 | Residual MLP | 16000 | **6.557227** | **5.915115** | **0.220409** |
| 1 | Linear | 16000 | **5.715049** | **4.991212** | **0.193929** |
| 1 | Residual MLP | 16000 | 4.583490 | 3.849948 | 0.166829 |

| Seed | Residual minus linear BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L F1 |
|---|---:|---:|---:|---:|---:|
| 0 | +2.683589 | +2.054444 | +1.519816 | **+1.130312** | +0.022513 |
| 1 | -3.429203 | -2.549412 | -1.612524 | **-1.141264** | -0.027100 |

ROUGE values above use the project's native 0-1 scale. These results are not
directly comparable to the released SpaMo checkpoint on the official test split.

## Repetition and prediction diversity

| Seed | Architecture | Outputs with repeated 4-grams / 768 | Unique predictions | Mean whitespace words |
|---|---|---:|---:|---:|
| 0 | Linear | 271 | 543 | 14.5130 |
| 0 | Residual MLP | **213** | **660** | 14.5052 |
| 1 | Linear | **253** | **598** | 14.6536 |
| 1 | Residual MLP | 274 | 421 | 13.9844 |

Residual reduces repetition and increases diversity at seed 0, but reverses both
patterns at seed 1. Duplicate-4gram flags are diagnostics: a flagged utterance is
not automatically incorrect, and legitimate repeated wording can be counted.
Whitespace lengths are not T5 decoder-token counts or proof of truncation.

## Dates, locations and temperatures

Analysis uses predeclared DEV-based German lexical/number rules. For an additional
paired comparison, the table below retains only clips with a recognized reference
value and supported parses in **both** predictions, so each pair has the same
denominator. An error is a value-set mismatch or a missing predicted value.

| Seed | Category | Shared eligible clips | Linear mismatches | Residual mismatches |
|---|---|---:|---:|---:|
| 0 | Date/day/month cues | 154 | 135 | **121** |
| 0 | Location/direction cues | 386 | **376** | 383 |
| 0 | Temperature values/ranges | 90 | 89 | 89 |
| 1 | Date/day/month cues | 155 | 142 | 142 |
| 1 | Location/direction cues | 394 | **388** | 392 |
| 1 | Temperature values/ranges | 90 | 90 | 90 |

Seed-0 residual improves recognized date cues, but there is no consistent
location/temperature grounding improvement. Both architectures still generate
generic weather phrases and incorrect numerical or geographic details.

Examples from completed seed-0 reports:

- `01August_2011_Monday_heute-4855`: reference mentions **Mittwoch**; linear
  prediction says **Samstag**, residual says **Dienstag**. Both are wrong weekdays.
- `01August_2011_Monday_heute-4850`: reference **Westen und Süden**; both models
  predict **Südosten und Südwesten**, shifting the recognized regions.
- `01August_2011_Monday_heute-4866`: reference reaches **dreißig Grad**; linear
  predicts **fünfzehn Grad**.
- `04July_2011_Monday_tagesschau-701`: reference **16 Grad an der Ostsee / 7 im
  Allgäu**; residual predicts **13 am Oberrhein / 19 im Südwesten**.

These are **rule-based lexical diagnostics, not human/visual correctness ratings**.
Values lose ordering, multiplicity and event/place/date associations. Calendar
year/relative-date reasoning, some location vocabulary, wind-versus-location roles,
decimal temperatures, missing-unit numbers and some degree-change expressions
remain unsupported or ambiguous. Unknown/unsupported cases are recorded explicitly
in each comparison JSON and excluded from the shared-support table. High mismatch
rates must not be presented as complete semantic error rates. Appearance audit
ratings remain untouched.

## Robustness and uncertainty

| Architecture | Two-seed mean Signer07 BLEU-4 | Sample standard deviation |
|---|---:|---:|
| Linear | 4.888008 | 0.145953 |
| Residual MLP | 4.882532 | 1.460294 |

Mean paired difference: **-0.005476 BLEU-4**. Paired differences are +1.130312 and
-1.141264; their sample standard deviation is 1.606247.

Exploratory paired broadcast-cluster bootstrap (1000 resamples, fixed bootstrap
seed 0, 69 inferred broadcast groups):

- Training seed 0 delta interval: **[+0.7256, +1.5401]**.
- Training seed 1 delta interval: **[-1.5474, -0.7535]**.

Intervals condition on the fixed reports/checkpoints and inferred broadcast
grouping. They do not capture training-seed uncertainty and are not a broad
significance or signer-invariance test. The opposing seed results are the key
robustness finding. Two seeds are insufficient for a strong research claim.
We cannot attribute instability separately to initialization, data order,
dropout or residual-branch initialization from this study.

## Conditional baseline for new descriptions

The prespecified candidate was residual 16000, selected from dev before held-out
scoring. Because its primary seed-0 gain transferred, the requested recipe is
saved as **`provisional_mixed_robustness`**, with seed 0 as the default reproduction
seed. The linear 16000 recipe remains a required comparator/reference; it should
not be discarded because a favorable residual seed was observed.

Repository recipe:
`configs/experiment/projector_residual16_provisional.yaml`.
Runtime recipe and decision:
`external-data/signer-pilot/projector-overnight-study-v1/future-description-baseline.json`.
No checkpoint/seed was selected using test scores, and no recipe was tuned from
test error examples. The other server's new description version is not available
in this study. When it arrives, bind its actual dataset/schema/source/text/encoder
identities and fresh features before a new comparison; preserve all translations
and mask failed/unknown appearance targets. The recipe is prepared, not a claim
that those new descriptions were trained or evaluated.

Recommended research wording: “Residual projection improved the primary seed's
held-out translation, but the matched second seed reversed the gain; aggregate
performance was unchanged and the residual variant showed greater sensitivity.”

## Artifact index

Study root: `/home/kan/datasets/despamo/signer-pilot/projector-overnight-study-v1`.

- `seed0-linear-signer07.json`, `seed0-residual-signer07.json`.
- `seed1-linear-signer07.json`, `seed1-residual-signer07.json`.
- `seed0-comparison.json`, `seed1-comparison.json`: full lexical diagnostics,
  bounded examples, unknown counts, signatures and exploratory bootstrap.
- `summary.json`, `future-description-baseline.json`, `study-state.json`.
- `seed1-initialization.json`: common hash
  `769848a2d06f6f02b38264a3258616a1f4ea343aeca9c7e61cfaea458625d87c`.
- Authoritative registry: `../projector-heldout-registry-v1.json`; four complete
  checkpoint-bound claims with durable 768-item journals and report hashes.

Seed-1 runs:

- `projector-linear-16000-0319401a7579e5fc6ee3a37748c01a441b2e0b75cf97ce9778f6fc60e1f457da`
- `projector-residual-16000-86f337958a8244e9538b3f36d7f6186859982f20fa1edd4ac71f614b8a54f353`

Selected checkpoint hashes (seed/architecture):

- 0/linear: `8cde697bf31ba67f0f24ddcb229d7f39829eb3d3cafc9c3f03b8171cf90fef4a`.
- 0/residual: `2c218623c1d725e733ac8fedc7db82a0f2e22ce6f12ab2777fa86d91946a46ea`.
- 1/linear: `5539b5e3c6c86fde600c7ff84e373f453c8c778ff61f2112f89b4403e0421661`.
- 1/residual: `8821bcaa3adfc2c18418240a0a28c7a5f7863fcbd2396c98fe5df3ba895f0579`.

## Verification and operational record

Integrated code verification: **874 tests passed, 4 skipped**, full Ruff clean,
targeted Pyright clean. Independent recovery findings were regression-fixed:
child/watchdog lifecycle, uncapped timeout accounting, completed-training adoption,
and registry-verified scoring reuse. A metadata-first checkpoint load was retained
after measuring 62-GiB RAM / 48-GiB availability on this host.

Every new training run retained two durable resume snapshots and all dev-selected
models. Both seed-1 runs completed at 16000 with closed budgets, and held-out scoring
used each dev-selected checkpoint exactly once logically. Completed reports were
revalidated through the registry before analysis. Workflow accounting includes
CPU setup/validation conservatively and totaled 13936.07 seconds, not a 10-hour
spending target. Existing caption records, historical results and source snapshots
remain provenance-bound artifacts; no commit, push or merge was performed.
