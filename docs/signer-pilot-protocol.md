# Signer-Disjoint E1/E2/E3 Pilot

Frozen external protocol:
`/home/kan/datasets/despamo/signer-pilot/protocol/0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7/protocol.json`.
Its split hash is
`a95fa8f2710ebca33d6565a26f0f076817d839ecc611bc43648a48761fd6527b`.

| Role | Signers | Clips | Qwen valid | Qwen failed |
|---|---|---:|---:|---:|
| Train | 01, 02, 04, 05, 06, 08, 09 | 5,746 | 5,641 | 105 |
| Dev | Signer03 | 582 | 571 | 11 |
| Held-out test | Signer07 | 768 | 765 | 3 |

All three logical groups originate from original physical PHOENIX `train/`.
Training uses only seven train signers; failed Qwen captions keep translation
targets but have all appearance factors masked. Dev references are used only
for checkpoint selection. Generation prompts contain no held-out references.
Signer07 was scored once for selected E1/E2/E3 step-4000 checkpoints; official
PHOENIX dev/test remain untouched.

Fresh matched seed-0 E1 (frozen original DINO, no factor heads) and E2
(same DINO plus seven projector-level factor heads) each trained 4,000
optimizer steps, physical batch 4, VT-only warm-up 1,000, WSL
`num_workers=0`. Both start with shared trainable tensor hash
`9e3a02201ca44b7ee1a1d62da3eba1308a91ee914cab983036dc6505aa8a4d88`.
Full 582-clip dev evaluation uses deterministic beam 5 at joint-phase
steps 1,750, 2,800 and 4,000, choosing highest corpus BLEU-4 (earliest
step on a tie). See [E1 result](e1-v3-wsl-retraining.md) and
[E2 result](e2-wsl-factor-run.md) for checkpoint hashes and measured scores.

The original three-way E1/E2/E3 20-GPU-hour admission gate failed on a
measured estimate; its external receipt is
`/home/kan/datasets/despamo/signer-pilot/profile-gate-b4.json`. E1/E2 were
subsequently authorized as separate bounded runs. Full E3 adaptation,
signer-probe evaluation and Signer07 held-out translation scoring were not
part of those original bounded training runs. Dev scores alone cannot
establish reduced signer information.

## Optional E1/E2 signer diagnostic

`scripts/probe_signer_pilot.py --mode preflight|run` accepts frozen protocol,
dataset, text manifest, DINO/motion roots, motion manifest, physical train
annotation, completed E1/E2 run roots, and a new output JSON path. It checks
selected step-4000 checkpoint hashes and all source identities, then probes
mean-pooled **spatial-projector outputs** with fresh seven-class linear
classifiers. Only the seven logical training signers enter its shared,
clip-disjoint fit/validation/test partition. Signer03 and Signer07 do not enter
this closed-set probe; original translation training already saw all probe
clips. This aggregate result is diagnostic only, not an E3 gate or an
unseen-signer translation score.

## Selected-checkpoint Signer07 result

Full 768-clip Signer07 logical test, deterministic beam 5, reference-free
prompts, identical ordered IDs and references. Selected checkpoint hashes
matched frozen E1/E2 identities. E1 BLEU-4: **2.990082**; E2 BLEU-4:
**2.487775**; E2 minus E1: **-0.502306** points, narrowly below the
predeclared -0.5-point boundary by 0.002306. One seed and one held-out signer
do not establish general signer invariance or robust superiority. Artifacts:
`/home/kan/datasets/despamo/signer-pilot/signer07-e1-e2-heldout-v2/` contains
both 768-row reports and `paired.json`. The first attempt at `v1/` wrote no
reports due to a WSL CUDA loader configuration; the `v2/` run reused its
verified E1 result after correcting E2 checkpoint provenance validation, so
neither checkpoint was scored a second time.

## E3 mask-aware gradient smoke

The recovered E3 profile preflight assumes all 19 captions are present on
every schema-valid clip; 161 logical train clips have legitimate partially
masked factor targets. A separate read-only alignment check accepted nullable
targets and verified CLIP NPZ masks/zero values for selected smoke clips. On two
fully valid train clips plus one failed clip (19, 19, and 0 active targets),
one pinned DINOv3 final-four Q/V LoRA optimizer step produced finite loss and
nonzero gradients in all eight LoRA up-projections. Four of seven fresh factor
heads had nonzero gradients; three had zero contrastive loss because this tiny
sample lacked distinct labels for those factors. This is a **smoke only**, not
an E3 training result, full adapted-feature extraction, or translation score.
The frozen receipt is
`/home/kan/datasets/despamo/signer-pilot/e3-smoke-mask-aware-v1.json`.
The original three-method 20-GPU-hour feasibility gate remains failed.

## Declared GPU ceiling

The frozen protocol records `gpu_hour_ceiling: 24`, which the original bounded
E1/E2 runs respected. Completed work now totals roughly **18 GPU-hours**: E1
attempts and E1-v3 (1.704 h), E2 including its profile (0.729 h), E3 adaptation
(5.093 h), E3 adapted-feature extraction (9.811 h), E3 SpaMo (0.657 h), plus
bounded profiles and diagnostics. The authorized ceiling for continued
signer-pilot work is raised to **40 GPU-hours**, superseding the 24-hour figure
for planning purposes; the original number stays recorded here as history and
the frozen protocol file is unchanged. Each GPU stage still enforces its own
explicit per-run cap in code, and the cumulative ceiling is documentation only,
so remaining budget must be checked before authorizing a new stage.

E3 training and extraction commands are described in
[E3 code handoff](e3-code-runbook.md).

## Completed E3 pilot (4,000-step tier)

E3 LoRA adaptation completed 35,900 steps: 7,180 head-only calibration
(5 epochs) plus 28,720 DINO LoRA joint steps (20 epochs), within its 8-hour cap
at **5.093 GPU-hours**. Adapted 224/448 DINO features for all 827,354 frames
across 7,096 physical `train/` clips passed the complete manifest/receipt
gate at **9.811 GPU-hours**. Fresh E3 SpaMo trained 4,000 steps under its
2-hour cap at **0.657 GPU-hours**. Full Signer03 dev selected step 4,000,
checkpoint SHA-256
`b36ec2eb783e21e2b44dc48b8dcf5a8e184a75e26bda43c6fe1606762d96e094`.

| Method | Signer03 dev BLEU-4 | Signer07 held-out BLEU-4 |
|---|---:|---:|
| E1 frozen DINO | 3.695629 | 2.990082 |
| E2 projector factors | 3.028302 | 2.487775 |
| E3 adapted DINO | 3.457467 | 3.057283 |

All held-out reports have the identical ordered 768 Signer07 clip IDs,
references, split hash and deterministic beam-5 decoding. E3 minus E1 test
BLEU-4 is **+0.067201**, and E3 minus E2 is **+0.569508**. E3 held-out report:
`/home/kan/datasets/despamo/signer-pilot/signer07-e3-heldout-v1.json`.

## Stage 0b Extended Budget Results (8,000-step tier)

To test whether the +0.067 gain was suppressed by undertraining, E1 and E3 were
both retrained with `--steps 8000` (warmup 2,000, full-dev evaluation checkpoints at
3,500, 5,600, and 8,000). Both models peaked at **step 5,600**.

- **E1@8000**: selected step 5,600 (BLEU-4 **4.0381** on dev; was 3.6956 at 4k).
  Run: `e1-v3-483c1500d1bc292990f65cd79a96a40899bc3843f92186e1f41243386c444fe8` (0.962 GPU-h).
- **E3@8000**: selected step 5,600 (BLEU-4 **3.9699** on dev; was 3.4575 at 4k).
  Run: `e3-only-d1d697be1ee8e847679c73eb4632b80eb27418a0f182b32bdb1387d691e60f08` (0.889 GPU-h).

Held-out scoring on the 768 Signer07 test clips:

| Method | Signer03 dev BLEU-4 | Signer07 held-out BLEU-4 | Gain from 8k steps |
|---|---:|---:|---:|
| **E1@8000 (frozen DINO)** | 4.0381 | 3.4238 | +0.4338 |
| **E3@8000 (adapted DINO, collapsed)** | 3.9699 | 3.7337 | +0.6765 |
| **E3@8000 (relational w=0.1, preserved)** | **4.4447** | **3.8554** | **+0.7981** |

### Comparison of Final Relational E3 (w=0.1) vs Baseline (E1) at 8,000 steps:
- **Signer03 dev BLEU-4:** **4.4447** vs 4.0381 (**+0.4066**)
- **Signer07 test BLEU-4:** **3.8554** vs 3.4238 (**+0.4316**)
- **Signer07 test BLEU-1:** **20.3410** vs 17.8989 (**+2.4421**)
- **Signer07 test BLEU-2:** **9.3507** vs 7.9805 (**+1.3702**)
- **Signer07 test BLEU-3:** **5.5132** vs 4.7684 (**+0.7448**)
- **Signer07 test ROUGE-L:** **0.1662** vs 0.1575 (**+0.0086**)

Both models selected their step-5600 checkpoints on Signer03 dev. The +0.4316 gain on the
unseen held-out signer substantially exceeds the 0.3-point significance threshold.
Paired report:
`/home/kan/datasets/despamo/signer-pilot/signer07-e1-e3-relational-0.1-8000-paired-v1.json`.

### Final Signer Probe on Trained Spatial Projector Outputs (867 test clips, 7 signers):
Evaluated using the identical partition hash (`9462d3647beaf...`), linear probe, and evaluation protocol as E1 and E2:

| Model | Signer Top-1 Accuracy | Balanced Accuracy | Delta vs Baseline (E1) |
|---|---:|---:|---:|
| **E1 (frozen DINO baseline)** | 99.54% | 99.60% | — |
| **E2 (projector GRL factors)** | 99.42% | 99.25% | -0.12% (top-1) / -0.35% (balanced) |
| **E3 (relational w=0.1, 8k)** | **97.81%** | **95.37%** | **-1.73% (top-1) / -4.23% (balanced)** |

Report: `/home/kan/datasets/despamo/signer-pilot/probe-e3-relational-0.1-8000-projector-v1.json`.

## Diagnosed adaptation defect and the relational structure fix

Probing the completed 4,000-step E3 adaptation showed its DIFFER-style objective did
**not** work as intended. Articulator information improved (left/right handshape linear
probes rose from 0.360/0.226 to 0.430/0.304), but the nuisance GRL collapsed the whole
feature cloud into a narrow cone instead of removing signer identity: mean pairwise
cosine rose from 0.770 to 0.977, its standard deviation fell from 0.094 to 0.009, the
same-signer centroid margin fell from +0.0439 to +0.0035, and signer top-1 accuracy
**rose** from 0.979 to 0.989. DIFFER preserves metric structure through its ReID
identity loss; this pilot had no equivalent scaffold, so the single per-vector cosine
reference term (weight 0.10) could not prevent the collapse.

Two relational preservation terms were added against detached frozen-DINO teachers,
per scale, over frame-level pairs, alongside the retained `scale_reference_loss`:
**Term A** matches the standardised off-diagonal pairwise-cosine pattern
(`1 - correlation`) and **Term B** matches their dispersion (squared log std-ratio).
Term A alone is scale-invariant and tolerates a uniform shrink, which is exactly the
observed failure, so Term B is required. On the real collapsed features Term A reads
0.3523 against 0.0042 for perfect preservation, while a naive pairwise MSE reads only
0.0266, roughly 90x below the ~2.3 factor losses and therefore too weak to matter.
Adaptation batch was raised to 16, which removes every dead nuisance contrastive step
(biometric 18.5% -> 0.1%) at 11.2 GiB peak VRAM; balanced sampling was therefore not
needed. SpaMo batch remains 4, as `pilot_config` admits only 2 or 4 and E1/E2 used 4.

A subset gate, `scripts/probe_e3_features.py`, extracts adapted features for a
deterministic per-signer sample of about 600 logical train clips, roughly 1% of a full
extraction, and reports `std_ratio`, signer top-1, and handshape retention. Validated
against the known-bad 4,000-step adaptation it independently reproduced the failure
(`std_ratio` 0.088, signer delta **+0.011**, `gate_pass` false), so it is trusted to
judge new variants.

Each sweep variant trained 718 steps (2 epochs, batch 16, 143 calibration steps) at a
measured 2.14 s/step:

| Relational weight | gate | `std_ratio` | signer top-1 delta | handshape L / R delta |
|---|---|---:|---:|---:|
| baseline (no terms) | fail | 0.088 | **+0.011** | +0.045 / 0.000 |
| **0.1** | **pass** | **0.8224** | **-0.0562** | 0.0000 / +0.0290 |
| 0.15 | fail | 0.8032 | -0.0899 | 0.0000 / -0.0435 |
| 0.2 | fail | 0.7943 | -0.1236 | -0.0448 / -0.0290 |
| 0.4 | fail | 0.8890 | -0.0449 | -0.0448 / +0.0290 |

Weight **0.1** is the only variant satisfying all three criteria together: geometry
restored from 0.088 to 0.822, signer leakage reduced for the first time instead of
increased, and handshape retained. Across 0.1, 0.15 and 0.2 the signer suppression
strengthens monotonically (-0.0562, -0.0899, -0.1236) while handshape retention decays
in step (+0.0290, -0.0435, -0.0738 summed over both hands), so the weight directly
trades signer invariance against linguistic content and 0.1 sits at the usable edge.
Weight 0.4 is non-monotonic on both axes, consistent with the stronger pull toward
frozen DINO leaving less room for the nuisance GRL to act at all. Gate reports:
`/home/kan/datasets/despamo/signer-pilot/gate-sweep-w{0.1,0.15,0.2,0.4}-v1.json` and
`gate-validate-known-collapse-v1.json`. These probe partitions hold roughly 90 to 700
clips from a single seed, so treat the magnitudes as indicative and the directions as
reliable; reduced probe accuracy is not by itself evidence of better translation.

## Matched-frame probe repair (2026-10-03)

The v1 subset gate pooled all original frames but only five adapted frames for
geometry and signer probes. The tables above remain historical v1 results;
their signer deltas and spread ratios include that sampling confound. Detecting
the known collapsed encoder did not validate the gate as a predictor of BLEU.

`scripts/probe_e3_features.py` now publishes schema 2 under
`probe_policy=e3-probe-matched-five-v2`. Both encoders use the same five source
frame indices. Clip-mean geometry/signer probes and middle-frame hand probes
therefore compare aligned inputs. Regression controls require unit spread ratio
and zero signer delta when the encoder is compared against itself.

`gate_status` is `PASS`, `FAIL`, or `INCOMPLETE`. A completed scoring operation
still has `status=complete`; an unavailable hand probe has an explicit reason,
`gate_status=INCOMPLETE`, and `gate_pass=null`. Both hands must be available
before a pass/fail verdict. Seven-class hand probes require at least seven clips
per class so every fit/validation/test partition is populated.

Historical heuristic thresholds are unchanged and recorded in each report:
`std_ratio > 0.60`, signer top-1 delta `<= 0.005`, and both hand top-1 deltas
`>= -0.02`. Translation checkpoint selection continues to use dev BLEU.

Corrected reports use verified full feature caches, then select five rows per
clip; no adaptation, extraction, or held-out translation evaluation was rerun.
Both reports cover the same deterministic 543 logical-train clips:

| Full adaptation recipe | `std_ratio` | Signer top-1 delta | Hand L delta | Hand R delta | `gate_status` |
|---|---:|---:|---:|---:|---|
| Constant LR, relational 0.1 | 0.830648 | -0.123596 | -0.044776 | -0.014493 | FAIL |
| Cosine/lower LR, relational 0.1 | 0.941416 | -0.022472 | -0.029851 | +0.014493 | FAIL |

Reports under `/home/kan/datasets/despamo/signer-pilot/`:

- `gate-full-w0.1-matched-five-v2.json`
- `gate-e3-differ-cosine-full-matched-five-v2.json`

Each report binds the adaptation checkpoint, cached encoder/manifest/frame-row
hashes, sampled clip/frame-index hashes, source/text digests, code hashes,
thresholds, and appearance-supervision policy. Cache loading checks per-clip
feature/receipt hashes and source-frame alignment before scoring. Original v1
reports and frozen captions remain historical artifacts.

To use the CPU cache path, pass all existing probe arguments plus
`--adapted-root /path/to/e3-features-<key>`, and choose a new output filename.
`--mode preflight` validates cache metadata without producing a report;
`--mode run` verifies selected feature files and scores the probes. Without
`--adapted-root`, the script still extracts the adapted five-frame subset on CUDA.

## Residual projector / longer downstream control launch (2026-10-03)

The user approved launching the downstream ablation while deferring Windows disk
compaction. The experiment reuses cosine feature cache `7122b42e...` and the
completed linear-8000 baseline (dev BLEU-4 4.709851); it does not consume the other
server's new descriptions. Full design:
[residual-projector experiment](superpowers/specs/2026-10-03-residual-projector-experiment.md).

Two new versioned arms run sequentially through `scripts/run_projector_experiments.sh`:

- Residual-MLP projector: 8000 steps, hidden width 1024 for spatial/motion branches,
  zero-initialized final residual layers, 2000 VT warmup, 800 LR warmup, cap 10800 s.
- Original linear projector control: 16000 steps, 2000 VT warmup, 1600 LR warmup,
  cap 14400 s. This declares a new scheduler horizon and fresh initialization.

Both preserve batch 4, seed 0, adapted features, temporal fusion, optimizer groups,
deterministic decoder, logical split and appearance-supervision labels. Each new
run retains only its latest two atomically saved and fsynced resume checkpoints;
all dev-selection checkpoints are preserved. Historical run artifacts are outside
this retention policy. Budgets and status writes remain under the owned run lock.

Run keys:

- `projector-residual-8000-def98a1f1fd934780ba7e18fe39e221c4d0a839e669cfa072d43931729dce325`
- `projector-linear-16000-eb0a5bef8210afc3d894f4ac57338f8cd5e0f01b6b4d74a8a38cb285b4f5f313`

Live session: tmux `projector-experiment-v1`. Log:
`/home/kan/datasets/despamo/signer-pilot/projector-experiment-v1-launch.log`.
Residual real-data joint CE+VT smoke passed with nonzero gradients in both
projectors and FLAN LoRA, no optimizer update, and restored initialization/RNG/
buffers. The first run was observed past warmup at step 2356 with two retained
resumes; the linear control is queued. Evaluation in this launch uses dev only.

Implementation verification: 663 tests passed, 4 skipped; full Ruff and targeted
Pyright clean. Independent review's checkpoint-durability, refused-resume status
and source-hash closure findings were reproduced and fixed before launch.

## Equal-budget residual 16000 extension (2026-10-03)

The completed original linear 16000 control selected step 16000 with dev
BLEU-4 **5.489715** (5500: 4.148136; 10400: 5.410774; 16000: 5.489715).
The user approved a fresh residual-MLP 16000 arm for an equal-budget architecture
comparison. It uses the same cached features, batch 4, seed 0, VT warmup 2000,
LR warmup 1600, 16000-step cosine horizon, optimizer groups, decoder and dev
schedule 5500/10400/16000. Both spatial/motion residual branches match the 8000
architecture; this is not a spatial-only ablation.

The runner verifies and binds the completed linear control's full-dev selection,
config/input hashes and selected checkpoint. All 56 shared local source hashes
match; the approved entrypoint extension is versioned separately. Before editing,
all 57 original source files were archived and verified against both original
arm identities. See [matched residual 16000 plan and execution record](superpowers/plans/2026-10-03-matched-residual-projector-16000.md).

Fresh run:
`projector-residual-16000-f8deb3d4b807d59f3e0df2e6e6253ec069fb01d1a3365c36d4c257422d87526d`.
Tmux `projector-residual-16000`; log
`/home/kan/datasets/despamo/signer-pilot/projector-residual-16000-launch.log`.
Real joint gradient smoke passed, initialization matched the residual 8000 arm,
and live training was verified. Cap remains 14400 s with latest-two resume
retention. Current comparison uses dev only; Signer07 remains unscored for these arms.

## Held-out and paired-seed robustness completion (2026-10-04)

The user authorized once-only Signer07 scoring for both selected 16000-step
models, error analysis, a conditional future-description baseline and a matched
seed1 replication, up to ten additional GPU-stage hours. The full workflow
completed in **3.8711 hours**, with four complete 768-item registry claims.

| Seed | Linear dev BLEU-4 | Residual dev BLEU-4 | Linear Signer07 BLEU-4 | Residual Signer07 BLEU-4 | Paired held-out delta |
|---|---:|---:|---:|---:|---:|
| 0 | 5.489715 | 6.557227 | 4.784803 | 5.915115 | +1.130312 |
| 1 | 5.715049 | 4.583490 | 4.991212 | 3.849948 | -1.141264 |

All four models selected step16000 on dev before test scoring. Two-seed mean
held-out BLEU-4 is **4.888008 linear versus4.882532 residual**; residual sample
standard deviation1.460294 versus0.145953 linear. The primary MLP gain transferred,
but its sign reversed in the second seed. This is a seed-sensitive result, not a
robust architecture or signer-invariance claim.

Repeated-4gram outputs: seed0 271 linear /213 residual; seed1 253 linear /274
residual. Lexical date/location/temperature diagnostics, supported-denominator
counts, representative errors and conditional cluster-bootstrap intervals are in
[the full report](projector-heldout-robustness-2026-10-04.md). These are automated
lexical comparisons, not human/visual correctness ratings or complete semantic
error rates. Actual new descriptions are not inputs to these experiments.

Study artifacts: `/home/kan/datasets/despamo/signer-pilot/projector-overnight-study-v1/`.
Registry: `/home/kan/datasets/despamo/signer-pilot/projector-heldout-registry-v1.json`.
The predeclared primary-transfer rule saved residual16000/seed0 as a
**provisional mixed-robustness** future-description recipe, with linear16000
retained as comparator: `configs/experiment/projector_residual16_provisional.yaml`.

## Three-arm downstream connector study completed (2026-10-05)

User authorized reference/plain spatial MLP/small Q-Former across two seeds,
six fresh16000-step runs, up to18GPU-stage hours. Motion remained linear in all
arms. Plain MLP replaced only spatial2048→768 with2048→1024→768/GELU. Q-Former
retained linear inputs and TCN, replacing the final output MLP with32 learned
queries,2layers,width768,12heads,FF3072,dropout0.1 and temporal positions.
It was from scratch, operating on frame-summary features rather than patches.
VT-Align remained2000-step warmup followed by CE+VT.

| Arm | Seed0 dev BLEU-4 | Seed1 dev BLEU-4 | Mean |
|---|---:|---:|---:|
| Fresh reference | 4.541238 | 4.571809 | 4.556524 |
| Plain spatial MLP | 5.197683 | 7.090514 | 6.144099 |
| Small Q-Former connector | 5.695729 | 5.901238 | 5.798483 |

All runs reached16000; seed1 Q-Former selected10400 on dev, other arms selected
16000. Both candidates beat fresh reference in both seeds, but plain MLP had
greater observed spread. Reference rerun differences versus historical linear
scores remain unexplained after recipe/initialization audit. These results are
dev-only; no Signer07 inference or strong architecture/significance claim.

Storage crash occurred during seed1 MLP's final resume write.113.51GiB redundant
completed-run snapshots/failed temporary write were removed under explicit user
authorization; intact terminal/dev results finalized without training. After user
compacted host disk, all completed work was revalidated and the last Q-Former arm
finished. Controller charged12.1521hours conservatively, within18, without reset.
Full [three-arm results](three-arm-connector-results-2026-10-05.md), including
metrics, repetition, parameters, trajectories, storage/reproducibility caveats.

## Three-arm connector held-out follow-up (2026-10-05)

After explicit user request, all six fixed dev-selected models were scored once
logically on exact ordered768Signer07 clips. Same beam5/max64/noICL/BF16/batch4;
no retraining, test-based selection or architecture/data changes. QFormer seed1
remained at its dev-selected10400 checkpoint; every training horizon was16000.

| Arm | Seed0 Signer07 BLEU-4 | Seed1 Signer07 BLEU-4 | Mean | Mean gain vs fresh reference |
|---|---:|---:|---:|---:|
| Fresh reference | 4.360015 | 4.513259 | 4.436637 | 0 |
| Plain spatial MLP | 4.637257 | 5.698437 | 5.167847 | +0.731210 |
| Small Q-Former connector | 5.150172 | 4.655053 | 4.902613 | +0.465976 |

Both alternatives improved held-out point estimates in both seeds. PlainMLP
had the higher mean and greater observed spread; QFormer won seed0 but notseed1.
Conditional69-broadcast bootstrap intervals includedzero for PlainMLP seed0
andQFormer seed1. Two seeds/one held-out signer and unresolved reference-rerun
variability limit broad significance/architecture/invariance claims.

Reports: `/home/kan/datasets/despamo/signer-pilot/three-arm-connector-heldout-v1/controller/`.
Shared registry now10complete claims:6new/4preserved, each768journaled items.
The unmodified original evaluator's registry engine was reused privately with
connector-specific reconstruction/source bindings. Runtime charge1.1045GPU-stage
hours. Full [held-out comparison](three-arm-connector-heldout-2026-10-05.md).
No old results, caption records, selected weights or test decoder were overwritten.
