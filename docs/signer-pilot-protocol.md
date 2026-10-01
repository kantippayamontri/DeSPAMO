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
| **E1@8000 (frozen DINO)** | **4.0381** | 3.4238 | +0.4338 |
| **E3@8000 (adapted DINO)** | 3.9699 | **3.7337** | **+0.6765** |
| **E3 minus E1** | -0.0682 | **+0.3099** | **+0.2427** |

Every metric on the unseen signer improved: BLEU-1 +0.87, BLEU-2 +0.67, BLEU-3 +0.41,
BLEU-4 **+0.3099**, ROUGE-L +0.0035. The +0.3099 gain on the unseen signer exceeds
the 0.3-point significance threshold. Paired report:
`/home/kan/datasets/despamo/signer-pilot/signer07-e1-e3-8000-paired-v1.json`.

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
