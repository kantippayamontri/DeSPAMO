# How2Sign Crop Probe (Stage 1 of the 2026-10-01 design, ported to How2Sign)

For end-to-end setup, dataset preparation, exact commands, recovery and HPC
handoff, see [the text-extraction reproduction runbook](../../docs/reproduce-qwen-text-extraction.md).

## Versioned queue drivers

Portable source copies of the active local queue drivers are in `scripts/`:

- `bash tools/how2sign/scripts/run_mouth_repair_queue.sh` repairs and resumes
  PHOENIX followed by How2Sign from existing `*_full_v1` roots.
- `bash tools/how2sign/scripts/run_how2sign_stable_queue.sh` waits for verified
  corrected How2Sign crop completion, extracts the stable factors, then exports
  the combined text-only handoff.

These scripts require the tool environment and dedicated Ollama servers on
ports 11435/11436. They resolve the checkout root automatically; override paths
with `DESPAMO_ROOT`, `DESPAMO_ARTIFACT_ROOT`, or `HOW2SIGN_VIDEO_ROOT` for another
host. Use one driver per output root. The host-specific driver instances and
all extracted artifacts remain in ignored `artifacts/`, outside source control.

## How2Sign four stable factors (approved 2026-10-06)

`how2sign_probe.stable` extracts biometric, clothing, hair and background once
per clip from five ordered whole frames. It imports the original
`despamo.appearance.schema.Stable` and canonical renderer via
`PYTHONPATH=src:tools/how2sign`; pydantic is locked in the isolated tool environment.
The stable-only prompt has a new version; the original frozen captions are not
modified. Source frame indices and decoded pixel hashes must match crop repair
receipts. Images are resized with aspect preserved to a longest edge of 640.

Output: `artifacts/how2sign_stable_wholeframe_v1/records/*/*.json`, bound by
`version.json` (source sample hash, prompt/schema hashes, model digest, code hash,
8192 context, deterministic options). Requests send all five images together and
use the measured non-thinking assistant-prefix path. Up to three attempts are
reserved before requests; final valid/failed records are retained on resume.
Failed clips receive four null targets. Whole-run completion requires every clip
to be terminal and a 98% schema-valid clip floor. Schema validity does not establish
visual accuracy; all outputs remain unreviewed and `human_review_status=not_assessed`.

The small live smoke processed four clips with four valid records and no retries:
139.9 seconds end-to-end, mean request time 33.68 seconds and 266–270 generated
tokens. This was a single smoke replica sharing the A30 with the crop run; it is
not a measured full-run throughput or representative visual-quality assessment.

`artifacts/run_how2sign_stable_queue.sh` waits for verified corrected crop completion,
then runs all 31,165 training clips on the two dedicated replicas and exports
`how2sign_qwen_19_targets_v1.json` plus a transfer checksum. This combines four
stable descriptions with fifteen corrected crop targets, checks matching clip
and source-pixel identity, and retains unreadable/failed clips with masks.
VIDEO_ID strata are stored as `video_stratum`, never fabricated as signer IDs.
No text embeddings, spatial features or training are included.

## Mouth-coordinate correction (2026-10-04)

**All v1 mouth labels and their distribution conclusions are invalid for research.**
Face-local landmarks were normalized by the face-crop dimensions, then interpreted
as full-frame normalized coordinates. This displaced mouth crops to image edges.
The corrected detector normalizes by the full-frame dimensions. Hand crops are
unaffected by this bug. Passing schema checks did not detect this visual error.

Recovery is additive: `artifacts/phoenix_full_mouth_v2/` and, after PHOENIX,
`artifacts/how2sign_full_mouth_v2/`. `how2sign_probe.repair` revisits every sampled
frame, keeps verified hand crops and completed hand journals, and never imports
an old mouth label, even if a new crop happens to share its old key.
`repair_clips/{clip}.json` tracks old mouth job keys/statuses, corrected boxes,
source-frame/crop hashes and hand labels reused. `repair_manifest.json` provides
the overall census. Per-clip receipts allow interrupted repairs to resume.

The replacement queue is `artifacts/run_mouth_repair_queue.sh`. It validates the
repaired detector gate, runs labeling and reporting, then checks every expected
job with `how2sign_probe.completion` before creating `driver_complete`. A killed
or failed stage cannot trigger the How2Sign handoff. Original v1 artifacts remain
available as historical evidence; earlier PASS verdicts are not visual-validity
or scientific-improvement evidence.

Feasibility probe for `docs/superpowers/specs/2026-10-01-cropped-articulator-supervision-design.md`
run against How2Sign realigned sentence videos instead of PHOENIX. Code lives in
`tools/how2sign/` with its own locked environment (mediapipe 0.10.21, opencv 4.11,
numpy 1.26.4); the frozen main and DINOv3 environments are untouched.

## Documented deviations from the design

| Design assumption | How2Sign reality |
|---|---|
| Stratify by signer (Signer03/07 excluded) | No signer mapping on this host (`-{N}-rgb_front` is a camera id), so sampling stratifies by `VIDEO_ID-camera`: 200 clips from 186 source videos. All train-split videos are eligible because this probe writes nothing to frozen artifacts. |
| 210×260 frames, hands ~57 px, crop magnifies 3.9×/5.5× | 1280×720 frames, median source crop 138 px; crops are usually downscaled, not magnified. |
| Five frame indices from frozen `source.json` | The probe defines them: the same 0.1–0.9 position recipe as `tools/dinov3/frames.py`. |
| Handshape probe on frozen DINO features | No DINO corpus exists for How2Sign and DINOv3 is gated; evidence is closed-vocabulary label statistics plus the detector report. |

Detector choice: MediaPipe legacy solutions (no GPU). FaceMesh on full 720p frames
misses faces, so mouth detection is two-stage: FaceDetection box → FaceMesh on the
padded face crop. Hand crops are the landmark bbox expanded 1.25×; mouth crops are
the lip-ring box sized by the nose→chin span.

## Stages

All commands from `tools/how2sign/` with its `.venv`; root is
`artifacts/how2sign_probe_v1` (git-ignored, strictly additive).

```bash
uv sync --locked                      # first time only
# 1. sample 200 clips (seed 0) + provenance: done
.venv/bin/python -u -m how2sign_probe.cli --root <root> sample \
  --csv ~/datasets/how2sign/realign_sentence_level/how2sign_realigned_train.csv \
  --video-root ~/datasets/how2sign/realign_sentence_level/train_sentence_videos --total 200 --seed 0
# 2. detect regions + write 224px crops, then apply the gate: done, gate passed
.venv/bin/python -u -m how2sign_probe.cli --root <root> detect --video-root <videos>
# 3. label every crop with qwen3-vl:8b (restart-safe journal; rerun to resume)
# Each server must run with OLLAMA_NUM_PARALLEL=4; the client request must NOT
# send `think` (see labeling.py: in raw mode think:false re-opens a think block).
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 ollama serve &
OLLAMA_HOST=127.0.0.1:11436 OLLAMA_NUM_PARALLEL=4 ollama serve &
.venv/bin/python -u -m how2sign_probe.cli --root <root> label --ports 11435 11436 --workers-per-port 2
# 4. aggregate
.venv/bin/python -u -m how2sign_probe.cli --root <root> report
```

Tests: `.venv/bin/python -m pytest tests -q` (85 offline tests; MediaPipe/Ollama
touched only through fakes or the detector smoke).

Handedness: MediaPipe reports handedness under the mirrored-selfie convention;
both datasets here are unmirrored recordings, so `hand_regions` flips the label
to the signer's anatomical side (`mirrored=False` in `side_label`). Runs from
before 2026-10-03 carried the unflipped side labels; aggregate field statistics
are unaffected, but any per-side (`LEFT`/`RIGHT`) numbers from those runs are
inverted.

## Measured costs on this host (A30 24 GiB)

- Detector: 200 clips / 985 frames / 2,723 crops in 264 s CPU.
- One labeled smoke: gate stats hand_rate 0.88, two_hand 0.80, mouth 1.0,
  below-floor 0.0, median crop 138 px → **detector gate passes**.
- VLM, naive config: `think:false` + assistant-`thinking`-field prefix still let the
  server generate ~1,200 hidden thinking tokens/call (~25 s/call; two replicas = ~5
  crops/min; 3rd/4th replica added nothing because generation, not loading, was the
  bottleneck).
- VLM, shipped config: raw ChatML whose assistant message `"\n\n"` closes an empty
  `` block and no `think` key is sent (raw + `think:false` re-opens the block
  — regression-tested): 34-60 generated tokens/call. Two `OLLAMA_NUM_PARALLEL=4`
  servers with 2 client streams each measured ~25 crops/min at 100% GPU util
  (~0.6 s/call amortized, prefill of ~1,500 image+prompt tokens dominates).
  Full 2,682-crop labeling: ~2 h.
- 3 unreadable videos in the sample (skip list in `detector_report.json`).

## Stage-1 results (completed 2026-10-02, seed 0, 200 clips / 985 frames)
Verdict: **PASS** (`label_report.json`). 2,681/2,682 crops labeled; 1 genuine
off-vocabulary failure (model persistently answered `finger_spread=extended`).
Latency mean 14.3 s incl. retries (8 concurrent streams), ~2 h total labeling.

| Field | distinct | top-1 share | uncertain rate |
|---|---:|---:|---:|
| handshape_class (all hands) | **12/12 classes observed** | 0.537 | 0.038 |
| left side / right side | 12 / 12 | 0.501 / 0.572 | |
| mouth openness | **2** | 0.956 | 0.044 |
| mouth lip_configuration | 4 | 0.906 | 0.055 |

Against the PHOENIX free-text baselines (3,787 distinct values, top-1 0.024,
1,363 singletons): closed vocabularies group perfectly — the label-space half of
the hypothesis holds, and at 720p the uncertain rate is ~2.6% overall, not the
blur-limited rate feared from PHOENIX resolution.

Caveats: the DINO handshape-probe criterion is not assessable here (no How2Sign
DINO corpus); extreme concentration (openness 2 values, thumb_position 3 of 5)
means some fields may still be too coarse for lexical content; hand labels are
ASL, not DGS, so no PHOENIX effect size transfers.

## PHOENIX 210x260 replication (completed 2026-10-02, seed 0, `artifacts/phoenix_probe_v1`)

The same stages, run on the design's original target: 200 train clips stratified
by the seven pilot signers (Signer03/Signer07 excluded by construction), frames
read from the release tar by index + single member-list extraction (the tarball
is uncompressed despite its `.gz` suffix). Detector gate **passed**: hand rate
0.756, two-hand 0.595, mouth 0.977, below-floor 0.0, median crop 49 px (~4.6x
magnified to 224) — MediaPipe is reliable at 210x260, so Experiment C's detector
risk is retired. Labeling 2,445/2,446 crops (same lone off-vocabulary answer,
`finger_spread=extended`), ~25 crops/min, mean 9.3 s/call.

| Field (PHOENIX crops) | distinct | top-1 share | uncertain rate |
|---|---:|---:|---:|
| handshape_class (all hands) | 11 of 12 | 0.920 | 0.002 |
| left side / right side | 10 / 10 | 0.917 / 0.922 | |
| mouth (all four content fields) | **2 each** | **0.982** | 0.018 |

Verdict **PASS** on every pre-declared threshold. The interpretation is double-edged:
the grouping mechanism is confirmed (top-1 0.92 vs the 0.024 free-text baseline,
zero singletons — contrastive positives will form), but low resolution collapses
the label space: the mouth fields became near-constants (98.2% one value, worse
than the free-text baseline's 34.4%), and handshape keeps only 10–11 usable
classes with one dominant. "Magnified but unreadable" was not the failure mode;
"readable but too coarse to carry lexical content" is the live risk — exactly the
one the design listed for a 12-way closed set.

## Interpretation limits

Labels are LLM-generated and unreviewed (`human_review_status=not_assessed`);
schema validity is not visual correctness. ASL handshapes are not DGS glosses, so
these numbers inform whether closed-vocabulary crop labeling groups at all — they
do not transfer a PHOENIX effect size. Stage 2 would still require its own
authorization, a new schema/prompt version, and E1–E3 reruns per the design doc.

## Full 7,096-clip run (launched 2026-10-03, `artifacts/phoenix_full_v1`)

`sample-phoenix --include-all-signers --total 7096` (all 9 signers, matching the
annotation census exactly), directory frame source, seed 0. Runs via
`artifacts/run_full.sh`: detect (35,480 frames) → detector gate → label
(~89k crops, ~2.5 days on 2 batched replicas) → report. Resumable; the driver
re-checks the gate before labeling and still reports if the label stage exits
non-zero (>max(5, jobs//200) failures).

Pre-launch code review (2026-10-03) fixed before this run:
- handedness: MediaPipe's mirrored-selfie labels are now flipped to the
  signer's anatomical side (`mirrored=False`); all earlier per-side numbers
  were inverted (aggregate statistics unaffected)
- `--include-all-signers` + a loud warning when `--total` exceeds the eligible
  population (Signer03/07 excluded by default, 5,746 eligible)
- label stage journals missing/corrupt crops instead of crashing the pool
- detect survives per-frame detector failures (records empty regions)
- label exit code + a 99% labeled-fraction floor in the verdict
- regions.json flushed every 10 clips; resume re-derives done clips from
  regions (stale-sample keys filtered); report skips `.partial-*` journals;
  `_jobs_from_regions` is one-pass
