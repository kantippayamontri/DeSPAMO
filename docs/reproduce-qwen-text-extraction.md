# Reproducing Qwen Text Descriptions: PHOENIX14T and How2Sign

This runbook documents the code on branch **`text_des`**, introduced in commit
`1caecea` (`feat(appearance): add resumable Qwen text extraction`). Commands below
run from the repository root on Linux. They extract **text descriptions only**;
they do not extract visual features, compute text embeddings, or train a model.

## 1. Method and scope

| Description family | Input | Output frequency |
|---|---|---|
| Anatomical left hand, right hand, mouth | MediaPipe region crops resized/letterboxed to 224 x 224 | Three possible targets per sampled frame |
| Biometric, clothing, hair, background | Five ordered whole frames, aspect-preserving resize with longest edge at most 640 pixels | Four clip-level targets from one Qwen request |

Five source frames are sampled at normalized positions **0.1, 0.3, 0.5, 0.7, 0.9**.
The zero-based index is `int((frame_count - 1) * position + 0.5)`; at least five
distinct source frames are required. Keep the same indices when later extracting
spatial features. PHOENIX filenames are resolved through sorted `imagesNNNN.png`
files: index zero is normally `images0001.png`, not `images0000.png`.

The crop method uses closed hand/mouth vocabularies in
`tools/how2sign/how2sign_probe/labels.py` and `schemas.py`. The whole-frame stable
method reuses `src/despamo/appearance/schema.py:Stable` and the original canonical
renderer with a separately versioned stable-only prompt in `stable.py`.

Current full-run scope:

- **PHOENIX14T:** all **7,096 training clips**, including Signer03 and Signer07.
  The prior 200/1,000-clip probes excluded those signers; full extraction must use
  `--include-all-signers`. Dev/test (519/642 clips) are not included here.
- **How2Sign:** all **31,165 realigned training clips**. Validation/test
  (1,741/2,357 clips) are not included here. The CSV provides video strata;
  these are not verified signer IDs.

Descriptions remain LLM-generated and unreviewed:
`supervision_policy=qwen-schema98-unreviewed-v1`,
`human_review_status=not_assessed`. Schema validity and completion do not prove
visual correctness. No human audit pass is fabricated.

## 2. Environment and data paths

In a checkout of `text_des`, set paths for your machine. The defaults below match
the extraction host; change them for HPC or another host.

```bash
set -euo pipefail
export REPO="$(pwd)"
export ARTIFACTS="$REPO/artifacts/reproduction_$(date -u +%Y%m%dT%H%M%SZ)"
export PHOENIX_ROOT="/home/jovyan/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"
export PHOENIX_FRAMES="$PHOENIX_ROOT/features/fullFrame-210x260px"
export PHOENIX_ANN="/home/jovyan/SLT/SpaMo/preprocess/Phoenix14T/train_info_ml.npy"
export HOW2SIGN_ROOT="/home/jovyan/datasets/how2sign/realign_sentence_level"
export HOW2SIGN_VIDEO_ROOT="$HOW2SIGN_ROOT/train_sentence_videos"
export PYTHONPATH="$REPO/src:$REPO/tools/how2sign${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2

uv sync --locked --project tools/how2sign
export PY="$REPO/tools/how2sign/.venv/bin/python"
mkdir -p "$ARTIFACTS"
```

The tool has its own Python 3.11 environment and committed `uv.lock`:
MediaPipe 0.10.21, OpenCV, NumPy, pydantic and pytest. The baseline and DINOv3
environments stay separate. No DINOv3 weights or Hugging Face token are needed
for this Qwen description extraction.

Expected inputs:

```text
PHOENIX_FRAMES/train/{clip_id}/images0001.png ... imagesNNNN.png
PHOENIX_ANN                            # SpaMo train_info_ml.npy
HOW2SIGN_ROOT/how2sign_realigned_train.csv
HOW2SIGN_VIDEO_ROOT/{SENTENCE_NAME}.mp4
```

The PHOENIX annotation file supplies `fileid`, `signer`, and `num_frames`.
Directory clip counts and frame counts are checked against it. How2Sign's
realigned CSV is tab-separated despite its `.csv` extension.

## 3. Ollama and GPU serving

The measured model identity on this host is:

```text
qwen3-vl:8b
digest: 901cae73216286ea8c5aba8b46d307ff7188f737285ec500c795a12f05225d28
```

On a new host, start the default Ollama service at `127.0.0.1:11434`, then install
the model and record the actual version/digest. Pulling a tag later may resolve
different weights; a different digest is a different extraction version.

```bash
ollama pull qwen3-vl:8b
ollama --version
ollama show qwen3-vl:8b --modelfile
ollama list
```

The sample/provenance stage looks at the default service on 11434. GPU labeling
uses dedicated replicas on 11435/11436. If those ports are already serving the
active extraction, reuse them rather than launch competing servers.

For a fresh run, launch these two commands in separate terminals and leave them
running. Both services must see the same Ollama model store.

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=30m ollama serve
```

```bash
OLLAMA_HOST=127.0.0.1:11436 OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=30m ollama serve
```

Requests pin `num_gpu=99`, temperature zero and seed zero. Crop requests use a
4,096-token context; the five-image stable requests use 8,192. On the measured
Ollama/model combination, the raw chat request with assistant content `"\n\n"`
and **no `think` or `format` field** bypasses lengthy thinking. This behavior is
model/server-specific: verify the small smoke after changing either one.

MediaPipe detection and video decoding are CPU operations. Qwen generation uses
the GPU. Two services share one A30; they are not two independent GPUs.

## 4. Offline verification

```bash
"$PY" -m pytest tools/how2sign/tests -q
uv run --frozen ruff check tools/how2sign
bash -n tools/how2sign/scripts/run_mouth_repair_queue.sh
bash -n tools/how2sign/scripts/run_how2sign_stable_queue.sh
uv lock --check --project tools/how2sign
```

At `1caecea`, **107 tests pass**. Tests cover frame alignment, anatomical
handedness, corrected mouth coordinates, strict label validation, resumable
journals, completion gates, stable/crop joining and 19-slot exports.

## 5. PHOENIX14T: directory frames to hand/mouth descriptions

For a fresh reproduction, choose a new artifact base so existing outputs are not
overwritten. The directory source avoids reading/extracting the large tar.

```bash
export P="$ARTIFACTS/phoenix_full_mouth_v2"

"$PY" -m how2sign_probe.cli --root "$P" sample-phoenix \
  --annotations "$PHOENIX_ANN" --frames-root "$PHOENIX_FRAMES" \
  --split train --total 7096 --seed 0 --include-all-signers

"$PY" -u -m how2sign_probe.cli --root "$P" detect --workers 1 \
  > "$P/detect.log" 2>&1
```

Detection exits nonzero if its gate fails. Check `detector_report.json` before
labeling. The current gate requires at least 100 frames and:
hand rate >= 0.65, two-hand rate >= 0.50, mouth rate >= 0.70, and
`below_floor_share` <= 0.20. These are detection-coverage checks, not visual audits.
Use `--workers 1`: real MediaPipe threading stalled in a prior full run despite
mocked parallel tests passing.

```bash
"$PY" -u -m how2sign_probe.cli --root "$P" label \
  --ports 11435 11436 --workers-per-port 2 > "$P/label.log" 2>&1
"$PY" -m how2sign_probe.cli --root "$P" report > "$P/report_stdout.txt"
"$PY" -m how2sign_probe.completion --root "$P" > "$P/completion_verified.json"
# Create this only after ALL commands above succeed:
touch "$P/driver_complete"
```

When wrapping these commands in a script, use `set -euo pipefail`. Do not create
a completion marker after an interrupted or unsuccessful stage.

Export the text-only handoff; use a new output filename if one already exists:

```bash
"$PY" -m how2sign_probe.export_targets --root "$P" \
  --output "$P/phoenix14t_qwen_text_targets_v2.json"
```

This export contains four masked stable slots because this PHOENIX run extracted
only hand/mouth labels. Earlier whole-frame PHOENIX stable descriptions are a
separate historical dataset and are not automatically imported.

## 6. How2Sign: crop descriptions with source-frame receipts

The whole-frame pass and combined export require `repair_clips/{clip}.json`
receipts containing decoded source-pixel hashes. To reproduce that layout from
scratch, first create a staging detection root, then publish the corrected root
with the repair module. The publishing pass also creates receipts; it redetects
mouths even when the initial detector already contains the coordinate fix.

```bash
export H_STAGE="$ARTIFACTS/how2sign_full_v1"
export H="$ARTIFACTS/how2sign_full_mouth_v2"
mkdir -p "$H"

"$PY" -m how2sign_probe.cli --root "$H_STAGE" sample \
  --csv "$HOW2SIGN_ROOT/how2sign_realigned_train.csv" --total 31165 --seed 0
"$PY" -u -m how2sign_probe.cli --root "$H_STAGE" detect \
  --video-root "$HOW2SIGN_VIDEO_ROOT" --workers 1 > "$H_STAGE/detect.log" 2>&1

"$PY" -u -m how2sign_probe.repair --source-root "$H_STAGE" --output-root "$H" \
  --video-root "$HOW2SIGN_VIDEO_ROOT" > "$H/repair.log" 2>&1
```

The repair module records gate results but does not exit solely on a failed gate. Check the
published report explicitly before the expensive label stage:

```bash
"$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["gate"]["pass"] else 2)' \
  "$H/detector_report.json"
"$PY" -u -m how2sign_probe.cli --root "$H" label \
  --ports 11435 11436 --workers-per-port 2 > "$H/label.log" 2>&1
"$PY" -m how2sign_probe.cli --root "$H" report > "$H/report_stdout.txt"
"$PY" -m how2sign_probe.completion --root "$H" > "$H/completion_verified.json"
touch "$H/driver_complete"
```

Unreadable videos are recorded rather than silently replaced. They remain in
the final combined clip census with explicit masked targets. Sampling does not
drop translation clips because appearance detection/labeling failed.

## 7. How2Sign: whole-frame four-factor descriptions and combined export

Run after corrected crop extraction completes. All five whole frames are sent
in a single request for each clip; this is **31,165 requests**, not four separate
requests per clip. Source indices and decoded pixel hashes must match the crop
repair receipts.

```bash
export S="$ARTIFACTS/how2sign_stable_wholeframe_v1"

# Smoke: four clips only, no full completion marker. Valid records are reusable.
"$PY" -u -m how2sign_probe.stable --crop-root "$H" --root "$S" \
  --video-root "$HOW2SIGN_VIDEO_ROOT" --ports 11435 --limit 4

# Full pass: rerun without --limit, with the SAME bound source and version.
"$PY" -u -m how2sign_probe.stable --crop-root "$H" --root "$S" \
  --video-root "$HOW2SIGN_VIDEO_ROOT" --ports 11435 11436 --workers-per-port 1 \
  > "$S/label.log" 2>&1

"$PY" -m how2sign_probe.combined_export --crop-root "$H" --stable-root "$S" \
  --output "$S/how2sign_qwen_19_targets_v1.json" > "$S/export_report.json"
```

The stable-only prompt must be used with the same model digest and code version
for smoke and full passes. The host's smoke used a temporary replica on 11437;
the full queue uses 11435/11436. `version.json` binds the model, prompt, schema,
source sample, source schema code and stable module code. If they change, use a
new output root. Stable records have up to three reserved attempts and terminal
`valid`/`failed` states; full completion requires no unfinished clips and at
least 98% schema-valid clips. Failed records mask all four stable targets.

To queue this automatically, after preparing `H_STAGE`/`H` as above:

```bash
export DESPAMO_ROOT="$REPO"
export DESPAMO_ARTIFACT_ROOT="$ARTIFACTS"
bash tools/how2sign/scripts/run_how2sign_stable_queue.sh
```

The driver waits for `H/driver_complete`, revalidates every expected crop job,
runs the stable pass, exports the combined JSON/checksum, and only then creates
`S/handoff_complete`. Run one driver per output root. The repair queue driver
`scripts/run_mouth_repair_queue.sh` operates on both existing `*_full_v1` roots;
it is not a replacement for their initial sample/detection stages.

### Running only the four stable factors on another server

The local crop repair is already complete, so the stable-factor pass can start
on a different GPU server **while crop labeling continues here**. Use the direct
`how2sign_probe.stable` command below rather than the queue driver, which waits
for crop-label completion.

Transfer these alignment inputs from the current host into one directory on
the new server, keeping the exact file bytes and directory names:

```text
/home/jovyan/SLT/DeSPAMO/artifacts/how2sign_full_mouth_v2/sample.json
/home/jovyan/SLT/DeSPAMO/artifacts/how2sign_full_mouth_v2/repair_clips/
```

They provide the 31,165-clip census and the five frame indices/source-pixel hashes
per readable video. Crop PNGs and crop-label journals are not required for this
standalone four-factor pass. Matching realigned training MP4s are still required
on the new server. Do not run from a code-only clone without the alignment inputs.

On the new server, clone the pushed branch, follow sections 2–4 for environment
setup and Ollama serving, then substitute your paths:

```bash
git clone --branch text_des git@github.com:kantippayamontri/DeSPAMO.git
cd DeSPAMO
export REPO="$(pwd)"
uv sync --locked --project tools/how2sign
export PYTHONPATH="$REPO/src:$REPO/tools/how2sign"
export PY="$REPO/tools/how2sign/.venv/bin/python"
export ALIGNMENT="/path/to/transferred/how2sign_alignment"
export VIDEOS="/path/to/how2sign/realign_sentence_level/train_sentence_videos"
export S="/path/to/output/how2sign_stable_wholeframe_v1"

# Smoke first; both replicas must have the same model digest for the full pass.
"$PY" -u -m how2sign_probe.stable --crop-root "$ALIGNMENT" --root "$S" \
  --video-root "$VIDEOS" --ports 11435 --limit 4

# Same version/root: reuse successful smoke records, then process all clips.
"$PY" -u -m how2sign_probe.stable --crop-root "$ALIGNMENT" --root "$S" \
  --video-root "$VIDEOS" --ports 11435 11436 --workers-per-port 1
```

After full stable completion, copy `S` back, including `sample.json`, `version.json`,
`provenance.json`, `records/` and `completion_verified.json`. Once crop labeling
also completes, run `how2sign_probe.combined_export` with the original corrected
crop root and the returned stable root (section 7). The exporter checks clip and
frame/pixel alignment. Run only one stable-factor writer per output root; use
either the local queued pass or the other-server pass for the final handoff.

## 8. Recovering the historical mouth bug without losing hands

Historical v1 mouth crops were wrong: face-local coordinates were divided by
the face-crop width/height, then interpreted at full-frame scale. Correct mapping
is `(local_x * face_width + x_offset) / source_width` (and equivalently for y).
The corrected boxes were checked by regression tests and image inspection.
Anatomical handedness also uses `mirrored=False` for these unmirrored datasets.

To recover an old run, use the same repair command in section 6 with a **new**
output root. For PHOENIX directory sources omit `--video-root`. The repair pass:

1. Visits all five sampled frames in every clip, including old missed mouths.
2. Preserves hand crops and copies only verified completed hand journals.
3. Never imports an old mouth journal, even when its key happens to match.
4. Writes per-clip receipts tracking old mouth status, corrected boxes and hashes.
5. Publishes `regions.json`, `repair_manifest.json` and the detector report.

Original artifacts remain intact. The previous low mouth-label diversity and PASS
verdicts based on displaced crops are not reliable evidence about label quality.

## 9. Resume, completion and monitoring

- **Detection:** rerun `detect` with the same sample/root/source. Regions checkpoint
  every ten clips; completed regions are authoritative for resume.
- **Repair:** rerun `repair` with the same source/new roots. Per-clip receipts
  avoid repeating completed work; source sample and regions hashes are bound.
- **Crop labeling:** rerun `label` with the same root/model. Completed journals
  are skipped; unfinished work resumes and failed jobs are retried on a new pass.
- **Stable labeling:** rerun `stable` with the same bound version. Terminal
  valid/failed records are retained; pending attempts continue.
- **Exports:** refuse to overwrite an existing output file. Use a new filename.

`completion` verifies expected job identities, schemas and terminal statuses,
requires at least 99% done, and limits crop failures to `max(5, jobs // 200)`.
It is an execution/schema check, independent of the statistical report verdict.
`report` alone does not establish complete extraction. A partial smoke/export
must never be marked as a completed full dataset.

For background operation, redirect the full log and use a detached process.
Do not pipe the running job into `tail`. Record exact process IDs; avoid broad
`pkill` patterns that can stop unrelated extraction or queue processes.

Monitor `label.log`, `repair.log`, stable `label.log`, verified completion files,
and `nvidia-smi`. Completion percentages refer to **crop jobs** for articulators
and **clips** for stable factors. A resumed crop counter includes reused labels,
which inflates its cumulative speed/ETA; estimate fresh-work speed separately.

Measured host: NVIDIA A30, 24 GiB. Crop labeling sustained approximately 24–25
jobs/min; corrected PHOENIX had 86,737 selected crop jobs and How2Sign 421,865.
The four-clip, five-image stable smoke took 139.9 seconds alongside active crop
labeling. Its ~34-second request latency is not a measured full-pass throughput.
Timings vary with hardware, server build, source decoding, retries and NFS I/O.

## 10. Text-only HPC handoff

Each JSON has `metadata` and `records`. There is one record per clip and exactly
19 targets in this order:

```text
biometric, clothing, hair, background,
frame1_left_hand, frame1_right_hand, frame1_mouth,
...,
frame5_left_hand, frame5_right_hand, frame5_mouth
```

`valid_mask[i]` is true exactly when `targets[i].text` is non-null. Encode only
non-null text strings; retain zero vectors in masked slots. Stable targets have
`frame_index=-1`; articulators retain their source frame indices. Null/uncertain
or failed targets are masked, not encoded as empty strings.

The PHOENIX handoff has four masked stable slots. The combined How2Sign handoff
fills those slots from whole-frame records when usable. It validates identical
clip censuses, frame indices and decoded pixel hashes between both passes.
How2Sign includes relative video paths, and unreadable clips retain 19 masked
slots and explicit source errors. Canonical text is derived from validated Qwen
fields, not a newly generated narrative caption.

Copy the final JSON and optionally its `.sha256` file to HPC. Neither crops nor
individual journals are needed just to encode text. Source videos/frames are
still needed for later visual spatial-feature extraction.

These exports match the 19-slot target ordering, but they are **not** the original
schema-2 frozen appearance dataset: `text_features.encode_dataset()` cannot load
them unchanged. Feed non-null strings to your chosen text encoder, preserving
the supplied masks and source alignment. No new translation training, frozen
dataset replacement or E1/E2/E3 protocol comparison is performed by this runbook.
