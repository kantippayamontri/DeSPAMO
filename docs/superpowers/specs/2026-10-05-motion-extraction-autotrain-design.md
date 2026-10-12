# Documented motion extraction → automatic fusion training

## Approval and intended outcome

User approved creating new motion features after the measured4–5hour estimate
and requested automatic training after extraction. Interpret this as a separate
**18000second/5hour extraction ceiling**, retaining the already approved
**64800second/18hour fusion-study ceiling**. Maximum combined charged time82800
seconds/23hours; neither stage borrows unused allowance from the other.
This interpretation is surfaced for review rather than silently raising budgets.

User approved this written design, including separate5hour/18hour caps and
automatic validated handoff. Implementation-plan review is the remaining gate
before implementation; execution method/current-checkout choices are preserved.

This is an architectural extension of the approved fusion study: a new documented
motion cache, strict admission into matched fresh training, and an automatic
stage coordinator. Implementation-plan review remains before implementation.
Preserve selected execution method (subagent-driven) and
user-approved current checkout; no repeated workspace/execution-method choice.

Success: complete all7096motion clips with contemporaneous frame-window receipts,
validate them, pass real-data fusion admission smokes, then start the six matched
training/dev runs automatically without another launch prompt. If any stage fails
or reaches its cap, stop incomplete and preserve recovery state; do not launch the
next stage. Approval does not authorize deletion, commits, pushes or Signer07
translation scoring.

## New records, not recovered historical records

Downloaded VideoMAE features lack original generating-window records locally.
New extraction creates a new cache with **its own original generating records**;
never describe it as recovery of the old downloaded cache's historical records.
Keep old motion arrays/manifests, spatial cache, source frames and captions intact.

Extract from exact recorded210x260 source images used by cosine DINO cache
`7122b42e35510611626c2a71d0731b79dd79b20de6e89bd49f62737c5c84a18e`.
Shared real frame ordinals become known by construction. Use all7096physical-train
clips/827354source frames, not only train/dev; this requires93232motion windows.
Frozen logical groups remain5746train/582Signer03dev/768Signer07test. Test feature
extraction is frozen-encoder inference only; no test translations or model selection.

All three fusion arms use the new motion cache and fresh within-study references.
Do not attribute differences from older downloaded-motion runs purely to fusion:
motion source pixels/preprocessing/data lineage have also changed.

## Frozen extraction recipe

- Model `MCG-NJU/videomae-large`, cached revision
  `12da269a02d3e1fbbb7011e610e591fea8061dca`; hash actual cached config/weights.
- Inference only, eval mode, FP32; no encoder training or model download.
- Use last encoder hidden-state tuple entry's first token:
  `outputs.hidden_states[-1][:,0]`, width1024. This is the local extractor's
  selection rule; do not substitute normalized `last_hidden_state` or pooling.
- RGB decode; resize shortest edge224 with bilinear resampling, centre crop224×224,
  rescale1/255, normalize mean/std[0.5,0.5,0.5]. These are explicitly chosen new
  preprocessing settings from the benchmark, not proven historical settings.
  Bind resolved processor configuration and installed preprocessing sources.
- 16consecutive source frames/window, stride8, ascending full windows; drop
  incomplete tails. For clips shorter than16, repeat last frame only for encoder
  input and mark synthetic repeats; real spatial support excludes them. Current
  7096clips all have at least16frames.
- Window batches up to16, two CPU threads, no BF16 or TF32 change relative to
  declared FP32 path. Record relevant CUDA/backend flags; optimized NumPy stacking
  followed by torch.from_numpy must match ordinary processor tensors on tested
  batch sizes1/8/16 and representative real shapes.
- Save `[window_count,1024]` finite float32 arrays. Keep motion feature payload
  approximately0.356GiB; provision2GiB for payload and records, verify actual size.

## Environment and implementation boundaries

Use a dedicated pinned motion-extraction environment with existing-compatible
torch2.0.1+cu117/transformers4.32.0/Pillow11.3.0 and explicit resolved dependency
identities. Main environment lacks Pillow; the speed benchmark's process-local
path bridge is not a production dependency mechanism. Preserve main and existing
extractor environments; reuse local package/model caches where available. Bind
the new environment lock and verify numerical preprocessing/model interfaces.
Environment/code setup is excluded from charged extraction runtime; model loading,
real extraction profiling, decoding/inference/writes and validation are included.

Proposed focused files: `tools/motion/` environment/lock, extraction/provenance and
storage helpers; `scripts/extract_motion_features.py`; new-motion admission helper;
`scripts/run_motion_fusion_pipeline.py`; scoped opt-in integration in the existing
new fusion runner/controller and relevant tests. Historical E1/E2/E3/projector/
connector runners and their policies remain unchanged.

## Extraction records and publication

New extraction root external to Git under
`/home/kan/datasets/despamo/signer-pilot/motion-features-<identity_hash>/`.
Feature identity binds source spatial/frame/annotation hashes, actual source image
bytes, encoder config/weights/revision, processor recipe, environment/code/hardware
and window/token rules. Exclude elapsed time and cap from feature content key;
separate budget/recovery records bind that key, authorized runtime and stage policy.

For each clip record ordered source paths/indices/hashes, actual model-input frame
indices for every row including repeats, real support/window centres, motion
array shape/dtype/hash and new generating-run identity. Decode/hash the same bytes
consumed by model so receipts cannot claim different source files. Compare source
identities against frozen spatial receipts; any mismatch stops extraction.

Write each array and receipt through owned regular paths using temporary output,
fsync and atomic publication. Interrupted clips are recomputed; completed clips
are reused only after byte/source/recipe/receipt verification. Do not delete any
pre-existing file or accept arbitrary outputs merely because filenames exist.
Durable charge accumulates across resume and does not reset after reboot.

Only publish complete FeatureManifest, source/window ledger, original-generating
provenance and timing artifact after validating all7096clips/93232rows, finite
float32,width1024, exact source joins and deterministic chronology. Completion
marker is atomic and binds every manifest/receipt/feature hash. Missing/failed
clips prevent completion and prevent automatic training; no clip dropping.

The timing artifact may reuse consistent Task1 schema with `original_receipt`
for the **new generating run**, clearly identifying origin as new extraction.
Evidence must be actual extractor receipts, not a synthetic attestation copied
from tests. Validate deterministic receipt facts in addition to evidence hashes.

## Opt-in fusion data integration

Existing frozen loaders and baseline configuration bind the old motion manifest;
simply swapping paths would fail or falsify provenance. Introduce explicit opt-in
new-motion package admission under distinct run/controller/profile identities,
for example `signer-pilot-time-aligned-fusion-new-motion-v1`.

First validate frozen protocol/source/splits/spatial cache and retained selected
baseline with their original motion lineage. Independently validate the complete
new motion package against exact same source frame ordinals. Construct new Phoenix
views with original annotations/spatial features and new motion root/manifest;
update only the new run's configuration/provenance. Bind original baseline
lineage and actual new training inputs separately; never pretend old manifests
identify new arrays or globally monkeypatch prior policy constants.

Common initialization and batch/RNG/optimizer recipe stay matched across all six
arms. Existing historical-motion mode retains its validation behavior. New-mode
profile source closure includes extractor/admission/pipeline sources and locks;
no reuse of old initialization/run identities. New study outputs use a distinct
external controller directory `time-aligned-fusion-new-motion-study-v1`.

## Automatic coordinator and budgets

Persist pipeline identity, owned process/boot/PID metadata, stage receipts and
separate spent clocks. Sequentially:

1. Validate inputs/environment/disk/GPU ownership and extraction identity.
2. Run bounded real extraction smoke/profile and full extraction, within18000s.
3. Validate complete package and new timing artifact; record exact byte bindings.
4. Launch new-motion fusion controller automatically with separate64800s budget.
5. Existing both-seed initialization and ordinary/long real-data joint CE+VT
   smokes run under training accounting; only passing smokes admit production fit.
6. Train seed0reference/fixed/learned then seed1same,16000updates each; select only
   full582clip dev at5500/10400/16000. Validate final six-run summary bytes.

Reuse neutral watchdog/owned atomic-storage primitives, not shell success alone.
Training child must validate extraction completion content before any GPU model
construction and again before launching later arms. If extraction cannot finish
within5hours or training within18hours, stop incomplete. No automatic retry
resets or cap increases; recovery resumes original identities/remaining clocks.

Keep existing training32400s/seed and10800s/run bounds, batch4, VTwarmup2000,
LRwarmup1600, BF16/clip1, adapter3e-4/FLAN5e-5, AdamW0.9/0.98 eps1e-8 decay.01,
cosinezero, beam5/max64/noICL; original spatialMLP/TCN/output/VTAlign recipe.
Retain latest2owned resumes and all new dev models; no old checkpoint cleanup.

Check Linux and Windows-host projected extraction plus remaining-training demand,
including environment/atomic-write allowance and50GiB reserve, before starting
and before each GPU stage. RTX4090 must be idle/owned; no overlapping processes.
Publication/queue handoff holds appropriate locks and checks live source/hardware
identities. Automatic chaining must survive disconnected client through owned
durable external process/watchdog state, not an agent promising to return later.

## Verification and reporting

TDD covers frame/window ledger, input tensor parity, feature row order and support,
clip/source tampering, finite outputs, atomic interruption/resume, capped elapsed
accounting, unsafe path/lock rejection, complete marker binding, old-mode parity,
new-mode view/config/identity joins, extraction failure/cap preventing training,
automatic successful handoff, duplicate launch prevention and final artifact drift.
Independent subagent task reviews and whole-change review precede real execution.

Report new-cache hashes, source/encoder/processor versions, exact counts/coverage,
actual extraction time/storage, training smoke peaks and full paired dev results.
Preserve `qwen-schema98-unreviewed-v1`/`human_review_status=not_assessed` where
existing provenance requires them. No new descriptions or human appearance gate.
No claim of recovered historical records or signer invariance. Publish honest
running/incomplete/complete status and both stage budgets; report extra implementation
wall time separately from5+18hours charged runtime.
