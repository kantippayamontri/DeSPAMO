# Single-scale CLIP and isolated cached dev evaluation

## Status and intent

Design and implementation plan requested on 2026-10-11, subsequently approved
for native WSL/CPU implementation. User requested CLIP without S² and fixes to
dev evaluation. Implementation evidence is in
`docs/clip-plain-safe-dev-implementation-2026-10-11.json`; production preparation
and GPU launch remain separately gated.

Success means a CPU-tested plain-CLIP pipeline that can later run with separately
reviewed GPU authorization, preserves experiment provenance, and does not reuse
the known unsafe evaluation cleanup. It does not mean the Windows crash is fixed.

### Decisions requiring review with the plan

1. "CLIP without S²" means frozen, single-scale **224 raw final hidden CLS,
   width1024**, with the existing CLIP preprocessing. It does not mean CLIP's
   post-layernorm pooler output, projected image embedding or normalized embedding.
2. Derive a new cache from the first1024 columns of the verified existing cache.
   No image re-encoding is necessary: original `s2_cls` explicitly concatenates
   `[global_cls, mean_tile_cls]` in that order.
3. Start a distinct plain-CLIP experiment from fresh seed initialization. The
   width2048 S² adapter/AdamW state cannot be represented as an exact width1024
   resume. Do not slice its weights, duplicate/zero-pad features, or rename it a
   plain-CLIP checkpoint. Preserve its1774updates and497.5533489369991s charges.
4. Replace in-process full-dev generation in the new variant with sequential,
   fresh inference workers. Training worker exits before inference loads a model.

The earlier proposed S² same-root migration bridge is superseded for this plan.
Restoring that old experiment would be a separate task. Safe-evaluation work
continues as part of the plain-CLIP variant.

## Evidence and existing code

- `tools/clip_s2/encoder.py:88-103`: raw global224 CLS first, averaged448-tile CLS
  second. Both are1024 wide; frozen revision is pinned below.
- `tools/clip_s2/data.py`: width2048 and S² policy are admission requirements.
- `tools/benchmark/translation_model.py:37-55`: current plain MLP has2048 input.
- `tools/clip_s2_inference/{generation,evaluation}.py`: tested explicit caching,
  batch1, per-batch autocast, bounded-report rules and no CUDA error cleanup.
  Its loader and GPU gate are tied to old S² data/hold, so it is a reference,
  not a directly callable new-variant evaluator.
- `tools/benchmark/compact_checkpoint.py`: mutable weights/buffers, AdamW,
  sampler, global RNG and prompt RNG with frozen-base/identity verification.
- `tools/followup/recipe.py`: approved training schedule and early stopping.
- `tools/dinov3/{storage,official_runtime,official_archive}.py`: publication,
  owner identity and native/backing-disk reserve primitives.
- `docs/dev-generation-code-audit-2026-10-11.md`: confirmed secondary-error
  masking by CUDA RNG restore; cacheFalse repeats decoder prefix computation.
  No proven initial kernel defect, malformed spatial input or memory leak.
- `docs/matched-dino-clip-gpu-comparison-2026-10-11.md`: small cached cases pass,
  but full dev/training stability and audio-isolation repair remain unproven.

Paths above are relative to the retained runtime worktree, except `src/` and
`docs/`, which refer to the main project. Full paths are in the plan.

## Global constraints

- Implementation is WSL code plus CPU tests only; production feature conversion, GPU validation and training require separate execution approval.
- Keep v3 manual-stop and automatic-recovery block intact; no Windows device, driver, registry, reboot or package changes.
- Preserve old source, feature, model, checkpoint, budget and test bytes; use only new `tools/clip_plain/` runtime modules and new variant-owned output namespaces.
- Use existing main `.venv`, Python3.11, torch2.0.1+cu117, Transformers4.32.0 and PEFT0.7.1; no dependency/lock changes or environment mixing.
- Frozen CLIP: `openai/clip-vit-large-patch14`, revision `32bd64288804d66eefd0ccbe215aa642df71cc41`; global224 raw final hidden CLS, FP32, width1024, no tiles or normalization.
- Frozen FLAN-T5-XL revision `7d6315df2c2fb742f0f5b556879d730926ca9001`; Q/V LoRA16/32/dropout0.1; prompt limit1024; train-only three-language ICL.
- Official train7096/dev519/test642, original order and source-frame provenance; unchanged motion1024 features; no official test inference/scoring.
- Training batch8/accumulation1/drop-last,887updates/epoch,max200,min40,dev every2epochs,patience10checks,minDelta0.10BLEU4; VT4000/LRwarmup10000,legacy_mean VT.
- Adapter LR3e-4→5e-5; FLAN LR5e-5→1e-5; AdamW betas0.9/0.98,eps1e-8,weight_decay0.01,gradient_clip1; training cacheFalse and gradient checkpointing.
- Dev batch1,beam5,max_length64,do_sampleFalse,use_cacheTrue; fresh optimizer-free inference process, per-batch BF16 autocast and fsynced telemetry.
- Retain old1774updates and497.5533489369991s accounting; plain-CLIP updates start at0 and aggregate historical charges exactly once at study level.
- No automatic GPU retries, automatic hold release, commits, pushes, artifact deletion or expensive production runs during implementation.
- Native and Windows backing storage require projected remaining demand plus50GiB reserve; low space blocks preparation/launch without deleting or relocating old artifacts.

Seeds0/1 remain the supported reference study configuration; adding an executable
controller does not authorize either run. Diagnostic subsets must never become
selection reports, including a bounded request whose size equals the corpus.

## Feature package and data contract

Parent native package:
`/home/kan/datasets/despamo/official-phoenix14t-clip-s2-features-v1/spatial-f0ea5e0b10af4a796375bcfd935f69c9988caec9e7ddb6c27de0e0c35a098488`.

Parent manifest SHA256:
`1e5280e35fa273f7fbc28d36802496d3ac1d60a2b62a80c5674339cd0dd587e4`.
Parent identity SHA256:
`653caacbaddfc597acff49a2079479864071fbd43aa94dedfc969e4571745903`.

Unchanged motion package:
`/home/kan/datasets/despamo/official-phoenix14t-features-v1/motion-1123ff445677bf310ac0b8c9b45cac4f690b9102f5dad1960aa2fef04937e82d`;
manifest SHA256
`023a054fe0d3b9e04b5c0472ef77ebb8d031130e604841d67acfe7d81f66fd03`.
Annotations: `/home/kan/Research/SpaMo/preprocess/Phoenix14T`.
Model cache: `/home/kan/datasets/spamo/hf_cache`.

New package base, only on approved preparation:
`/home/kan/datasets/despamo/official-phoenix14t-clip-plain-features-v1`.
Package key: `spatial-<digest(identity)>`, policy
`official-frozen-clip-global224-spatial-v1`.

Identity includes parent root/manifest/identity hashes, producer-source closure,
model/processor binding, original feature layout, derivation `columns=[0,1024]`,
width1024 and dtypefloat32. Validate the pinned parent layout before slicing.
Each child receipt binds the parent array AND parent receipt hashes and retains
source rows, frame indices, clip/split IDs, frame count and child array SHA256.

Derived array is a contiguous independent copy of `parent[:, :1024]`, bitwise
equal to that half. Parent must be finite FP32 `[frames,2048]`; no silent casts.
Never publish a derived array under the parent filename/root. Reuse generic
`write_clip`, `finish_package` and `verify_package` with explicit width1024,
including current source-frame validation. Check parent bytes before/after each
conversion and all parent bindings before final manifest publication.

Full package requires8257 clips and947756 frames. Raw child payload is
3.6154022216796875GiB, plus NPY headers, copied receipts, manifest and staging
overhead. Calculate actual projected bytes before writing; keep50GiB reserve.
Existing C: observations near51GiB mean conversion may be storage-blocked.

Interrupted files are preserved and reported. Already completed child receipts
are verified for idempotent continuation; unreceipted/conflicting files stop the
converter. Publish complete metadata atomically only after exact inventory checks.
All raw data/cache preparation is explicit CPU CLI, never a side effect of tests.

Loader uses explicit plain spatial/motion/annotation roots. It rejects old S²,
DINO, partial packages, wrong widths, reordered/missing IDs, drifted arrays or
changed source rows before they reach the model. It shares existing `Phoenix14T`
and `collate_phoenix` behavior, including full-frame evaluation and no worker pool.

## Model, identity and accounting

Spatial MLP:1024→1024→768 with GELU; motion1024→768; original concat-reference
temporal K5/P2/K5/P2 stack; connector768→2048→2048. FLAN and VT objective retain
existing definitions. Build deterministically for seed0/1 on CPU with CPU-only
RNG forks. Unaffected initialized tensors need not match the S² model bitwise;
publish a plain-variant initialization fingerprint and do not claim a one-factor
S² ablation (representation width and projector parameter count also change).

New study identifier: `official-phoenix14t-translation-clip-plain-spamo200-v1`;
native root under `/home/kan/datasets/despamo`, matching B: archive suffix.
This is a user-requested representation change, not a reset of the failed budget.
`lineage.json` references old S² root/checkpoint/charges and the feature parent.

Historical study/host-hold root:
`/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v3`.
Historical run:
`translation-reference-seed0-fe5c22c66305d9e3bea3074792c0f64955b2c01c4b61bf92656448c0f7ae5671`.
Its `resume-epoch-2.pt` SHA256 is
`d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9`.
Capture current budget and manual-stop hashes at implementation entry for
preservation checks; do not overwrite either record with this design's values.

Run identity policy `official-clip-global224-safe-dev-spamo200-v1` binds new
source/config/data hashes, initialization, frozen base, recipe, runtime and
evaluation-policy hash. Old compact snapshots fail identity/shape admission.
Resume is supported only for checkpoints produced by this new plain variant.

Own `budget.json` records completed updates, monotonically increasing elapsed
charges, owner PID/boot/start ticks, stage and status. A controller-owned attempt
journal charges setup, training, dev and failed attempts once, including intervals
whose worker died. On an unclean host stop, explicitly retain last recorded time
and charge documented unreconciled wall time before manual recovery; never erase
elapsed cost. No time-based training stop is introduced.

Study cost summary uses `historical_charge_seconds` once, plus new conversion and
per-run charges;1774 historical updates remain a separate provenance field.

New-run retention follows the inherited latest2-resume/all-dev-candidates policy.
Only an explicitly approved runtime receipt with `allow_owned_resume_pruning=true`
permits pruning this new run's obsolete full resumes, after atomic/fsynced next
resume publication and verification. All model-only dev candidates remain
hash-verified on B:, actual-best weights remain native, and the two newest full
resumes remain native. Without that permission, preserve all files and project
storage accordingly. Never prune or rewrite any pre-existing study artifact.

Storage projection uses measured compact checkpoint sizes, remaining dev epochs,
both seeds, native staging/retention, archive candidates and50GiB reserves. The
old S² profile is not a size/headroom admission certificate for this new model.

## Isolated process lifecycle

The parent controller never constructs a model or initializes CUDA. Sequential
state machine for each seed:

```text
created -> train chunk -> awaiting_dev -> dev worker -> select/stop
                 ^                                      |
                 +--------------- continue -------------+
any child/identity/storage error -> held (no automatic retry)
```

Training chunk runs to the next declared dev epoch (normally two epochs), saves
full compact resume each epoch and final model-only candidate, publishes a durable
handoff and exits0. Parent waits for **actual process exit**, not only a handoff
file, then starts inference. Resume after epoch1 uses sampler state and continues
to epoch2; resume after even epoch with missing dev must run dev before update.

Inference worker loads only frozen base and mutable candidate weights. Compact
receipt uses the same run identity; no optimizer construction/state restoration.
It publishes full-dev report only on complete ordered coverage and exits. Next
train worker restores full AdamW/sampler/global+prompt RNG from training snapshot
after model/device initialization; evaluation has no access to that live RNG.

On any device error, worker emits best-effort CPU error evidence and exits
nonzero. No CUDA RNG restore, memory query, synchronize, empty_cache, tensor
transfer or save-after-fault. Boolean module-state/hook restoration is safe.
Logging failures attach context but never replace the primary exception.
Controller sets new-study hold, invalidates active scope, retains checkpoint/
charges and stops. A later explicit recovery must verify state/owners first.
Retain frozen-base and CE-collapse guards and the post-warmup identical-dev-output
guard as fail-stop checks. Their old `RecoverableFault` classification must not
enable a retry in this new controller.

## GPU authorization and ownership

Default CLI operation is CPU preflight. GPU commands require an explicit
`--approval <receipt>` issued only after a later user-reviewed scope; no code
path manufactures that approval. Receipt binds purpose, experiment/config/source
hashes, existing v3 hold hash, current boot, driver-review receipt, allowed stages,
seeds, batch limits and a unique attempt token. A current new-study manual-stop
always blocks launch until separately reviewed; expired/consumed tokens block.

An explicit receipt can authorize this **new variant only** while retaining v3's
hold. Its reference to the reviewed v3 hold is not permission to clear it or run
old runners. CUDA/NVML queries themselves happen only after scope validation.
Each controller attempt claims token atomically; child tickets are limited to
the pending stage and validated against exact parent owner/boot/start ticks.

Refuse concurrent GPU workers and reused same-boot PIDs. An old-boot owner is
historical, not live; state labels alone cannot establish ownership. Use nonblocking
locks and existing process-owner format. No watchdog/retry daemon in this design.

## Generation, telemetry and selection

New evaluator adapts the tested safe helper into `tools/clip_plain/`; do not
monkeypatch its hard-coded loader or old hold. Explicit cacheTrue is per call,
training config remains cacheFalse. Validate visual shapes, bool masks and
finite valid tokens; masked padding must not falsely reject an otherwise valid
sample. Use one BF16 autocast context per batch, with inference_mode.

Telemetry: attempt/source/evaluation hashes, UTC, PID/boot, checkpoint, clip,
spatial/motion/joint shapes and lengths, decode policy, batch timing and allocated/
reserved peaks. Flush/fsync `batch_start` and `generation_start` before expensive
work. Decoder hooks record step/input length before forward and finite raw final
logits after forward, with dtype/stride. Nonfinite raw logits abort; do not reject
intentional negative-infinity masks in processed beam scores. Hooks remove on
all exits. No vocabulary-sized tensors/logits or hidden states retained in logs.

Policy ID `clip-plain-cached-b1-beam5-v1` and its digest are part of reports and
checkpoint selection. CPU synthetic tests establish supported behavior, not
exact full-model BF16 equality across batch/cache settings. Store actual output
IDs/lengths as well as text, but do not claim cached and uncached outputs identical
without a corresponding comparison.

Only complete ordered519-clip dev reports with matching checkpoint/source/data/
policy bindings enter selection. Bounded probes have `metrics=null` and
`diagnostic_only=true`, even if they happen to cover519 clips. Actual maximum
BLEU4 selects weights; significant-improvement anchor only controls patience.
Earlier ties win. Keep candidates/receipts; selection is an atomic pointer and
verified native copy, never deletion of old studies' artifacts.

## Validation stages and acceptance

Implementation acceptance is CPU-only: synthetic file-backed feature packages,
tiny real T5/PEFT generation, injected primary/observer faults, strict snapshot
and report identities, actual subprocess ordering with CPU workers, interrupted
handoff/accounting recovery, and blocked GPU CLI before device calls.

After review, production CPU preparation is a separate explicit run, subject to
capacity/provenance checks. Future GPU rollout is separately authorized: fixed
short/long-clip probe, complete dev, then bounded training-to-dev handoff/profile
before any long training. Width1024 plain model needs its own profile; old S²
profile/checkpoint and audio-isolation results cannot authorize it.

A later batch2/4 cached experiment is a policy change outside this plan, not an
automatic fallback. Original batch4/cacheFalse reproduction, Windows changes,
test scoring, new encoders/losses and S² checkpoint surgery are outside scope.
