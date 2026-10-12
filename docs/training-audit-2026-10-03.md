# DIFFER / SpaMo training audit and performance recommendations

Date: 2026-10-03. Evidence: current source, pinned dependency configurations,
checkpoint-bound historical code, frozen run receipts, local SpaMo source, and
the DIFFER and SpaMo papers. This is an audit and proposed experiment order,
not an approved architecture change or authorization for additional training.

## 1. Measured translation result

The completed cosine/lower-LR E3 recipe is the best measured translation run.
Checkpoint selection used Signer03 dev only; its selected step-8000 checkpoint
then produced one completed Signer07 evaluation on the ordered 768 clips.

| Model | Selected downstream step | Signer03 dev BLEU-4 | Signer07 BLEU-4 | Signer07 ROUGE-L F1 |
|---|---:|---:|---:|---:|
| E1, frozen DINO | 5600 | 4.0381 | 3.423840 | 0.157542 |
| E3, relational 0.1, constant-LR adaptation | 5600 | 4.444680 | 3.855402 | 0.166160 |
| E3, relational 0.1, cosine/lower-LR adaptation | 8000 | **4.709851** | **4.361798** | **0.184918** |

New E3 versus constant-LR E3: +0.506396 BLEU-4; versus E1: +0.937958.
The ordered clip IDs and references match across all three held-out reports.
These are single-seed point estimates, not a statistical-significance or
signer-invariance claim. The official PHOENIX test split and released SpaMo
checkpoint are different experiments and are not directly comparable.

New E3 dev trajectory: 3500 -> 4.036741; 5600 -> 4.464992; 8000 -> 4.709851.
The last sampled checkpoint was best, which motivates testing more downstream
training rather than declaring convergence.

### Bound artifacts

All paths below are relative to `/home/kan/datasets/despamo/signer-pilot/`:

- Adaptation: `e3-adapt-71a3c8c0805813a73c445ca2141b5e8b702933a809c641f483cb5d7e61c3ad8a`.
- Adaptation checkpoint SHA-256: `5c4d2f62491998512e596ac2cb5f59efc4fab4f3de1054deaa5e04a158f87e30`.
- Features: `e3-features-7122b42e35510611626c2a71d0731b79dd79b20de6e89bd49f62737c5c84a18e`.
- Downstream run: `e3-only-0402717f5ef91cca106caa881b689ce1eba75a3a87976302c99c654bea189690`.
- Selected translation checkpoint SHA-256: `56745b433fd37ddaaf73b768cbd4da5bc75d76f72bfde849ca948d8cabb420a7`.
- Held-out report: `signer07-e3-cosine-8975-8000-v1.json`.
- Comparators: `signer07-e1-8000-heldout-v1.json` and
  `signer07-e3-w0.1-8000-heldout-v1.json`.

Adaptation spent 5.2107 GPU-hours, extraction approximately 9.37, and downstream training
0.9263. Held-out evaluation is not recorded in the training budget JSONs;
summing those JSONs alone is not complete experiment accounting. The preflight
downstream run key differs from the actual run key because the GPU cap is part
of the identity; the actual completed key above is authoritative.

All appearance-supervised results retain `qwen-schema98-unreviewed-v1` and
`human_review_status=not_assessed`.

## 2. Corrections to earlier conclusions

1. The best previous E3 translation scores came from the **8975-step**
   constant-LR adaptation, not the 718-step sweep that passed the small probe.
   Feature receipt `e3-features-c344a07c.../budget.json` binds it to
   `e3-adapt-108e440f...`.
2. The new run is **not a cosine-only ablation**. The historical adaptation
   code matching SHA-256 `4624c9eb...` was found at commit `6d3ecc1eaa`.
   It used AdamW groups with LoRA LR `1e-4` and head LR `3e-4`, and the pinned
   optimizer's default weight decay `0.01`.
3. A failed handshape probe does not establish failed translation performance.
   Both 25-epoch E3 recipes failed that probe and both improved translation
   relative to E1. The new recipe improves translation further.
4. A feature-spread ratio close to one is not a measurement of unchanged
   features. It measures one dispersion statistic, not feature displacement
   or preservation of every semantic attribute.

| Adaptation setting | Previous constant-LR E3 | New cosine/lower-LR E3 |
|---|---:|---:|
| Steps / batch | 8975 / 16 | 8975 / 16 |
| Head calibration | 1795 | 1795 |
| LoRA LR | 1e-4 | 1e-5 peak |
| Head LR | 3e-4 | 3e-5 peak |
| Weight decay | 0.01 | 0.05 |
| LR warmup | none | 1795, from zero |
| LR decay | none | cosine to zero |
| Relational weight | 0.1 | 0.1 |

The combined recipe is supported by measured results. Attribution of its gain
to any individual setting requires a matched control.

## 3. Training-concept audit

### DIFFER-inspired adaptation

`tools/dinov3/e3_model.py:32-47` attaches rank-8/alpha-8 Q/V LoRA to DINOv3
ViT-L blocks 20-23. The base encoder is frozen. At two resolutions, each of
five sampled frames contributes a CLS vector; concatenation yields
`[batch, 5, 2048]`. Targets are frozen text vectors of shape
`[batch, 19, 768]`, with boolean masks for invalid descriptions.

The four nuisance factors use gradient reversal on clip-mean features.
Left hand, right hand, and mouth use positive frame-level contrastive losses.
Identical description strings are multi-positive examples; distinct strings
are treated as different targets. The temperature is 0.07. Per-vector teacher
cosine matching and relational pattern/spread matching discourage collapse.
`e3_data.py:12-97` checks train-only signers, exact sampled frame indices,
feature/image/target hashes, and zero-valued masked targets.

This is DIFFER-inspired, not a literal reproduction. DIFFER's paper, page 5,
uses EVA02-CLIP-L, SGD, peak LR 2e-6, initial warmup LR 8.42e-7, weight decay
0.05, 60 epochs, and identity classification plus triplet loss. It preserves
biometric identity while removing clothes-related nuisances. DeSpaMo instead
treats biometric identity as nuisance, preserves articulators, uses AdamW and
DINO LoRA, and replaces identity supervision with teacher geometry.
The paper does not specify a 20% warmup duration. Our duration is a pilot
convention, not a paper-derived requirement. The code's unsupported "42x"
update-pressure claim (`e3_train.py:39-41`) should not be treated as evidence.

Calibration and LR warmup are separate mechanisms. During calibration,
`e3_train.py:95` disables the encoder gradient. Therefore the current
1795-step LR warmup warms the heads, while LoRA receives its first gradient
at approximately peak LR after unfreezing. Equality between calibration and
warmup is not inherently required; neither is automatically the right LoRA
warmup. Any future change should specify both schedules deliberately.

Teacher geometry preserves nuisance structure as well as useful structure;
it cannot replace task-specific translation or handshape supervision.
Full-image CLS heads also receive no explicit hand/mouth region selection.
These are objective-design limitations, not proven causes of the observed
translation errors.

### SpaMo downstream training

`visual_adapter.py:17-44` projects spatial and motion features, concatenates
their valid sequences, applies the K5/P2/K5/P2 temporal encoder, and maps to
the 2048-dimensional FLAN embedding space. This temporal implementation
passed parity against the local SpaMo reference.

`flan_t5.py:56-73` enables rank-16/alpha-32 Q/V FLAN LoRA. Thus the 23.5M
trainable parameters are **not just a projector**: they include the visual
adapter and FLAN LoRA. Frozen FLAN base weights do not imply all language
model parameters are frozen.

The first approximately 2000 steps optimize VT alignment only; subsequent
steps optimize translation cross-entropy plus VT alignment. Text targets
come from the fixed embedding layer, as in SpaMo. Pooling is mask-aware.
Actual pilot LR groups (`pilot_factory.py:70-109`) are 3e-4 for the visual
adapter and 5e-5 for FLAN LoRA, with AdamW betas (0.9, 0.98), weight decay
0.01, 10% LR warmup, and cosine decay to zero. The generic config value
`optimizer.learning_rate: 0.0006` is not the pilot's actual group LR.

At batch 4, 8000 steps correspond to about 5.57 passes over 5746 clips;
only about 4.18 passes have translation supervision. SpaMo's paper reports
40 epochs and 4000 VT-only steps on PHOENIX. Its appendix describes a nonzero
LR floor, while the local upstream `spamo/t5_slt.py:611-643` uses a 10%
warmup/zero-floor cosine scheduler. Paper and local code are distinct
references; do not silently change the frozen pilot to make either match.

The downstream E1/E3 configuration, shared initialization and logical split
are explicitly matched (`train_e3_pilot.py:59-76`). The new spatial source
is the intervention at this stage. No reference translations are inserted
into prompts at evaluation.

## 4. Probe defects and evidence

### Unmatched temporal sampling

`scripts/probe_e3_features.py:87-90,97-98,117` loads every original frame but
extracts only the five sampled adapted frames. Geometry and signer probes
then pool these different inventories. The comment about a middle frame
also disagrees with the actual mean pooling.

Read-only recomputation using the existing frozen/adapted features and the
same five source indices demonstrates the confound on 543 clips:

- Same frozen encoder, full-frame versus five-frame pooling:
  spread ratio 0.995219; signer accuracy **0.966292 versus 0.977528**.
- Matched five-frame constant E3: spread ratio **0.830648**; signer delta
  **-0.123596**; hand deltas **-0.044776 / -0.014493**.
- Matched five-frame cosine E3: spread ratio **0.941416**; signer delta
  **-0.022472**; hand deltas **-0.029851 / +0.014493**.

These recomputations were diagnostic, not replacements for published reports.
The minimal future repair is to select the same five original frame rows in
`_load_subset_features` before either probe. Its regression control should
compare an encoder against itself and require identical probe inputs, a
spread ratio of one, and zero signer delta.

### Weak handshape surrogate and incomplete-gate behavior

`probe_e3_features.py:123-154` chooses seven most frequent free-text hand
configuration strings. Its left-hand test has 67 examples, so the cosine
top-1 decrease is **two predictions**, not evidence of a general erosion
law. Frozen left/right top-1 (0.2687/0.2899) is below majority top-1
(0.3433/0.3333). Left balanced accuracy improves from 0.2311 to 0.2391
despite the top-1 decline. These are noisy, class-imbalanced probes trained
on unreviewed descriptions.

`probe_e3_features.py:186-208` silently omits a hand probe when it cannot be
formed; `all(...)` over an empty dictionary is true. A report can therefore
pass without measuring both hands. A repaired/versioned gate should require
both hands, include per-class counts and balanced scores, and explicitly
report an unavailable diagnostic instead of interpreting it as success.

Actual current code thresholds are `std_ratio > 0.60`, signer delta
`<= 0.005`, and available hand deltas `>= -0.02`. Earlier descriptions of
0.80 / 0 / 0 were not this code's policy. None of these thresholds has been
validated as a predictor of BLEU. The current code defects remain open;
they were not modified as part of this analysis-only audit.

## 5. Highest-value next experiments

Success means better dev-selected translation on the fixed signer split,
with preserved provenance and separately reported identity/articulator
diagnostics. A cheaper proxy is not a replacement for translation BLEU.

### First: dev-only decoding experiment on the existing best checkpoint

Stored dev predictions show a major decoding failure mode:

| Selected model | Outputs with a repeated 4-gram | Unique outputs / 582 | Hypothesis/reference token-length ratio |
|---|---:|---:|---:|
| E1 | 253 | 371 | 1.0145 |
| Constant-LR E3 | 239 | 395 | 1.0139 |
| Cosine/lower-LR E3 | 231 | 478 | 1.0482 |

The repeated-4-gram statistic flags repetition, not a guarantee that every
flag is wrong. Nevertheless, outputs such as many repetitions of "tiefer"
or "schwach" establish actual degenerate decoding. Dates, temperatures and
locations also remain poorly grounded in sample predictions.

Test baseline deterministic beam decoding versus one explicit repetition
constraint, for example `no_repeat_ngram_size=3`, on a fixed dev subset,
then full dev if promising. Measure corpus BLEU, ROUGE, repetitions,
lengths and truncation. Do not edit or re-score the already completed
Signer07 report. Decoder changes require a new protocol/report identity.
This requires no feature extraction or model training, and is the cheapest
plausible BLEU intervention identified by the audit. Improvement is not yet
measured.

### Second: more joint translation training with cached cosine features

The cosine model's dev BLEU is still rising at step 8000 and it has only
about four translation-supervised passes. Compare the existing recipe with
a deliberately longer downstream horizon using the **same** cached
features, shared initialization and train/dev split. Declare the new LR
schedule before launch and select on dev. Simply resuming an exhausted
zero-LR scheduler adds no useful updates. A longer horizon is not the same
as adding steps to the existing scheduler.

This avoids another approximately 9.4-hour extraction. It still needs a
new supported step tier and a separate training authorization. It should
not be bundled with batch-size or loss changes.

### Third: improve the positive semantic signal, then localize it

Follow the existing `2026-10-01-cropped-articulator-supervision-design.md`
sequence: closed-vocabulary hand labels first (Experiment L), region crops
second (Experiment C). Free-text equality can turn synonymous handshape
descriptions into false contrastive negatives. Full-image CLS supervision
does not isolate the small articulators needed for translation.

Derived targets or regenerated descriptions need a new version and source
binding; preserve frozen caption records and mask unknown/failed targets.
Measure label support and actual encoder gradient contributions before a
long run. Task-positive alignment and localized features are promising
mechanisms, not performance claims.

A larger real contrastive batch can also improve VT alignment, whose batch
4 has only three other examples per anchor. Ordinary gradient accumulation
alone does **not** enlarge that negative set; embedding gathering or a
separately specified queue would be a different objective. Treat this as
a later isolated experiment.

### Separate efficiency work

- `baseline_module.py:125-141` executes a full FLAN translation-loss forward
  even during VT-only warmup, then discards that loss from the objective.
  A warmup-specific path could save compute. It changes RNG consumption and
  logged losses, so it needs a regression check and new code provenance.
- The new downstream run wrote **16 resume snapshots totaling 88.16 GiB**
  (about 5.52 GiB each). Future lightweight trainable-state snapshots or
  explicit retention policy would reduce the demonstrated WSL storage
  problem. Published checkpoints remain provenance-bound artifacts.
- `train_e3_pilot.py:133-135` hashes only its entrypoint and `e3_runtime.py`,
  not the full optimizer/model/loss implementation closure. Future run
  identities should bind those dependencies before experimentation changes
  them; current recorded runs should not be rewritten.

## 6. Verification performed

Configuration: root `pyproject.toml` declares pytest and Ruff, torch 2.0.1,
transformers 4.32.0, Python 3.11. DINO has a separate locked environment
with torch 2.5.1+cu121 and transformers 4.56.2. Pyright is installed globally
(1.1.411), but is neither a declared root dev dependency nor configured by
a project Pyright configuration file.

### Tests

- Targeted main training/models/E3 runtime/E3 CLI/evaluation/parity checks:
  **99 passed, 2 skipped** after correcting the import path.
- DINO schedule, relational-loss, training, data and extraction tests were
  checked before the broader DINO suite below.
- Full main CPU suite: **635 passed, 5 deselected**.
  Command: `PYTHONPATH=src:. OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 uv run --locked --offline pytest tests -m 'not gpu and not integration' -q`.
- Full DINO CPU suite: **190 passed, 1 deselected**.
  Command: `UV_PROJECT_ENVIRONMENT=/tmp/opencode/despamo-e3-smoke-env PYTHONPATH=src:. OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 uv run --offline --project tools/dinov3 --locked pytest tools/dinov3/tests -m 'not gpu and not integration' -q`.
- Local SpaMo temporal parity: **1 passed**.
  Command: `SPAMO_PROJECT_PATH=/home/kan/Research/SpaMo PYTHONPATH=src:. OMP_NUM_THREADS=2 uv run --locked --offline pytest tests/parity/test_spamo_visual_adapter.py -q`.
- Completed real extraction, 8000-step training, dev checkpoint selection,
  and Signer07 evaluation were verified against receipts and checkpoint
  hashes. CPU tests are not being substituted for those experiment results.

### Lint, typing and shell checks

- Targeted Ruff: passed. `uv run --locked --offline ruff check .`: passed.
- E3 main train/evaluate entrypoints under root `.venv`: **0 errors**.
- E3 DINO model/train/extract under the DINO interpreter: **0 errors**.
- Broad Pyright over `src scripts` with the main interpreter:
  **98 errors** across existing source, wrappers and older runners.
- Broad Pyright over DINO implementation files with its interpreter:
  **6 errors**, in `cli.py` (3), `encoder.py` (1), `frames.py` (1), and
  `storage.py` (1).
- `bash -n` over the adaptation, probe, extraction, training and new
  held-out driver scripts: passed.

The wider type errors are pre-existing in unchanged files; many involve
dynamic PyTorch/Lightning attributes, optional-value narrowing, or
OmegaConf container unions. They remain unresolved and prevent a claim of
repository-wide type cleanliness. No unrelated code was changed to make
these checks pass.

### Failures and fixes during this task

- Initial pytest invocation failed collection with missing `tests` and
  `scripts`. Cause: repository root absent from import path. Corrected to
  `PYTHONPATH=src:.`; targeted and broad tests passed.
- Initial combined Pyright invocation used the SpaMo environment for DINO,
  producing a missing DINOv3 symbol and old-torch export diagnostics.
  Selecting each pinned interpreter removed those environment errors.
- Foreground held-out generation exceeded the shell's 300-second timeout.
  No completed output existed and no scorer process remained. A detached
  evaluation driver completed successfully.
- One detached launch had a mistyped dataset hash and failed provenance
  validation before scoring. Corrected the path in
  `/tmp/opencode/run_cosine_e3_eval.sh`; the completed report matches the
  selected checkpoint and all 768 expected references.

## 7. Remaining concerns

The temporal mismatch, unavailable-hand gate path, weak handshape surrogate,
unsupported schedule-only attribution, narrow code-hash closure, large
resume snapshots, and pre-existing broad type diagnostics remain follow-up
items. The handoff and project instruction text also contains older
"Signer07 not evaluated" statements; actual versioned receipts supersede
those historical status notes.

No extra adaptation, annotation generation, long training experiment, or
decoder change is authorized by this audit. Next recommended bounded
experiment: dev-only repetition-constrained decoding using the existing
best cosine checkpoint, under a new decoder identity.

## 8. Follow-up: first-stage probe repair implemented

The earlier sections describe the pre-repair audit snapshot. The subsequently
approved first stage now aligns both encoders to the same five source frames,
reports unavailable hand probes explicitly as `INCOMPLETE` with
`gate_pass=null`, and handles insufficient per-class support without crashing.
Regression tests cover the identical-encoder control, either/both missing hand
probes, cache-only execution, and corrupted checkpoint/frame/feature bindings.

New schema-2 reports were generated from existing feature caches on CPU for
the constant-LR and cosine/lower-LR full adaptations. They publish the matched
diagnostic values from section 4, together with cache, sampling, threshold and
code provenance. See [matched-frame probe repair](signer-pilot-protocol.md#matched-frame-probe-repair-2026-10-03)
for report paths and the updated outcome contract.

The free-text label limitations, small hand test sets, unvalidated relationship
between probe thresholds and BLEU, broad type diagnostics and efficiency work
remain open. No labeling or translation-training behavior changed in this repair.

Post-repair verification: the full root `uv run --locked --offline pytest -q`
suite passed **646 tests**, with **4 skipped**; the **15 probe tests** also
passed in the pinned extractor environment. Full-project Ruff passed, and
targeted Pyright for the probe script reported zero errors with either pinned
interpreter. The wider historical type diagnostics in section 6 were not
repaired or reclassified as clean.
