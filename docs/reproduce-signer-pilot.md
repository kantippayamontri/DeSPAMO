# Reproducing the Signer-Disjoint Pilot (E1 / E2 / E3)

This document lets a second machine reproduce, verify, or extend the signer pilot.
Every number here was measured on the original host, not estimated. Where a figure is
an estimate it says so.

Read [signer pilot protocol](signer-pilot-protocol.md) first for the frozen data
contract, and [E3 code handoff](e3-code-runbook.md) for the raw command surface.

## 0. What this repository does and does not contain

Git carries **code, tests, and documentation only**. It does **not** carry the dataset,
features, or checkpoints, which total roughly 170 GiB and live outside the tree.

| Needed on the new host | Size | In git? |
|---|---|---|
| PHOENIX14T frames, 210x260 PNG | large | no |
| PHOENIX14T annotations (`*_info_ml.npy`) | small | no |
| Qwen appearance dataset `b4d52678...` | ~10 GiB | no |
| Frozen CLIP text targets (inside the dataset key) | — | no |
| Original DINOv3 features `1e84e43d...` | ~35 GiB | no |
| SpaMo motion features (`mae_feat_Phoenix14T`) | large | no |
| Frozen protocol JSON `0d7df319...` | small | no |
| Any E1/E2/E3 checkpoint or adapted feature version | ~100 GiB+ | no |

`external-data` and `recovery-worktree` in the repository root are **git-ignored
symlinks** to local paths. They will not exist after a clone and must be recreated or
ignored. Recorded run configurations embed absolute paths, so the simplest path is to
mirror the original layout; otherwise every command needs new arguments.

## 1. Environments

Two separate, pinned environments are required. They must stay separate: the main
stack is pinned to torch 2.0.1 for SpaMo/Lightning parity with E1/E2, while DINOv3
needs transformers 4.56.2 and torch 2.5.1+cu121. Both `uv.lock` files are committed.

```bash
# main stack: SpaMo training, evaluation, probes
uv sync --locked                      # python >=3.11,<3.12

# extractor stack: DINO adaptation and feature extraction
uv sync --locked --project tools/dinov3
```

Pinned versions, for reference: main stack `numpy==1.26.4`, `torch==2.0.1`,
`transformers==4.32.0`, `pytorch-lightning==1.9.5`, `torchmetrics==1.8.2`;
extractor stack `torch==2.5.1+cu121`, `transformers==4.56.2`, `pillow==11.3.0`,
`huggingface-hub==0.35.3`.

### WSL / CUDA note

On the original WSL host, cuDNN could not resolve `libcuda.so` and every CUDA job
hung silently until this was set. Export it before any GPU command:

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
```

Also set `HF_HUB_OFFLINE=1` so a pinned revision can never be silently re-resolved,
and keep `OMP_NUM_THREADS=2`.

## 2. Pinned model identities

These are enforced in code; a mismatch aborts before any GPU work.

| Role | Identity |
|---|---|
| DINOv3 base | `facebook/dinov3-vitl16-pretrain-lvd1689m` revision `ea8dc2863c51be0a264bab82070e3e8836b02d51` |
| FLAN-T5-XL | revision `7d6315df2c2fb742f0f5b556879d730926ca9001` |
| CLIP text encoder | `openai/clip-vit-large-patch14` revision `32bd64288804d66eefd0ccbe215aa642df71cc41`, width 768 |
| Appearance VLM | `qwen3-vl:8b` via Ollama, prompt version `seven-factor-v2` |

DINOv3 is gated on Hugging Face. Access must be granted to the account whose token is
used, and the token belongs in an ignored `.env`, never in git.

## 3. Frozen data identities to verify first

Run this before anything expensive. If any value differs, stop: the data is not the
data these results came from.

```
protocol_hash          0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7
split_hash             a95fa8f2710ebca33d6565a26f0f076817d839ecc611bc43648a48761fd6527b
dataset_key            b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a
source_hash            c405bcc08d43235783ae6df1322944d1e6f365b2a74e5743c1acff27865a61c7
records_hash           65cf8d79ad45d567462c7b6c6b0693fc7f4f7a559eafacf6b40be3353927211a
text_manifest_hash     0f304c0602aea13bfc8393eeec0e17bf999735b16586262ad36f86e686c02419
spatial_manifest_hash  558fbcb9bb1336c26586da2ade9706b94a713fd8c2228b717edbae57bd9e8cf1
motion_manifest_hash   8edfbcbf51f75e35860611f07657548406884ba8618665fb68437ee8660cce8a
frame_rows_hash        538516cc3e7cee868207bda87d959818f197cf40f1235c4c5a363f2f8ee7a7bb
annotation_hash        592cd07c4f3fe65733bf716254bbb813ebe3737466f2c40eef6977bda7b89ed3
```

Split: train 5,746 clips over signers 01, 02, 04, 05, 06, 08, 09; dev 582 clips
Signer03; held-out test 768 clips Signer07. All three originate from physical PHOENIX
`train/`, so official PHOENIX dev/test remain untouched throughout.

Decoder, fixed everywhere: deterministic beam 5, `max_length` 64, no in-context
references. Seed 0. Supervision policy `qwen-schema98-unreviewed-v1` with
`human_review_status=not_assessed`.

## 4. Offline verification (no GPU, no data)

Always runs, even with none of the datasets present:

```bash
PYTHONPATH=src:. uv run python -m pytest tests/unit -q        # expect 635 passed
PYTHONPATH=. uv run --project tools/dinov3 \
  python -m pytest tools/dinov3/tests -q                       # expect 183 passed, 1 skipped
uv run ruff check scripts src tests tools/dinov3               # expect clean
```

The single skip is an opt-in live-encoder test. If counts differ, the checkout does
not match this document.

## 5. Reproduction order

Every stage has a read-only `--mode preflight` that validates identities, paths, and
budget without touching the GPU. **Always run preflight first.** Every `--mode run`
requires an explicit authorization flag and an explicit GPU cap, by design: nothing
long-running starts implicitly.

Stages are independent in the sense that the SpaMo stages only need the *original*
DINO features. Only E3 requires the adaptation and extraction stages.

### 5.1 E1 baseline, frozen DINO

`scripts/retrain_e1_authorized.py --steps {4000|6000|8000}`. Dev-selection checkpoints
follow the tier: 4,000 -> (1,750, 2,800, 4,000) with warmup 1,000; 8,000 -> (3,500,
5,600, 8,000) with warmup 2,000. Physical batch 4, `num_workers=0`.

A quirk worth knowing: this script requires a *stale parent run* (policy
`signer-pilot-e1-fresh-retry-v2`, status `running`, dead PID) as a recovery
precondition. On a fresh host that parent does not exist, so either carry the parent
run directory across or relax that gate deliberately and record the change.

### 5.2 E2 projector factors

`scripts/train_e2_pilot.py`, 4,000 steps only in the recorded results. Adds seven
projector-level factor heads on frozen DINO features.

### 5.3 E3 DINO adaptation (extractor env)

`python -m tools.dinov3.e3_train --mode run --authorize-e3-run`, with
`--adapt-steps`, `--calibration-steps`, `--gpu-cap-seconds`, `--batch-size {4,8,16}`,
`--relational-weight W`.

Epoch arithmetic: 5,746 clips at batch 16 gives `5746 // 16 = 359` steps per epoch
(2 clips dropped, reshuffled each epoch). The 25-epoch schedule is therefore 8,975
steps with 1,795 calibration. At batch 4 the same 25 epochs is 35,900 steps with
7,180 calibration. Clips seen per epoch are identical; only the update count changes.

LoRA is rank 8 / alpha 8 on Q and V in DINO layers 20-23 only. LoRA LR `1e-4`, heads
LR `3e-4`, nuisance weight 0.05, articulator 0.10, reference 0.10. Snapshots every 500
steps carry optimizer, sampler, RNG, step, and elapsed GPU spend, so interrupted runs
resume without rewinding.

Note on the LR: batch 16 gives 4x fewer updates than batch 4 at the same LR. That was
a deliberate choice, left unscaled, and it is why the sweep in 5.6 was run entirely at
batch 16 so the selected weight matches the regime it will be used in.

### 5.4 E3 adapted-feature extraction (extractor env)

`python -m tools.dinov3.e3_extract --mode run --authorize-e3-extraction`.
Re-extracts **all 7,096 physical train clips, all 827,354 frames**, at both 224 and
448, and publishes a new versioned root only after a complete manifest, row-map, and
per-clip receipt gate. Partial output is never promoted; interrupted runs reuse only
verified clips. Needs roughly 100 GiB free.

### 5.5 E3 SpaMo translation (main env)

`scripts/train_e3_pilot.py --steps 8000 --mode run --authorize-e3-run`, pointed at the
adapted feature root and at a **completed E1 run of the same tier**. The config match
enforces equality with that E1 run except for the spatial source, and now also
enforces an equal step tier, so unequal budgets cannot be compared silently. SpaMo
batch stays 4 because `pilot_config` admits only 2 or 4 and E1/E2 used 4.

### 5.6 Relational weight sweep, gated

Short adaptations (718 steps = 2 epochs at batch 16, 143 calibration) at weights 0.1,
0.15, 0.2, 0.4, each followed immediately by `scripts/probe_e3_features.py`. The gate
extracts adapted features for ~600 clips, about 1% of a full extraction, and reports
`std_ratio`, signer top-1, and handshape retention.

Validate the gate before trusting it: pointed at the known-bad batch-4 adaptation it
must reproduce the failure (`std_ratio` ~0.088, signer delta ~+0.011, `gate_pass`
false). If it does not, the gate is wrong, not the variant.

### 5.7 Held-out Signer07 scoring, once only

`scripts/evaluate_e1_8000.py` for E1 and `scripts/evaluate_e3_pilot.py` for E3. Both
refuse an existing output and verify the selected checkpoint against its own dev
reports. Score Signer07 **only after** dev selection, and only once per checkpoint.

## 6. Measured costs on the original host

RTX 4080 SUPER, 16 GiB. Treat these as the reference, not a guarantee.

| Stage | Measured |
|---|---:|
| E1 @ 4,000 steps | 0.588 GPU-h |
| E1 @ 8,000 steps | 0.962 GPU-h |
| E2 @ 4,000 steps | 0.729 GPU-h |
| E3 adaptation, batch 4, 35,900 steps | 5.093 GPU-h |
| E3 adaptation, batch 16, 718 steps | ~0.41-0.45 GPU-h each |
| E3 full extraction, 7,096 clips | 9.811 GPU-h |
| E3 SpaMo @ 4,000 / 8,000 steps | 0.657 / 0.889 GPU-h |

Throughput: adaptation at batch 16 measured **2.14 s/step** end to end. An isolated
forward/backward benchmark showed 1.42 s/step; the difference is per-step image
loading (10 PNG decodes at two resolutions per clip), so trust 2.14 for planning.
Peak VRAM at batch 16 was 11.24 GiB of 16.4 GiB, so batch 16 fits but batch 32 was
not tested.

Cumulative spend to date is about 21.5 GPU-h. The declared ceiling was raised from
24 h to 40 h; per-stage caps are enforced in code, the cumulative ceiling is
documentation only and must be checked manually before authorizing a stage.

## 7. Expected results

Signer03 dev, full 582 clips, highest corpus BLEU-4 with earliest step on a tie:

| Run | tier | selected step | dev BLEU-4 |
|---|---:|---:|---:|
| E1 | 4,000 | 4,000 | 3.6956 |
| E2 | 4,000 | 4,000 | 3.0283 |
| E3 | 4,000 | 4,000 | 3.4575 |
| E1 | 8,000 | 5,600 | **4.0381** |
| E3 | 8,000 | 5,600 | 3.9699 |

Signer07 held-out, 768 clips, identical ordered IDs and references, beam 5:

| Tier | E1 | E2 | E3 | E3 - E1 |
|---|---:|---:|---:|---:|
| 4,000 | 2.9901 | 2.4878 | 3.0573 | +0.0672 |
| 8,000 | 3.4238 | — | **3.7337** | **+0.3099** |

Both 8,000-step runs peaked at dev step 5,600, and the extra budget mattered more to
E3 (+0.6765) than to E1 (+0.4338), so 4,000 steps had been undertraining. Only the
8,000-step gap exceeds the 0.3-point threshold; the 4,000-step gap does not and should
not be described as an improvement.

Probe gate, ~600 clips, batch 16, 718 steps per variant:

| Variant | gate | `std_ratio` | signer top-1 delta | handshape L / R |
|---|---|---:|---:|---:|
| baseline, no relational terms | fail | 0.088 | **+0.011** | +0.045 / 0.000 |
| **w = 0.1** | **pass** | **0.822** | **-0.056** | 0.000 / +0.029 |
| w = 0.15 | fail | 0.803 | -0.090 | 0.000 / -0.044 |
| w = 0.2 | fail | 0.794 | -0.124 | -0.045 / -0.029 |
| w = 0.4 | fail | 0.889 | -0.045 | -0.045 / +0.029 |

## 8. Determinism and what will not reproduce bit-for-bit

Seed 0, `num_workers=0`, deterministic beam-5 decoding, and a seeded clip sampler make
the pipeline *intended* to be reproducible. It will not be bit-identical across hosts:
GPU kernel selection, cuDNN autotuning, bf16 autocast ordering, and driver version all
perturb low-order bits, and BLEU on 582 or 768 clips is sensitive enough that small
drift is visible. Expect small deviations; treat a shift beyond roughly 0.3 BLEU-4 as a
signal to investigate rather than accept.

Run keys are content hashes over identity, config, and code hashes, so **any code edit
produces a new run directory**. That is deliberate: it protects completed artifacts and
makes accidental mixing of incomparable runs impossible. It also means your run keys
will differ from the ones quoted in the protocol document even when everything else
matches.

## 9. Known gotchas

- **Stale empty run directories block a fresh start.** A killed run leaves a directory
  containing only `.writer.lock` and a step-0 budget; the fresh-run guard then refuses
  to proceed. Confirm the PID is dead and no snapshots exist, then archive, do not
  delete, the empty directory.
- **Do not pipe a run command straight into `tail` under `set -o pipefail`.** It
  swallows the traceback and leaves only a banner. Redirect the full stream to a file
  and tail the file.
- **The two environments are not interchangeable.** The extractor env lacks `pydantic`,
  so modules importing `despamo.appearance.provenance` fail there;
  `scripts/probe_e3_features.py` reads JSON locally for exactly this reason.
- **Original and adapted feature arrays index frames differently.** Original arrays
  hold every frame and are addressed by absolute source index; adapted arrays hold only
  the 5 sampled frames, addressed 0-4. Mixing the conventions raises
  `IndexError: index 26 is out of bounds for axis 0 with size 5`.
- **`train_signer_pilot.py` must not be edited** without accepting that E1-v3's
  `checkpoint_bound_code_hashes` gate will reject its own prior artifacts.
- Two one-off scripts, `scripts/run_e3_pipeline.sh` and
  `scripts/paired_signer07_8000.py`, hardcode run keys from the original host and are
  not portable as written.

## 10. Extending: new appearance text version

If the goal is a new text-description dataset rather than reproducing these numbers,
read [cropped articulator supervision design](superpowers/specs/2026-10-01-cropped-articulator-supervision-design.md).

The short version: hand targets are free text and therefore near-unique (3,787 distinct
values over 34,782 rows, 1,363 of them singletons), which leaves the contrastive
articulator objective largely inert because it needs at least two positions sharing a
label. Mouth targets use `Literal` closed sets and have only 53 distinct values. The
cause is the schema in `src/despamo/appearance/schema.py`, not the VLM.

That design separates two experiments deliberately, because bundling them makes a
positive result unattributable:

- **Experiment L**: closed hand vocabularies on the same whole frames. No detector, no
  crops, directly targets the mechanism. Cheapest and most likely to matter.
- **Experiment C**: add detector-driven crops for 3.9x magnification on hands and 5.5x
  on mouths. Needs a detector, which is **not currently installed** and is unverified at
  210x260 resolution.

Any new schema or prompt changes `schema_hash` and `prompt_hash`, which forces a new
dataset key and a new protocol hash. E1, E2, and E3 would all need re-running against
it, and results across two dataset keys must never appear in one comparison table.

Only `qwen3-vl:8b` is present in Ollama on the original host; no detector package is.
Labels stay LLM-generated and unreviewed regardless, so schema validity still does not
demonstrate visual correctness.
