# DeSpaMo

## PHOENIX14T Data And Checkpoint Preparation

SpaMo annotations, spatial features, and motion features cover 7,096 train, 519 dev,
and 642 test clips. Raw video frames are in the release's
`PHOENIX-2014-T/features/fullFrame-210x260px/{train,dev,test}/` directories;
clip-directory IDs match all three SpaMo annotation splits. Sampled PNGs measure
210 × 260 pixels. These frames are for later appearance supervision; baseline
loads the already-extracted spatial and motion feature arrays.

The released SpaMo checkpoint mapped 871 tensors. Conversion against the
cached Flan-T5-XL target schema checked every key, shape, and dtype; strict
CPU loading reported zero missing and zero unexpected tensors. Converted
checkpoint and schema live outside Git. Source checkpoint SHA-256:
`06a432cdd8e1da4ce0b0e4cff246b20ad7d6a60406f32dfdbdfd974f94d3eee6`.
Target schema SHA-256:
`6bd3ec3de4d4d7a10101fdf530734561605f632d202f23a2a3f95dab51d47d77`.
Released-checkpoint translation results appear below.

## PHOENIX14T Baseline Smoke

The bounded smoke completed with frozen `google/flan-t5-xl`, one epoch, 1% of
training batches, and one validation batch. Set `PHOENIX14T_ANNOTATION_ROOT`,
`DESPAMO_FEATURE_ROOT`, and `DESPAMO_HF_CACHE` to local paths; on WSL, make the
driver library visible to cuDNN. Exact successful command:

```bash
LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" uv run --frozen python scripts/train.py \
  --config configs/data/phoenix14t.yaml \
  --config configs/model/spamo_flan_t5_xl.yaml \
  --config configs/experiment/phoenix14t_baseline.yaml \
  --config configs/experiment/phoenix14t_smoke.yaml \
  trainer.default_root_dir=artifacts/phoenix14t_smoke_retry2
```

Observed: 35 optimizer steps, 35 finite training-loss rows, one finite validation-loss
row, and strict loading of all 583 checkpoint tensors with zero missing/unexpected
keys. Checkpoint:
`artifacts/phoenix14t_smoke_retry2/lightning_logs/version_0/checkpoints/epoch=0-step=35.ckpt`.
This verifies pipeline execution, not translation quality. Earlier failed attempts
remain in separate ignored run directories. See
`docs/superpowers/plans/2026-09-22-phoenix14t-baseline.md` for baseline plan.
To resume a run with the same config and output root, pass `--resume PATH --trusted-resume`
only for a checkpoint you trust. Resume fails if safe checkpoint preflight rejects it.
Resume restores model, optimizer, scheduler, and prompt RNG state, but does not restore
minibatch order, random spatial crops, PyTorch/LoRA dropout RNG, or worker RNG. Outputs
can differ from uninterrupted training; use fresh runs for controlled comparisons.
Use one training writer per `trainer.default_root_dir`; concurrent manifest writes are not
serialized.

## PHOENIX14T Baseline Evaluation

Task12 bounded GPU smoke passed: optimizer steps, validation, finite losses, and
checkpoint reload were observed. Released-checkpoint evaluation covered all
642 test clips, passed upstream-compatible metric tolerances, and produced
byte-identical deterministic artifacts on two runs. With PHOENIX14T features,
manifests, converted checkpoint, and CUDA GPU available,
set `PHOENIX14T_ANNOTATION_ROOT`, `DESPAMO_FEATURE_ROOT`, and `DESPAMO_HF_CACHE`.
Evaluate upstream-compatible sampling with ordered config layers:
Write outputs outside DeSpaMo repository or to paths Git ignores; `artifacts/` is ignored.

```bash
LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" uv run --frozen python scripts/evaluate.py \
  --config configs/data/phoenix14t.yaml \
  --config configs/model/spamo_flan_t5_xl.yaml \
  --config configs/experiment/phoenix14t_baseline.yaml \
  --checkpoint /home/kan/datasets/spamo/ckpt/despamo-spamo-baseline.pt \
  --generation upstream \
  --accept-baseline \
  --output artifacts/phoenix14t_baseline/upstream.json
```

For deterministic runs, change to `--generation deterministic` and **remove
`--accept-baseline`**; use distinct output files
`artifacts/phoenix14t_baseline/deterministic-1.json` and
`artifacts/phoenix14t_baseline/deterministic-2.json`.
Each successful artifact includes clip-level predictions and references, BLEU-1 through
BLEU-4, ROUGE-L, checkpoint SHA-256, config, and runtime provenance. Every run
requires exactly 642 clips; only upstream runs with `--accept-baseline` enforce
BLEU-4 within 1.0 point of 25.08 and ROUGE-L F1 within 0.01 of 0.4698. General
research evaluations omit that flag.

| Decoding | Clips | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L F1 |
|---|---:|---:|---:|---:|---:|---:|
| Upstream-compatible | 642 | 50.5658 | 37.8416 | 29.8984 | 24.7104 | 0.464884 |
| Deterministic (both runs) | 642 | 51.0865 | 38.3082 | 30.2984 | 25.0307 | 0.469417 |

The upstream result is within both acceptance tolerances. Deterministic
artifacts match byte-for-byte (SHA-256:
`4dc0917ae85c3ea7f601e8056a88f4ac68877f13c716261f29cd20c74b7e6f64`).
Converted checkpoint SHA-256 recorded in evaluation artifacts:
`44ace3e8536817691f6c6b3104f8881fd65e6e1ac8ff92364a4af16efd93f45f`.
