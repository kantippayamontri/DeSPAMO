# E3 code and completed pilot handoff

Run commands from the isolated E3 worktree after reviewing every preflight
receipt. Full adaptation, extraction, translation training and held-out scoring
are separate actions; each stage refuses existing completed output. The
frozen E3 run and Signer07 result are recorded in
[signer pilot protocol](signer-pilot-protocol.md); do not rerun or overwrite
their artifacts. Initial one-step diagnostic receipt is
`/home/kan/datasets/despamo/signer-pilot/e3-smoke-mask-aware-v1.json`.
Original three-method 20-GPU-hour admission failed; this code does not reverse
that result or supply a new GPU budget. Allocate explicit per-stage caps.

Use a persistent native-Linux `E3_BASE` with sufficient free space. Existing
frozen roots on this host:

```bash
export E3_BASE=/home/kan/datasets/despamo/signer-pilot
export PROTOCOL="$E3_BASE/protocol/0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7/protocol.json"
export DATASET=/home/kan/datasets/despamo/appearance/datasets/b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a
export TEXT="$DATASET/text/24ad915ab5bde47335bfc8839f654b0a54ad66ac1253fc4218e1c415951fa571/manifest.json"
export ORIGINAL=/home/kan/datasets/despamo/features/1e84e43d142abc6242cce75c3e65abe9e1e03a6c61e3770181d7be381f3bc31d
export FRAMES=/home/kan/datasets/despamo/phoenix-frames-210x260px
export DINO_CACHE=/home/kan/datasets/despamo/hf_cache
export MOTION=/home/kan/datasets/spamo/features/mae_feat_Phoenix14T
export MOTION_MANIFEST=/home/kan/datasets/spamo/features/manifests/phoenix14t_motion.json
export ANNOTATION=/home/kan/Research/SpaMo/preprocess/Phoenix14T/train_info_ml.npy
export FLAN_CACHE=/home/kan/datasets/spamo/hf_cache
export E1_RUN="$E3_BASE/e1-v3-79d14f8b66093483f683102f06bf28a23253d82d056ca03be0771725a170b533"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
```

From `/home/kan/Research/DeSpaMo-signer-probe`, use locked separate extractor
environment for DINO commands (`uv sync --project tools/dinov3 --locked` once).
Set `ADAPT_STEPS`, `CALIBRATION_STEPS`, `DINO_GPU_CAP_SECONDS` explicitly;
calibration must be positive and shorter than adaptation. Preflight prints
versioned `run_key` without starting CUDA. Add `--authorize-e3-run` only to
the eventual `--mode run` invocation:

```bash
uv run --project tools/dinov3 --locked python -m tools.dinov3.e3_train \
  --mode preflight --protocol "$PROTOCOL" --dataset "$DATASET" \
  --text-manifest "$TEXT" --dino-root "$ORIGINAL" --frames "$FRAMES" \
  --cache "$DINO_CACHE" --output-base "$E3_BASE" \
  --adapt-steps "$ADAPT_STEPS" --calibration-steps "$CALIBRATION_STEPS" \
  --gpu-cap-seconds "$DINO_GPU_CAP_SECONDS"
```

After authorized adaptation completes, set `ADAPT_RUN` to printed run-key
directory. Feature preflight prints deterministic `feature_key`; full version
will be `$E3_BASE/e3-features-$feature_key`. Set `EXTRACT_GPU_CAP_SECONDS`
and use `--authorize-e3-extraction` only for extraction run:

```bash
uv run --project tools/dinov3 --locked python -m tools.dinov3.e3_extract \
  --mode preflight --protocol "$PROTOCOL" --dataset "$DATASET" \
  --adapt-run "$ADAPT_RUN" --original-dino-root "$ORIGINAL" \
  --frames "$FRAMES" --cache "$DINO_CACHE" --output-base "$E3_BASE" \
  --gpu-cap-seconds "$EXTRACT_GPU_CAP_SECONDS"
```

For main SpaMo stack, start a separate shell with the repository's pinned
baseline environment (not the extractor environment):

```bash
export PYTHONPATH="$PWD/src:$PWD"
export SPAMO_GPU_CAP_SECONDS=21600   # explicitly choose a positive value <=21600
/home/kan/Research/DeSpaMo/.venv/bin/python scripts/train_e3_pilot.py \
  --mode preflight --protocol "$PROTOCOL" --dataset "$DATASET" \
  --text-manifest "$TEXT" --dino-root "$ORIGINAL" \
  --motion-root "$MOTION" --motion-manifest "$MOTION_MANIFEST" \
  --annotation "$ANNOTATION" --adapted-root "$ADAPTED_ROOT" \
  --e1-run "$E1_RUN" --hf-cache "$FLAN_CACHE" --output-base "$E3_BASE" \
  --gpu-cap-seconds "$SPAMO_GPU_CAP_SECONDS"
```

`--mode run --authorize-e3-run` performs 4,000 fresh E3 SpaMo steps and
selects a checkpoint on Signer03 dev. Do not use Signer07 before that
selection. Once selected, `scripts/evaluate_e3_pilot.py --mode preflight|run`
accepts the same frozen input paths plus `--adapted-root`, `--e3-run`, and a
new `--output` for a *single* complete Signer07 result. Optional
`scripts/probe_e3_pilot.py` accepts E1/E2 aggregate probe JSON as
`--split-manifest`; probe is diagnostic, not an E3 gate. All new runs and
results carry `qwen-schema98-unreviewed-v1` and
`human_review_status=not_assessed`.
