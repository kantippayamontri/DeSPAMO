#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/kan/Research/DeSpaMo
export PYTHONPATH="$ROOT/src:$ROOT"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1

# Grace is inside the approved hour; watchdog remains if the controller disappears.
timeout --signal=INT --kill-after=14s 3585s \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/diagnose_projector_grounding.py" \
  --mode run \
  --output-dir /home/kan/datasets/despamo/signer-pilot/projector-dev-grounding-v1 \
  --gpu-cap-seconds 3600 \
  --count 128
