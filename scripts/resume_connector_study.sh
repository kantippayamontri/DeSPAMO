#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/kan/Research/DeSpaMo
export PYTHONPATH="$ROOT/src:$ROOT"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1

# Windows hosts the VHDX. Linux free space alone does not prevent host exhaustion.
"$ROOT/.venv/bin/python" -c '
import os
for path, minimum in (("/", 50), ("/mnt/c", 50)):
    stats = os.statvfs(path)
    available = stats.f_bavail * stats.f_frsize / 2**30
    if available < minimum:
        raise SystemExit(f"Resume blocked: {path} has {available:.2f} GiB free; require {minimum} GiB.")
    print(f"Storage admission: {path} {available:.2f} GiB available")
'

exec "$ROOT/.venv/bin/python" "$ROOT/scripts/run_connector_study.py" \
  --study-root /home/kan/datasets/despamo/signer-pilot/three-arm-connector-study-v1
