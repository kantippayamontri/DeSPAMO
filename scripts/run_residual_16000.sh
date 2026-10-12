#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/kan/Research/DeSpaMo
B=/home/kan/datasets/despamo/signer-pilot
KEY=b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a
D=/home/kan/datasets/despamo/appearance/datasets/$KEY
PROTOCOL="$B/protocol/0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7/protocol.json"
TEXT="$D/text/24ad915ab5bde47335bfc8839f654b0a54ad66ac1253fc4218e1c415951fa571/manifest.json"

export PYTHONPATH="$ROOT/src:$ROOT"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2

"$ROOT/.venv/bin/python" "$ROOT/scripts/train_projector_pilot.py" \
  --mode "${1:-run}" \
  --authorize-projector-run \
  --projector residual \
  --steps 16000 \
  --gpu-cap-seconds 14400 \
  --protocol "$PROTOCOL" \
  --dataset "$D" \
  --text-manifest "$TEXT" \
  --dino-root /home/kan/datasets/despamo/features/1e84e43d142abc6242cce75c3e65abe9e1e03a6c61e3770181d7be381f3bc31d \
  --motion-root /home/kan/datasets/spamo/features/mae_feat_Phoenix14T \
  --motion-manifest /home/kan/datasets/spamo/features/manifests/phoenix14t_motion.json \
  --annotation /home/kan/Research/SpaMo/preprocess/Phoenix14T/train_info_ml.npy \
  --adapted-root "$B/e3-features-7122b42e35510611626c2a71d0731b79dd79b20de6e89bd49f62737c5c84a18e" \
  --baseline-run "$B/e3-only-0402717f5ef91cca106caa881b689ce1eba75a3a87976302c99c654bea189690" \
  --matched-control-run "$B/projector-linear-16000-eb0a5bef8210afc3d894f4ac57338f8cd5e0f01b6b4d74a8a38cb285b4f5f313" \
  --output-base "$B"
