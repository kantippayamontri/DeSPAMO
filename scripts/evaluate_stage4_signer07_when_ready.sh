#!/usr/bin/env bash
set -euo pipefail

# Wait for Stage 4 pipeline to complete, then score the selected E3 checkpoint on Signer07.
ROOT=/home/kan/Research/DeSpaMo-signer-probe
BASE=/home/kan/datasets/despamo/signer-pilot
DATASET=/home/kan/datasets/despamo/appearance/datasets/b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a
PROTOCOL="$BASE/protocol/0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7/protocol.json"
TEXT="$DATASET/text/24ad915ab5bde47335bfc8839f654b0a54ad66ac1253fc4218e1c415951fa571/manifest.json"
ORIG=/home/kan/datasets/despamo/features/1e84e43d142abc6242cce75c3e65abe9e1e03a6c61e3770181d7be381f3bc31d
FRAMES=/home/kan/datasets/despamo/phoenix-frames-210x260px
CACHE=/home/kan/datasets/despamo/hf_cache
MOTION=/home/kan/datasets/spamo/features/mae_feat_Phoenix14T
MOTION_MANIFEST=/home/kan/datasets/spamo/features/manifests/phoenix14t_motion.json
ANNOTATION=/home/kan/Research/SpaMo/preprocess/Phoenix14T/train_info_ml.npy
MAIN=/home/kan/Research/DeSpaMo/.venv/bin/python

export PYTHONPATH="$ROOT/src:$ROOT"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2

echo "Waiting for Stage 4 (e3-stage4 tmux session) to complete..."
while tmux has-session -t e3-stage4 2>/dev/null; do
    dead=$(tmux list-panes -t e3-stage4 -F '#{pane_dead}' 2>/dev/null || echo "1")
    if [ "$dead" = "1" ]; then
        echo "e3-stage4 pane is finished."
        break
    fi
    sleep 60
done

echo "Checking pipeline log..."
if ! grep -q "STAGE 4 COMPLETE" /tmp/opencode/e3_stage4.log 2>/dev/null; then
    echo "ERROR: Stage 4 log does not indicate successful completion!"
    tail -30 /tmp/opencode/e3_stage4.log
    exit 1
fi

echo "Locating newly trained E3 run..."
E3_RUN=$(python3 -c "
import json, pathlib
base = pathlib.Path('$BASE')
candidates = []
for p in base.glob('e3-only-*'):
    sel = p / 'selected-checkpoint.json'
    b = p / 'budget.json'
    if sel.exists() and b.exists():
        bj = json.loads(b.read_text())
        if bj.get('status') == 'closed' and bj.get('identity', {}).get('steps') == 8000:
            candidates.append((p.stat().st_mtime, p))
candidates.sort()
if candidates:
    print(candidates[-1][1])
")

if [ -z "$E3_RUN" ]; then
    echo "ERROR: Could not find completed E3@8000 run directory!"
    exit 1
fi
echo "Found E3 run: $E3_RUN"

ADAPTED_ROOT=$(python3 -c "
import json, pathlib
b = json.loads((pathlib.Path('$E3_RUN') / 'budget.json').read_text())
print(b['identity']['config']['data']['spatial_root'])
")
echo "Found adapted features: $ADAPTED_ROOT"

OUT_FILE="$BASE/signer07-e3-w0.1-8000-heldout-v1.json"
echo "Evaluating on Signer07 held-out test -> $OUT_FILE"
"$MAIN" "$ROOT/scripts/evaluate_e3_pilot.py" --mode run \
    --protocol "$PROTOCOL" \
    --dataset "$DATASET" \
    --text-manifest "$TEXT" \
    --dino-root "$ORIG" \
    --motion-root "$MOTION" \
    --motion-manifest "$MOTION_MANIFEST" \
    --annotation "$ANNOTATION" \
    --adapted-root "$ADAPTED_ROOT" \
    --e3-run "$E3_RUN" \
    --output "$OUT_FILE"

echo "Generating paired comparison against E1@8000..."
python3 - << 'PY'
import json, pathlib
base = pathlib.Path('/home/kan/datasets/despamo/signer-pilot')
p1 = base / 'signer07-e1-8000-heldout-v1.json'
p3 = base / 'signer07-e3-w0.1-8000-heldout-v1.json'
d1 = json.loads(p1.read_text())
d3 = json.loads(p3.read_text())

m1 = d1['metrics']
m3 = d3['metrics']

paired = {
    "status": "complete",
    "protocol_hash": d1["protocol_hash"],
    "split_hash": d1["split_hash"],
    "test_clips": 768,
    "decoding": d1["decoding"],
    "checkpoint_hashes": {
        "e1_frozen_8000": d1["checkpoint_hash"],
        "e3_relational_0.1_8000": d3["checkpoint_hash"],
    },
    "scores": {
        "e1_frozen_8000": m1,
        "e3_relational_0.1_8000": m3,
    },
    "e3_minus_e1": {k: m3[k] - m1[k] for k in m1},
    "adaptation_checkpoint_hash": d3.get("adaptation_checkpoint_hash"),
    "supervision_policy": "qwen-schema98-unreviewed-v1",
    "human_review_status": "not_assessed",
}
out_path = base / 'signer07-e1-e3-relational-0.1-8000-paired-v1.json'
out_path.write_text(json.dumps(paired, indent=2) + '\n')
print("Wrote paired report:", out_path)
print(f"E3(w=0.1) - E1 BLEU-4: {paired['e3_minus_e1']['bleu4']:+.4f}")
PY

echo "Signer07 evaluation and paired comparison completed successfully."
