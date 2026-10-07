#!/bin/bash
# Portable copy of the approved stable-factor queue; starts after crop completion.
set -euo pipefail
REPO=${DESPAMO_ROOT:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")}
ARTIFACTS=${DESPAMO_ARTIFACT_ROOT:-$REPO/artifacts}
PY=$REPO/tools/how2sign/.venv/bin/python
CROP=$ARTIFACTS/how2sign_full_mouth_v2
STABLE=$ARTIFACTS/how2sign_stable_wholeframe_v1
VIDEOS=${HOW2SIGN_VIDEO_ROOT:-$HOME/datasets/how2sign/realign_sentence_level/train_sentence_videos}
export PYTHONPATH="$REPO/src:$REPO/tools/how2sign"
cd "$REPO"
mkdir -p "$STABLE"
echo "[stable] $(date -u) driver PID $$; waiting for verified corrected crop labels"
while [ ! -f "$CROP/driver_complete" ]; do sleep 300; done
"$PY" -m how2sign_probe.completion --root "$CROP" > "$STABLE/crop_completion_checked.json"
"$PY" -u -m how2sign_probe.stable --crop-root "$CROP" --root "$STABLE" \
    --video-root "$VIDEOS" --ports 11435 11436 --workers-per-port 1 \
    >> "$STABLE/label.log" 2>&1
OUTPUT=$STABLE/how2sign_qwen_19_targets_v1.json
if [ ! -f "$OUTPUT" ]; then
    "$PY" -m how2sign_probe.combined_export --crop-root "$CROP" \
        --stable-root "$STABLE" --output "$OUTPUT" > "$STABLE/export_report.json"
fi
(
    cd "$STABLE"
    sha256sum how2sign_qwen_19_targets_v1.json > how2sign_qwen_19_targets_v1.json.sha256
)
touch "$STABLE/handoff_complete"
echo "[stable] $(date -u) handoff complete"
