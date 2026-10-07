#!/bin/bash
# Portable copy of the active versioned mouth-repair queue.
set -euo pipefail
REPO=${DESPAMO_ROOT:-$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")}
TOOLS=$REPO/tools/how2sign
ARTIFACTS=${DESPAMO_ARTIFACT_ROOT:-$REPO/artifacts}
VIDEOS=${HOW2SIGN_VIDEO_ROOT:-$HOME/datasets/how2sign/realign_sentence_level/train_sentence_videos}
PY=$TOOLS/.venv/bin/python
cd "$TOOLS"
echo "[repair] $(date -u) driver PID $$"
for dataset in phoenix how2sign; do
    OLD=$ARTIFACTS/${dataset}_full_v1
    NEW=$ARTIFACTS/${dataset}_full_mouth_v2
    mkdir -p "$NEW"
    echo "[repair] $(date -u) $dataset: regenerate mouths, reuse completed hands"
    EXTRA=()
    if [ "$dataset" = how2sign ]; then EXTRA=(--video-root "$VIDEOS"); fi
    "$PY" -u -m how2sign_probe.repair --source-root "$OLD" --output-root "$NEW" \
        "${EXTRA[@]}" >> "$NEW/repair.log" 2>&1
    "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["gate"]["pass"] else 2)' \
        "$NEW/detector_report.json"
    "$PY" -c 'import urllib.request; [urllib.request.urlopen("http://127.0.0.1:%d/api/tags"%p, timeout=30).close() for p in (11435,11436)]'
    "$PY" -u -m how2sign_probe.cli --root "$NEW" label --ports 11435 11436 \
        --workers-per-port 2 >> "$NEW/label.log" 2>&1
    "$PY" -m how2sign_probe.cli --root "$NEW" report > "$NEW/report_stdout.txt" 2>&1
    "$PY" -m how2sign_probe.completion --root "$NEW" > "$NEW/completion_verified.json"
    touch "$NEW/driver_complete"
    echo "[repair] $(date -u) $dataset: complete, expected jobs verified"
done
