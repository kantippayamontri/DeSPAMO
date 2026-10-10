"""Read-only budget lineage for retrying the stopped validation-only training start."""

import math
from pathlib import Path

from despamo.appearance.provenance import file_hash, read_json
from tools.motion.recovery import _stopped

BASE = Path("/home/kan/datasets/despamo/signer-pilot")
OLD = BASE / "time-aligned-fusion-new-motion-study-v1"
NEW = BASE / "time-aligned-fusion-new-motion-validated-study-v1"


def validate_training_recovery(path: Path, root: Path, package: Path) -> dict:
    if (path != OLD / "validation-recovery.json" or root != NEW or path.resolve() != path
            or not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError("fixed training recovery record/root required")
    value = read_json(path)
    if (value.get("schema_version") != 1 or value.get("policy") != "validation-recovery-v1"
            or value.get("motion_package") != str(package)
            or value.get("budget_seconds") != 64800):
        raise ValueError("training recovery policy/input/cap mismatch")
    required = {str(OLD / "state.json"), str(OLD / "seed0/study-state.json"),
                str(OLD / "seed1/study-state.json"),
                str(BASE / "motion-fusion-pipeline-measured-profile-v1/state.json"),
                str(package / "complete/completion.json")}
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or not required <= set(artifacts):
        raise ValueError("training recovery evidence missing")
    for name, sha in artifacts.items():
        item = Path(name)
        if (not item.is_relative_to(BASE) or item.resolve() != item or item.is_symlink()
                or not item.is_file() or item.stat().st_nlink != 1 or file_hash(item) != sha):
            raise ValueError("training recovery evidence changed")
    old = read_json(OLD / "state.json")
    phase0 = read_json(OLD / "seed0/study-state.json")
    phase1 = read_json(OLD / "seed1/study-state.json")
    pipeline = read_json(BASE / "motion-fusion-pipeline-measured-profile-v1/state.json")
    if (old.get("status") != "incomplete" or phase0.get("runs") or phase1.get("runs")
            or phase1.get("spent_seconds") != 0 or pipeline.get("status") != "failed"):
        raise ValueError("only stopped setup with no training runs is recoverable")
    for state in (phase0, phase1):
        for stage in state["stages"].values():
            _stopped({"pid": stage.get("pid"), "identity": stage.get("process_identity", {})})
    _stopped(pipeline)
    amounts = [phase0["spent_seconds"], phase1["spent_seconds"],
               pipeline["clocks"]["training"]["outer_seconds"],
               pipeline["clocks"]["training"]["inner_bound_seconds"]]
    if any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in amounts):
        raise ValueError("invalid old training charges")
    credit = max(amounts[0] + amounts[1], amounts[2], amounts[3])
    if value.get("inherited_seconds") != credit or not 0 < credit < 10800:
        raise ValueError("old training setup charge cannot be refunded")
    return {"path": str(path), "sha256": file_hash(path), "inherited_seconds": credit}
