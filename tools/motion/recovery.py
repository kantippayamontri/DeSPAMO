"""Read-only recovery authorization for the forecast-rejected, stopped attempt."""

import math
from pathlib import Path

from despamo.appearance.provenance import file_hash, read_json

PILOT = Path("/home/kan/datasets/despamo/signer-pilot")
PIPELINE_NAME = "motion-fusion-pipeline-measured-profile-v1"
BOOTSTRAP_NAME = "motion-extraction-bootstrap-measured-profile-v1"
AUTHORIZATION = {"extraction_cap_seconds": 18000, "training_cap_seconds": 64800,
                 "cap_transfer": False}


def _number(value: object) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        raise ValueError("invalid recovery credit")
    return float(value)


def _stopped(value: object) -> None:
    if isinstance(value, dict):
        pid = value.get("pid")
        identity = value.get("identity", value)
        if type(pid) is int and isinstance(identity, dict) and identity.get("start_ticks"):
            path = Path(f"/proc/{pid}/stat")
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                fields = []
            if fields and fields[0] != "Z" and fields[19] == str(identity["start_ticks"]):
                raise ValueError("previous recovery process still active")
        for item in value.values():
            _stopped(item)
    elif isinstance(value, list):
        for item in value:
            _stopped(item)


def validate_recovery_record(path: Path, *, pilot: Path = PILOT) -> dict:
    old = pilot / "motion-fusion-pipeline-v1"
    boot = pilot / "motion-extraction-bootstrap-v1"
    expected_path = old / "measured-profile-recovery.json"
    if path != expected_path or path.resolve() != path or not path.is_file():
        raise ValueError("fixed regular recovery record required")
    record = read_json(path)
    if (record.get("schema_version") != 1
            or record.get("policy") != "measured-profile-recovery-v1"
            or record.get("authorization") != AUTHORIZATION):
        raise ValueError("recovery policy/authorization mismatch")
    artifacts = record.get("artifacts")
    required = {str(old / "state.json"), str(boot / "budget.json"), str(boot / "status.json")}
    if not isinstance(artifacts, dict) or not required <= set(artifacts):
        raise ValueError("recovery evidence missing")
    for name, sha in artifacts.items():
        item = Path(name)
        if (not (item.is_relative_to(old) or item.is_relative_to(boot)
                 or item.parent.parent == pilot and item.parent.name.startswith("motion-features-"))
                or item.resolve() != item or item.is_symlink() or not item.is_file()
                or item.stat().st_nlink != 1 or file_hash(item) != sha):
            raise ValueError("recovery evidence bytes/path changed")
        if item.suffix == ".json":
            _stopped(read_json(item))
    state, budget, status = (read_json(old / "state.json"), read_json(boot / "budget.json"),
                             read_json(boot / "status.json"))
    # A failed coordinator retains its last anchor for accounting; the verified
    # process receipts above must be quiescent, not artificially cleared.
    if (state.get("status") != "failed"
            or budget.get("status") != "capped" or budget.get("active") is not None
            or budget.get("child") is not None
            or budget.get("authorization") != AUTHORIZATION
            or state["identity"].get("authorization") != AUTHORIZATION
            or status.get("error") != (
                "TimeoutError: profile projects beyond remaining extraction cap")):
        raise ValueError("only the stopped forecast rejection is recoverable")
    active = state.get("active")
    if active is not None and (not isinstance(active, dict)
                              or active.get("bucket") != "extraction"
                              or not isinstance(active.get("owner"), dict)):
        raise ValueError("invalid previous active accounting anchor")
    clocks = state["clocks"]
    extraction = max(_number(clocks["extraction"]["outer_seconds"]),
                     _number(clocks["extraction"]["inner_bound_seconds"]),
                     _number(budget["spent_seconds"]))
    training = max(_number(clocks["training"]["outer_seconds"]),
                   _number(clocks["training"]["inner_bound_seconds"]))
    if (training != 0 or extraction >= 18000 - 30
            or record.get("credit") != {"extraction": extraction, "training": training}):
        raise ValueError("recovery credit cannot reset or transfer allowance")
    return {"path": str(path), "sha256": file_hash(path), "credit": record["credit"]}


def recovery_for(args) -> dict | None:
    path = getattr(args, "recovery_record", None)
    return validate_recovery_record(Path(path)) if path is not None else None


def pipeline_bootstrap(identity: dict) -> Path:
    name = BOOTSTRAP_NAME if identity.get("recovery") is not None else (
        "motion-extraction-bootstrap-v1")
    return Path(identity["paths"]["output_base"]) / name
