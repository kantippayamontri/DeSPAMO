"""Explicit user-authorized time policy for the existing six-run fusion study.

Checkpoint/model/data identities remain historical and exact. This execution
policy binds upgraded time-control source bytes separately and never refunds
elapsed work. Ordinary runs without the recorded authorization stay bounded.
"""

import math
import os
import re
import subprocess
import time
from pathlib import Path

from despamo.appearance.provenance import file_hash, read_json

STUDY = (Path("/home/kan/datasets/despamo/signer-pilot")
         / "time-aligned-fusion-new-motion-validated-study-v1")
RECORD = STUDY / "time-waiver.json"


def policy() -> dict | None:
    value = os.environ.get("DESPAMO_FUSION_TIME_WAIVER")
    if value is None:
        return None
    path = Path(value)
    if path != RECORD or path.resolve() != path or not path.is_file():
        raise ValueError("fixed owned time-waiver record required")
    data = read_json(path)
    if (data.get("schema_version") != 1 or data.get("policy") != "fusion-time-waiver-v1"
            or data.get("study_root") != str(STUDY)
            or data.get("authorization") != "just train and do not care the time"
            or data.get("steps") != 16000 or data.get("seeds") != [0, 1]
            or data.get("arms") != ["reference", "aligned_fixed", "aligned_learned"]
            or data.get("effective_duration_limit_seconds") is not None):
        raise ValueError("time-waiver authorization/scope mismatch")
    updates = data.get("source_updates", {})
    if set(updates) != {"scripts/train_fusion_study.py", "scripts/run_fusion_study.py",
                       "scripts/fusion_time_waiver.py"}:
        raise ValueError("time waiver permits only the time-control source upgrade")
    if updates.get("scripts/fusion_time_waiver.py") != file_hash(
            Path(__file__)):
        raise ValueError("time-waiver execution helper bytes changed")
    return data


def time_waived(identity: dict | None = None) -> bool:
    data = policy()
    if data is None:
        return False
    if identity is not None and (identity.get("initialization_profile_path") != str(
            STUDY / "common-initialization.json")
            or identity.get("motion_package", {}).get("root") != data["motion_package"]):
        raise ValueError("time waiver cannot apply to another model/data study")
    return True


def execution_fields(identity: dict | None = None) -> dict:
    if not time_waived(identity):
        return {}
    return {"execution_time_policy": "fusion-time-waiver-v1",
            "time_waiver_path": str(RECORD), "time_waiver_sha256": file_hash(RECORD),
            "effective_duration_limit_seconds": None}


def compatible_sources(actual: dict[str, str], kind: str, *, record: dict | None = None) -> dict:
    data = policy() if record is None else record
    if data is None:
        return actual
    original_runner = data.get("original_runner_sources")
    if original_runner is None:
        profile = Path(data["original_profile_path"])
        if profile != STUDY / "common-initialization.json" or file_hash(profile) != data[
                "original_profile_sha256"]:
            raise ValueError("time-waiver original initialization proof changed")
        original_runner = read_json(profile)["seeds"]["0"]["code_hashes"]
    current = data.get("current_sources", {
        **original_runner, **data.get("original_controller_sources", {}),
        **data.get("source_updates", {}),
    })
    for name, sha in actual.items():
        if current.get(name) != sha:
            raise ValueError(f"time-waiver upgraded source mismatch: {name}")
    original = original_runner if kind == "runner" else data["original_controller_sources"]
    if not set(original) <= set(actual):
        raise ValueError("time-waiver source closure incomplete")
    return dict(original)


def run_without_deadline(phase, name: str, command: list[str], validator=None):
    """Owned process recovery and monotonic charging; no duration watchdog."""
    from scripts.run_projector_overnight import command_signature, process_identity, stop_group

    if phase.active_arm not in {"reference", "aligned_fixed", "aligned_learned"}:
        raise ValueError("waived stage owner invalid")
    if phase.state["controller_sources"] != phase._sources():
        raise ValueError("waived controller source changed before stage")
    record = phase.state["stages"].setdefault(name, {
        "status": "pending", "elapsed_seconds": 0.0, "attempts": 0,
        "command_hash": command_signature(command), "owned_arm": phase.active_arm,
    })
    if record.get("owned_arm") != phase.active_arm:
        raise ValueError("waived stage owner mismatch")
    if record["command_hash"] != command_signature(command):
        raise ValueError("waived stage identity/command mismatch")
    previous = record.get("command")
    if previous is not None and previous != command:
        record.setdefault("previous_attempt_commands", []).append({
            "command": previous, "command_hash": record["command_hash"]})
    if validator is not None and validator():
        record.update(status="complete", adopted=True)
        phase.save()
        return
    if record["status"] == "complete":
        if validator is not None:
            record["status"] = "invalid"
            phase.save()
            raise RuntimeError("waived completion artifacts missing")
        return
    started = time.monotonic()
    charged = 0.0
    process = None
    rd, wr = os.pipe()
    # GNUtimeout duration zero explicitly disables its timeout while retaining
    # the stable supervisor PID/process group used by existing orphan recovery.
    guarded = ["/usr/bin/gnutimeout", "--signal=INT", "--kill-after=15s", "0",
               command[0], "-c",
               "import os,sys; fd=int(sys.argv[1]); t=os.read(fd,1); os.close(fd); "
               "sys.exit(125) if t!=b'1' else os.execvpe(sys.argv[2],sys.argv[2:],os.environ)",
               str(rd), *command]
    log = phase.root / f"{name}.log"
    try:
        record.update(status="launching", pid=None, command=command,
                      watchdog_command=guarded, boot_id=phase.boot_id,
                      last_wall=time.time(), last_monotonic=started,
                      attempts=record["attempts"] + 1, log=str(log),
                      **execution_fields())
        phase.state.update(status="running", **execution_fields())
        phase.save()
        with log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(guarded, cwd=Path(__file__).resolve().parents[1],
                                       env=phase.env, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(rd,))
            identity = process_identity(process.pid)
            while identity is not None and identity["command"] != guarded:
                time.sleep(.001)
                identity = process_identity(process.pid)
            if identity is None:
                raise RuntimeError("waived child exited before ownership publication")
            record.update(status="running", pid=process.pid, process_identity=identity)
            phase.save()
            os.write(wr, b"1")
            while process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                phase._charge(record, now - started - charged, now)
                charged = now - started
                phase.save()
            record.update(status="complete" if process.returncode == 0 else "failed",
                          exit_code=process.returncode, timed_out=False)
    except BaseException:
        record["status"] = "interrupted"
        raise
    finally:
        os.close(rd)
        os.close(wr)
        if process is not None:
            stop_group(process.pid)
            process.wait()
        now = time.monotonic()
        phase._charge(record, max(0, now - started - charged), now)
        phase.save()
    if record["status"] != "complete":
        raise RuntimeError(f"waived stage failed: {name}; see {log}")
    if validator is not None and not validator():
        record["status"] = "invalid"
        phase.save()
        raise RuntimeError("waived completion artifacts invalid")


def resume_ledger(cls, root, identity, checkpoint_seconds, checkpoint_step, monotonic, wall):
    budget = read_json(root / "budget.json")
    recorded = budget.get("gpu_seconds")
    if (budget.get("identity") != identity or type(recorded) not in (float, int)
            or not math.isfinite(recorded) or not 0 <= checkpoint_seconds <= recorded
            or type(checkpoint_step) is not int
            or not 0 <= checkpoint_step <= budget["optimizer_step"]):
        raise ValueError("waived resume identity/checkpoint/accounting mismatch")
    if budget["status"] not in ("closed", "paused"):
        elapsed = wall() - budget["wall_timestamp"]
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("waived resume wall clock moved backwards")
        recorded += elapsed
    result = cls(root, identity, recorded, checkpoint_step, monotonic, wall)
    result._write("resumed")
    return result


def validate_waived_checkpoint(path: Path, output: Path, identity: dict, spent: float) -> dict:
    """Preserve replay validation and actual accounting, without the legacy 24h cap."""
    import torch

    from despamo.training.pilot_replay import verify_resume

    candidate = re.fullmatch(r"(?:resume|step)-(\d+)\.ckpt", path.name)
    if (candidate is None or path.is_symlink() or not path.is_file()
            or path.resolve().parent != output.resolve(strict=True)):
        raise ValueError("waived checkpoint must be an owned complete snapshot")
    step = int(candidate.group(1))
    saved = [int(m.group(1)) for entry in output.glob("*.ckpt")
             if (m := re.fullmatch(r"(?:resume|step)-(\d+)\.ckpt", entry.name))]
    if not saved or step != max(saved):
        raise ValueError("waived resume requires latest complete saved checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    seconds = checkpoint.get("gpu_seconds_cumulative")
    if (type(seconds) not in (int, float) or type(spent) not in (int, float)
            or not math.isfinite(seconds) or not math.isfinite(spent)
            or not 0 <= seconds <= spent):
        raise ValueError("waived checkpoint accounting reset/invalid")
    # All non-time replay checks stay identical; actual clocks are checked above.
    verify_resume({**checkpoint, "gpu_seconds_cumulative": 0}, identity, step, 0)
    if (checkpoint.get("run_metadata") != identity
            or checkpoint["pilot_sampler"].get("batches_yielded") != step):
        raise ValueError("waived checkpoint identity/sampler mismatch")
    return checkpoint
