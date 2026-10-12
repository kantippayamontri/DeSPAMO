"""Budgeted, sequential held-out comparison and paired seed-1 robustness workflow."""

import argparse
import ast
import fcntl
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/home/kan/datasets/despamo/signer-pilot")
PYTHON = ROOT / ".venv/bin/python"
STUDY_POLICY = "projector-paired-seed-heldout-study-v1"
DATASET = Path("/home/kan/datasets/despamo/appearance/datasets/") / (
    "b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a")
PROTOCOL_KEY = "0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7"
TEXT_KEY = "24ad915ab5bde47335bfc8839f654b0a54ad66ac1253fc4218e1c415951fa571"
ORIGINAL_KEY = "1e84e43d142abc6242cce75c3e65abe9e1e03a6c61e3770181d7be381f3bc31d"
ADAPTED_KEY = "7122b42e35510611626c2a71d0731b79dd79b20de6e89bd49f62737c5c84a18e"
INPUTS = {
    "protocol": BASE / "protocol" / PROTOCOL_KEY / "protocol.json",
    "dataset": DATASET,
    "text-manifest": DATASET / "text" / TEXT_KEY / "manifest.json",
    "dino-root": Path("/home/kan/datasets/despamo/features") / ORIGINAL_KEY,
    "motion-root": Path("/home/kan/datasets/spamo/features/mae_feat_Phoenix14T"),
    "motion-manifest": Path("/home/kan/datasets/spamo/features/manifests/phoenix14t_motion.json"),
    "annotation": Path("/home/kan/Research/SpaMo/preprocess/Phoenix14T/train_info_ml.npy"),
    "adapted-root": BASE / f"e3-features-{ADAPTED_KEY}",
}
REFERENCE = BASE / ("e3-only-" "0402717f5ef91cca106caa881b689ce1eba75a3a87976302c99c654bea189690")
SEED0 = {
    "linear": BASE / ("projector-linear-16000-"
                      "eb0a5bef8210afc3d894f4ac57338f8cd5e0f01b6b4d74a8a38cb285b4f5f313"),
    "residual": BASE / ("projector-residual-16000-"
                        "f8deb3d4b807d59f3e0df2e6e6253ec069fb01d1a3365c36d4c257422d87526d"),
}


def stage_allowance(spent: float, total: float, stage_cap: float, stage_spent: float) -> float:
    values = (spent, total, stage_cap, stage_spent)
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
            or spent < 0 or stage_spent < 0 or not 0 < total <= 36000 or stage_cap <= 0):
        raise ValueError("invalid overnight budget")
    available = min(total - spent, stage_cap - stage_spent)
    if available <= 0:
        raise RuntimeError("overnight overall/stage budget exhausted")
    return available


def baseline_decision(analyses: list[dict]) -> dict:
    by_seed = {a["training_seed"]: a for a in analyses}
    if len(analyses) != 2 or set(by_seed) != {0, 1}:
        raise ValueError("baseline decision requires both paired seeds")
    deltas = [by_seed[s]["metric_deltas"]["bleu4"] for s in (0, 1)]
    if not all(isinstance(d, (float, int)) and math.isfinite(d) for d in deltas):
        raise ValueError("baseline decision requires finite held-out deltas")
    promote = deltas[0] > 0
    return {
        "projector": "residual" if promote else "linear", "steps": 16000, "default_seed": 0,
        "status": ("replicated_two_seeds" if promote and deltas[1] > 0 else
                   "provisional_mixed_robustness" if promote else "primary_gain_did_not_transfer"),
        "heldout_bleu4_deltas": deltas, "mean_delta": statistics.mean(deltas),
        "sample_std_delta": statistics.stdev(deltas),
        "rule": "Prespecified residual candidate promoted only on positive seed0 transfer; "
                "seed1 qualifies robustness. No test-based checkpoint/seed selection.",
        "strong_research_claim": False,
    }


def command_signature(command: list[str]) -> str:
    # Recovery changes checkpoint pointer, not the declared model/data/budget recipe.
    normalized = []
    index = 0
    while index < len(command):
        if command[index] == "--resume-checkpoint":
            if index + 1 >= len(command):
                raise ValueError("resume checkpoint argument missing")
            index += 2
        else:
            normalized.append(command[index])
            index += 1
    return digest(normalized)


def process_identity(pid: int) -> dict | None:
    """Linux PID reuse protection; zombies no longer consume child-stage wall time."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    except FileNotFoundError:
        return None
    if fields[0] == "Z":
        return None
    return {"start_ticks": fields[19], "pgid": int(fields[2]), "session": int(fields[3]),
            "command": [os.fsdecode(arg) for arg in command]}


def group_active(pgid: int) -> bool:
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            try:
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                continue
            if fields[0] != "Z" and int(fields[2]) == pgid:
                return True
    return False


def signal_group(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


@contextmanager
def termination_guard():
    # Ignore repeated termination process-wide: a thread-only signal mask can
    # still deliver Python handlers via another unmasked interpreter thread.
    previous = {signum: signal.signal(signum, signal.SIG_IGN)
                for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def stop_group(pid: int, grace: float = 0) -> None:
    signal_group(pid, signal.SIGINT)
    until = time.monotonic() + grace
    while group_active(pid) and time.monotonic() < until:
        time.sleep(min(0.05, max(0, until - time.monotonic())))
    if group_active(pid):
        signal_group(pid, signal.SIGKILL)
    # SIGKILL delivery/reaping can be delayed by the kernel. Never start another
    # child while a surviving group is still consuming the authorized allowance.
    while group_active(pid):
        time.sleep(0.01)


def watchdog_command(command: list[str], available: float, gate: int) -> list[str]:
    timeout = shutil.which("gnutimeout") or shutil.which("timeout")
    if timeout is None:
        raise RuntimeError("GNU timeout watchdog required")
    version = subprocess.run([timeout, "--version"], capture_output=True, text=True,
                             check=True, timeout=5).stdout
    if "GNU coreutils" not in version:
        raise RuntimeError("GNU timeout watchdog required")
    reserve = min(1.0, available / 10)
    grace = min(15.0, (available - reserve) / 2)
    soft = available - reserve - grace
    if soft <= 0:
        raise RuntimeError("overnight budget lacks watchdog startup reserve")
    # EOF before durable PID ownership means no GPU command may start. timeout
    # survives controller SIGKILL and bounds both this gate and the real child.
    gate_code = ("import os, sys; fd=int(sys.argv[1]); token=os.read(fd, 1); os.close(fd); "
                 "sys.exit(125) if token != b'1' else os.execvpe(sys.argv[2], "
                 "sys.argv[2:], os.environ)")
    return [timeout, "--signal=INT", f"--kill-after={grace:.6f}s", f"{soft:.6f}s",
            sys.executable, "-c", gate_code, str(gate), *command]


class Study:
    def __init__(self, root: Path, budget: int):
        if type(budget) is not int or not 0 < budget <= 36000:
            raise ValueError("study budget exceeds user authorization")
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = (root / ".study.lock").open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._load(budget)
        except BaseException:
            self.close()
            raise

    def _load(self, budget: int):
        root = self.root
        self.path = root / "study-state.json"
        self.state = read_json(self.path) if self.path.exists() else {
            "study_policy": STUDY_POLICY, "budget_seconds": budget, "spent_seconds": 0.0,
            "status": "created", "stages": {},
            "runs": {"seed0": {k: str(v) for k, v in SEED0.items()}},
        }
        if self.state.get("study_policy") != STUDY_POLICY or self.state["budget_seconds"] != budget:
            raise ValueError("study identity/budget mismatch")
        self.env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", "OMP_NUM_THREADS": "2",
                    "HF_HUB_OFFLINE": "1",
                    "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        try:
            for name, record in self.state["stages"].items():
                if record["status"] in {"running", "launching"}:
                    self._recover(name, record)
        except BaseException as error:
            self.state.update(status="incomplete", error=type(error).__name__,
                              error_message=str(error))
            self.save()
            raise
        self.save()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.lock.close()

    def save(self):
        if self.lock.closed:
            raise RuntimeError("study lock must be owned before writes")
        self.state["updated_at"] = time.time()
        atomic_json(self.path, self.state)

    def _charge(self, record: dict, elapsed: float, clock: float):
        self.state["spent_seconds"] += elapsed
        record["elapsed_seconds"] += elapsed
        record["reserved_seconds"] = max(0, record.get("reserved_seconds", 0) - elapsed)
        record.update(last_wall=time.time(), last_monotonic=clock, boot_id=self.boot_id)

    def _recover(self, name: str, record: dict):
        pid = record.get("pid")
        current = process_identity(pid) if pid else None
        try:
            if pid and current is not None:
                expected = record.get("process_identity")
                if (expected is None and "watchdog_command" not in record
                        and "boot_id" not in record):
                    # Original controller had no watchdog/start-tick receipt. Admit
                    # its exact command signature only if this session provably
                    # predates the heartbeat (btime has one-second resolution).
                    boot_wall = next(int(line.split()[1]) for line in
                                     Path("/proc/stat").read_text().splitlines()
                                     if line.startswith("btime "))
                    started_wall = (boot_wall + int(current["start_ticks"])
                                    / os.sysconf("SC_CLK_TCK"))
                    verified = (command_signature(current["command"]) == record["command_hash"]
                                and started_wall + 1 <= record["last_wall"])
                else:
                    verified = (record.get("boot_id") == self.boot_id and current == expected
                                and current["command"] == record.get("watchdog_command")
                                and command_signature(record.get("command", []))
                                == record["command_hash"])
                if not verified or current["pgid"] != pid or current["session"] != pid:
                    raise RuntimeError(f"stage child PID/command identity mismatch: {name}")
                # Stop an owned orphan before inspecting snapshots or starting any
                # preflight. A resumed trainer then owns the newest durable snapshot.
                with termination_guard():
                    stop_group(pid)
            elif pid and group_active(pid):
                raise RuntimeError(f"stage group survives without verifiable PID identity: {name}")
        finally:
            clock = time.monotonic()
            if record.get("boot_id") == self.boot_id and "last_monotonic" in record:
                elapsed = max(0, clock - record["last_monotonic"])
            else:
                elapsed = max(0, time.time() - record["last_wall"],
                              record.get("reserved_seconds", 0))
            # No durable end receipt: include downtime conservatively, never clamp.
            self._charge(record, elapsed, clock)
            self.save()
        record.update(status="interrupted", recovered=True)
        self.save()

    def run_stage(self, name: str, command: list[str], cap: int,
                  completion_validator: Callable[[], bool] | None = None):
        stages = self.state["stages"]
        record = stages.setdefault(name, {
            "status": "pending", "elapsed_seconds": 0.0,
            "command_hash": command_signature(command), "attempts": 0,
        })
        if record["command_hash"] != command_signature(command):
            raise ValueError(f"stage command changed: {name}")
        if (self.state["spent_seconds"] > self.state["budget_seconds"]
                or record["elapsed_seconds"] > cap):
            raise RuntimeError("overnight overall/stage budget overrun")
        if completion_validator is not None:
            try:
                complete = completion_validator()
                if record["status"] == "complete" and not complete:
                    raise RuntimeError(f"stage completion artifacts missing: {name}")
            except BaseException:
                record["status"] = "invalid"
                self.save()
                raise
            if complete:
                record.update(status="complete", adopted=True)
                self.save()
                return
        if record["status"] == "complete":
            return
        available = stage_allowance(self.state["spent_seconds"], self.state["budget_seconds"],
                                    cap, record["elapsed_seconds"])
        log_path = self.root / f"{name}.log"
        read_gate, write_gate = os.pipe()
        process = None
        start = time.monotonic()
        charged = 0.0
        try:
            # Include watchdog verification/setup conservatively. The launch record
            # is durable before Popen, and the GPU gate opens only after PID save.
            guarded = watchdog_command(command, available, read_gate)
            record.update(status="launching", pid=None, last_wall=time.time(),
                          last_monotonic=start, boot_id=self.boot_id,
                          reserved_seconds=available, command=command, watchdog_command=guarded,
                          attempts=record["attempts"] + 1, log=str(log_path))
            self.state["status"] = "running"
            self.save()
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(guarded, cwd=ROOT, env=self.env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           pass_fds=(read_gate,))
                record["pid"] = process.pid
                identity = process_identity(process.pid)
                while identity is not None and identity["command"] != guarded:
                    if time.monotonic() - start >= min(1, available / 10):
                        raise RuntimeError("watchdog exec identity not established")
                    time.sleep(0.001)
                    identity = process_identity(process.pid)
                if identity is None:
                    raise RuntimeError("watchdog exited before durable child ownership")
                record.update(status="running", process_identity=identity)
                self.save()
                if time.monotonic() - start >= min(1, available / 10):
                    raise RuntimeError("watchdog startup exhausted deadline reserve")
                os.write(write_gate, b"1")
                while process.poll() is None:
                    try:
                        process.wait(timeout=min(10, max(0.001, available -
                                                       (time.monotonic() - start))))
                    except subprocess.TimeoutExpired:
                        if time.monotonic() - start >= available:
                            stop_group(process.pid)
                    clock = time.monotonic()
                    now = clock - start
                    self._charge(record, now - charged, clock)
                    charged = now
                    self.save()
                process.wait()
                if group_active(process.pid):
                    stop_group(process.pid)
            elapsed = time.monotonic() - start
            timed_out = process.returncode in {124, 137, -signal.SIGKILL} or elapsed > available
            record.update(status="complete" if process.returncode == 0 and not timed_out
                          else "failed", exit_code=process.returncode, timed_out=timed_out)
        except BaseException:
            with termination_guard():
                if process is not None:
                    grace = min(15, max(0, start + available - min(1, available / 10)
                                        - time.monotonic()))
                    stop_group(process.pid, grace)
                    process.wait()
                record["status"] = "interrupted"
            raise
        finally:
            os.close(read_gate)
            os.close(write_gate)
            clock = time.monotonic()
            self._charge(record, max(0, clock - start - charged), clock)
            if record["status"] == "complete" and (
                    self.state["spent_seconds"] > self.state["budget_seconds"]
                    or record["elapsed_seconds"] > cap):
                record.update(status="failed", timed_out=True)
            self.save()
        if record["status"] != "complete":
            raise RuntimeError(f"stage failed: {name}; see {log_path}")
        if completion_validator is not None:
            try:
                if not completion_validator():
                    raise RuntimeError(f"stage completion artifacts missing: {name}")
            except BaseException:
                record["status"] = "invalid"
                self.save()
                raise

    def cpu_query(self, command: list[str]) -> dict:
        # Read-only queries are child wall time too, including recovery/validation.
        sequence = self.state.get("query_sequence", 0) + 1
        self.state["query_sequence"] = sequence
        name = f"preflight-{sequence}"
        self.run_stage(name, command, 300)
        for line in reversed((self.root / f"{name}.log").read_text().splitlines()):
            try:
                value = json.loads(line)
            except (ValueError, SyntaxError):
                try:
                    value = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue
            if isinstance(value, dict):
                return value
        raise ValueError("preflight returned no structured identity")


def common_arguments() -> list[str]:
    return [value for key, path in INPUTS.items() for value in ("--" + key, str(path))]


def training_command(kind: str, profile: Path, control: Path | None, mode="run") -> list[str]:
    command = [str(PYTHON), str(ROOT / "scripts/train_projector_pilot.py"), *common_arguments(),
               "--baseline-run", str(REFERENCE), "--output-base", str(BASE),
               "--mode", mode, "--projector", kind, "--steps", "16000", "--seed", "1",
               "--gpu-cap-seconds", "14400", "--initialization-profile", str(profile),
               "--authorize-projector-run"]
    if control is not None:
        command.extend(("--matched-control-run", str(control)))
    return command


def training_preflight_command(kind: str, profile: Path, control: Path | None) -> list[str]:
    # The public CLI prints only part of the identity. Capture the exact identity
    # from its existing preparation API without constructing/loading a model.
    code = """from scripts import train_projector_pilot as trainer
ready = {}
prepare = trainer.prepare_projector
def capture(args):
    result = prepare(args)
    ready.update(identity=result[0], output=str(result[1]), run_key=result[1].name)
    return result
trainer.prepare_projector = capture
trainer.main()
print(ready)
"""
    command = training_command(kind, profile, control, "preflight")
    return [command[0], "-c", code, *command[2:]]


def validate_training_completion(study: Study, run: Path, expected: dict) -> bool:
    selected = run / "selected-checkpoint.json"
    status_path = run / "run-status.json"
    status = read_json(status_path) if status_path.exists() else {}
    if not selected.exists() and status.get("status") != "complete":
        return False
    code = """import json, sys
from pathlib import Path
from despamo.appearance.provenance import digest, read_json
from scripts.evaluate_projector_pilot import validate_projector_selection
protocol = read_json(Path(sys.argv[2]))
selection = validate_projector_selection(Path(sys.argv[1]), protocol,
    tuple(protocol['split']['groups']['dev']['clip_ids']))
if digest(selection['identity']) != digest(json.loads(sys.argv[3])):
    raise ValueError('completed projector full training identity mismatch')
print({'status': 'complete'})
"""
    result = study.cpu_query([str(PYTHON), "-c", code, str(run), str(INPUTS["protocol"]),
                              json.dumps(expected, sort_keys=True)])
    if result.get("status") != "complete":
        raise ValueError("completed projector selection validation failed")
    return True


def evaluation_command(run: Path, output: Path, registry: Path) -> list[str]:
    return [str(PYTHON), str(ROOT / "scripts/evaluate_projector_pilot.py"),
            *common_arguments(), "--mode", "run", "--projector-run", str(run),
            "--output", str(output), "--registry", str(registry), "--gpu-cap-seconds", "1800"]


def validate_evaluation_completion(study: Study, command: list[str]) -> bool:
    preflight = list(command)
    preflight[preflight.index("--mode") + 1] = "preflight"
    ready = study.cpu_query(preflight)
    if ready.get("status") != "ready":
        raise ValueError("held-out evaluation preflight validation failed")
    return ready.get("registry_status") == "complete"


def run_study(study: Study):
    reports, analyses = {}, []
    registry = BASE / "projector-heldout-registry-v1.json"
    for kind, run in SEED0.items():
        output = study.root / f"seed0-{kind}-signer07.json"
        command = evaluation_command(run, output, registry)
        study.run_stage(f"seed0-{kind}-heldout", command, 1800,
                        completion_validator=partial(validate_evaluation_completion,
                                                     study, command))
        reports[f"seed0-{kind}"] = output
    profile = study.root / "seed1-initialization.json"
    study.run_stage("seed1-initialization",
                    training_command("linear", profile, None, "initialize"), 600)
    seed1 = {}
    for kind in ("linear", "residual"):
        control = seed1.get("linear") if kind == "residual" else None
        ready = study.cpu_query(training_preflight_command(kind, profile, control))
        run = Path(ready["output"])
        expected = ready["identity"]
        seed1[kind] = run
        study.state["runs"]["seed1"] = {k: str(v) for k, v in seed1.items()}
        study.save()
        command = training_command(kind, profile, control)
        completed = validate_training_completion(study, run, expected)
        if run.exists() and not completed:
            snapshots = [(int(match.group(1)), p) for p in run.glob("*.ckpt")
                         if (match := re.fullmatch(r"(?:step|resume)-(\d+)\.ckpt", p.name))]
            if not snapshots:
                raise RuntimeError("partial seed1 run has no recoverable checkpoint")
            command.extend(("--resume-checkpoint", str(max(snapshots)[1])))
        study.run_stage(f"seed1-{kind}-train", command, 14400,
                        completion_validator=partial(validate_training_completion,
                                                     study, run, expected))
        output = study.root / f"seed1-{kind}-signer07.json"
        evaluation = evaluation_command(run, output, registry)
        study.run_stage(f"seed1-{kind}-heldout", evaluation, 1800,
                        completion_validator=partial(validate_evaluation_completion,
                                                     study, evaluation))
        reports[f"seed1-{kind}"] = output
    from scripts.analyze_projector_study import compare_projector_reports

    # Even an already-complete study must revalidate registry, report bytes,
    # selected checkpoint and source identity before regenerating its analysis.
    for seed, runs in ((0, SEED0), (1, seed1)):
        for kind, run in runs.items():
            command = evaluation_command(run, reports[f"seed{seed}-{kind}"], registry)
            if not validate_evaluation_completion(study, command):
                raise RuntimeError("held-out registry incomplete before analysis")
    for seed in (0, 1):
        linear, residual = (read_json(reports[f"seed{seed}-{kind}"])
                            for kind in ("linear", "residual"))
        analysis = compare_projector_reports(linear, residual, bootstrap_replicates=1000, seed=0)
        analysis["source_files"] = {kind: {"path": str(reports[f"seed{seed}-{kind}"]),
                                          "sha256": file_hash(reports[f"seed{seed}-{kind}"])}
                                    for kind in ("linear", "residual")}
        atomic_json(study.root / f"seed{seed}-comparison.json", analysis)
        analyses.append(analysis)
    decision = baseline_decision(analyses)
    summary = {
        "study_policy": STUDY_POLICY, "status": "complete", "training_seeds": [0, 1],
        "budget_seconds": study.state["budget_seconds"],
        "spent_gpu_stage_seconds": study.state["spent_seconds"],
        "runs": study.state["runs"], "report_paths": {k: str(v) for k, v in reports.items()},
        "paired_bleu4_deltas": decision["heldout_bleu4_deltas"], "baseline_decision": decision,
        "paired_analysis_paths": [str(study.root / f"seed{seed}-comparison.json")
                                  for seed in (0, 1)],
        "limitations": [
            "Two seeds, one held-out signer; no broad significance or signer-invariance claim.",
            "Weather categories are lexical-rule diagnostics, not human/visual ratings.",
            "New descriptions are not inputs; baseline prepared for their future version.",
        ],
    }
    atomic_json(study.root / "summary.json", summary)
    baseline = {
        **decision, "training": {"steps": 16000, "seed": 0, "batch_size": 4,
            "vt_warmup_steps": 2000, "lr_warmup_steps": 1600, "adapter_lr": 3e-4,
            "flan_lora_lr": 5e-5, "weight_decay": 0.01, "cosine_decay": True,
            "resume_retention": 2,
            "residual_hidden_dim": 1024 if decision["projector"] == "residual" else None},
        "validation_report": str(study.root / "summary.json"),
        "supervision_policy": "qwen-schema98-unreviewed-v1", "human_review_status": "not_assessed",
        "new_description_status": "awaiting_external_versioned_artifacts",
    }
    atomic_json(study.root / "future-description-baseline.json", baseline)
    study.state["status"] = "complete"
    study.state["summary"] = str(study.root / "summary.json")
    study.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=BASE / "projector-overnight-study-v1")
    parser.add_argument("--budget-seconds", type=int, default=36000)
    args = parser.parse_args()
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    old_term = signal.signal(signal.SIGTERM, terminate)
    try:
        with Study(args.study_root, args.budget_seconds) as study:
            try:
                run_study(study)
            except BaseException as error:
                study.state.update(status="incomplete", error=type(error).__name__,
                                   error_message=str(error))
                study.save()
                raise
            print(json.dumps({"status": "complete", "summary": study.state["summary"]}))
    finally:
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    main()
