"""Six dev-only connector ablations with reused durable watchdog accounting."""

import argparse
import fcntl
import json
import math
import re
import signal
import statistics
from pathlib import Path

from despamo.appearance.provenance import atomic_json, file_hash, read_json
from scripts.run_projector_overnight import BASE, INPUTS, PYTHON, REFERENCE, ROOT, Study

POLICY = "three-arm-connector-controller-v1"
ARMS = ("reference", "plain_mlp", "qformer")
PHASE_CAP = 32400
RUN_CAP = 10800


class ConnectorPhase(Study):
    """Two independently watchdog-bounded nine-hour phases cannot exceed eighteen hours."""

    def __init__(self, root: Path, seed: int):
        if type(seed) is not int or seed not in (0, 1):
            raise ValueError("connector phase seed invalid")
        self.connector_seed = seed
        try:
            super().__init__(root, PHASE_CAP)
            if self.state.get("connector_policy", POLICY) != POLICY:
                raise ValueError("connector controller policy mismatch")
            if self.state.get("connector_seed", seed) != seed:
                raise ValueError("connector phase seed mismatch")
            self.state.update(connector_policy=POLICY, connector_seed=seed)
            self.save()
        except BaseException:
            if hasattr(self, "lock") and not self.lock.closed:
                self.close()
            raise


def common_arguments() -> list[str]:
    return [value for key, path in INPUTS.items() for value in ("--" + key, str(path))]


def stage_command(arm: str, seed: int, profile: Path, mode: str) -> list[str]:
    if arm not in ARMS or seed not in (0, 1) or mode not in ("initialize", "preflight", "run"):
        raise ValueError("connector command arm/seed/mode invalid")
    return [str(PYTHON), str(ROOT / "scripts/train_connector_study.py"),
            *common_arguments(), "--mode", mode, "--arm", arm, "--seed", str(seed),
            "--steps", "16000", "--gpu-cap-seconds", str(RUN_CAP),
            "--initialization-profile", str(profile), "--baseline-run", str(REFERENCE),
            "--output-base", str(BASE), "--authorize-connector-run"]


def admission_smoke_command(arm: str, profile: Path, output: Path) -> list[str]:
    code = """import sys,json
from pathlib import Path
from despamo.appearance.provenance import atomic_json,digest,read_json
from despamo.data.batch import collate_phoenix
from despamo.training.pilot_factory import shared_tensor_hash
from scripts import train_connector_study as runner
out=Path(sys.argv.pop(1));prepare=runner.prepare_connector
def smoke(args):
    result=prepare(args);identity,run,protocol,views,config=result
    if out.exists():
        saved=read_json(out)
        if saved['identity_hash'] != digest(identity) or saved['status'] != 'passed':
            raise ValueError('admission smoke identity mismatch')
        return result
    model=runner.build_connector_model(config,args.arm,args.seed,identity['initial_base_hash'])
    runner.validate_effective_optimizer(model)
    before=shared_tensor_hash(model);model.to('cuda')
    batch=collate_phoenix([views['train'][i] for i in range(4)])
    receipt=runner.connector_gradient_smoke(model,batch,args.arm)
    if shared_tensor_hash(model) != before:raise RuntimeError('admission smoke mutated weights')
    atomic_json(out,{**receipt,'identity_hash':digest(identity),'arm':args.arm,'seed':args.seed,
                    'variant_hash':before,'clip_ids_hash':digest(batch.clip_ids)})
    return result
runner.prepare_connector=smoke
runner.main()
"""
    command = stage_command(arm, 0, profile, "preflight")
    return [command[0], "-c", code, str(output), *command[2:]]


def validate_completion(phase: Study, run: Path, expected: dict) -> bool:
    status = read_json(run / "run-status.json") if (run / "run-status.json").exists() else {}
    if not (run / "selected-checkpoint.json").exists() and status.get("status") != "complete":
        return False
    code = """import sys,json
from pathlib import Path
from despamo.appearance.provenance import read_json
from scripts.train_connector_study import validate_connector_selection
p=read_json(Path(sys.argv[2]))
s=validate_connector_selection(Path(sys.argv[1]),p,tuple(p['split']['groups']['dev']['clip_ids']),json.loads(sys.argv[3]))
print(json.dumps({'status':'complete','checkpoint_hash':s['checkpoint_hash']}))
"""
    result = phase.cpu_query([str(PYTHON), "-c", code, str(run), str(INPUTS["protocol"]),
                              json.dumps(expected, sort_keys=True)])
    if result.get("status") != "complete":
        raise ValueError("connector completion validation failed")
    return True


def prediction_diagnostics(items: list[dict]) -> dict:
    repeated = 0
    for item in items:
        words = item["prediction"].casefold().split()
        grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
        repeated += len(grams) != len(set(grams))
    return {"clips": len(items), "repeated_4gram_outputs": repeated,
            "unique_predictions": len({item["prediction"] for item in items}),
            "mean_whitespace_words": statistics.mean(len(i["prediction"].split()) for i in items)}


def collect_result(seed: int, arm: str, run: Path) -> dict:
    selected = read_json(run / "selected-checkpoint.json")
    report = read_json(run / f"dev-{selected['step']}.json")
    budget = read_json(run / "budget.json")
    initialization = read_json(run / "initialization.json")
    smoke = read_json(run / "gradient-smoke.json")
    return {
        "seed": seed, "arm": arm, "run_key": run.name, "run": str(run),
        "selected_step": selected["step"], "checkpoint_hash": selected["checkpoint_hash"],
        "metrics": selected["metrics"],
        "prediction_diagnostics": prediction_diagnostics(report["items"]),
        "gpu_seconds": budget["gpu_seconds"], "initialization": initialization,
        "gradient_smoke": smoke,
        "selected_dev_report_hash": file_hash(run / f"dev-{selected['step']}.json"),
        "dev_trajectory": [
            {"step": step, "bleu4": read_json(run / f"dev-{step}.json")["metrics"]["bleu4"]}
            for step in (5500, 10400, 16000)
        ],
    }


def aggregate_connector_results(results: list[dict]) -> dict:
    pairs = {(r["seed"], r["arm"]): r for r in results}
    if len(results) != 6 or set(pairs) != {(seed, arm) for seed in (0, 1) for arm in ARMS}:
        raise ValueError("connector summary requires exactly three arms and both seeds")
    arms = {}
    for arm in ARMS:
        values = [pairs[seed, arm]["metrics"]["bleu4"] for seed in (0, 1)]
        deltas = [values[seed] - pairs[seed, "reference"]["metrics"]["bleu4"] for seed in (0, 1)]
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values + deltas):
            raise ValueError("connector summary scores must be finite")
        arms[arm] = {
            "dev_bleu4": values, "mean_dev_bleu4": statistics.mean(values),
            "sample_std_dev_bleu4": statistics.stdev(values), "paired_bleu4_deltas": deltas,
            "mean_paired_bleu4_delta": statistics.mean(deltas),
            "improved_both_seeds": all(delta > 0 for delta in deltas),
        }
        metric_keys = set(pairs[0, "reference"]["metrics"])
        if any(set(pairs[seed, arm]["metrics"]) != metric_keys
               or set(pairs[seed, "reference"]["metrics"]) != metric_keys for seed in (0, 1)):
            raise ValueError("connector metric inventory differs across paired runs")
        arms[arm]["metric_deltas"] = {
            key: [pairs[seed, arm]["metrics"][key] - pairs[seed, "reference"]["metrics"][key]
                  for seed in (0, 1)] for key in sorted(metric_keys)
        }
    return {"study_policy": POLICY, "status": "complete", "comparison_scope": "dev_only",
            "signer07_scored": False, "training_seeds": [0, 1], "arms": arms, "results": results,
            "limitations": ["Only two training seeds and one dev signer.",
                            "Q-Former changes token count, normalization and temporal attention.",
                            "No test-set generalization or signer-invariance claim."]}


def run_phase(phase: ConnectorPhase, profile: Path) -> list[dict]:
    if phase.connector_seed == 0:
        phase.run_stage("common-initialization",
                        stage_command("reference", 0, profile, "initialize"), 600)
        for arm in ARMS:
            phase.run_stage(
                f"{arm}-admission-smoke",
                admission_smoke_command(arm, profile, phase.root / f"{arm}-admission-smoke.json"),
                300,
            )
    elif not profile.is_file():
        raise ValueError("common initialization missing before second seed")
    results = []
    for arm in ARMS:
        ready = phase.cpu_query(stage_command(arm, phase.connector_seed, profile, "preflight"))
        if ready.get("status") != "ready":
            raise ValueError("connector preflight failed")
        identity, run = ready["identity"], Path(ready["output"])
        phase.state.setdefault("connector_runs", {})[arm] = str(run)
        phase.save()
        command = stage_command(arm, phase.connector_seed, profile, "run")
        if run.is_dir() and not (run / "selected-checkpoint.json").exists():
            candidates = [(int(match.group(1)), p) for p in run.glob("*.ckpt")
                          if (match := re.fullmatch(r"(?:step|resume)-(\d+)\.ckpt", p.name))]
            if not candidates:
                raise RuntimeError("partial connector run has no durable recovery checkpoint")
            command.extend(("--resume-checkpoint", str(max(candidates)[1])))
        phase.run_stage(
            f"{arm}-train", command, RUN_CAP,
            completion_validator=lambda run=run, identity=identity:
                validate_completion(phase, run, identity),
        )
        result = collect_result(phase.connector_seed, arm, run)
        results.append(result)
        atomic_json(phase.root / f"{arm}-result.json", result)
    phase.state["status"] = "complete"
    phase.save()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=BASE / "three-arm-connector-study-v1")
    args = parser.parse_args()
    if args.study_root.is_symlink() or args.study_root.resolve().is_relative_to(ROOT):
        raise ValueError("connector study output must be external and regular")
    args.study_root.mkdir(parents=True, exist_ok=True)
    lock = (args.study_root / ".connector-controller.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    old_term = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    profile = args.study_root / "common-initialization.json"
    state_path = args.study_root / "state.json"
    try:
        metadata = {"study_policy": POLICY, "budget_seconds": 64800,
                    "per_seed_budget_seconds": PHASE_CAP,
                    "budget_split": "two nine-hour phases; unused seed budget is not transferred",
                    "status": "running", "profile": str(profile)}
        if state_path.exists() and read_json(state_path).get("study_policy") != POLICY:
            raise ValueError("controller state policy mismatch")
        atomic_json(state_path, metadata)
        results, spent = [], 0.0
        for seed in (0, 1):
            with ConnectorPhase(args.study_root / f"seed{seed}", seed) as phase:
                try:
                    results.extend(run_phase(phase, profile))
                except BaseException as error:
                    phase.state.update(status="incomplete", error=type(error).__name__,
                                       error_message=str(error))
                    phase.save()
                    raise
                spent += phase.state["spent_seconds"]
                metadata.update(completed_seeds=list(range(seed + 1)), spent_seconds=spent)
                atomic_json(state_path, metadata)
        summary = aggregate_connector_results(results)
        summary.update(
            budget_seconds=64800, spent_gpu_stage_seconds=spent,
            common_profile_hash=file_hash(profile), controller_code_hash=file_hash(Path(__file__)),
        )
        atomic_json(args.study_root / "summary.json", summary)
        metadata.update(status="complete", spent_seconds=spent,
                        summary=str(args.study_root / "summary.json"))
        atomic_json(state_path, metadata)
        print(json.dumps({"status": "complete", "summary": metadata["summary"]}))
    except BaseException as error:
        atomic_json(state_path, {"study_policy": POLICY, "status": "incomplete",
                                "error": type(error).__name__, "error_message": str(error)})
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
        lock.close()


if __name__ == "__main__":
    main()
