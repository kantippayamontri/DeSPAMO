"""Sequential once-only scoring of six fixed dev-selected connector checkpoints."""

import argparse
import fcntl
import json
import math
import signal
import statistics
from pathlib import Path

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from scripts.run_connector_study import ARMS, prediction_diagnostics
from scripts.run_projector_overnight import BASE, PYTHON, ROOT, Study, common_arguments

POLICY = "three-arm-connector-heldout-controller-v1"
SOURCE = BASE / "three-arm-connector-study-v1/summary.json"
REGISTRY = BASE / "projector-heldout-registry-v1.json"


def evaluation_command(run: Path, output: Path, mode="run") -> list[str]:
    return [str(PYTHON), str(ROOT / "scripts/evaluate_connector_study.py"),
            *common_arguments(), "--mode", mode, "--connector-run", str(run),
            "--output", str(output), "--registry", str(REGISTRY), "--gpu-cap-seconds", "1800"]


def validated_complete(study: Study, run: Path, output: Path, expected: dict) -> bool:
    ready = study.cpu_query(evaluation_command(run, output, "preflight"))
    if ready.get("status") != "ready":
        raise ValueError("connector held-out preflight invalid")
    for field, source_field in (("checkpoint_hash", "checkpoint_hash"),
                                ("checkpoint_step", "selected_step"), ("seed", "seed"),
                                ("arm", "arm")):
        if ready.get(field) != expected.get(source_field):
            raise ValueError("held-out preflight differs from frozen dev selection")
    return ready.get("registry_status") == "complete"


def summarize_heldout(reports: list[dict]) -> dict:
    by = {(r["seed"], r["arm"]): r for r in reports}
    if len(reports) != 6 or set(by) != {(s, a) for s in (0, 1) for a in ARMS}:
        raise ValueError("held-out summary requires exactly three arms and two seeds")
    first = reports[0]
    ordered = [(i["clip_id"], i["reference"]) for i in first["items"]]
    metric_keys = set(first["metrics"])
    for report in reports:
        if (report.get("split") != "test" or report["split_hash"] != first["split_hash"]
                or report["decoding"] != first["decoding"] or set(report["metrics"]) != metric_keys
                or [(i["clip_id"], i["reference"]) for i in report["items"]] != ordered
                or not all(type(v) in (int, float) and math.isfinite(v)
                           for v in report["metrics"].values())):
            raise ValueError("held-out report split/decoder/references/metrics differ")
    arms = {}
    for arm in ARMS:
        scores = [by[s, arm]["metrics"]["bleu4"] for s in (0, 1)]
        deltas = [scores[s] - by[s, "reference"]["metrics"]["bleu4"] for s in (0, 1)]
        arms[arm] = {"bleu4": scores, "mean_bleu4": statistics.mean(scores),
                     "sample_std_bleu4": statistics.stdev(scores), "bleu4_deltas": deltas,
                     "mean_delta": statistics.mean(deltas),
                     "positive_both_seeds": all(x > 0 for x in deltas),
                     "metric_deltas": {k: [by[s, arm]["metrics"][k]
                                           - by[s, "reference"]["metrics"][k] for s in (0, 1)]
                                       for k in sorted(metric_keys)}}
    results = [{"seed": r["seed"], "arm": r["arm"], "checkpoint_hash": r["checkpoint_hash"],
                "selected_step": r["checkpoint_step"], "metrics": r["metrics"],
                "repetition": prediction_diagnostics(r["items"])} for r in reports]
    return {"study_policy": POLICY, "status": "complete", "arms": arms, "results": results,
            "split": "test", "test_clips": len(ordered), "decoding": first["decoding"],
            "ordered_ids_references_hash": digest(ordered), "test_based_selection": False,
            "limitations": ["Two seeds and one held-out signer only.",
                            "Intervals condition on fixed checkpoints, not seed uncertainty.",
                            "Reference rerun variability remains unexplained."]}


def run(study: Study, source: dict, source_hash: str):
    rows = source.get("results", [])
    if (source.get("status") != "complete" or not isinstance(rows, list) or len(rows) != 6
            or any(type(x.get("seed")) is not int for x in rows)
            or {(x.get("seed"), x.get("arm")) for x in rows}
            != {(s, a) for s in (0, 1) for a in ARMS}):
        raise ValueError("connector source requires exact six complete seed/arm pairs")
    binding = {
        "policy": POLICY, "source_study_hash": source_hash,
        "checkpoints": {f"{x['seed']}-{x['arm']}": x["checkpoint_hash"] for x in source["results"]},
        "controller_hash": file_hash(Path(__file__)),
    }
    if ("connector_heldout_binding" in study.state
            and study.state["connector_heldout_binding"] != binding):
        raise ValueError("held-out controller manifest/source changed")
    study.state["connector_heldout_binding"] = binding
    study.save()
    records, paths = [], {}
    for item in source["results"]:
        name = f"seed{item['seed']}-{item['arm']}"
        path = study.root / f"{name}-signer07.json"
        run_root = Path(item["run"])
        study.run_stage(name, evaluation_command(run_root, path), 1800,
                        completion_validator=lambda run_root=run_root, path=path, item=item:
                            validated_complete(study, run_root, path, item))
        report = read_json(path)
        if (report["checkpoint_hash"] != item["checkpoint_hash"]
                or report["checkpoint_step"] != item["selected_step"]
                or report["seed"] != item["seed"] or report["arm"] != item["arm"]
                or len(report["items"]) != 768):
            raise ValueError("connector held-out result differs from frozen selection")
        records.append(report)
        paths[name] = {"path": str(path), "sha256": file_hash(path)}
    if file_hash(SOURCE) != source_hash:
        raise ValueError("completed dev study changed during scoring")
    summary = summarize_heldout(records)
    from scripts.analyze_projector_study import compare_projector_reports

    by = {(r["seed"], r["arm"]): r for r in records}
    analyses = {}
    for seed in (0, 1):
        for arm in ("plain_mlp", "qformer"):
            analysis = compare_projector_reports(by[seed, "reference"], by[seed, arm],
                                                bootstrap_replicates=1000, seed=0)
            analysis["comparison_arms"] = {"linear_label": "reference", "residual_label": arm}
            name = f"seed{seed}-{arm}-vs-reference"
            output = study.root / f"{name}-comparison.json"
            atomic_json(output, analysis)
            analyses[name] = {"path": str(output), "sha256": file_hash(output)}
    summary.update(budget_seconds=10800, spent_gpu_stage_seconds=study.state["spent_seconds"],
                   reports=paths, analyses=analyses, frozen_dev_study=str(SOURCE),
                   frozen_dev_study_hash=source_hash)
    atomic_json(study.root / "summary.json", summary)
    study.state.update(status="complete", connector_summary=str(study.root / "summary.json"))
    study.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "three-arm-connector-heldout-v1")
    args = parser.parse_args()
    if args.output_dir.is_symlink() or args.output_dir.resolve().is_relative_to(ROOT):
        raise ValueError("held-out controller output must be external")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".connector-heldout.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
        try:
            source_hash = file_hash(SOURCE)
            source = read_json(SOURCE)
            if file_hash(SOURCE) != source_hash:
                raise ValueError("source study changed during read")
            with Study(args.output_dir / "controller", 10800) as study:
                try:
                    run(study, source, source_hash)
                except BaseException as error:
                    study.state.update(status="incomplete", error=type(error).__name__,
                                       error_message=str(error))
                    study.save()
                    raise
        finally:
            signal.signal(signal.SIGTERM, previous)
    print(json.dumps({"status": "complete",
                      "summary": str(args.output_dir / "controller/summary.json")}))


if __name__ == "__main__":
    main()
