"""Optimizer-free dev worker with strict full-corpus selection contracts."""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import torch

from despamo.evaluation.metrics import evaluate_translations
from tools.clip_plain.data import admit_data, eval_batches
from tools.clip_plain.generation import generate_cached_text, guard_device, memory_snapshot
from tools.clip_plain.model import build_model
from tools.clip_plain.policy import (
    evaluation_policy,
    load_config,
    require_exclusive_gpu,
    require_gpu_scope,
    source_hashes,
)
from tools.clip_plain.state import load_weights
from tools.clip_s2.guards import check_dev_report
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.storage import atomic_json, writer
from tools.followup.recipe import PromptPool


def score_dev(
    model, admission: dict, pool: PromptPool, job: dict, *, on_event, scope: dict | None = None
) -> dict:
    expected = admission["ids"]["dev"]
    if job.get("split", "dev") != "dev" or not expected or len(set(expected)) != len(expected):
        raise ValueError("Ordered nonempty dev split required")
    limit = job.get("max_batches")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Positive diagnostic batch bound required")
    pool.guard_references(tuple(r["text"] for r in admission["datasets"]["dev"].records))
    device = next(model.parameters()).device
    guard_device(device, scope)
    modes = [(m, m.training) for m in model.modules()]
    items = []
    try:
        model.eval()
        for index, batch in enumerate(eval_batches(admission, max_batches=limit)):
            if batch.clip_ids != expected[index : index + 1]:
                raise ValueError("Dev order mismatch")
            meta = dict(
                batch_index=index,
                clip_ids=list(batch.clip_ids),
                spatial_shape=list(batch.spatial.shape),
                motion_shape=list(batch.motion.shape),
                spatial_lengths=batch.spatial_mask.sum(1).tolist(),
                motion_lengths=batch.motion_mask.sum(1).tolist(),
            )
            start = time.perf_counter()
            try:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                on_event(dict(phase="batch_start", **meta, memory=memory_snapshot(device)))
                prompts, exemplars = pool.inference(batch.clip_ids)
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with torch.inference_mode(), context:
                    visual, mask = model.visual_adapter(
                        batch.spatial.to(device),
                        batch.spatial_mask.to(device),
                        batch.motion.to(device),
                        batch.motion_mask.to(device),
                    )
                    result = generate_cached_text(
                        model.language_model,
                        visual,
                        mask,
                        prompts,
                        on_event=lambda e, meta=meta: on_event({**meta, **e}),
                        scope=scope,
                    )
                items.append(
                    dict(
                        clip_id=batch.clip_ids[0],
                        prediction=result["texts"][0],
                        reference=batch.texts[0].lower(),
                        token_ids=result["token_ids"][0],
                        output_length=result["output_lengths"][0],
                        exemplar_ids=[exemplars[0]],
                    )
                )
                on_event(
                    dict(
                        phase="batch_complete",
                        **meta,
                        seconds=time.perf_counter() - start,
                        memory=memory_snapshot(device),
                    )
                )
            except Exception as error:
                try:
                    on_event(
                        dict(
                            phase="batch_error",
                            **meta,
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                    )
                except Exception as observer:
                    error.add_note(f"Error telemetry failed: {observer}")
                raise
    finally:
        for m, flag in modes:
            m.training = flag
    requested = expected if limit is None else expected[:limit]
    if tuple(i["clip_id"] for i in items) != requested:
        raise ValueError("Incomplete dev generation")
    complete = limit is None
    metrics = (
        evaluate_translations([i["prediction"] for i in items], [i["reference"] for i in items])
        if complete
        else None
    )
    if metrics is not None and not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError("Nonfinite metrics")
    return dict(
        schema_version=1,
        split="dev",
        epoch=job["epoch"],
        checkpoint_sha256=job["checkpoint_sha256"],
        run_identity_hash=job["run_identity_hash"],
        evaluation_policy_hash=job["evaluation_policy_hash"],
        data_identity_hash=job["data_identity_hash"],
        complete=complete,
        diagnostic_only=not complete,
        metrics=metrics,
        items=items,
        decoding=evaluation_policy(),
        exemplar_id=pool.eval_id,
    )


def validate_report(report: dict, job: dict, admission: dict) -> None:
    if (
        report.get("split") != "dev"
        or report.get("complete") is not True
        or report.get("diagnostic_only") is not False
        or report.get("metrics") is None
        or job.get("max_batches") is not None
        or tuple(i["clip_id"] for i in report["items"]) != admission["ids"]["dev"]
        or report.get("decoding") != evaluation_policy()
        or report.get("evaluation_policy_hash") != digest(evaluation_policy())
    ):
        raise ValueError("Only complete full-policy dev reports may select")
    for key in [
        "epoch",
        "checkpoint_sha256",
        "run_identity_hash",
        "evaluation_policy_hash",
        "data_identity_hash",
    ]:
        if report.get(key) != job.get(key):
            raise ValueError(f"Dev report {key} mismatch")
    if report["data_identity_hash"] != digest(admission["identity"]) or not all(
        math.isfinite(v) for v in report["metrics"].values()
    ):
        raise ValueError("Dev data/metric mismatch")
    texts = [r["text"].strip().lower() for r in admission["datasets"]["dev"].records]
    if job.get("seed") is not None:
        expected_exemplar = PromptPool(admission["datasets"]["train"].records, job["seed"]).eval_id
        if report.get("exemplar_id") != expected_exemplar:
            raise ValueError("Dev inference exemplar changed")
    else:
        expected_exemplar = report.get("exemplar_id")
    for row, text in zip(report["items"], texts, strict=True):
        reference = text if text.endswith(".") else text + "."
        if row["reference"] != reference:
            raise ValueError("Dev reference changed")
        if (
            row.get("exemplar_ids") != [expected_exemplar]
            or not row.get("token_ids")
            or row.get("output_length") != len(row["token_ids"])
        ):
            raise ValueError("Dev exemplar/token evidence invalid")
    actual_metrics = evaluate_translations(
        [r["prediction"] for r in report["items"]], [r["reference"] for r in report["items"]]
    )
    if actual_metrics != report["metrics"]:
        raise ValueError("Dev metrics changed relative to predictions")
    if "telemetry" in report and sha256_file(Path(report["telemetry"])) != report.get(
        "telemetry_sha256"
    ):
        raise ValueError("Dev telemetry bytes changed")


def run_dev_job(config: dict, job: dict, approval: Path | None) -> dict:
    scope = require_gpu_scope(config, "dev", approval, job=job)
    require_exclusive_gpu()
    run = Path(job["run_root"])
    identity = json.loads((run / "identity.json").read_text())
    if (
        identity["code_hashes"] != source_hashes()
        or identity["config_sha256"] != digest(config)
        or digest(identity) != job["run_identity_hash"]
    ):
        raise ValueError("Dev source/config/identity drift")
    admission = admit_data(config)
    if identity["data"] != admission["identity"] or job["data_identity_hash"] != digest(
        admission["identity"]
    ):
        raise ValueError("Dev data changed")
    candidate = Path(job["checkpoint"])
    if (
        candidate.parent.resolve() != run.resolve()
        or sha256_file(candidate) != job["checkpoint_sha256"]
    ):
        raise ValueError("Dev checkpoint changed")
    with writer(run / "dev-worker"):
        model, base_hash = build_model(job["seed"], config["cache"])
        state = load_weights(candidate, model, identity, base_hash)
        if state["step"] != job["epoch"] * identity["recipe"]["steps_per_epoch"]:
            raise ValueError("Dev checkpoint epoch mismatch")
        del state
        model.to("cuda").eval()
        pool = PromptPool(admission["datasets"]["train"].records, job["seed"])
        diagnostic = job.get("max_batches") is not None
        root = run / "diagnostics" / job["attempt_token"] if diagnostic else run / "dev"
        target = root / f"dev-epoch-{job['epoch']}.json"
        if target.exists():
            raise ValueError("Owned dev report already exists")
        root.mkdir(parents=True, exist_ok=True)
        telemetry = root / f"events-epoch-{job['epoch']}-{job['attempt_token']}.jsonl"
        with telemetry.open("x") as handle:

            def emit(event):
                handle.write(
                    json.dumps(
                        dict(utc=datetime.now(UTC).isoformat(), job_hash=digest(job), **event)
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())

            report = score_dev(model, admission, pool, job, on_event=emit, scope=scope)
        if not diagnostic:
            validate_report(report, job, admission)
            check_dev_report(report, job["epoch"], identity["recipe"])
        report.update(
            checkpoint_path=str(candidate),
            telemetry=str(telemetry),
            telemetry_sha256=sha256_file(telemetry),
        )
        atomic_json(target, report)
        result = dict(
            status="complete",
            stage="dev",
            completed_updates=job["epoch"] * identity["recipe"]["steps_per_epoch"],
            durable_updates=job["epoch"] * identity["recipe"]["steps_per_epoch"],
            report=str(target),
            report_sha256=sha256_file(target),
            diagnostic_only=diagnostic,
        )
        atomic_json(Path(job["result_path"]), result)
        return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--approval", type=Path)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    config = load_config(a.config)
    job = json.loads(a.job.read_text())
    if a.check_only:
        admission = admit_data(config)
        print(json.dumps(dict(status="cpu_preflight", dev_clips=len(admission["ids"]["dev"]))))
        return
    print(json.dumps(run_dev_job(config, job, a.approval)))


if __name__ == "__main__":
    main()
