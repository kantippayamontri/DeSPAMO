"""Original S² inputs in an optimizer-free worker; only new-bound full reports select."""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import torch

from despamo.data.phoenix14t import normalize_text
from despamo.evaluation.metrics import evaluate_translations
from tools.clip_plain.generation import generate_cached_text, guard_device, memory_snapshot
from tools.clip_s2 import data as original_data
from tools.clip_s2.guards import check_dev_report
from tools.clip_s2.train import load_config as original_config
from tools.clip_s2_safe.accounting import require_storage
from tools.clip_s2_safe.policy import (
    evaluation_policy,
    load_config,
    require_exclusive_gpu,
    require_scope,
    verify_original,
)
from tools.clip_s2_safe.state import checkpoint_bundle, load_weights, read_binding
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.storage import atomic_json, writer
from tools.followup.model import build_model
from tools.followup.recipe import PromptPool


def admit_original(config: dict, binding: dict) -> dict:
    c = original_config(Path(config["study_config"]))
    a = original_data.admit_data(Path(c["spatial_root"]), c["spatial_manifest_sha256"])
    if a["identity"] != binding["original_identity"]["data"]:
        raise ValueError("Original full S² data identity changed")
    return a


def score_dev(
    model, admission: dict, pool, job: dict, binding: dict, *, on_event, scope: dict | None = None
) -> dict:
    expected = admission["ids"]["dev"]
    limit = job.get("max_batches")
    if (
        job.get("stage") != "dev"
        or job.get("seed") != 0
        or job.get("binding_hash") != digest(binding)
        or not expected
        or len(set(expected)) != len(expected)
        or digest(expected) != binding["original_identity"]["data"]["ordered_ids"]["dev"]
    ):
        raise ValueError("Original ordered S² dev and exact job binding required")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Positive diagnostic clip bound required")
    pool.guard_references(tuple(r["text"] for r in admission["datasets"]["dev"].records))
    if pool.eval_id != binding["original_identity"]["decoder"]["exemplar_id"]:
        raise ValueError("Original exemplar drift")
    device = next(model.parameters()).device
    guard_device(device, scope)
    modes = [(m, m.training) for m in model.modules()]
    items = []
    core_tag = None
    try:
        model.eval()
        requested = expected[:limit]
        bounded = {**admission, "ids": {**admission["ids"], "dev": requested}}
        for index, batch in enumerate(original_data.eval_batches(bounded, "dev", batch_size=1)):
            if batch.clip_ids != expected[index : index + 1] or batch.spatial.shape[-1] != 2048:
                raise ValueError("S² dev order or input width mismatch")
            meta = dict(
                batch_index=index,
                clip_ids=list(batch.clip_ids),
                spatial_lengths=batch.spatial_mask.sum(1).tolist(),
                motion_lengths=batch.motion_mask.sum(1).tolist(),
            )
            start = time.perf_counter()
            try:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                on_event(dict(phase="batch_start", **meta, memory=memory_snapshot(device)))
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
                    prompts, exemplars = pool.inference(batch.clip_ids)
                    result = generate_cached_text(
                        model.language_model,
                        visual,
                        mask,
                        prompts,
                        scope=scope,
                        on_event=lambda e, meta=meta: on_event({**meta, **e}),
                    )
                core_tag = result["decoding"]["id"]
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
        for module, flag in modes:
            module.training = flag
    if tuple(row["clip_id"] for row in items) != requested:
        raise ValueError("Incomplete S² dev generation")
    full = job["mode"] == "production" and limit is None
    metrics = (
        evaluate_translations([r["prediction"] for r in items], [r["reference"] for r in items])
        if full
        else None
    )
    if metrics is not None and not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError("Nonfinite dev metrics")
    return dict(
        schema_version=1,
        stage="dev",
        mode=job["mode"],
        split="dev",
        epoch=job.get("epoch"),
        step=job["target_step"],
        binding_hash=digest(binding),
        original_identity_hash=binding["original_identity_hash"],
        checkpoint_path=job["checkpoint"],
        checkpoint_sha256=job["checkpoint_sha256"],
        evaluation_policy=evaluation_policy(),
        evaluation_policy_hash=binding["evaluation_policy_hash"],
        data_identity_hash=digest(admission["identity"]),
        dev_ids_sha256=digest(expected),
        complete=full,
        diagnostic_only=not full,
        metrics=metrics,
        items=items,
        generation_core_policy_tag=core_tag,
        exemplar_id=pool.eval_id,
    )


def validate_report(report: dict, job: dict, binding: dict, admission: dict) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("complete") is not True
        or report.get("diagnostic_only") is not False
        or report.get("mode") != "production"
        or report.get("split") != "dev"
        or report.get("stage") != "dev"
        or job.get("mode") != "production"
        or job.get("max_batches") is not None
        or report.get("binding_hash") != digest(binding)
        or report.get("original_identity_hash") != binding["original_identity_hash"]
        or report.get("evaluation_policy") != evaluation_policy()
        or report.get("evaluation_policy_hash") != binding["evaluation_policy_hash"]
        or report.get("data_identity_hash") != digest(admission["identity"])
        or report.get("dev_ids_sha256") != digest(admission["ids"]["dev"])
        or tuple(row["clip_id"] for row in report["items"]) != admission["ids"]["dev"]
    ):
        raise ValueError("Only exact new-bound complete production S² reports may select")
    for field, jobfield in [
        ("epoch", "epoch"),
        ("step", "target_step"),
        ("checkpoint_sha256", "checkpoint_sha256"),
    ]:
        if report[field] != job.get(jobfield):
            raise ValueError("Dev checkpoint/step/epoch mismatch")
    exemplar = binding["original_identity"]["decoder"]["exemplar_id"]
    if report.get("exemplar_id") != exemplar:
        raise ValueError("Dev exemplar changed")
    for item, raw in zip(report["items"], admission["datasets"]["dev"].records, strict=True):
        if (
            item["reference"] != normalize_text(raw["text"])
            or item["exemplar_ids"] != [exemplar]
            or not item["token_ids"]
            or item["output_length"] != len(item["token_ids"])
        ):
            raise ValueError("Dev reference/exemplar/token evidence drift")
    metrics = evaluate_translations(
        [r["prediction"] for r in report["items"]], [r["reference"] for r in report["items"]]
    )
    if report.get("metrics") != metrics or not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError("Dev metric drift")
    if (
        "telemetry" in report
        and sha256_file(Path(report["telemetry"])) != report["telemetry_sha256"]
    ):
        raise ValueError("Dev telemetry drift")


def report_path(config: dict, job: dict) -> Path:
    root = Path(config["runtime_root"])
    return (
        root / "dev" / f"epoch-{job['epoch']}.json"
        if job["mode"] == "production" and job.get("max_batches") is None
        else root / "validation" / job["token"] / f"dev-step-{job['target_step']}.json"
    )


def validate_probe_report(report: dict, job: dict, binding: dict, admission: dict) -> None:
    expected = admission["ids"]["dev"][: job.get("max_batches")]
    if (
        report.get("schema_version") != 1
        or report.get("stage") != "dev"
        or report.get("split") != "dev"
        or report.get("complete") is not False
        or report.get("diagnostic_only") is not True
        or report.get("metrics") is not None
        or report.get("mode") != job["mode"]
        or report.get("step") != job["target_step"]
        or report.get("checkpoint_sha256") != job["checkpoint_sha256"]
        or report.get("binding_hash") != digest(binding)
        or report.get("original_identity_hash") != binding["original_identity_hash"]
        or report.get("evaluation_policy_hash") != binding["evaluation_policy_hash"]
        or report.get("evaluation_policy") != evaluation_policy()
        or report.get("data_identity_hash") != digest(admission["identity"])
        or report.get("dev_ids_sha256") != digest(admission["ids"]["dev"])
        or not expected
        or tuple(r["clip_id"] for r in report.get("items", [])) != expected
    ):
        raise ValueError("Incomplete/mismatched diagnostic handoff report")
    exemplar = binding["original_identity"]["decoder"]["exemplar_id"]
    for item, raw in zip(
        report["items"], admission["datasets"]["dev"].records[: len(expected)], strict=True
    ):
        if (
            item.get("reference") != normalize_text(raw["text"])
            or item.get("exemplar_ids") != [exemplar]
            or not item.get("token_ids")
            or item.get("output_length") != len(item["token_ids"])
        ):
            raise ValueError("Diagnostic reference/token/exemplar drift")
    if (
        "telemetry" in report
        and sha256_file(Path(report["telemetry"])) != report["telemetry_sha256"]
    ):
        raise ValueError("Diagnostic telemetry drift")


def run_dev_job(config: dict, job: dict, approval: Path | None) -> dict:
    scope = require_scope(config, "dev", approval, job=job)
    require_exclusive_gpu()
    b = read_binding(config)
    verify_original(config)
    require_storage(config, b, job)
    admission = admit_original(config, b)
    path = Path(job["checkpoint"])
    bundle = checkpoint_bundle(path, b)
    if (
        sha256_file(path) != job["checkpoint_sha256"]
        or bundle["receipt"]["step"] != job["target_step"]
    ):
        raise ValueError("Dev candidate/step mismatch")
    count = (
        config["expected_dev_clips"]
        if job.get("max_batches") is None
        else min(job["max_batches"], config["expected_dev_clips"])
    )
    if count > scope["max_dev_clips"]:
        raise RuntimeError("Dev scope cap exceeded")
    with writer(Path(config["runtime_root"]) / "dev-worker"):
        target = report_path(config, job)
        if target.exists():
            raise RuntimeError("Owned report already exists; reconcile rather than retry")
        model, base = build_model(0, "reference", config["cache"])
        state = load_weights(path, model, b, base_hash=base)
        del state
        model.to("cuda").eval()
        pool = PromptPool(admission["datasets"]["train"].records, 0)
        target.parent.mkdir(parents=True, exist_ok=True)
        log = target.parent / f"{target.stem}-{job['token']}.jsonl"
        with log.open("x") as handle:

            def emit(event):
                handle.write(
                    json.dumps(
                        dict(utc=datetime.now(UTC).isoformat(), job_hash=digest(job), **event)
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
                if event["phase"] == "batch_complete":
                    atomic_json(
                        Path(job["progress_path"]),
                        dict(
                            completed_updates=job["target_step"],
                            durable_updates=job["target_step"],
                            completed_clips=event["batch_index"] + 1,
                        ),
                    )

            report = score_dev(model, admission, pool, job, b, on_event=emit, scope=scope)
        if report["complete"]:
            validate_report(report, job, b, admission)
            check_dev_report(report, job["epoch"], b["recipe"])
        else:
            validate_probe_report(report, job, b, admission)
        report.update(telemetry=str(log), telemetry_sha256=sha256_file(log))
        atomic_json(target, report)
        result = dict(
            status="complete",
            stage="dev",
            mode=job["mode"],
            completed_updates=job["target_step"],
            durable_updates=job["target_step"],
            completed_clips=len(report["items"]),
            report=str(target),
            report_sha256=sha256_file(target),
            diagnostic_only=report["diagnostic_only"],
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
    c = load_config(a.config)
    if a.check_only:
        print(
            json.dumps(
                dict(
                    status="cpu_preflight", anchor_step=verify_original(c)["anchor_receipt"]["step"]
                )
            )
        )
        return
    print(json.dumps(run_dev_job(c, json.loads(a.job.read_text()), a.approval)))


if __name__ == "__main__":
    main()
