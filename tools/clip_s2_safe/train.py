"""Retained original training math/state; no in-process generation or post-fault save."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from tools.benchmark.compact_checkpoint import capture_state
from tools.benchmark.translation_model import optimizer_for
from tools.clip_s2.data import batch_for
from tools.clip_s2.guards import assert_base_intact, check_ce_collapse
from tools.clip_s2_safe.accounting import require_storage
from tools.clip_s2_safe.evaluate import admit_original, validate_report
from tools.clip_s2_safe.policy import (
    load_config,
    require_exclusive_gpu,
    require_scope,
    source_hashes,
    verify_original,
)
from tools.clip_s2_safe.state import (
    load_weights,
    publish_checkpoint,
    read_binding,
    restore_training,
    validate_guard_state,
)
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.dinov3.storage import atomic_json, writer
from tools.followup.model import build_model, update
from tools.followup.recipe import PromptPool


def next_boundary(step: int, recipe: dict) -> int:
    for epoch in recipe["dev_epochs"]:
        if epoch * recipe["steps_per_epoch"] > step:
            return epoch * recipe["steps_per_epoch"]
    raise ValueError("Training ceiling reached")


def check_guard(model, base_hash: str, values: list, step: int, recipe: dict) -> None:
    assert_base_intact(model, base_hash)
    check_ce_collapse(values, step, recipe)


def require_prior_dev(config: dict, binding: dict, admission: dict, start: int) -> None:
    per = binding["recipe"]["steps_per_epoch"]
    for epoch in binding["recipe"]["dev_epochs"]:
        if epoch * per > start:
            break
        path = Path(config["runtime_root"]) / "dev" / f"epoch-{epoch}.json"
        if not path.is_file():
            raise RuntimeError("Pending production dev must precede next update")
        report = json.loads(path.read_text())
        cp = Path(report["checkpoint_path"])
        from tools.clip_s2_safe.state import checkpoint_bundle

        bundle = checkpoint_bundle(cp, binding)
        job = dict(
            mode="production",
            epoch=epoch,
            target_step=epoch * per,
            checkpoint_sha256=bundle["receipt"]["sha256"],
            max_batches=None,
        )
        validate_report(report, job, binding, admission)


def train_chunk(
    model,
    optimizer,
    sampler,
    admission: dict,
    pool,
    binding: dict,
    guard: dict,
    job: dict,
    *,
    device: torch.device,
    on_progress,
) -> dict:
    start = job["start_step"]
    target = job["target_step"]
    recipe = binding["recipe"]
    config = binding["config"]
    if (
        job.get("mode") not in ["production", "handoff_probe"]
        or job["stage"] != "train"
        or job["seed"] != 0
        or job["binding_hash"] != digest(binding)
        or start != sampler.batches_yielded
        or target <= start
        or target > recipe["total_steps"]
    ):
        raise ValueError("Exact retained start/target/sampler and binding required")
    if job["mode"] == "production" and target != next_boundary(start, recipe):
        raise ValueError("Production must stop at next declared dev boundary")
    validate_guard_state(
        guard, checkpoint_sha256=guard["checkpoint_sha256"], binding=binding, step=start
    )
    values = list(guard["ce_values"])
    durable = start
    publication = {}
    model.train()
    for k in range(start + 1, target + 1):
        batch = batch_for(admission, "train", sampler.next_ids())
        batch = replace(
            batch,
            spatial=batch.spatial.to(device),
            spatial_mask=batch.spatial_mask.to(device),
            motion=batch.motion.to(device),
            motion_mask=batch.motion_mask.to(device),
        )
        result = update(model, optimizer, batch, k, recipe, pool)
        if k > recipe["vt_warmup_steps"]:
            if result["translation_loss"] is None:
                raise ValueError("Missing eligible CE observation")
            values.append(float(result["translation_loss"]))
            del values[:-200]
        if k % recipe["steps_per_epoch"] == 0 or k == target:
            check_guard(
                model, binding["original_identity"]["frozen_fingerprint"], values, k, recipe
            )
            if source_hashes() != binding["sources"]:
                raise ValueError("Runtime source drift before snapshot")
            for p, sha in admission.get("bindings", {}).items():
                if sha256_file(Path(p)) != sha:
                    raise ValueError("Data binding drift")
            snapshot = capture_state(
                model,
                optimizer,
                binding["original_identity"],
                k,
                sampler.state_dict(),
                base_hash=binding["original_identity"]["frozen_fingerprint"],
            )
            publication = publish_checkpoint(
                config, snapshot, dict(last_update=k, ce_values=values), mode=job["mode"]
            )
            del snapshot
            durable = k
        on_progress(dict(completed_updates=k, durable_updates=durable, phase="train"))
    return dict(
        status="awaiting_dev",
        stage="train",
        mode=job["mode"],
        diagnostic_only=job["mode"] == "handoff_probe",
        completed_updates=target,
        durable_updates=durable,
        **publication,
    )


def run_train_job(config: dict, job: dict, approval: Path | None) -> dict:
    scope = require_scope(config, "train", approval, job=job)
    require_exclusive_gpu()
    b = read_binding(config)
    verify_original(config)
    require_storage(config, b, job)
    admission = admit_original(config, b)
    if job["mode"] == "production":
        require_prior_dev(config, b, admission, job["start_step"])
    if job["target_step"] - job["start_step"] > scope["max_train_updates"]:
        raise RuntimeError("Train scope cap exceeded")
    checkpoint = Path(job["checkpoint"])
    if sha256_file(checkpoint) != job["checkpoint_sha256"]:
        raise ValueError("Train input checkpoint hash drift")
    with writer(Path(config["runtime_root"]) / "train-worker"):
        model, base = build_model(0, "reference", config["cache"])
        state = load_weights(checkpoint, model, b, base_hash=base)
        if state["step"] != job["start_step"]:
            raise ValueError("Train checkpoint start mismatch")
        optimizer = optimizer_for(model)
        sampler = OfficialBatchSampler(admission["ids"]["train"], seed=0, batch_size=8)
        pool = PromptPool(admission["datasets"]["train"].records, 0)
        for split in ["dev", "test"]:
            pool.guard_references(tuple(r["text"] for r in admission["datasets"][split].records))
        model.to("cuda").train()
        restore_training(state, model, optimizer, sampler, device=torch.device("cuda"), scope=scope)
        guard = state["safe_guard"]
        del state
        last = [job["start_step"]]
        durable = [guard["last_update"]]

        def progress(value):
            last[0] = value["completed_updates"]
            durable[0] = value["durable_updates"]
            if (
                value["completed_updates"] % 25 == 0
                or value["durable_updates"] == value["completed_updates"]
            ):
                atomic_json(Path(job["progress_path"]), value)

        try:
            result = train_chunk(
                model,
                optimizer,
                sampler,
                admission,
                pool,
                b,
                guard,
                job,
                device=torch.device("cuda"),
                on_progress=progress,
            )
            atomic_json(Path(job["result_path"]), result)
            return result
        except BaseException as error:
            try:
                atomic_json(
                    Path(job["progress_path"]),
                    dict(
                        completed_updates=last[0],
                        durable_updates=durable[0],
                        error=str(error),
                        attempted_step=sampler.batches_yielded,
                        execution_uncertain=True,
                    ),
                )
            except Exception as observer:
                error.add_note(f"Failure progress write failed: {observer}")
            raise


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
    print(json.dumps(run_train_job(c, json.loads(a.job.read_text()), a.approval)))


if __name__ == "__main__":
    main()
