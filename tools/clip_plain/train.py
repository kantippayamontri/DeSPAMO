"""One train chunk ends at dev boundary; generation occurs in another process."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from tools.benchmark.compact_checkpoint import capture_state
from tools.benchmark.translation_model import optimizer_for
from tools.clip_plain.data import admit_data, batch_for
from tools.clip_plain.model import build_model
from tools.clip_plain.policy import (
    load_config,
    recipe,
    require_exclusive_gpu,
    require_gpu_scope,
    source_hashes,
)
from tools.clip_plain.state import (
    load_weights,
    make_identity,
    publish_epoch,
    require_storage,
    restore_training,
)
from tools.clip_s2.guards import COLLAPSE_WINDOW, assert_base_intact, check_ce_collapse
from tools.dinov3.identity import sha256_file
from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.dinov3.storage import atomic_json, writer
from tools.followup.model import update, validate_prompt_lengths
from tools.followup.recipe import PromptPool


def next_dev_epoch(completed_updates: int, recipe: dict) -> int:
    for epoch in recipe["dev_epochs"]:
        if epoch * recipe["steps_per_epoch"] > completed_updates:
            return epoch
    raise ValueError("Training ceiling reached")


def validate_training_boundary(run_root: Path, completed: int, recipe: dict) -> None:
    epoch = completed // recipe["steps_per_epoch"]
    if (
        epoch in recipe["dev_epochs"]
        and not (run_root / "dev" / f"dev-epoch-{epoch}.json").is_file()
    ):
        raise ValueError("Pending even-epoch dev must finish before training")


def epoch_guard(model, base_hash: str, history: list, step: int, recipe: dict) -> None:
    assert_base_intact(model, base_hash)
    check_ce_collapse(history, step, recipe)


def train_to_boundary(
    model,
    optimizer,
    sampler,
    admission,
    pool,
    identity: dict,
    run_root: Path,
    target_epoch: int,
    base_hash: str,
    device: torch.device,
    on_progress,
) -> dict:
    recipe = identity["recipe"]
    per = recipe["steps_per_epoch"]
    target = target_epoch * per
    if target_epoch not in recipe["dev_epochs"] or target <= sampler.batches_yielded:
        raise ValueError("Exact next dev boundary required")
    history = []
    publication = {}
    durable = sampler.batches_yielded
    for k in range(sampler.batches_yielded + 1, target + 1):
        batch = batch_for(admission, "train", sampler.next_ids())
        batch = (
            replace(
                batch,
                spatial=batch.spatial.to(device),
                spatial_mask=batch.spatial_mask.to(device),
                motion=batch.motion.to(device),
                motion_mask=batch.motion_mask.to(device),
            )
            if hasattr(batch, "__dataclass_fields__")
            else batch
        )
        values = update(model, optimizer, batch, k, recipe, pool)
        if values["translation_loss"] is not None:
            history.append(values["translation_loss"])
            del history[:-COLLAPSE_WINDOW]
        if k % per == 0:
            epoch_guard(model, base_hash, history, k, recipe)
            if (
                identity.get("code_hashes") is not None
                and source_hashes() != identity["code_hashes"]
            ):
                raise ValueError("Source drift before checkpoint")
            for p, sha in admission.get("bindings", {}).items():
                if sha256_file(Path(p)) != sha:
                    raise ValueError("Data binding drift")
            snapshot = capture_state(
                model, optimizer, identity, k, sampler.state_dict(), base_hash=base_hash
            )
            publication = publish_epoch(run_root, snapshot, k // per, dev_candidate=k == target)
            del snapshot
            durable = k
        on_progress(dict(completed_updates=k, durable_updates=durable, phase="train"))
    return dict(
        status="awaiting_dev",
        stage="train",
        completed_updates=target,
        durable_updates=durable,
        **publication,
    )


def run_train_chunk(config: dict, job: dict, approval: Path | None) -> dict:
    scope = require_gpu_scope(config, "train", approval, job=job)
    require_exclusive_gpu()
    run = Path(job["run_root"])
    if run.resolve() != (Path(config["study_root"]) / "runs" / f"seed{job['seed']}").resolve():
        raise ValueError("Foreign run root")
    admission = admit_data(config)
    require_storage(config, run, prune_allowed=scope.get("allow_owned_resume_pruning", False))
    with writer(run / "train-worker"):
        model, base_hash = build_model(job["seed"], config["cache"])
        identity = make_identity(config, admission, job["seed"], model, base_hash)
        path = run / "identity.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("Plain run identity changed")
        if not path.exists():
            atomic_json(path, identity)
        pool = PromptPool(admission["datasets"]["train"].records, job["seed"])
        for split in ["dev", "test"]:
            pool.guard_references(tuple(r["text"] for r in admission["datasets"][split].records))
        validate_prompt_lengths(model, pool)
        sampler = OfficialBatchSampler(admission["ids"]["train"], seed=job["seed"], batch_size=8)
        state = None
        if job.get("checkpoint"):
            cp = Path(job["checkpoint"])
            if cp.parent.resolve() != run.resolve() or sha256_file(cp) != job["checkpoint_sha256"]:
                raise ValueError("Resume hash/path mismatch")
            state = load_weights(cp, model, identity, base_hash)
            if (
                state["step"] != state["sampler"]["batches_yielded"]
                or state["step"] % identity["recipe"]["steps_per_epoch"]
            ):
                raise ValueError("Resume sampler/epoch mismatch")
            validate_training_boundary(run, state["step"], identity["recipe"])
        optimizer = optimizer_for(model)
        model.to("cuda").train()
        if state is not None:
            restore_training(state, model, optimizer, sampler)
            del state
        target = next_dev_epoch(sampler.batches_yielded, identity["recipe"])
        if target != job["epoch"]:
            raise ValueError("Controller target dev epoch changed")

        def progress(row):
            atomic_json(Path(job["progress_path"]), row)

        result = train_to_boundary(
            model,
            optimizer,
            sampler,
            admission,
            pool,
            identity,
            run,
            target,
            base_hash,
            torch.device("cuda"),
            progress,
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
        print(json.dumps(dict(status="cpu_preflight", recipe=recipe(config))))
        return
    print(json.dumps(run_train_chunk(config, job, a.approval)))


if __name__ == "__main__":
    main()
