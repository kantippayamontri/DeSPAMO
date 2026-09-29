"""Matched fresh E2 factor pilot under its own cap, with overall GPU accounting."""

import argparse
import shutil
import signal
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.evaluation.signer_pilot import choose_checkpoint
from despamo.training.e1_budget import CEILING_SECONDS, E1BudgetGuard, E1Ledger
from despamo.training.pilot_factory import PilotDataModule, shared_tensor_hash
from despamo.training.pilot_runtime import load_pilot_views, pilot_config
from scripts.train_signer_pilot import (
    E1_SHARED_HASH,
    ROOT,
    E1ResumeSnapshots,
    FullDevCheckpoints,
    _model,
    _train_data,
    validate_e1_resume_checkpoint,
)
from tools.dinov3.storage import verify_location, writer

RUN_POLICY = "signer-pilot-e2-projector-bounded-v1"
OVERALL_CEILING_SECONDS = 24 * 3600
CHECKPOINT_STORAGE_BYTES = 160 * 1024**3


def build_e2(args, protocol: dict, e1_root: Path, profile_path: Path, *, resume: bool = False):
    if not getattr(args, "authorize_e2_six_hour_run", False):
        raise ValueError("E2-only run requires explicit six-hour authorization")
    if args.physical_batch != 4:
        raise ValueError("E2 requires matched E1 physical batch 4")
    if (
        e1_root.is_symlink()
        or e1_root.resolve(strict=False).parent != args.output_base.resolve()
        or not e1_root.is_dir()
        or profile_path.resolve(strict=False).parent != args.output_base.resolve()
    ):
        raise ValueError("E2 E1 run/profile must be local external pilot artifacts")
    prior = read_json(e1_root / "run-status.json")
    e1_budget = read_json(e1_root / "budget.json")
    e1_identity = e1_budget.get("identity", {})
    selected = read_json(e1_root / "selected-checkpoint.json")
    if (
        prior.get("status") != "complete"
        or prior.get("global_step") != 4000
        or e1_budget.get("status") != "closed"
        or e1_identity.get("run_policy") != "signer-pilot-e1-fresh-retry-v3"
        or e1_identity.get("protocol_hash") != protocol.get("protocol_hash")
        or e1_identity.get("initial_shared_hash") != E1_SHARED_HASH
        or selected.get("step") != 4000
        or selected.get("checkpoint_hash") != file_hash(e1_root / "step-4000.ckpt")
    ):
        raise ValueError("E2 matched E1 selected checkpoint/protocol identity mismatch")
    profile = read_json(profile_path)
    factor = profile.get("input_identity", {}).get("factor_provenance", {})
    train = protocol["split"]["groups"]["train"]
    if (
        profile.get("status") != "complete"
        or profile.get("variant") != "E2_projector"
        or profile.get("protocol_hash") != protocol["protocol_hash"]
        or profile.get("physical_batch") != 4
        or profile.get("initial_shared_tensor_hash") != E1_SHARED_HASH
        or factor.get("supervision_policy") != "qwen-schema98-unreviewed-v1"
        or factor.get("human_review_status") != "not_assessed"
        or factor.get("valid_count") != 5641
        or factor.get("failed_count") != 105
        or factor.get("records_hash") != protocol["records_hash"]
        or factor.get("text_manifest_hash") != protocol["text_manifest_hash"]
        or factor.get("logical_train_ids_hash") != digest(tuple(train["clip_ids"]))
    ):
        raise ValueError("E2 factor profile/initialization/held-out clip admission mismatch")
    config = pilot_config(
        args.hf_cache,
        args.dino_root,
        args.motion_root,
        args.annotation.parent,
        args.dino_root / "complete/manifest.json",
        args.motion_manifest,
        4000,
        4,
        args.output_base,
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    if (
        resolved != e1_identity.get("config")
        or resolved != profile["input_identity"].get("config")
        or config.data.num_workers != 0
        or config.trainer.accumulate_grad_batches != 1
    ):
        raise ValueError("E2 does not match E1/config/WSL worker policy")
    measured = profile.get("gpu_seconds")
    step_rate = profile.get("joint_seconds_per_step")
    dev_rate = profile.get("dev_seconds_per_full_split")
    if (
        not all(
            isinstance(x, (int, float)) and 0 < x < CEILING_SECONDS
            for x in (measured, step_rate, dev_rate)
        )
        or measured + 4000 * step_rate + 3 * dev_rate + 900 >= CEILING_SECONDS
    ):
        raise ValueError("E2 projected six-hour budget admission failed")
    prior_total = prior.get("overall_recorded_seconds")
    if (
        not isinstance(prior_total, (int, float))
        or prior_total + CEILING_SECONDS >= OVERALL_CEILING_SECONDS
    ):
        raise ValueError("E2 overall 24-hour GPU budget admission failed")
    code_files = (
        "scripts/train_e2_pilot.py",
        "scripts/train_signer_pilot.py",
        "src/despamo/training/e1_budget.py",
        "src/despamo/training/pilot_factory.py",
        "src/despamo/training/pilot_replay.py",
        "src/despamo/training/pilot_runtime.py",
        "src/despamo/training/baseline_module.py",
        "src/despamo/training/factor_module.py",
        "src/despamo/data/factors.py",
        "src/despamo/models/factor_adapter.py",
        "src/despamo/models/factor_heads.py",
        "src/despamo/losses/factor_contrast.py",
        "src/despamo/evaluation/signer_pilot.py",
        "uv.lock",
    )
    identity = {
        "run_policy": RUN_POLICY,
        "protocol_hash": protocol["protocol_hash"],
        "variant": "E2_projector",
        "seed": 0,
        "config": resolved,
        "factor_provenance": factor,
        "initial_shared_hash": E1_SHARED_HASH,
        "matched_e1_run": e1_root.name,
        "matched_e1_checkpoint_hash": selected["checkpoint_hash"],
        "e2_profile_hash": file_hash(profile_path),
        "e2_profile_gpu_seconds": measured,
        "prior_other_gpu_seconds": prior_total,
        "new_run_cap_seconds": CEILING_SECONDS,
        "overall_cap_seconds": OVERALL_CEILING_SECONDS,
        "code_hashes": {name: file_hash(ROOT / name) for name in code_files},
        "wsl_num_workers": 0,
    }
    output = args.output_base / ("e2-only-" + digest(identity))
    if output.is_symlink() or (output.exists() if not resume else not output.is_dir()):
        raise ValueError("E2 run root exists or required resume root missing")
    if shutil.disk_usage(args.output_base).free < CHECKPOINT_STORAGE_BYTES:
        raise ValueError("E2 requires 160 GiB free native checkpoint storage")
    if Path("/mnt/c").is_dir() and shutil.disk_usage("/mnt/c").free < 80 * 1024**3:
        raise ValueError("E2 requires 80 GiB Windows C: headroom for WSL checkpoints")
    return identity, output


def charge_model_loaded(ledger: E1Ledger) -> float:
    return ledger.charge(step=ledger.step, phase="model_loaded")


def run_e2(args, protocol: dict, views: dict, e1_root: Path, profile_path: Path) -> dict:
    """Fresh matched E2, never initialize from E1 checkpoint weights."""
    resume = getattr(args, "resume_checkpoint", None)
    identity, output = build_e2(args, protocol, e1_root, profile_path, resume=resume is not None)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("E2 requires 12 GiB free CUDA memory")
    ledger = (
        E1Ledger.create(output, identity, identity["e2_profile_gpu_seconds"])
        if resume is None
        else None
    )
    trainer = None
    old_term = signal.getsignal(signal.SIGTERM)
    old_alarm = signal.getsignal(signal.SIGALRM)
    alarm_armed = False

    def terminate(signum, frame):
        raise KeyboardInterrupt("E2 received SIGTERM")

    def timeout(signum, frame):
        raise RuntimeError("E2 six-hour cap expired")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGALRM, timeout)
    try:
        with writer(output):
            if resume is not None:
                if (output / "selected-checkpoint.json").exists():
                    raise ValueError("E2 already completed")
                spent = read_json(output / "budget.json")["gpu_seconds"]
                checkpoint = validate_e1_resume_checkpoint(resume, output, identity, spent)
                checkpoint_seconds, checkpoint_step = (
                    checkpoint["gpu_seconds_cumulative"],
                    checkpoint["global_step"],
                )
                del checkpoint
                ledger = E1Ledger.resume(
                    output, identity, checkpoint_seconds, checkpoint_step=checkpoint_step
                )
            signal.setitimer(signal.ITIMER_REAL, max(1, CEILING_SECONDS - ledger.last - 2))
            alarm_armed = True
            config = OmegaConf.create(identity["config"])
            config.trainer.default_root_dir = str(output)
            train, collate, actual_factor = _train_data(
                "E2_projector", views["train"], args, protocol
            )
            if actual_factor != identity["factor_provenance"]:
                raise ValueError("E2 train-only factor provenance changed after preflight")
            model = _model(config, "E2_projector", 768, 4000, actual_factor)
            if shared_tensor_hash(model) != E1_SHARED_HASH:
                raise ValueError("E2 fresh shared initialization differs from E1")
            model.run_metadata = identity
            charge_model_loaded(ledger)
            data = PilotDataModule(train, 4, 0, 4000, collate)
            dev_ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
            dev = FullDevCheckpoints(
                views["dev"],
                dev_ids,
                protocol["split_hash"],
                output,
                4000,
                1000,
                4,
                "E2_projector",
                ledger=ledger,
            )
            if (
                config.data.num_workers != 0
                or data.train_dataloader().num_workers != 0
                or dev.loader.num_workers != 0
            ):
                raise ValueError("WSL E2 forbids DataLoader workers")
            trainer = pl.Trainer(
                **dict(config.trainer),
                callbacks=[E1BudgetGuard(ledger), dev, E1ResumeSnapshots(output, ledger)],
            )
            trainer.fit(model, datamodule=data, ckpt_path=str(resume) if resume else None)
            if trainer.global_step == 4000:
                reports = [read_json(output / f"dev-{step}.json") for step in (1750, 2800, 4000)]
                best = choose_checkpoint(reports, dev_ids)
                atomic_json(
                    output / "selected-checkpoint.json",
                    {
                        "step": best["checkpoint_step"],
                        "checkpoint_hash": best["checkpoint_hash"],
                        "metrics": best["metrics"],
                        "protocol_hash": protocol["protocol_hash"],
                        "model_variant": "E2_projector",
                    },
                )
            result = {
                "status": "complete" if trainer.global_step == 4000 else "partial",
                "global_step": trainer.global_step,
                "run_key": output.name,
                "protocol_hash": protocol["protocol_hash"],
            }
    except BaseException as error:
        result = {
            "status": "partial",
            "global_step": trainer.global_step if trainer is not None else 0,
            "run_key": output.name,
            "protocol_hash": protocol["protocol_hash"],
            "error_type": type(error).__name__,
        }
        if ledger is not None:
            atomic_json(output / "run-status.json", result)
        raise
    finally:
        if alarm_armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if ledger is not None:
            ledger.close()
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGALRM, old_alarm)
    result["e2_gpu_seconds"] = ledger.last
    result["overall_recorded_seconds"] = identity["prior_other_gpu_seconds"] + ledger.last
    if result["overall_recorded_seconds"] >= OVERALL_CEILING_SECONDS:
        raise RuntimeError("E2 overall 24-hour GPU cap exceeded")
    atomic_json(output / "run-status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    for key in (
        "protocol",
        "dataset",
        "text-manifest",
        "dino-root",
        "motion-root",
        "motion-manifest",
        "annotation",
        "hf-cache",
        "output-base",
        "e1-run",
        "e2-profile",
    ):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--physical-batch", type=int, default=4)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--authorize-e2-six-hour-run", action="store_true")
    args = parser.parse_args()
    verify_location(args.output_base, ROOT)
    protocol, views = load_pilot_views(
        args.protocol,
        args.dataset,
        args.text_manifest,
        args.dino_root,
        args.motion_root,
        args.motion_manifest,
        args.annotation,
    )
    if args.mode == "preflight":
        identity, output = build_e2(
            args, protocol, args.e1_run, args.e2_profile, resume=args.resume_checkpoint is not None
        )
        print(
            {
                "run_key": output.name,
                "profile_gpu_seconds": identity["e2_profile_gpu_seconds"],
                "overall_prior_seconds": identity["prior_other_gpu_seconds"],
                "train": len(views["train"]),
                "dev": len(views["dev"]),
                "workers": 0,
            }
        )
        return
    print(run_e2(args, protocol, views, args.e1_run, args.e2_profile))


if __name__ == "__main__":
    main()
