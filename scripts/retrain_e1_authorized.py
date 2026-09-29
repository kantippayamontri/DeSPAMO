"""User-authorized fresh E1-v3: WSL workers=0, new six-hour cap, old spend bound."""

import argparse
import shutil
import signal
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.data.batch import collate_phoenix
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
    validate_e1_resume_checkpoint,
)
from tools.dinov3.storage import verify_location, writer

RUN_POLICY = "signer-pilot-e1-fresh-retry-v3"
PARENT_POLICY = "signer-pilot-e1-fresh-retry-v2"
OVERALL_CEILING_SECONDS = 24 * 3600


def build_v3(args, protocol: dict, parent: Path, *, resume: bool = False) -> tuple[dict, Path]:
    """Keep old ledger intact; new budget counts only GPU residency after fresh start."""
    if not getattr(args, "authorize_new_six_hour_run", False):
        raise ValueError("E1-v3 requires explicit new six-hour authorization")
    if args.physical_batch != 4:
        raise ValueError("E1-v3 requires physical batch 4")
    if parent.is_symlink() or parent.resolve(strict=False).parent != args.output_base.resolve():
        raise ValueError("E1-v3 parent must be one owned run under output root")
    old = read_json(parent / "budget.json")
    if old.get("status") != "running" or Path(f"/proc/{old.get('pid')}").exists():
        raise ValueError("E1-v3 parent must be stale, with no active process")
    previous = old.get("identity", {})
    if previous.get("run_policy") != PARENT_POLICY or previous.get("protocol_hash") != protocol.get(
        "protocol_hash"
    ):
        raise ValueError("E1-v3 predecessor protocol/policy mismatch")
    originals = previous.get("checkpoint_bound_code_hashes", {})
    if not originals or any(
        not (ROOT / name).is_file() or file_hash(ROOT / name) != expected
        for name, expected in originals.items()
    ):
        raise ValueError("E1-v3 original checkpoint-bound code/lock drift")
    previous_retry_hash = previous.get("retry_code_hash")
    if previous_retry_hash and previous_retry_hash != file_hash(ROOT / "scripts/retrain_e1_wsl.py"):
        raise ValueError("E1-v3 predecessor runner code drift")
    spent = old.get("gpu_seconds")
    if (
        not isinstance(spent, (int, float))
        or not 0 < spent < OVERALL_CEILING_SECONDS
        or spent + CEILING_SECONDS >= OVERALL_CEILING_SECONDS
    ):
        raise ValueError("E1-v3 prior recorded GPU spend exceeds overall cap")
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
    if config.data.num_workers != 0 or config.trainer.accumulate_grad_batches != 1:
        raise ValueError("WSL E1-v3 requires num_workers=0 and physical batch4")
    identity = {
        "run_policy": RUN_POLICY,
        "protocol_hash": protocol["protocol_hash"],
        "variant": "E1_frozen",
        "seed": 0,
        "config": OmegaConf.to_container(config, resolve=True),
        "initial_shared_hash": E1_SHARED_HASH,
        "parent_run_key": parent.name,
        "parent_budget_hash": file_hash(parent / "budget.json"),
        "prior_recorded_gpu_seconds": spent,
        "parent_unclean_gap_policy": "new authorized run; gap excluded from GPU residency",
        "new_run_cap_seconds": CEILING_SECONDS,
        "overall_cap_seconds": OVERALL_CEILING_SECONDS,
        "checkpoint_bound_code_hashes": dict(originals),
        "predecessor_retry_code_hash": previous_retry_hash,
        "v3_code_hash": file_hash(Path(__file__)),
        "wsl_num_workers": 0,
    }
    output = args.output_base / ("e1-v3-" + digest(identity))
    if output.is_symlink() or (output.exists() if not resume else not output.is_dir()):
        raise ValueError("E1-v3 output exists or required resume root missing")
    if shutil.disk_usage(args.output_base).free < 160 * 1024**3:
        raise ValueError("E1-v3 requires 160 GiB free native checkpoint storage")
    if Path("/mnt/c").is_dir() and shutil.disk_usage("/mnt/c").free < 80 * 1024**3:
        raise ValueError("E1-v3 requires 80 GiB Windows C: headroom for WSL checkpoints")
    return identity, output


def charge_model_loaded(ledger: E1Ledger) -> float:
    """Fresh starts at zero; resume preserves checkpoint step without rewind."""
    return ledger.charge(step=ledger.step, phase="model_loaded")


def run_v3(args, protocol: dict, views: dict, parent: Path) -> dict:
    """Fresh E1 with an independently authorized cap and inherited overall spend."""
    resume = getattr(args, "resume_checkpoint", None)
    identity, output = build_v3(args, protocol, parent, resume=resume is not None)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("E1-v3 requires 12 GiB free CUDA memory")
    ledger = E1Ledger.create(output, identity, 0.0) if resume is None else None
    trainer = None
    old_term = signal.getsignal(signal.SIGTERM)
    old_alarm = signal.getsignal(signal.SIGALRM)
    alarm_armed = False

    def terminate(signum, frame):
        raise KeyboardInterrupt("E1-v3 received SIGTERM")

    def timeout(signum, frame):
        raise RuntimeError("E1-v3 six-hour cap expired")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGALRM, timeout)
    try:
        with writer(output):
            if resume is not None:
                if (output / "selected-checkpoint.json").exists():
                    raise ValueError("E1-v3 already completed")
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
            model = _model(
                config,
                "E1_frozen",
                768,
                4000,
                {
                    "supervision_policy": "qwen-schema98-unreviewed-v1",
                    "human_review_status": "not_assessed",
                },
            )
            if shared_tensor_hash(model) != E1_SHARED_HASH:
                raise ValueError("E1-v3 fresh shared initialization hash mismatch")
            model.run_metadata = identity
            charge_model_loaded(ledger)
            data = PilotDataModule(views["train"], 4, 0, 4000, collate_phoenix)
            dev_ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
            dev = FullDevCheckpoints(
                views["dev"],
                dev_ids,
                protocol["split_hash"],
                output,
                4000,
                1000,
                4,
                "E1_frozen",
                ledger=ledger,
            )
            if (
                config.data.num_workers != 0
                or data.train_dataloader().num_workers != 0
                or dev.loader.num_workers != 0
            ):
                raise ValueError("WSL E1-v3 forbids DataLoader workers")
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
    result["gpu_seconds"] = ledger.last
    result["overall_recorded_seconds"] = identity["prior_recorded_gpu_seconds"] + ledger.last
    if result["overall_recorded_seconds"] >= OVERALL_CEILING_SECONDS:
        raise RuntimeError("E1-v3 overall 24-hour cap exceeded")
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
        "parent-run",
    ):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--physical-batch", type=int, default=4)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--authorize-new-six-hour-run", action="store_true")
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
        identity, output = build_v3(
            args, protocol, args.parent_run, resume=args.resume_checkpoint is not None
        )
        print(
            {
                "run_key": output.name,
                "old_recorded_gpu_seconds": identity["prior_recorded_gpu_seconds"],
                "new_cap_seconds": identity["new_run_cap_seconds"],
                "train": len(views["train"]),
                "dev": len(views["dev"]),
                "workers": 0,
            }
        )
        return
    print(run_v3(args, protocol, views, args.parent_run))


if __name__ == "__main__":
    main()
