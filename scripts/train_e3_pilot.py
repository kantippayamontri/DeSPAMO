"""Fresh bounded E3 SpaMo pilot on verified adapted DINO spatial features."""

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
from despamo.training.e3_runtime import load_e3_views, validate_adapted_manifest
from despamo.training.pilot_factory import PilotDataModule, shared_tensor_hash
from despamo.training.pilot_runtime import pilot_config
from scripts.train_signer_pilot import (
    E1ResumeSnapshots,
    FullDevCheckpoints,
    _model,
    checkpoint_steps,
    validate_e1_resume_checkpoint,
)
from tools.dinov3.storage import verify_location, writer

E1_SHARED_HASH = "9e3a02201ca44b7ee1a1d62da3eba1308a91ee914cab983036dc6505aa8a4d88"
POLICY = "qwen-schema98-unreviewed-v1"
WARMUP = {4000: 1000, 6000: 2000, 8000: 2000}


def _validate_matched_e1(root: Path, protocol: dict, steps: int) -> dict:
    if not root.is_dir():
        raise ValueError("matched E1 run root missing")
    status = read_json(root / "run-status.json")
    budget = read_json(root / "budget.json")
    selected = read_json(root / "selected-checkpoint.json")
    identity = budget.get("identity", {})
    if (status.get("status") != "complete" or status.get("global_step") != steps
            or budget.get("status") != "closed"
            or identity.get("run_policy") != "signer-pilot-e1-fresh-retry-v3"
            or identity.get("variant") != "E1_frozen"
            or identity.get("protocol_hash") != protocol["protocol_hash"]
            or identity.get("initial_shared_hash") != E1_SHARED_HASH):
        raise ValueError("matched E1 run identity/status mismatch")
    schedule = checkpoint_steps(steps, WARMUP[steps])
    step = selected.get("step")
    if step not in schedule:
        raise ValueError("matched E1 selected step outside tier schedule")
    checkpoint = root / f"step-{step}.ckpt"
    if (not checkpoint.is_file()
            or file_hash(checkpoint) != selected.get("checkpoint_hash")):
        raise ValueError("matched E1 selected checkpoint hash mismatch")
    return {"checkpoint": checkpoint, "checkpoint_hash": selected["checkpoint_hash"],
            "step": step, "identity": identity, "run_key": root.name}


def require_matched_e3_config(e1_identity: dict, e3_config: dict, adapted_root: str,
                              *, steps: int) -> None:
    if e1_identity.get("initial_shared_hash") != E1_SHARED_HASH:
        raise ValueError("E3 matched E1 shared initialization mismatch")
    if steps not in WARMUP:
        raise ValueError("E3 unsupported SpaMo step tier")
    expected = OmegaConf.create(e1_identity["config"])
    expected.data.spatial_root = adapted_root
    expected.data.spatial_manifest = str(Path(adapted_root) / "complete/manifest.json")
    if (OmegaConf.to_container(expected, resolve=True) != e3_config
            or e3_config["data"]["num_workers"] != 0
            or e3_config["data"]["batch_size"] != 4
            or e3_config["model"]["warm_up_steps"] != WARMUP[steps]
            or e3_config["model"]["use_in_context"]
            or e3_config["trainer"]["max_steps"] != steps
            or e3_config["trainer"]["accumulate_grad_batches"] != 1
            or e3_config["seed"] != 0):
        raise ValueError("E3 SpaMo config not matched to frozen E1 except spatial source")


def build_e3_identity(protocol: dict, adapted: dict, config: dict,
                      gpu_cap_seconds: int, steps: int = 4000) -> dict:
    if (not 0 < gpu_cap_seconds <= CEILING_SECONDS
            or adapted.get("clips") != 7096
            or steps not in WARMUP
            or protocol.get("supervision_policy") != POLICY):
        raise ValueError("E3 adaptation/budget/policy identity invalid")
    result = {"run_policy": "signer-pilot-e3-adapted-bounded-v1",
              "variant": "E3_dino_lora", "protocol_hash": protocol["protocol_hash"],
              "split_hash": protocol["split_hash"], "seed": 0, "steps": steps,
              "config": config, "initial_shared_hash": E1_SHARED_HASH,
              "adaptation_checkpoint_hash": adapted["checkpoint_hash"],
              "adapted_manifest_hash": adapted["manifest_hash"],
              "adapted_frame_rows_hash": adapted["frame_rows_hash"],
              "adapted_encoder_key": adapted["encoder_key"],
              "gpu_cap_seconds": gpu_cap_seconds,
              "supervision_policy": POLICY, "human_review_status": "not_assessed"}
    return {**result, "run_key": "e3-only-" + digest(result)}


def select_e3_checkpoint(reports: list[dict], dev_ids: tuple[str, ...], steps: int) -> dict:
    if steps not in WARMUP:
        raise ValueError("E3 unsupported SpaMo step tier")
    expected = list(checkpoint_steps(steps, WARMUP[steps]))
    if [report.get("checkpoint_step") for report in reports] != expected:
        raise ValueError("E3 complete full-dev checkpoint schedule required")
    return choose_checkpoint(reports, dev_ids)


def _prepare(args):
    steps = getattr(args, "steps", 4000)
    if steps not in WARMUP:
        raise ValueError("unsupported SpaMo step tier")
    args.steps = steps
    original = {name: getattr(args, name) for name in
                ("protocol", "dataset", "text_manifest", "dino_root", "motion_root",
                 "motion_manifest", "annotation")}
    adaptation = read_json(args.adapted_root / "complete/identity.json")
    protocol, views = load_e3_views(original, args.adapted_root, adaptation)
    if protocol["protocol_hash"] != args.protocol.parent.name:
        raise ValueError("E3 pilot frozen protocol path mismatch")
    e1 = _validate_matched_e1(args.e1_run, protocol, args.steps)
    e1_identity = e1["identity"]
    config = pilot_config(args.hf_cache, args.adapted_root, args.motion_root,
                          args.annotation.parent, args.adapted_root / "complete/manifest.json",
                          args.motion_manifest, args.steps, 4, args.output_base)
    resolved = OmegaConf.to_container(config, resolve=True)
    require_matched_e3_config(e1_identity, resolved, str(args.adapted_root), steps=args.steps)
    ids = tuple(sorted(record["fileid"] for view in views.values() for record in view.records))
    adapted = validate_adapted_manifest(args.adapted_root, adaptation, ids)
    identity = build_e3_identity(protocol, adapted, resolved, args.gpu_cap_seconds, args.steps)
    identity["matched_e1_checkpoint_hash"] = e1["checkpoint_hash"]
    identity["adaptation_identity_hash"] = file_hash(args.adapted_root / "complete/identity.json")
    identity["code_hashes"] = {str(path): file_hash(Path(__file__).resolve().parents[1] / path)
                               for path in ("scripts/train_e3_pilot.py",
                                            "src/despamo/training/e3_runtime.py")}
    # Bind output name to final, complete identity including E1 and code hashes.
    identity["run_key"] = "e3-only-" + digest({k: v for k, v in identity.items()
                                               if k != "run_key"})
    output = args.output_base / identity["run_key"]
    verify_location(output, Path(__file__).resolve().parents[1])
    return identity, output, protocol, views, config


def run_e3(args):
    if not args.authorize_e3_run or not 0 < args.gpu_cap_seconds <= CEILING_SECONDS:
        raise ValueError("E3 translation training requires explicit six-hour-or-less cap")
    identity, output, protocol, views, config = _prepare(args)
    resume = args.resume_checkpoint
    if (resume is None and output.exists()) or (resume is not None and not output.is_dir()):
        raise ValueError("E3 output exists or resume root missing")
    if shutil.disk_usage(args.output_base).free < 160 * 1024**3:
        raise ValueError("E3 SpaMo checkpoint storage below 160 GiB")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("E3 SpaMo requires 12 GiB free CUDA memory")
    ledger = E1Ledger.create(output, identity, 0) if resume is None else None
    trainer = None
    old_alarm = signal.getsignal(signal.SIGALRM)

    def timeout(signum, frame):
        raise TimeoutError("E3 SpaMo GPU-time cap expired")

    try:
        with writer(output):
            if resume is not None:
                if (output / "selected-checkpoint.json").exists():
                    raise ValueError("E3 run already completed")
                spent = read_json(output / "budget.json")["gpu_seconds"]
                checkpoint = validate_e1_resume_checkpoint(resume, output, identity, spent)
                elapsed, step = checkpoint["gpu_seconds_cumulative"], checkpoint["global_step"]
                del checkpoint
                ledger = E1Ledger.resume(output, identity, elapsed, checkpoint_step=step)
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL,
                             max(1, args.gpu_cap_seconds - ledger.last - 1))
            model = _model(config, "E3_dino_lora", 768, args.steps,
                           {"supervision_policy": POLICY, "human_review_status": "not_assessed"})
            if shared_tensor_hash(model) != E1_SHARED_HASH:
                raise ValueError("E3 fresh shared initialization differs from E1/E2")
            model.run_metadata = identity
            ledger.charge(step=ledger.step, phase="model_loaded")
            data = PilotDataModule(views["train"], 4, 0, args.steps, collate_phoenix)
            dev_ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
            dev = FullDevCheckpoints(views["dev"], dev_ids, protocol["split_hash"],
                                     output, args.steps, WARMUP[args.steps], 4,
                                     "E3_dino_lora", ledger=ledger)
            if data.train_dataloader().num_workers or dev.loader.num_workers:
                raise ValueError("E3 WSL DataLoader worker policy changed")
            trainer = pl.Trainer(**dict(config.trainer),
                                 callbacks=[E1BudgetGuard(ledger), dev,
                                            E1ResumeSnapshots(output, ledger)])
            trainer.fit(model, datamodule=data, ckpt_path=str(resume) if resume else None)
            if trainer.global_step != args.steps:
                raise RuntimeError(f"E3 SpaMo did not complete {args.steps} optimizer steps")
            schedule = checkpoint_steps(args.steps, WARMUP[args.steps])
            reports = [read_json(output / f"dev-{step}.json") for step in schedule]
            best = select_e3_checkpoint(reports, dev_ids, args.steps)
            atomic_json(output / "selected-checkpoint.json",
                        {"step": best["checkpoint_step"],
                         "checkpoint_hash": best["checkpoint_hash"],
                         "metrics": best["metrics"], "protocol_hash": protocol["protocol_hash"],
                         "model_variant": "E3_dino_lora",
                         "adaptation_checkpoint_hash": identity["adaptation_checkpoint_hash"],
                         "human_review_status": "not_assessed", "supervision_policy": POLICY})
            result = {"status": "complete", "global_step": args.steps, "run_key": output.name,
                      "protocol_hash": protocol["protocol_hash"]}
            atomic_json(output / "run-status.json", result)
            return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        if ledger is not None:
            ledger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    for name in ("protocol", "dataset", "text-manifest", "dino-root", "motion-root",
                 "motion-manifest", "annotation", "adapted-root", "e1-run", "hf-cache",
                 "output-base"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--gpu-cap-seconds", required=True, type=int)
    parser.add_argument("--steps", type=int, default=4000, choices=sorted(WARMUP))
    parser.add_argument("--authorize-e3-run", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        identity, output, protocol, views, config = _prepare(args)
        print({"status": "ready", "run_key": output.name,
               "train": len(views["train"]), "dev": len(views["dev"]),
               "test_unscored": len(views["test"]),
               "adaptation_checkpoint_hash": identity["adaptation_checkpoint_hash"]})
        return
    print(run_e3(args))


if __name__ == "__main__":
    main()
