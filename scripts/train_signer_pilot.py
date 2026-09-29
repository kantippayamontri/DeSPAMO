"""Fresh matched signer-pilot preflight, bounded SpaMo profile and gated E1 run."""

import argparse
import gc
import math
import os
import re
import shutil
import signal
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.data.batch import collate_phoenix
from despamo.data.factors import UNREVIEWED_POLICY, FactorDataset, collate_factors
from despamo.data.signer_split import PhoenixTrainView
from despamo.evaluation.signer_pilot import choose_checkpoint, score_pilot_batches
from despamo.factory import build_model
from despamo.training.e1_budget import (
    CEILING_SECONDS,
    PRIOR_RECEIPTS,
    E1BudgetGuard,
    E1Ledger,
    initial_pilot_seconds,
)
from despamo.training.pilot_factory import (
    PilotBaselineModule,
    PilotDataModule,
    PilotFactorModule,
    PilotRunState,
    make_variant,
    shared_tensor_hash,
)
from despamo.training.pilot_replay import freeze_tier, verify_resume
from despamo.training.pilot_runtime import load_pilot_views, pilot_config
from tools.dinov3.storage import verify_location, writer

ROOT = Path(__file__).resolve().parents[1]
E1_SHARED_HASH = "9e3a02201ca44b7ee1a1d62da3eba1308a91ee914cab983036dc6505aa8a4d88"


def e1_only_identity(
    protocol_hash: str, config: dict, prior_seconds: float, receipts: dict
) -> dict:
    if (
        not protocol_hash
        or config.get("seed") != 0
        or config.get("data", {}).get("batch_size") != 4
        or config.get("model", {}).get("warm_up_steps") != 1000
        or config.get("trainer", {}).get("max_steps") != 4000
        or config.get("trainer", {}).get("accumulate_grad_batches") != 1
        or not 0 < prior_seconds < CEILING_SECONDS
        or not receipts
    ):
        raise ValueError("E1-only seed/batch/steps/budget/receipt policy mismatch")
    return {
        "run_policy": "signer-pilot-e1-only-bounded-v1",
        "protocol_hash": protocol_hash,
        "variant": "E1_frozen",
        "seed": 0,
        "config": config,
        "budget_ceiling_seconds": CEILING_SECONDS,
        "prior_gpu_seconds": prior_seconds,
        "prior_receipt_hashes": dict(receipts),
        "initial_shared_hash": E1_SHARED_HASH,
    }


def e1_only_preflight(args, protocol: dict, *, resume: bool = False) -> tuple[dict, Path]:
    """Dry immutable, budget and storage admission; no FLAN weights loaded."""
    if args.physical_batch != 4:
        raise ValueError("E1-only requires physical batch 4")
    paths = {name: args.output_base / name for name in PRIOR_RECEIPTS}
    prior_seconds, hashes = initial_pilot_seconds(paths, protocol["protocol_hash"])
    sampled = read_json(args.output_base / "profile-E1_frozen-b4.json")
    if sampled.get("initial_shared_tensor_hash") != E1_SHARED_HASH:
        raise ValueError("E1-only fresh shared initialization differs from measured profile")
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
    identity = e1_only_identity(
        protocol["protocol_hash"],
        OmegaConf.to_container(config, resolve=True),
        prior_seconds,
        hashes,
    )
    identity["code_hashes"] = {
        name: file_hash(ROOT / name)
        for name in (
            "scripts/train_signer_pilot.py",
            "src/despamo/training/e1_budget.py",
            "src/despamo/training/pilot_factory.py",
            "src/despamo/training/pilot_replay.py",
            "src/despamo/training/pilot_runtime.py",
            "src/despamo/training/baseline_module.py",
            "src/despamo/evaluation/signer_pilot.py",
            "uv.lock",
        )
    }
    output = args.output_base / ("e1-only-" + digest(identity))
    if output.is_symlink() or (output.exists() if not resume else not output.is_dir()):
        raise ValueError("E1-only run root exists or required resume root missing")
    if shutil.disk_usage(args.output_base).free < 160 * 1024**3:
        raise ValueError("E1-only external storage below 160 GiB checkpoint reserve")
    return identity, output


def validate_e1_resume_checkpoint(path: Path, output: Path, identity: dict, spent: float) -> dict:
    """Accept only latest complete checkpoint owned by this immutable E1 run."""
    candidate = re.fullmatch(r"(?:resume|step)-(\d+)\.ckpt", path.name)
    if (
        candidate is None
        or path.is_symlink()
        or path.resolve(strict=False).parent != output.resolve(strict=True)
        or not path.is_file()
    ):
        raise ValueError("E1 checkpoint must be owned complete run snapshot")
    snapshots = [
        int(match.group(1))
        for entry in output.glob("*.ckpt")
        if (match := re.fullmatch(r"(?:resume|step)-(\d+)\.ckpt", entry.name))
    ]
    step = int(candidate.group(1))
    if not snapshots or step != max(snapshots):
        raise ValueError("E1 resume requires latest complete saved checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)  # own locked run
    verify_resume(checkpoint, identity, step, spent)
    if checkpoint.get("run_metadata") != identity:
        raise ValueError("E1 checkpoint run identity changed")
    if checkpoint["pilot_sampler"].get("batches_yielded") != step:
        raise ValueError("E1 checkpoint sampler/step mismatch")
    return checkpoint


def checkpoint_steps(steps: int, warmup: int) -> tuple[int, int, int]:
    if steps not in (4000, 6000, 8000) or warmup != {4000: 1000, 6000: 2000, 8000: 2000}[steps]:
        raise ValueError("invalid frozen pilot checkpoint schedule")
    joint = steps - warmup
    return tuple(warmup + math.ceil(joint * fraction) for fraction in (0.25, 0.60, 1.0))


def dev_profile_ids(expected_ids: tuple[str, ...], count: int) -> tuple[str, ...]:
    if (
        not expected_ids
        or len(set(expected_ids)) != len(expected_ids)
        or not 2 <= count <= len(expected_ids)
    ):
        raise ValueError("invalid bounded dev profile count/IDs")
    return tuple(expected_ids[i * (len(expected_ids) - 1) // (count - 1)] for i in range(count))


def reject_failed_gate(gate: dict, protocol_hash: str) -> None:
    if (
        gate.get("protocol_hash") == protocol_hash
        and gate.get("status") == "failed_20_gpu_hour_feasibility_gate"
    ):
        raise ValueError("pilot GPU budget gate failed; E1 comparison cannot start")


def require_frozen_tier(receipt: dict, protocol_hash: str, physical_batch: int) -> None:
    if (
        receipt.get("protocol_hash") != protocol_hash
        or receipt.get("physical_batch") != physical_batch
        or len(receipt.get("source_profile_hashes", [])) != 2
    ):
        raise ValueError("pilot tier missing source/profile identity")
    try:
        expected = freeze_tier(receipt["profile"], receipt["spent_gpu_seconds_at_freeze"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("pilot tier lacks complete measured GPU profile") from exc
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("pilot tier differs from frozen measured profile")


class StepTimer(pl.Callback):
    def __init__(self):
        super().__init__()
        self.last = None
        self.last_step = 0
        self.per_step = {}

    def on_train_start(self, trainer, pl_module):
        torch.cuda.synchronize()
        self.last = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step != self.last_step:
            torch.cuda.synchronize()
            now = time.monotonic()
            self.per_step[trainer.global_step] = now - self.last
            self.last, self.last_step = now, trainer.global_step


class FullDevCheckpoints(pl.Callback):
    def __init__(
        self, dev, expected, split_hash, output, steps, warmup, batch_size, variant, ledger=None
    ):
        super().__init__()
        self.loader = DataLoader(
            dev,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_phoenix,
            generator=torch.Generator().manual_seed(0),
        )
        self.expected = expected
        self.split_hash = split_hash
        self.output = output
        self.variant = variant
        self.ledger: E1Ledger | None = ledger
        self.steps = checkpoint_steps(steps, warmup)
        self.completed = set()

    def on_train_start(self, trainer, pl_module):
        for step in self.steps:
            target = self.output / f"step-{step}.ckpt"
            report_path = self.output / f"dev-{step}.json"
            if report_path.exists():
                report = read_json(report_path)
                if (
                    not target.is_file()
                    or tuple(item["clip_id"] for item in report["items"]) != self.expected
                    or report.get("checkpoint_hash") != file_hash(target)
                    or report.get("checkpoint_step") != step
                    or report.get("split_hash") != self.split_hash
                ):
                    raise ValueError("pilot saved dev report/checkpoint mismatch on resume")
                self.completed.add(step)
            elif target.is_file():
                if trainer.global_step != step:
                    raise ValueError(
                        "pilot incomplete dev report requires matching checkpoint resume"
                    )
                self._score_saved(pl_module, step, target)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step not in self.steps or step in self.completed:
            return
        target = self.output / f"step-{step}.ckpt"
        if target.exists():
            raise ValueError("pilot checkpoint exists; refusing overwrite")
        if self.ledger is not None:
            self.ledger.require_remaining(300, step=step, phase="before_checkpoint")
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_name(f".{target.name}.{os.getpid()}.pending")
        if pending.exists():
            raise ValueError("pilot pending checkpoint exists; refusing overwrite")
        trainer.save_checkpoint(pending)
        os.replace(pending, target)
        self._score_saved(pl_module, step, target)

    def _score_saved(self, pl_module, step, target):
        try:
            on_batch_end = None
            if self.ledger is not None:
                self.ledger.require_remaining(300, step=step, phase="before_dev")
                seen = 0

                def on_batch_end(count):
                    nonlocal seen
                    seen += count
                    self.ledger.require_remaining(
                        30, step=step, phase=f"dev-{seen}/{len(self.expected)}"
                    )

            report = score_pilot_batches(
                pl_module,
                self.loader,
                self.expected,
                "dev",
                self.split_hash,
                **({"on_batch_end": on_batch_end} if on_batch_end is not None else {}),
            )
            report["checkpoint_step"] = step
            report["checkpoint_hash"] = file_hash(target)
            report["model_variant"] = self.variant
            atomic_json(self.output / f"dev-{step}.json", report)
            self.completed.add(step)
        finally:
            pl_module.train()


class E1ResumeSnapshots(pl.Callback):
    """Immutable full Lightning snapshots before first dev and during joint training."""

    def __init__(self, output: Path, ledger: E1Ledger, interval: int = 500):
        super().__init__()
        if interval < 1:
            raise ValueError("E1 resume interval must be positive")
        self.output, self.ledger, self.interval = output, ledger, interval
        self.completed: set[int] = set()

    def on_train_start(self, trainer, pl_module):
        self.completed = {
            step
            for step in range(self.interval, trainer.global_step + 1, self.interval)
            if (self.output / f"resume-{step}.ckpt").is_file()
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step <= 0 or step % self.interval or step in self.completed:
            return
        target = self.output / f"resume-{step}.ckpt"
        pending = target.with_name(f".{target.name}.{os.getpid()}.pending")
        if target.exists() or pending.exists():
            raise ValueError("E1 resume snapshot exists; refusing overwrite")
        self.ledger.require_remaining(300, step=step, phase="before_resume_snapshot")
        trainer.save_checkpoint(pending)
        os.replace(pending, target)
        self.ledger.charge(step=step, phase="resume_snapshot_saved")
        self.completed.add(step)


def _model(config, variant, width, steps, provenance):
    def wrap(base):
        if variant == "E2_projector":
            return PilotFactorModule(base, width, steps, provenance)
        return PilotBaselineModule(base, steps)

    return make_variant(variant, lambda: build_model(config), wrap, seed=0)


def _train_data(variant, train, args, protocol):
    if variant != "E2_projector":
        return train, collate_phoenix, None
    joined = FactorDataset(
        train,
        args.dataset,
        args.text_manifest,
        args.dino_root / "complete/frame_rows.json",
        args.dino_root / "complete/manifest.json",
        UNREVIEWED_POLICY,
        logical_clip_ids=tuple(protocol["split"]["groups"]["train"]["clip_ids"]),
        expected_parent_records_hash=protocol["records_hash"],
    )
    if joined.provenance["valid_count"] != 5641 or joined.provenance["failed_count"] != 105:
        raise ValueError("E2 training factor masks/count changed")
    return joined, collate_factors, joined.provenance


def _fit_variant(args, protocol, views, variant, *, profile, tier=None):
    profile_path = args.output_base / f"profile-{variant}-b{args.physical_batch}.json"
    if profile and profile_path.exists():
        raise ValueError("pilot profile already exists; preserving measured artifact")
    steps = 4000 if profile else tier["spaMo_steps"]
    config = pilot_config(
        args.hf_cache,
        args.dino_root,
        args.motion_root,
        args.annotation.parent,
        args.dino_root / "complete/manifest.json",
        args.motion_manifest,
        steps,
        args.physical_batch,
        args.output_base,
    )
    train, collate, provenance = _train_data(variant, views["train"], args, protocol)
    identity = {
        "protocol_hash": protocol["protocol_hash"],
        "variant": variant,
        "seed": 0,
        "config": OmegaConf.to_container(config, resolve=True),
        "package_lock_hash": file_hash(ROOT / "uv.lock"),
        "trainer_code_hash": file_hash(Path(__file__)),
        "factor_provenance": provenance,
        "tier": tier,
    }
    output = args.output_base / digest(identity)
    if output.exists():
        raise ValueError("pilot run/profile key exists; refusing to overwrite prior artifacts")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("pilot profile/training needs at least 12 GiB free GPU")
    started = time.monotonic()
    model = _model(
        config,
        variant,
        768,
        steps,
        provenance
        or {"supervision_policy": UNREVIEWED_POLICY, "human_review_status": "not_assessed"},
    )
    model.run_metadata = identity
    shared_hash = shared_tensor_hash(model)
    if profile:
        model.warm_up_steps = 2
        config.trainer.max_steps = 6
    sampler = PilotDataModule(
        train,
        args.physical_batch,
        0,
        (6 if profile else steps) * config.trainer.accumulate_grad_batches,
        collate,
    )
    callbacks = [PilotRunState(identity, 0.0 if profile else tier["spent_gpu_seconds_at_freeze"])]
    timer = StepTimer()
    callbacks.append(timer)
    if not profile:
        callbacks.append(
            FullDevCheckpoints(
                views["dev"],
                tuple(protocol["split"]["groups"]["dev"]["clip_ids"]),
                protocol["split_hash"],
                output,
                steps,
                config.model.warm_up_steps,
                args.physical_batch,
                variant,
            )
        )
    trainer = pl.Trainer(**dict(config.trainer), callbacks=callbacks)
    trainer.fit(model, datamodule=sampler)
    if profile:
        rate = max(timer.per_step[5], timer.per_step[6])
        full_ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
        sampled_ids = dev_profile_ids(full_ids, args.dev_profile_clips)
        sample_view = PhoenixTrainView(views["dev"].parent, sampled_ids)
        dev_started = time.monotonic()
        report = score_pilot_batches(
            model,
            DataLoader(sample_view, batch_size=args.physical_batch, collate_fn=collate_phoenix),
            sampled_ids,
            "dev",
            protocol["split_hash"],
        )
        dev_seconds = time.monotonic() - dev_started
        record = {
            "status": "complete",
            "variant": variant,
            "protocol_hash": protocol["protocol_hash"],
            "initial_shared_tensor_hash": shared_hash,
            "physical_batch": args.physical_batch,
            "joint_seconds_per_step": rate,
            "dev_seconds_per_full_split": dev_seconds * len(full_ids) / len(sampled_ids) * 1.5,
            "dev_profile_elapsed_seconds": dev_seconds,
            "dev_profile_count": len(report["items"]),
            "dev_full_count": len(full_ids),
            "dev_estimate_safety_factor": 1.5,
            "gpu_seconds": time.monotonic() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "input_identity": identity,
        }
        atomic_json(profile_path, record)
        return record
    return {"step": trainer.global_step, "output": str(output), "shared_hash": shared_hash}


def run_e1_only(args, protocol: dict, views: dict) -> dict:
    """Start one separately bounded E1 run; no failed three-method tier required."""
    resume = getattr(args, "resume_checkpoint", None)
    identity, output = e1_only_preflight(args, protocol, resume=resume is not None)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("E1-only requires at least 12 GiB free CUDA memory")
    ledger = (
        E1Ledger.create(output, identity, identity["prior_gpu_seconds"]) if resume is None else None
    )
    trainer = None
    original_term = signal.getsignal(signal.SIGTERM)
    original_alarm = signal.getsignal(signal.SIGALRM)
    alarm_armed = False

    def stop_on_term(signum, frame):
        raise KeyboardInterrupt("E1-only received termination signal")

    def stop_on_alarm(signum, frame):
        raise RuntimeError("E1 six-hour GPU wall budget expired")

    signal.signal(signal.SIGTERM, stop_on_term)
    signal.signal(signal.SIGALRM, stop_on_alarm)
    try:
        with writer(output):
            if resume is not None:
                if (output / "selected-checkpoint.json").exists():
                    raise ValueError("E1-only run already complete")
                spent = read_json(output / "budget.json")["gpu_seconds"]
                checkpoint = validate_e1_resume_checkpoint(resume, output, identity, spent)
                checkpoint_seconds = checkpoint["gpu_seconds_cumulative"]
                checkpoint_step = checkpoint["global_step"]
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
                {"supervision_policy": UNREVIEWED_POLICY, "human_review_status": "not_assessed"},
            )
            if shared_tensor_hash(model) != E1_SHARED_HASH:
                raise ValueError("E1-only fresh shared initialization differs from preflight")
            model.run_metadata = identity
            ledger.charge(step=0, phase="model_loaded")
            data = PilotDataModule(views["train"], 4, 0, 4000, collate_phoenix)
            dev_ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
            trainer = pl.Trainer(
                **dict(config.trainer),
                callbacks=[
                    E1BudgetGuard(ledger),
                    FullDevCheckpoints(
                        views["dev"],
                        dev_ids,
                        protocol["split_hash"],
                        output,
                        4000,
                        1000,
                        4,
                        "E1_frozen",
                        ledger=ledger,
                    ),
                    E1ResumeSnapshots(output, ledger),
                ],
            )
            trainer.fit(model, datamodule=data, ckpt_path=str(resume) if resume else None)
            if trainer.global_step == 4000:
                reports = [read_json(output / f"dev-{step}.json") for step in (1750, 2800, 4000)]
                selected = choose_checkpoint(reports, dev_ids)
                atomic_json(
                    output / "selected-checkpoint.json",
                    {
                        "step": selected["checkpoint_step"],
                        "checkpoint_hash": selected["checkpoint_hash"],
                        "metrics": selected["metrics"],
                        "protocol_hash": protocol["protocol_hash"],
                    },
                )
            outcome = {
                "status": "complete" if trainer.global_step == 4000 else "partial",
                "global_step": trainer.global_step,
                "run_key": output.name,
                "protocol_hash": protocol["protocol_hash"],
            }
    except BaseException as error:
        outcome = {
            "status": "partial",
            "global_step": trainer.global_step if trainer is not None else 0,
            "run_key": output.name,
            "protocol_hash": protocol["protocol_hash"],
            "error_type": type(error).__name__,
        }
        if ledger is not None:
            atomic_json(output / "run-status.json", outcome)
        raise
    finally:
        if alarm_armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if ledger is not None:
            ledger.close()
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGALRM, original_alarm)
    outcome["gpu_seconds"] = ledger.last
    atomic_json(output / "run-status.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight", "preflight-e1-only", "profile", "run-e1", "run-e1-only"),
        required=True,
    )
    for name in (
        "protocol",
        "dataset",
        "text-manifest",
        "dino-root",
        "motion-root",
        "motion-manifest",
        "annotation",
        "hf-cache",
        "output-base",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--physical-batch", type=int, choices=(2, 4), default=2)
    parser.add_argument("--tier", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--profile-variant", choices=("E1_frozen", "E2_projector"))
    parser.add_argument("--dev-profile-clips", type=int, default=24)
    args = parser.parse_args()
    if not 2 <= args.dev_profile_clips <= 24:
        raise ValueError("bounded dev profiling requires 2–24 clips")
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
        print({role: len(view) for role, view in views.items()})
        return
    if args.mode == "preflight-e1-only":
        identity, output = e1_only_preflight(args, protocol)
        print({"run_key": output.name, "prior_gpu_seconds": identity["prior_gpu_seconds"]})
        return
    if args.mode == "run-e1-only":
        print(run_e1_only(args, protocol, views))
        return
    if args.resume_checkpoint is not None:
        raise ValueError("resume checkpoint is supported only by run-e1-only")
    if args.mode == "profile":
        choices = ("E1_frozen", "E2_projector")
        for variant in (args.profile_variant,) if args.profile_variant else choices:
            print(_fit_variant(args, protocol, views, variant, profile=True))
            gc.collect()
            torch.cuda.empty_cache()
        return
    failed_gate = args.output_base / "profile-gate-b4.json"
    if failed_gate.is_file():
        reject_failed_gate(read_json(failed_gate), protocol["protocol_hash"])
    if args.tier is None:
        raise ValueError("E1 requires frozen GPU tier receipt")
    tier = read_json(args.tier)
    require_frozen_tier(tier, protocol["protocol_hash"], args.physical_batch)
    print(_fit_variant(args, protocol, views, "E1_frozen", profile=False, tier=tier))


if __name__ == "__main__":
    main()
