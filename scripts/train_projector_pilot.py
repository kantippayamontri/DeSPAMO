"""Versioned residual-MLP and longer-linear downstream ablation on cached E3 features."""

import argparse
import gc
import math
import os
import random
import re
import shutil
import signal
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import cast

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.data.batch import collate_phoenix
from despamo.evaluation.signer_pilot import choose_checkpoint
from despamo.factory import build_model
from despamo.models.residual_projector import ResidualMLPProjector, enable_residual_projectors
from despamo.models.visual_adapter import SpaMoVisualAdapter
from despamo.training.e1_budget import CEILING_SECONDS, E1BudgetGuard, E1Ledger
from despamo.training.e3_runtime import load_e3_views, validate_adapted_manifest
from despamo.training.pilot_factory import PilotBaselineModule, PilotDataModule, shared_tensor_hash
from despamo.training.pilot_replay import capture_rng, restore_rng
from scripts.evaluate_e3_pilot import DECODER, _validate_e3_selection
from scripts.train_e3_pilot import E1_SHARED_HASH, POLICY
from scripts.train_signer_pilot import (
    E1ResumeSnapshots,
    FullDevCheckpoints,
    _model,
    validate_e1_resume_checkpoint,
)
from tools.dinov3.storage import verify_location, writer

RUN_POLICY = "signer-pilot-projector-ablation-v1"
ARMS = {"residual": (8000, 16000), "linear": (16000,)}
WARMUP = 2000
HIDDEN = 1024
INITIALIZATION_POLICY = "projector-common-initialization-v1"


def build_seeded_projector_model(config: DictConfig, steps: int, seed: int) -> PilotBaselineModule:
    """Construct the common linear model; residual attachment is a separate seeded operation."""
    if type(seed) is not int or seed not in (0, 1):
        raise ValueError("unsupported projector initialization seed")
    if seed == 0:
        return cast(PilotBaselineModule, _model(
            config, "E3_dino_lora", 768, steps,
            {"supervision_policy": POLICY, "human_review_status": "not_assessed"},
        ))
    if type(config.get("seed")) is not int or config.seed != seed:
        raise ValueError("projector model/config seed mismatch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    base = build_model(config)
    initial_hash = shared_tensor_hash(base)
    model = PilotBaselineModule(base, steps)
    if shared_tensor_hash(model) != initial_hash:
        raise ValueError("projector wrapper altered fresh shared tensors")
    model.rng = random.Random(seed)
    return model


def build_seeded_projector_data(train, steps: int, seed: int) -> PilotDataModule:
    """Reseed a fresh local sampler without relaxing the original pilot's seed-0 guard."""
    if type(seed) is not int or seed not in (0, 1):
        raise ValueError("unsupported projector data seed")
    data = PilotDataModule(train, 4, 0, steps, collate_phoenix)
    if seed == 1:
        data.sampler.seed = seed
        data.sampler.generator.manual_seed(seed)
        data.sampler.order = torch.randperm(len(train), generator=data.sampler.generator).tolist()
    return data


def projector_checkpoint_steps(steps: int) -> tuple[int, int, int]:
    if type(steps) is not int or steps not in (8000, 16000):
        raise ValueError("unsupported projector experiment horizon")
    points = tuple(WARMUP + math.ceil((steps - WARMUP) * f) for f in (0.25, 0.60, 1.0))
    return points[0], points[1], points[2]


def _require_projector_config(adapted, config, projector, steps, cap, seed):
    if (projector not in ARMS or steps not in ARMS[projector]
            or type(seed) is not int or seed not in (0, 1) or (seed == 1 and steps != 16000)
            or type(cap) is not int or not 0 < cap <= CEILING_SECONDS
            or adapted.get("clips") != 7096 or config.get("seed") != seed
            or config["model"]["warm_up_steps"] != WARMUP
            or config["trainer"]["max_steps"] != steps
            or config["trainer"]["accumulate_grad_batches"] != 1
            or config["data"]["batch_size"] != 4 or config["data"]["num_workers"] != 0):
        raise ValueError("projector experiment architecture/horizon/config/budget invalid")


def _initialization_profile_binding(protocol, adapted, config, steps, seed,
                                    baseline_hash, code_hashes) -> dict:
    return {
        "profile_policy": INITIALIZATION_POLICY, "seed": seed, "steps": steps,
        "config_hash": digest(config), "code_hashes": code_hashes,
        "input_identity": {
            "protocol_hash": protocol["protocol_hash"], "split_hash": protocol["split_hash"],
            "protocol_identity_hash": digest(protocol),
            "baseline_checkpoint_hash": baseline_hash,
            "adaptation_checkpoint_hash": adapted["checkpoint_hash"],
            "adapted_encoder_key": adapted["encoder_key"],
            "adapted_manifest_hash": adapted["manifest_hash"],
            "adapted_frame_rows_hash": adapted["frame_rows_hash"], "clips": adapted["clips"],
        },
    }


def _validate_initialization_profile(profile, binding) -> dict:
    """Validate a canonical attestation against freshly verified inputs, without model loading."""
    if not isinstance(profile, dict) or set(profile) != set(binding) | {
        "initial_base_hash", "profile_hash"
    }:
        raise ValueError("projector initialization profile missing or malformed")
    initial_hash = profile["initial_base_hash"]
    if (type(profile.get("seed")) is not int or profile["seed"] != 1
            or any(profile.get(key) != value for key, value in binding.items())
            or not isinstance(initial_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", initial_hash) is None
            or initial_hash == E1_SHARED_HASH
            or profile["profile_hash"] != digest({key: value for key, value in profile.items()
                                                  if key != "profile_hash"})):
        raise ValueError("projector initialization profile seed/config/code/input/hash mismatch")
    return profile


def _read_initialization_profile(path, binding) -> dict:
    if path is None or path.is_symlink() or not path.is_file():
        raise ValueError("seed1 requires a regular verified initialization profile")
    try:
        profile = read_json(path)
    except (ValueError, OSError) as error:
        raise ValueError("projector initialization profile unreadable") from error
    return _validate_initialization_profile(profile, binding)


def build_projector_identity(protocol, adapted, config, projector, steps, cap,
                              baseline_hash, code_hashes, seed=0, *,
                              initial_base_hash=E1_SHARED_HASH, initialization_profile=None):
    """Keep legacy seed-0 identities exact; seed 1 must carry a bound CPU attestation."""
    _require_projector_config(adapted, config, projector, steps, cap, seed)
    if seed == 0 and (initial_base_hash != E1_SHARED_HASH or initialization_profile is not None):
        raise ValueError("seed0 retains frozen initialization without an initialization profile")
    if seed == 1:
        binding = _initialization_profile_binding(protocol, adapted, config, steps, seed,
                                                  baseline_hash, code_hashes)
        initialization_profile = _validate_initialization_profile(
            initialization_profile, binding)
        if initial_base_hash != initialization_profile["initial_base_hash"]:
            raise ValueError("projector initialization profile common hash mismatch")
    identity = {
        "run_policy": RUN_POLICY, "variant": f"E3_projector_{projector}",
        "projector": projector, "steps": steps, "seed": seed,
        "residual_hidden_dim": HIDDEN if projector == "residual" else None,
        "residual_seed": seed + 1 if projector == "residual" else None,
        "warmup_steps": WARMUP, "lr_warmup_steps": int(steps * 0.1),
        "dev_checkpoints": list(projector_checkpoint_steps(steps)),
        "resume_retention": 2, "snapshot_interval": 500,
        "protocol_hash": protocol["protocol_hash"], "split_hash": protocol["split_hash"],
        "config": config, "initial_base_hash": initial_base_hash,
        "baseline_checkpoint_hash": baseline_hash,
        "adaptation_checkpoint_hash": adapted["checkpoint_hash"],
        "adapted_encoder_key": adapted["encoder_key"],
        "adapted_manifest_hash": adapted["manifest_hash"],
        "adapted_frame_rows_hash": adapted["frame_rows_hash"],
        "gpu_cap_seconds": cap, "code_hashes": code_hashes,
        "supervision_policy": POLICY, "human_review_status": "not_assessed",
    }
    if seed == 1:
        assert isinstance(initialization_profile, dict)
        identity["initialization_profile_hash"] = initialization_profile["profile_hash"]
    return {**identity, "run_key": f"projector-{projector}-{steps}-" + digest(identity)}


class BoundedResumeSnapshots(E1ResumeSnapshots):
    """Prune only this new-policy run's older resumes after a successful atomic save."""

    def __init__(self, output, ledger, keep=2):
        if (type(keep) is not int or keep < 2 or output.resolve() != ledger.root.resolve()
                or ledger.identity.get("run_policy") != RUN_POLICY):
            raise ValueError("bounded retention requires an owned projector run and two snapshots")
        super().__init__(output, ledger)
        self.keep = keep

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step <= 0 or step % self.interval or step in self.completed:
            return
        target = self.output / f"resume-{step}.ckpt"
        pending = target.with_name(f".{target.name}.{os.getpid()}.pending")
        if target.exists() or pending.exists():
            raise ValueError("projector resume snapshot exists; refusing overwrite")
        self.ledger.require_remaining(300, step=step, phase="before_resume_snapshot")
        try:
            trainer.save_checkpoint(pending)
            with pending.open("rb") as saved:
                os.fsync(saved.fileno())
            os.replace(pending, target)
            directory = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            pending.unlink(missing_ok=True)
        self.ledger.charge(step=step, phase="resume_snapshot_saved")
        self.completed.add(step)
        snapshots = []
        for path in self.output.glob("resume-*.ckpt"):
            match = re.fullmatch(r"resume-(\d+)\.ckpt", path.name)
            if match is None or path.is_symlink() or path.resolve().parent != self.output.resolve():
                raise ValueError("unsafe projector resume snapshot")
            snapshots.append((int(match.group(1)), path))
        snapshots.sort()
        for old_step, path in snapshots[:-self.keep]:
            path.unlink()
            self.completed.discard(old_step)
        directory = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def projector_gradient_smoke(model, batch) -> dict:
    """Check joint CE+VT gradients, then restore RNG, buffers and modes without updating weights."""
    rng = capture_rng()
    prompt_rng = model.rng.getstate()
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    losses = None
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if device.type == "cuda" else nullcontext())
        with context:
            losses = model.compute_losses(batch)
            joint = losses["loss"] + model.vt_weight * losses["contra_loss"]
        if not torch.isfinite(joint):
            raise FloatingPointError("projector joint gradient smoke loss is non-finite")
        joint.backward()
        norms = {}
        for name, projector in (("spatial", model.visual_adapter.spatial_projector),
                                ("motion", model.visual_adapter.motion_projector)):
            if isinstance(projector, ResidualMLPProjector):
                output = projector.refinement[3]
                assert isinstance(output, torch.nn.Linear)
                parameter = output.weight
            elif isinstance(projector, torch.nn.Linear):
                parameter = projector.weight
            else:
                raise ValueError("unsupported projector gradient smoke module")
            if (parameter.grad is None or not torch.isfinite(parameter.grad).all()
                    or parameter.grad.norm() == 0):
                raise RuntimeError(f"projector gradient smoke missing {name} gradient")
            norms[name] = float(parameter.grad.norm().detach().cpu())
        language = [param.grad for name, param in model.named_parameters()
                    if name.startswith("language_model.")
                    and param.requires_grad and param.grad is not None]
        if not language or not all(torch.isfinite(grad).all() for grad in language):
            raise RuntimeError("projector smoke language gradients missing/non-finite")
        norms["language"] = sum(float(grad.norm().detach().cpu()) for grad in language)
        if norms["language"] == 0:
            raise RuntimeError("projector smoke language gradients are zero")
        return {"status": "passed", "objective": "joint-ce-vt", "optimizer_updates": 0,
                "joint_loss": float(joint.detach().cpu()), "gradient_norms": norms,
                "cuda_peak_bytes": (torch.cuda.max_memory_allocated()
                                    if device.type == "cuda" else 0)}
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        for module, mode in modes:
            module.training = mode
        model.rng.setstate(prompt_rng)
        restore_rng(rng)
        del losses
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def projector_code_hashes(project: Path) -> dict[str, str]:
    sources = list((project / "src/despamo").rglob("*.py")) + [
        project / name for name in (
            "scripts/train_projector_pilot.py", "scripts/train_signer_pilot.py",
            "scripts/train_e3_pilot.py", "scripts/evaluate_e3_pilot.py",
            "tools/dinov3/storage.py", "tools/dinov3/identity.py", "pyproject.toml", "uv.lock",
        )
    ]
    return {str(path.relative_to(project)): file_hash(path) for path in sorted(sources)}


def validate_matched_linear_control(root: Path, candidate: dict,
                                    dev_ids: tuple[str, ...]) -> dict:
    """Bind a completed equal-budget linear control, including its full-dev selection."""
    budget = read_json(root / "budget.json")
    status = read_json(root / "run-status.json")
    selected = read_json(root / "selected-checkpoint.json")
    control = budget.get("identity", {})
    profile_hash = candidate.get("initialization_profile_hash")
    shared = (
        "protocol_hash", "split_hash", "steps", "seed", "warmup_steps", "lr_warmup_steps",
        "dev_checkpoints", "resume_retention", "snapshot_interval", "initial_base_hash",
        "baseline_checkpoint_hash", "adaptation_checkpoint_hash", "adapted_encoder_key",
        "adapted_manifest_hash", "adapted_frame_rows_hash", "gpu_cap_seconds",
        "supervision_policy", "human_review_status", "config",
        "initialization_profile_hash",
    )
    if (root.is_symlink() or budget.get("status") != "closed"
            or status.get("status") != "complete" or status.get("global_step") != 16000
            or control.get("run_policy") != RUN_POLICY or control.get("projector") != "linear"
            or control.get("variant") != "E3_projector_linear" or control.get("steps") != 16000
            or control.get("run_key") != root.name
            or candidate.get("projector") != "residual" or candidate.get("steps") != 16000
            or (candidate.get("seed") == 1 and (
                not isinstance(profile_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", profile_hash) is None))
            or any(control.get(key) != candidate.get(key) for key in shared)):
        raise ValueError("matched control must be complete linear 16000 with the same recipe")
    entrypoint = "scripts/train_projector_pilot.py"
    control_sources = {key: value for key, value in control.get("code_hashes", {}).items()
                       if key != entrypoint}
    candidate_sources = {key: value for key, value in candidate.get("code_hashes", {}).items()
                         if key != entrypoint}
    if not control_sources or control_sources != candidate_sources:
        raise ValueError("matched control reused training/model/storage source hashes differ")
    reports = [read_json(root / f"dev-{step}.json") for step in projector_checkpoint_steps(16000)]
    if any(report.get("checkpoint_step") != step or report.get("decoding") != DECODER
           or report.get("split_hash") != candidate["split_hash"]
           or report.get("model_variant") != control["variant"]
           for step, report in zip(projector_checkpoint_steps(16000), reports, strict=True)):
        raise ValueError("matched control dev schedule/decoder identity mismatch")
    best = choose_checkpoint(reports, dev_ids)
    checkpoint = root / f"step-{selected.get('step')}.ckpt"
    if (selected.get("step") != best["checkpoint_step"]
            or selected.get("checkpoint_hash") != best["checkpoint_hash"]
            or selected.get("metrics") != best["metrics"]
            or selected.get("model_variant") != control["variant"]
            or selected.get("protocol_hash") != candidate["protocol_hash"]
            or selected.get("adaptation_checkpoint_hash") != candidate["adaptation_checkpoint_hash"]
            or checkpoint.is_symlink() or not checkpoint.is_file()
            or file_hash(checkpoint) != selected["checkpoint_hash"]):
        raise ValueError("matched control selected checkpoint/hash mismatch")
    return {"run_key": root.name, "identity_hash": digest(control),
            "checkpoint_hash": selected["checkpoint_hash"], "selected_step": selected["step"],
            "metrics": selected["metrics"], "budget_hash": file_hash(root / "budget.json"),
            "selection_hash": file_hash(root / "selected-checkpoint.json")}


def _prepare_projector_inputs(args):
    """Verify frozen inputs and resolve the common recipe before any profile is required."""
    seed = getattr(args, "seed", 0)
    if type(seed) is not int or seed not in (0, 1):
        raise ValueError("unsupported projector training seed")
    original = {name: getattr(args, name) for name in
                ("protocol", "dataset", "text_manifest", "dino_root", "motion_root",
                 "motion_manifest", "annotation")}
    adaptation = read_json(args.adapted_root / "complete/identity.json")
    protocol, views = load_e3_views(original, args.adapted_root, adaptation)
    dev_ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
    baseline = _validate_e3_selection(args.baseline_run, protocol, dev_ids)
    reference = baseline["identity"]
    if (reference.get("steps") != 8000 or reference["adapted_encoder_key"] != digest(adaptation)
            or adaptation["checkpoint_hash"] != reference["adaptation_checkpoint_hash"]):
        raise ValueError("projector baseline/cached feature identity mismatch")
    config = OmegaConf.create(reference["config"])
    assert isinstance(config, DictConfig)
    if config.seed != 0:
        raise ValueError("projector requires the verified seed0 baseline config")
    required = {"spatial_root": str(args.adapted_root),
                "spatial_manifest": str(args.adapted_root / "complete/manifest.json"),
                "motion_root": str(args.motion_root), "motion_manifest": str(args.motion_manifest),
                "annotation_root": str(args.annotation.parent)}
    if any(config.data[name] != value for name, value in required.items()):
        raise ValueError("projector data paths differ from verified baseline")
    config.trainer.max_steps = args.steps
    config.trainer.default_root_dir = str(args.output_base)
    if seed == 1:
        config.seed = seed
    ids = tuple(sorted(row["fileid"] for view in views.values() for row in view.records))
    adapted = validate_adapted_manifest(args.adapted_root, adaptation, ids)
    project = Path(__file__).resolve().parents[1]
    hashes = projector_code_hashes(project)
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    _require_projector_config(adapted, resolved, args.projector, args.steps,
                              args.gpu_cap_seconds, seed)
    return protocol, views, config, adapted, baseline["checkpoint_hash"], hashes


def initialize_projector(args) -> dict:
    """Publish/reuse seed-1 common initialization on CPU; no optimizer or training is built."""
    path = getattr(args, "initialization_profile", None)
    if (getattr(args, "seed", 0) != 1 or args.projector != "linear" or args.steps != 16000
            or path is None or getattr(args, "resume_checkpoint", None) is not None
            or getattr(args, "matched_control_run", None) is not None):
        raise ValueError("initialize requires seed1 linear16000 and --initialization-profile")
    if path.is_symlink():
        raise ValueError("unsafe initialization profile symlink")
    protocol, _, config, adapted, baseline_hash, hashes = _prepare_projector_inputs(args)
    resolved = OmegaConf.to_container(config, resolve=True)
    binding = _initialization_profile_binding(protocol, adapted, resolved, args.steps, 1,
                                              baseline_hash, hashes)
    project = Path(__file__).resolve().parents[1]
    verify_location(path, project)
    with writer(path.parent):
        if path.exists() or path.is_symlink():
            profile = _read_initialization_profile(path, binding)
            status = "reused"
        else:
            model = build_seeded_projector_model(config, args.steps, 1)
            if any(value.device.type != "cpu" for value in
                   (*model.parameters(), *model.buffers())):
                raise ValueError("initialization profile requires CPU common tensors")
            initial_hash = shared_tensor_hash(model)
            del model
            gc.collect()
            if projector_code_hashes(project) != hashes:
                raise ValueError("initialization profile code changed during CPU construction")
            profile = {**binding, "initial_base_hash": initial_hash}
            profile["profile_hash"] = digest(profile)
            _validate_initialization_profile(profile, binding)
            atomic_json(path, profile)
            status = "initialized"
    return {"status": status, "initialization_profile": str(path),
            "initialization_profile_hash": profile["profile_hash"],
            "initial_base_hash": profile["initial_base_hash"], "seed": 1}


def prepare_projector(args):
    matched_control = getattr(args, "matched_control_run", None)
    if args.projector == "residual" and args.steps == 16000:
        if matched_control is None:
            raise ValueError("residual 16000 requires a completed matched linear control")
    elif matched_control is not None:
        raise ValueError("matched control is only supported for residual 16000")
    protocol, views, config, adapted, baseline_hash, hashes = _prepare_projector_inputs(args)
    seed = getattr(args, "seed", 0)
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    profile = None
    initial_hash = E1_SHARED_HASH
    if seed == 1:
        binding = _initialization_profile_binding(protocol, adapted, resolved, args.steps, seed,
                                                  baseline_hash, hashes)
        profile = _read_initialization_profile(
            getattr(args, "initialization_profile", None), binding)
        initial_hash = profile["initial_base_hash"]
    elif getattr(args, "initialization_profile", None) is not None:
        raise ValueError("seed0 does not use an initialization profile")
    identity = build_projector_identity(protocol, adapted, resolved, args.projector,
                                        args.steps, args.gpu_cap_seconds,
                                        baseline_hash, hashes, seed=seed,
                                        initial_base_hash=initial_hash,
                                        initialization_profile=profile)
    if matched_control is not None:
        dev_ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
        identity["matched_linear_control"] = validate_matched_linear_control(
            matched_control, identity, dev_ids)
        identity["run_key"] = f"projector-{args.projector}-{args.steps}-" + digest({
            key: value for key, value in identity.items() if key != "run_key"
        })
    output = args.output_base / identity["run_key"]
    verify_location(output, Path(__file__).resolve().parents[1])
    return identity, output, protocol, views, config


@contextmanager
def owned_projector_ledger(args, identity, output):
    """Only accepted invocations publish failure status or close budgets, under the run lock."""
    ledger = E1Ledger.create(output, identity, 0) if args.resume_checkpoint is None else None
    with writer(output):
        if args.resume_checkpoint is not None:
            if (output / "selected-checkpoint.json").exists():
                raise ValueError("projector run already completed")
            spent = read_json(output / "budget.json")["gpu_seconds"]
            checkpoint = validate_e1_resume_checkpoint(
                args.resume_checkpoint, output, identity, spent)
            seconds, step = checkpoint["gpu_seconds_cumulative"], checkpoint["global_step"]
            del checkpoint
            ledger = E1Ledger.resume(output, identity, seconds, checkpoint_step=step)
        assert ledger is not None
        try:
            if ledger.last >= args.gpu_cap_seconds - 300:
                raise ValueError("projector resume cap lacks startup reserve")
            yield ledger
        except BaseException as error:
            atomic_json(output / "run-status.json", {
                "status": "partial", "global_step": ledger.step, "error": type(error).__name__,
            })
            raise
        finally:
            ledger.close()


def run_projector(args):
    if not args.authorize_projector_run:
        raise ValueError("projector training requires explicit authorization")
    identity, output, protocol, views, config = prepare_projector(args)
    resume = args.resume_checkpoint
    if (resume is None and output.exists()) or (resume is not None and not output.is_dir()):
        raise ValueError("projector output exists or resume root missing")
    if shutil.disk_usage(args.output_base).free < 50 * 1024**3:
        raise ValueError("projector storage below bounded-checkpoint 50 GiB reserve")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("projector experiment requires 12 GiB free CUDA memory")
    trainer = None
    old_alarm = signal.getsignal(signal.SIGALRM)

    def timeout(signum, frame):
        raise TimeoutError("projector run time cap expired")

    try:
        with owned_projector_ledger(args, identity, output) as ledger:
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL, args.gpu_cap_seconds - ledger.last - 1)
            model = build_seeded_projector_model(config, args.steps, identity["seed"])
            if shared_tensor_hash(model) != identity["initial_base_hash"]:
                raise ValueError("projector attested shared initialization changed")
            assert isinstance(model.visual_adapter, SpaMoVisualAdapter)
            if args.projector == "residual":
                enable_residual_projectors(model.visual_adapter, HIDDEN,
                                           seed=identity["residual_seed"])
            initial_hash = shared_tensor_hash(model)
            model.run_metadata = identity  # pyright: ignore[reportArgumentType]
            atomic_json(output / "initialization.json", {
                "initial_base_hash": identity["initial_base_hash"],
                "initial_variant_hash": initial_hash,
                "trainable_parameters": sum(p.numel() for p in model.parameters()
                                            if p.requires_grad),
            })
            ledger.charge(step=ledger.step, phase="model_loaded")
            if resume is None:
                model.to("cuda")
                batch = collate_phoenix([views["train"][i] for i in range(4)])
                smoke = projector_gradient_smoke(model, batch)
                if shared_tensor_hash(model) != initial_hash:
                    raise RuntimeError("gradient smoke altered initialization")
                atomic_json(output / "gradient-smoke.json", {
                    **smoke, "clip_ids_hash": digest(batch.clip_ids),
                    "initial_variant_hash": initial_hash,
                })
                del batch
                ledger.charge(step=ledger.step, phase="gradient_smoke_passed")
            data = build_seeded_projector_data(views["train"], args.steps, identity["seed"])
            schedule = projector_checkpoint_steps(args.steps)
            dev_ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
            dev = FullDevCheckpoints(views["dev"], dev_ids, protocol["split_hash"], output,
                                     args.steps, WARMUP, 4, identity["variant"], ledger=ledger,
                                     checkpoint_schedule=schedule)
            trainer = pl.Trainer(**dict(config.trainer), callbacks=[E1BudgetGuard(ledger), dev,
                                 BoundedResumeSnapshots(output, ledger)])
            trainer.fit(model, datamodule=data, ckpt_path=str(resume) if resume else None)  # pyright: ignore[reportArgumentType]
            if trainer.global_step != args.steps:
                raise RuntimeError("projector experiment ended before its declared horizon")
            reports = [read_json(output / f"dev-{step}.json") for step in schedule]
            if [report["checkpoint_step"] for report in reports] != list(schedule):
                raise ValueError("projector full-dev selection schedule incomplete")
            best = choose_checkpoint(reports, dev_ids)
            atomic_json(output / "selected-checkpoint.json", {
                "step": best["checkpoint_step"], "checkpoint_hash": best["checkpoint_hash"],
                "metrics": best["metrics"], "model_variant": identity["variant"],
                "protocol_hash": protocol["protocol_hash"],
                "adaptation_checkpoint_hash": identity["adaptation_checkpoint_hash"],
                "supervision_policy": POLICY, "human_review_status": "not_assessed",
            })
            result = {"status": "complete", "global_step": args.steps, "run_key": output.name,
                      "projector": args.projector, "selected_step": best["checkpoint_step"]}
            atomic_json(output / "run-status.json", result)
            return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "initialize", "run"), required=True)
    for name in ("protocol", "dataset", "text-manifest", "dino-root", "motion-root",
                 "motion-manifest", "annotation", "adapted-root", "baseline-run", "output-base"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--projector", choices=tuple(ARMS), required=True)
    parser.add_argument("--steps", choices=(8000, 16000), type=int, required=True)
    parser.add_argument("--gpu-cap-seconds", type=int, required=True)
    parser.add_argument("--seed", choices=(0, 1), type=int, default=0)
    parser.add_argument("--initialization-profile", type=Path)
    parser.add_argument("--authorize-projector-run", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--matched-control-run", type=Path)
    args = parser.parse_args()
    if args.mode == "initialize":
        print(initialize_projector(args))
    elif args.mode == "preflight":
        identity, output, _, views, _ = prepare_projector(args)
        print({"status": "ready", "run_key": output.name, "output": str(output),
               "train": len(views["train"]), "dev": len(views["dev"]),
               "test_unscored": len(views["test"]), "projector": args.projector,
                "steps": args.steps, "resume_retention": identity["resume_retention"],
                "seed": identity["seed"],
                "initialization_profile_hash": identity.get("initialization_profile_hash"),
               "matched_linear_control": identity.get("matched_linear_control")})
    else:
        print(run_projector(args))


if __name__ == "__main__":
    main()
