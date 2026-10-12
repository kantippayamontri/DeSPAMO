"""Independent, dev-only three-arm connector study with common CPU attestations."""

import argparse
import copy
import gc
import json
import math
import os
import re
import shutil
import signal
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.data.batch import collate_phoenix
from despamo.evaluation.signer_pilot import choose_checkpoint, score_pilot_batches
from despamo.training.e1_budget import E1BudgetGuard, E1Ledger
from despamo.training.pilot_factory import shared_tensor_hash
from despamo.training.pilot_replay import capture_rng, restore_rng
from scripts import train_projector_pilot as legacy
from scripts.evaluate_e3_pilot import DECODER
from scripts.train_e3_pilot import E1_SHARED_HASH, POLICY
from scripts.train_projector_pilot import build_seeded_projector_data, build_seeded_projector_model
from scripts.train_signer_pilot import (
    E1ResumeSnapshots,
    FullDevCheckpoints,
    validate_e1_resume_checkpoint,
)
from tools.dinov3.storage import verify_location, writer

RUN_POLICY = "signer-pilot-three-arm-connectors-v1"
INITIALIZATION_POLICY = "connector-common-initialization-v1"
ARMS = ("reference", "plain_mlp", "qformer")
STEPS = 16000
WARMUP = 2000
DEV_STEPS = (5500, 10400, 16000)
MAX_CAP_SECONDS = 10800
PROJECT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/train_connector_study.py", "scripts/train_projector_pilot.py",
    "scripts/train_signer_pilot.py", "scripts/train_e3_pilot.py",
    "scripts/evaluate_e3_pilot.py", "scripts/evaluate_projector_pilot.py",
    "scripts/run_connector_study.py", "scripts/run_projector_overnight.py",
    "tools/dinov3/storage.py", "tools/dinov3/identity.py", "pyproject.toml", "uv.lock",
)
RECIPE = {
    "steps": STEPS, "warmup_steps": WARMUP, "lr_warmup_steps": 1600,
    "batch_size": 4, "accumulate_grad_batches": 1, "precision": "bf16",
    "optimizer": "AdamW", "adapter_lr": 3e-4, "flan_lr": 5e-5,
    "betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.01,
    "gradient_clip_val": 1.0, "lr_schedule": "cosine-to-zero",
    "dev_checkpoints": list(DEV_STEPS), "resume_retention": 2, "snapshot_interval": 500,
    "decoder": DECODER, "motion_tcn": "original-K5/P2/K5/P2",
}


def arm_configs() -> dict:
    """Lazy import permits CPU test collection before the model agent publishes its module."""
    from despamo.models.connector_adapter import ARM_CONFIGS

    configs = copy.deepcopy(ARM_CONFIGS)
    if set(configs) != set(ARMS):
        raise ValueError("connector architecture inventory mismatch")
    # Enforce the public JSON interface, including no hidden tensor/config objects.
    return json.loads(json.dumps(configs, sort_keys=True, allow_nan=False))


def connector_code_hashes(project: Path) -> dict[str, str]:
    """Hash all source files, training/evaluation closure, storage and dependency lock."""
    sources = {path for path in (project / "src").rglob("*") if path.is_file()
               and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")}
    sources.update(project / name for name in SOURCE_FILES)
    sources.update((project / "scripts").glob("evaluate_*.py"))
    if any(path.is_symlink() or not path.is_file() for path in sources):
        raise ValueError("connector source closure requires regular files")
    return {str(path.relative_to(project)): file_hash(path) for path in sorted(sources)}


def _require_args(args) -> None:
    if (args.arm not in ARMS or type(args.seed) is not int or args.seed not in (0, 1)
            or type(args.steps) is not int or args.steps != STEPS
            or type(args.gpu_cap_seconds) is not int
            or not 0 < args.gpu_cap_seconds <= MAX_CAP_SECONDS):
        raise ValueError("connector arm/seed/horizon/cap invalid")


def _safe_destination(path: Path, args) -> None:
    if path.is_symlink():
        raise ValueError("unsafe connector destination symlink")
    resolved = path.resolve()
    for name in ("baseline_run", "adapted_root", "dataset", "dino_root", "motion_root"):
        source = getattr(args, name, None)
        if source is not None and resolved.is_relative_to(source.resolve()):
            raise ValueError("connector destination lies in prior run/input")
    verify_location(path, PROJECT)


def _prepare_inputs(args):
    _require_args(args)
    # This read-only legacy path retains its original input/baseline guards unchanged.
    controlled = argparse.Namespace(**vars(args), projector="linear")
    prepared = legacy._prepare_projector_inputs(controlled)
    protocol, views, config, adapted, baseline_hash, _ = prepared
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("connector config must resolve to a mapping")
    legacy._require_projector_config(adapted, resolved, "linear", STEPS,
                                     args.gpu_cap_seconds, args.seed)
    if (config.trainer.precision != "bf16" or config.trainer.gradient_clip_val != 1.0
            or config.optimizer.weight_decay != 0.01
            or config.model.use_in_context or protocol.get("decoder") != DECODER):
        raise ValueError("connector fixed BF16/optimizer/decoder recipe changed")
    return protocol, views, config, adapted, baseline_hash, connector_code_hashes(PROJECT)


def _profile_binding(protocol, adapted, config, baseline_hash, code, cap) -> dict:
    return {
        "seed": config["seed"], "steps": STEPS, "config_hash": digest(config),
        "code_hashes": code, "schema_hash": code["src/despamo/appearance/schema.py"],
        "architecture_schema_hash": digest(arm_configs()), "gpu_cap_seconds": cap,
        "recipe": RECIPE,
        "input_identity": {
            "protocol_hash": protocol["protocol_hash"], "split_hash": protocol["split_hash"],
            "protocol_identity_hash": digest(protocol), "baseline_checkpoint_hash": baseline_hash,
            "adaptation_checkpoint_hash": adapted["checkpoint_hash"],
            "adapted_encoder_key": adapted["encoder_key"],
            "adapted_manifest_hash": adapted["manifest_hash"],
            "adapted_frame_rows_hash": adapted["frame_rows_hash"], "clips": adapted["clips"],
        },
    }


def _bindings(protocol, adapted, config, baseline_hash, code, cap) -> dict:
    result = {}
    for seed in (0, 1):
        resolved = copy.deepcopy(OmegaConf.to_container(config, resolve=True))
        if not isinstance(resolved, dict):
            raise ValueError("connector config must resolve to a mapping")
        resolved["seed"] = seed
        result[str(seed)] = _profile_binding(protocol, adapted, resolved, baseline_hash, code, cap)
    return result


def _validate_profile(profile, bindings) -> dict:
    if (not isinstance(profile, dict) or set(profile) != {"profile_policy", "seeds", "profile_hash"}
            or profile["profile_policy"] != INITIALIZATION_POLICY
            or not isinstance(profile["seeds"], dict) or set(profile["seeds"]) != {"0", "1"}
            or profile["profile_hash"] != digest({k: v for k, v in profile.items()
                                                 if k != "profile_hash"})):
        raise ValueError("connector common initialization profile malformed/hash mismatch")
    for seed, binding in bindings.items():
        entry = profile["seeds"][seed]
        if (not isinstance(entry, dict) or set(entry) != set(binding) | {"initial_base_hash"}
                or type(entry.get("seed")) is not int
                or any(entry.get(k) != v for k, v in binding.items())
                or not isinstance(entry["initial_base_hash"], str)
                or re.fullmatch(r"[0-9a-f]{64}", entry["initial_base_hash"]) is None
                or (seed == "0" and entry["initial_base_hash"] != E1_SHARED_HASH)):
            raise ValueError("connector profile seed/config/source/input/schema/cap mismatch")
    return profile


def _read_profile(path, bindings) -> dict:
    if path is None or path.is_symlink() or not path.is_file():
        raise ValueError("all connector arms require a regular common initialization profile")
    return _validate_profile(read_json(path), bindings)


def initialize_connector(args) -> dict:
    """Attest fresh base on CPU for BOTH seeds; no connector, optimizer or GPU model."""
    _require_args(args)
    path = args.initialization_profile
    if args.arm != "reference" or path is None or args.resume_checkpoint is not None:
        raise ValueError("initialize requires reference, --initialization-profile and no resume")
    _safe_destination(path, args)
    protocol, _, config, adapted, baseline_hash, code = _prepare_inputs(args)
    bindings = _bindings(protocol, adapted, config, baseline_hash, code, args.gpu_cap_seconds)
    with writer(path.parent):
        if path.exists():
            profile = _read_profile(path, bindings)
            status = "reused"
        else:
            entries = {}
            rng = capture_rng()
            try:
                for seed in (0, 1):
                    seeded = copy.deepcopy(config)
                    seeded.seed = seed
                    model = build_seeded_projector_model(seeded, STEPS, seed)
                    if any(value.device.type != "cpu" for value in
                           (*model.parameters(), *model.buffers())):
                        raise ValueError("connector common initialization requires CPU tensors")
                    common_hash = shared_tensor_hash(model)
                    if seed == 0 and common_hash != E1_SHARED_HASH:
                        raise ValueError("connector seed0 fresh hash differs from verified E1")
                    entries[str(seed)] = {**bindings[str(seed)], "initial_base_hash": common_hash}
                    del model
                    gc.collect()
            finally:
                restore_rng(rng)
            if connector_code_hashes(PROJECT) != code:
                raise ValueError("connector source changed during common CPU initialization")
            profile = {"profile_policy": INITIALIZATION_POLICY, "seeds": entries}
            profile["profile_hash"] = digest(profile)
            _validate_profile(profile, bindings)
            atomic_json(path, profile)
            status = "initialized"
    return {"status": status, "initialization_profile": str(path),
            "initialization_profile_hash": profile["profile_hash"],
            "initial_base_hashes": {s: e["initial_base_hash"] for s, e in profile["seeds"].items()}}


def _run_key(identity) -> str:
    payload = {k: v for k, v in identity.items() if k != "run_key"}
    return f"connector-{identity['arm']}-seed{identity['seed']}-16000-" + digest(payload)


def prepare_connector(args):
    """Read-only preflight: verify inputs and both attestations without constructing models."""
    protocol, views, config, adapted, baseline_hash, code = _prepare_inputs(args)
    bindings = _bindings(protocol, adapted, config, baseline_hash, code, args.gpu_cap_seconds)
    profile = _read_profile(args.initialization_profile, bindings)
    common = profile["seeds"][str(args.seed)]
    identity = {
        "run_policy": RUN_POLICY, "variant": f"E3_connector_{args.arm}", "arm": args.arm,
        "seed": args.seed, "module_seed": 100 + args.seed, "steps": STEPS,
        "architecture": arm_configs()[args.arm], "recipe": RECIPE,
        "warmup_steps": WARMUP, "lr_warmup_steps": 1600, "dev_checkpoints": list(DEV_STEPS),
        "resume_retention": 2, "snapshot_interval": 500, "gpu_cap_seconds": args.gpu_cap_seconds,
        "config": OmegaConf.to_container(config, resolve=True),
        "initial_base_hash": common["initial_base_hash"],
        "initialization_profile_hash": profile["profile_hash"],
        "input_identity": common["input_identity"], "code_hashes": code,
        "schema_hash": common["schema_hash"], "architecture_schema_hash": digest(arm_configs()),
        **{k: v for k, v in common["input_identity"].items() if k != "clips"},
        "supervision_policy": POLICY, "human_review_status": "not_assessed",
    }
    identity["run_key"] = _run_key(identity)
    output = args.output_base / identity["run_key"]
    _safe_destination(output, args)
    return identity, output, protocol, views, config


def report_binding(identity: dict) -> dict:
    return {
        "run_policy": RUN_POLICY, "run_identity_hash": digest(identity),
        "model_variant": identity["variant"], "protocol_hash": identity["protocol_hash"],
        "adaptation_checkpoint_hash": identity["adaptation_checkpoint_hash"],
        "adapted_encoder_key": identity["adapted_encoder_key"],
        "adapted_manifest_hash": identity["adapted_manifest_hash"],
        "supervision_policy": POLICY, "human_review_status": "not_assessed",
    }


def _validate_report(run: Path, report: dict, step: int, identity: dict, dev_ids: tuple) -> Path:
    content_hash = digest({k: v for k, v in report.items() if k != "content_hash"})
    if report.get("content_hash") != content_hash:
        raise ValueError("connector dev report content hash mismatch")
    checkpoint = run / f"step-{step}.ckpt"
    if (report.get("checkpoint_step") != step or report.get("split") != "dev"
            or report.get("split_hash") != identity["split_hash"]
            or report.get("decoding") != DECODER
            or tuple(item.get("clip_id") for item in report.get("items", ())) != dev_ids
            or any(report.get(k) != v for k, v in report_binding(identity).items())
            or not report.get("metrics")
            or any(type(v) not in (float, int) or not math.isfinite(v)
                   for v in report["metrics"].values())
            or checkpoint.is_symlink() or not checkpoint.is_file()
            or report.get("checkpoint_hash") != file_hash(checkpoint)):
        raise ValueError("connector full ordered dev/decoder/policy/cache/checkpoint mismatch")
    return checkpoint


def _dev_report_hashes(run: Path) -> dict[str, str]:
    hashes = {}
    for step in DEV_STEPS:
        path = run / f"dev-{step}.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("connector dev report hash inventory incomplete/unsafe")
        hashes[path.name] = file_hash(path)
    return hashes


def validate_connector_selection(run: Path, protocol: dict, dev_ids: tuple,
                                 expected_identity: dict | None = None) -> dict:
    """Strict completion adoption by controller; never generates held-out translations."""
    if (run.is_symlink() or not run.is_dir() or not dev_ids or len(set(dev_ids)) != len(dev_ids)
            or dev_ids != tuple(protocol["split"]["groups"]["dev"]["clip_ids"])):
        raise ValueError("connector completion requires owned root and exact protocol dev IDs")
    for name in ("budget.json", "run-status.json", "selected-checkpoint.json"):
        if (run / name).is_symlink():
            raise ValueError("unsafe connector completion record")
    budget = read_json(run / "budget.json")
    status = read_json(run / "run-status.json")
    selected = read_json(run / "selected-checkpoint.json")
    identity = budget.get("identity", {})
    cap = identity.get("gpu_cap_seconds")
    seconds = budget.get("gpu_seconds")
    if (identity.get("run_policy") != RUN_POLICY or identity.get("arm") not in ARMS
            or type(identity.get("seed")) is not int or identity["seed"] not in (0, 1)
            or identity.get("variant") != f"E3_connector_{identity['arm']}"
            or identity.get("steps") != STEPS or identity.get("recipe") != RECIPE
            or identity.get("dev_checkpoints") != list(DEV_STEPS)
            or identity.get("run_key") != run.name or _run_key(identity) != run.name
            or (expected_identity is not None and identity != expected_identity)
            or identity.get("protocol_hash") != protocol["protocol_hash"]
            or identity.get("split_hash") != protocol["split_hash"]
            or identity.get("supervision_policy") != POLICY
            or identity.get("human_review_status") != "not_assessed"
            or type(cap) is not int or not 0 < cap <= MAX_CAP_SECONDS
            or type(seconds) not in (float, int) or not math.isfinite(seconds)
            or not 0 <= seconds < cap
            or budget.get("ceiling_seconds") != cap or budget.get("status") != "closed"
            or budget.get("optimizer_step") != STEPS or status.get("status") != "complete"
            or status.get("global_step") != STEPS or status.get("run_key") != run.name):
        raise ValueError("connector completion identity/global steps/budget mismatch")
    reports = []
    if selected.get("dev_report_hashes") != _dev_report_hashes(run):
        raise ValueError("connector selected dev report byte hashes mismatch")
    for step in DEV_STEPS:
        path = run / f"dev-{step}.json"
        if path.is_symlink():
            raise ValueError("unsafe connector dev record")
        report = read_json(path)
        checkpoint_path = _validate_report(run, report, step, identity, dev_ids)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (checkpoint.get("run_metadata") != identity
                or checkpoint.get("pilot_identity") != identity
                or type(checkpoint.get("global_step")) is not int
                or checkpoint["global_step"] != step):
            raise ValueError("connector checkpoint full identity/global step mismatch")
        del checkpoint
        reports.append(report)
    best = choose_checkpoint(reports, dev_ids)
    if (selected.get("step") != best["checkpoint_step"]
            or selected.get("checkpoint_hash") != best["checkpoint_hash"]
            or selected.get("metrics") != best["metrics"]
            or any(selected.get(k) != v for k, v in report_binding(identity).items())):
        raise ValueError("connector selected checkpoint/dev identity mismatch")
    return {"checkpoint": run / f"step-{selected['step']}.ckpt",
            "checkpoint_hash": selected["checkpoint_hash"], "step": selected["step"],
            "identity": identity, "metrics": selected["metrics"]}


class ConnectorLedger(E1Ledger):
    """Reuse durable E1 accounting with this policy's shorter immutable cap."""

    def _write(self, status: str, phase: str | None = None) -> None:
        atomic_json(self.root / "budget.json", {
            "identity": self.identity, "gpu_seconds": self.last, "optimizer_step": self.step,
            "pid": os.getpid(), "phase": phase, "status": status, "wall_timestamp": self.wall(),
            "ceiling_seconds": self.identity["gpu_cap_seconds"],
        })

    def charge(self, *, step: int, phase: str) -> float:
        spent = super().charge(step=step, phase=phase)
        if spent >= self.identity["gpu_cap_seconds"]:
            raise RuntimeError("connector run cap exhausted")
        return spent

    def require_remaining(self, reserve_seconds: float, *, step: int, phase: str) -> float:
        if not math.isfinite(reserve_seconds) or reserve_seconds < 0:
            raise ValueError("connector reserve invalid")
        spent = self.charge(step=step, phase=phase)
        if spent + reserve_seconds >= self.identity["gpu_cap_seconds"]:
            raise RuntimeError("connector cap lacks time reserve")
        return spent


def _refuse_completed(output: Path) -> None:
    selected = output / "selected-checkpoint.json"
    status = output / "run-status.json"
    if selected.exists() or selected.is_symlink():
        raise ValueError("connector run already completed")
    if status.is_symlink():
        raise ValueError("unsafe connector run status")
    if status.is_file() and read_json(status).get("status") == "complete":
        raise ValueError("connector run already completed")


@contextmanager
def owned_connector_ledger(args, identity, output):
    """No failure/status/budget writes before acceptance, or after releasing own run lock."""
    if identity.get("run_policy") != RUN_POLICY or output.name != identity.get("run_key"):
        raise ValueError("connector ledger requires own policy/root")
    resume = args.resume_checkpoint
    _refuse_completed(output)
    if resume is None:
        if output.exists() or output.is_symlink():
            raise ValueError("connector output already exists")
        output.mkdir()  # Exclusive root creation; writer cannot lock/create an older run.
    elif output.is_symlink() or not output.is_dir():
        raise ValueError("connector resume root missing/unsafe")
    with writer(output):
        _refuse_completed(output)
        if resume is not None:
            if (output / "budget.json").is_symlink():
                raise ValueError("unsafe connector resume budget")
            budget = read_json(output / "budget.json")
            if (budget.get("identity") != identity
                    or budget.get("ceiling_seconds") != identity["gpu_cap_seconds"]):
                raise ValueError("connector resume ledger identity mismatch")
            spent = budget["gpu_seconds"]
            effective = spent
            if budget.get("status") not in ("closed", "paused"):
                elapsed = time.time() - budget["wall_timestamp"]
                if elapsed < 0 or not math.isfinite(elapsed):
                    raise ValueError("connector resume wall clock invalid")
                effective += elapsed
            if (not math.isfinite(effective)
                    or not 0 <= effective < identity["gpu_cap_seconds"] - 300):
                raise ValueError("connector resume cap lacks startup reserve")
            checkpoint = validate_e1_resume_checkpoint(resume, output, identity, spent)
            if (type(checkpoint.get("global_step")) is not int
                    or not 0 < checkpoint["global_step"] <= STEPS
                    or type(checkpoint.get("gpu_seconds_cumulative")) not in (float, int)
                    or not math.isfinite(checkpoint["gpu_seconds_cumulative"])):
                raise ValueError("connector resume checkpoint step/seconds invalid")
            ledger = ConnectorLedger.resume(output, identity, checkpoint["gpu_seconds_cumulative"],
                                             checkpoint_step=checkpoint["global_step"])
            del checkpoint
        else:
            ledger = ConnectorLedger(output, identity, 0, 0, time.monotonic, time.time)
            ledger._write("created")
        try:
            ledger.require_remaining(300, step=ledger.step, phase="startup")
            yield ledger
        except BaseException as error:
            atomic_json(output / "run-status.json", {
                "status": "partial", "global_step": ledger.step, "run_key": output.name,
                "error": type(error).__name__,
            })
            raise
        finally:
            ledger.close()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_checkpoint(trainer, target: Path) -> None:
    pending = target.with_name(f".{target.name}.{os.getpid()}.pending")
    if target.exists() or target.is_symlink() or pending.exists() or pending.is_symlink():
        raise ValueError("connector checkpoint exists; refusing overwrite")
    try:
        trainer.save_checkpoint(pending)
        if pending.is_symlink() or not pending.is_file():
            raise ValueError("unsafe connector pending checkpoint")
        with pending.open("rb") as saved:
            os.fsync(saved.fileno())
        os.replace(pending, target)
        _fsync_directory(target.parent)
    finally:
        pending.unlink(missing_ok=True)


class ConnectorResumeSnapshots(E1ResumeSnapshots):
    """Latest two owned resumes, pruned only after new snapshot's durable publication."""

    def __init__(self, output, ledger, keep=2):
        if (type(keep) is not int or keep != 2 or output.resolve() != ledger.root.resolve()
                or output.is_symlink() or ledger.identity.get("run_policy") != RUN_POLICY):
            raise ValueError("connector retention requires owned new-policy root and latest two")
        super().__init__(output, ledger, interval=500)
        self.keep = keep

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step <= 0 or step % self.interval or step in self.completed:
            return
        snapshots = []
        for path in self.output.glob("resume-*.ckpt"):
            match = re.fullmatch(r"resume-(\d+)\.ckpt", path.name)
            if (match is None or path.is_symlink() or not path.is_file()
                    or path.resolve().parent != self.output.resolve()):
                raise ValueError("unsafe connector resume snapshot")
            snapshots.append((int(match.group(1)), path))
        self.ledger.require_remaining(300, step=step, phase="before_resume_snapshot")
        target = self.output / f"resume-{step}.ckpt"
        _durable_checkpoint(trainer, target)
        self.ledger.charge(step=step, phase="resume_snapshot_saved")
        self.completed.add(step)
        snapshots.append((step, target))
        for old_step, path in sorted(snapshots)[:-self.keep]:
            path.unlink()
            self.completed.discard(old_step)
        _fsync_directory(self.output)


class ConnectorDevCheckpoints(FullDevCheckpoints):
    """Reuse dev scorer, adding durable saves and strict connector cache/policy bindings."""

    def on_train_start(self, trainer, pl_module):
        ledger = self.ledger
        assert ledger is not None
        for step in self.steps:
            target = self.output / f"step-{step}.ckpt"
            report_path = self.output / f"dev-{step}.json"
            if step < trainer.global_step and not report_path.exists():
                raise ValueError("connector incomplete prior dev report requires its matching step")
            if report_path.exists():
                if report_path.is_symlink():
                    raise ValueError("unsafe connector dev report")
                _validate_report(self.output, read_json(report_path), step,
                                 ledger.identity, self.expected)
                self.completed.add(step)
            elif target.exists():
                if target.is_symlink() or trainer.global_step != step:
                    raise ValueError("connector incomplete dev requires matching owned checkpoint")
                self._score_saved(pl_module, step, target)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        ledger = self.ledger
        assert ledger is not None
        step = trainer.global_step
        if step not in self.steps or step in self.completed:
            return
        ledger.require_remaining(300, step=step, phase="before_checkpoint")
        target = self.output / f"step-{step}.ckpt"
        _durable_checkpoint(trainer, target)
        self._score_saved(pl_module, step, target)

    def _score_saved(self, pl_module, step, target):
        ledger = self.ledger
        assert ledger is not None
        ledger.require_remaining(300, step=step, phase="before_dev")
        seen = 0

        def on_batch_end(count):
            nonlocal seen
            seen += count
            ledger.require_remaining(30, step=step, phase=f"dev-{seen}/{len(self.expected)}")

        try:
            report = score_pilot_batches(pl_module, self.loader, self.expected, "dev",
                                         self.split_hash, on_batch_end=on_batch_end)
            report.update(checkpoint_step=step, checkpoint_hash=file_hash(target),
                          **report_binding(ledger.identity))
            report["content_hash"] = digest(report)
            _validate_report(self.output, report, step, ledger.identity, self.expected)
            atomic_json(self.output / f"dev-{step}.json", report)
            self.completed.add(step)
        finally:
            pl_module.train()


def connector_gradient_smoke(model, batch, arm: str) -> dict:
    """Joint CE+VT; aggregate group norms; no optimizer; restore all mutable training state."""
    if arm not in ARMS:
        raise ValueError("unknown connector smoke arm")
    rng = capture_rng()
    prompt_rng = model.rng.getstate()
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if device.type == "cuda" else nullcontext())
        with context:
            losses = model.compute_losses(batch)
            joint = losses["loss"] + model.vt_weight * losses["contra_loss"]
        if not torch.isfinite(joint):
            raise FloatingPointError("connector joint smoke loss non-finite")
        joint.backward()
        adapter = model.visual_adapter
        groups = {
            "spatial": list(adapter.spatial_projector.parameters()),
            "motion": list(adapter.motion_projector.parameters()),
            "connector": list(adapter.multimodal_projector.parameters()),
            "flan": list(model.language_model.parameters()),
        }
        if arm == "qformer":
            connector = adapter.multimodal_projector
            groups["learned_queries"] = [p for name, p in connector.named_parameters()
                                         if "quer" in name and p.ndim in (2, 3)]
            groups["attention"] = [p for module in connector.modules()
                                   if isinstance(module, torch.nn.MultiheadAttention)
                                   for p in module.parameters()]
        norms = {}
        for name, parameters in groups.items():
            trainable = [p for p in parameters if p.requires_grad]
            grads = [p.grad for p in trainable if p.grad is not None]
            if not trainable or not grads or not all(torch.isfinite(g).all() for g in grads):
                raise RuntimeError(f"connector smoke missing/non-finite {name} gradient")
            norm = sum(float(g.detach().float().norm().cpu()) for g in grads)
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError(f"connector smoke zero {name} gradient")
            norms[name] = norm
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        if peak > 16 * 1024**3:
            raise RuntimeError("connector gradient smoke exceeds 16 GiB fit")
        return {"status": "passed", "objective": "joint-ce-vt", "optimizer_updates": 0,
                "joint_loss": float(joint.detach().cpu()), "gradient_norms": norms,
                "cuda_peak_bytes": peak, "fit_limit_bytes": 16 * 1024**3,
                "fit_16gib": device.type == "cuda" and peak <= 16 * 1024**3}
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        for module, mode in modes:
            module.training = mode
        model.rng.setstate(prompt_rng)
        restore_rng(rng)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def build_connector_model(config, arm: str, seed: int, initial_base_hash: str):
    """Construct discarded legacy branches first; attachment owns seed100+training_seed RNG."""
    from despamo.models.connector_adapter import enable_connector_arm

    model = build_seeded_projector_model(config, STEPS, seed)
    if shared_tensor_hash(model) != initial_base_hash:
        raise ValueError("connector attested common initialization changed")
    # The public module takes TRAINING seed and forks RNG internally at 100 + seed.
    model.visual_adapter = enable_connector_arm(model.visual_adapter, arm, seed)
    return model


def validate_effective_optimizer(model) -> dict:
    """Validate actual pilot groups; generic baseline config LR is not their effective LR."""
    configured = model.configure_optimizers()
    optimizer = configured.get("optimizer") if isinstance(configured, dict) else configured
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("connector optimizer must be pilot AdamW")
    expected = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("language_model."):
            group = "flan"
        elif name.startswith(("visual_adapter.", "vt_align.")):
            group = "adapter"
        else:
            raise ValueError(f"unexpected connector optimizer parameter: {name}")
        expected[id(parameter)] = group
    seen = set()
    rates = {}
    for group in optimizer.param_groups:
        parameters = group["params"]
        labels = {expected.get(id(parameter)) for parameter in parameters}
        if len(labels) != 1 or None in labels:
            raise ValueError("connector optimizer has empty/mixed/unowned parameter group")
        label = labels.pop()
        if (label in rates or group["lr"] != RECIPE[f"{label}_lr"]
                or group["betas"] != tuple(RECIPE["betas"])
                or group["eps"] != RECIPE["eps"]
                or group["weight_decay"] != RECIPE["weight_decay"]):
            raise ValueError("connector optimizer effective group LR/recipe mismatch")
        rates[label] = group["lr"]
        for parameter in parameters:
            if id(parameter) in seen:
                raise ValueError("connector optimizer has duplicated parameter")
            seen.add(id(parameter))
    if seen != set(expected) or set(rates) != {"adapter", "flan"}:
        raise ValueError("connector optimizer missing trainable parameter/group")
    return _effective_optimizer_recipe()


def _effective_optimizer_recipe() -> dict:
    return {"optimizer": "AdamW", "group_learning_rates": {
        "adapter": RECIPE["adapter_lr"], "flan": RECIPE["flan_lr"]},
            "betas": RECIPE["betas"], "eps": RECIPE["eps"],
            "weight_decay": RECIPE["weight_decay"]}


def _recover_terminal_dev(model, data, dev, resume: Path, output: Path,
                          identity: dict, ledger: E1Ledger) -> None:
    """Owned step-16000 replay is evaluation only: no Trainer, optimizer or backward pass."""
    target = output / f"step-{STEPS}.ckpt"
    # If both latest snapshots exist, score the exact step checkpoint named by the dev report.
    source = target if target.exists() or target.is_symlink() else resume
    checkpoint = validate_e1_resume_checkpoint(source, output, identity, ledger.last)
    if checkpoint["global_step"] != STEPS or "prompt_rng_state" not in checkpoint:
        raise ValueError("connector terminal recovery requires full step/RNG metadata")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    data.load_state_dict(checkpoint["pilot_sampler"])
    model.rng.setstate(checkpoint["prompt_rng_state"])
    restore_rng(checkpoint["pilot_rng"])
    del checkpoint
    ledger.charge(step=STEPS, phase="terminal_checkpoint_loaded")
    if not target.exists():
        ledger.require_remaining(300, step=STEPS, phase="before_terminal_dev_checkpoint")
        # Preserve exact owned snapshot bytes, including optimizer/RNG metadata; no new update.
        copy_only = SimpleNamespace(save_checkpoint=lambda path: shutil.copyfile(source, path))
        _durable_checkpoint(copy_only, target)
    dev.on_train_start(SimpleNamespace(global_step=STEPS), model)


def run_connector(args) -> dict:
    if not args.authorize_connector_run:
        raise ValueError("connector run requires --authorize-connector-run")
    identity, output, protocol, views, config = prepare_connector(args)
    if shutil.disk_usage(args.output_base).free < 50 * 1024**3:
        raise ValueError("connector storage below 50 GiB reserve")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("connector runtime requires 12 GiB free CUDA memory")
    with owned_connector_ledger(args, identity, output) as ledger:
        old_alarm = signal.getsignal(signal.SIGALRM)

        def timeout(signum, frame):
            raise TimeoutError("connector run cap expired")

        try:
            signal.signal(signal.SIGALRM, timeout)
            signal.setitimer(signal.ITIMER_REAL,
                             max(0.001, identity["gpu_cap_seconds"] - ledger.last - 1))
            model = build_connector_model(config, args.arm, args.seed,
                                          identity["initial_base_hash"])
            terminal = args.resume_checkpoint is not None and ledger.step == STEPS
            optimizer_recipe = (_effective_optimizer_recipe() if terminal
                                else validate_effective_optimizer(model))
            variant_hash = shared_tensor_hash(model)
            model.run_metadata = identity
            initialization = {
                "initial_base_hash": identity["initial_base_hash"],
                "initial_variant_hash": variant_hash,
                "initialization_profile_hash": identity["initialization_profile_hash"],
                "effective_optimizer": optimizer_recipe,
                "trainable_parameters": sum(p.numel() for p in model.parameters()
                                            if p.requires_grad),
                "adapter_trainable_parameters": sum(
                    p.numel() for p in model.visual_adapter.parameters() if p.requires_grad),
                "flan_trainable_parameters": sum(
                    p.numel() for p in model.language_model.parameters() if p.requires_grad),
            }
            path = output / "initialization.json"
            if args.resume_checkpoint is not None:
                if path.is_symlink() or read_json(path) != initialization:
                    raise ValueError("connector resume initialization record mismatch")
            else:
                atomic_json(path, initialization)
            ledger.charge(step=ledger.step, phase="model_loaded")
            model.to("cuda")
            if not terminal:
                batch = collate_phoenix([views["train"][i] for i in range(4)])
                smoke = connector_gradient_smoke(model, batch, args.arm)
                if shared_tensor_hash(model) != variant_hash:
                    raise RuntimeError("connector gradient smoke altered initialized parameters")
                if not smoke["fit_16gib"]:
                    raise RuntimeError("connector smoke did not attest 16 GiB CUDA fit")
                atomic_json(output / "gradient-smoke.json", {
                    **smoke, "clip_ids_hash": digest(batch.clip_ids),
                    "initial_variant_hash": variant_hash,
                    "initial_base_hash": identity["initial_base_hash"], "arm": args.arm,
                })
                del batch
                ledger.charge(step=ledger.step, phase="gradient_smoke_passed")
            if connector_code_hashes(PROJECT) != identity["code_hashes"]:
                raise ValueError("connector source changed before production training")
            data = build_seeded_projector_data(views["train"], STEPS, args.seed)
            dev_ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
            dev = ConnectorDevCheckpoints(views["dev"], dev_ids, protocol["split_hash"], output,
                                         STEPS, WARMUP, 4, identity["variant"], ledger=ledger,
                                         checkpoint_schedule=DEV_STEPS)
            if terminal:
                assert args.resume_checkpoint is not None
                _recover_terminal_dev(model, data, dev, args.resume_checkpoint, output,
                                      identity, ledger)
            else:
                trainer = pl.Trainer(**dict(config.trainer), callbacks=[E1BudgetGuard(ledger), dev,
                                     ConnectorResumeSnapshots(output, ledger)])
                checkpoint_path = str(args.resume_checkpoint) if args.resume_checkpoint else None
                trainer.fit(model, datamodule=data, ckpt_path=checkpoint_path)
                if trainer.global_step != STEPS:
                    raise RuntimeError("connector run ended before declared horizon")
            if connector_code_hashes(PROJECT) != identity["code_hashes"]:
                raise ValueError("connector source changed during production training")
            reports = [read_json(output / f"dev-{step}.json") for step in DEV_STEPS]
            for step, report in zip(DEV_STEPS, reports, strict=True):
                _validate_report(output, report, step, identity, dev_ids)
            best = choose_checkpoint(reports, dev_ids)
            ledger.charge(step=STEPS, phase="dev_selection_complete")
            atomic_json(output / "selected-checkpoint.json", {
                "step": best["checkpoint_step"], "checkpoint_hash": best["checkpoint_hash"],
                "metrics": best["metrics"], **report_binding(identity),
                "dev_report_hashes": _dev_report_hashes(output),
            })
            result = {"status": "complete", "global_step": STEPS, "run_key": output.name,
                      "arm": args.arm, "seed": args.seed, "selected_step": best["checkpoint_step"]}
            atomic_json(output / "run-status.json", result)
            return result
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("initialize", "preflight", "run"), required=True)
    for name in ("protocol", "dataset", "text-manifest", "dino-root", "motion-root",
                 "motion-manifest", "annotation", "adapted-root", "baseline-run", "output-base"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", choices=(0, 1), type=int, default=0)
    parser.add_argument("--steps", choices=(STEPS,), type=int, default=STEPS)
    parser.add_argument("--gpu-cap-seconds", type=int, default=MAX_CAP_SECONDS)
    parser.add_argument("--initialization-profile", type=Path, required=True)
    parser.add_argument("--authorize-connector-run", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.mode == "initialize":
        result = initialize_connector(args)
    elif args.mode == "preflight":
        identity, output, _, views, _ = prepare_connector(args)
        result = {"status": "ready", "identity": identity, "output": str(output),
                  "run_key": identity["run_key"], "train": len(views["train"]),
                  "dev": len(views["dev"]), "test_unscored": len(views["test"])}
    else:
        result = run_connector(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
