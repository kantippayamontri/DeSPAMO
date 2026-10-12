"""Time-aligned fusion runtime. Real chronology must be admitted before any study output."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import version as torch_version

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.data.batch import collate_phoenix
from despamo.data.motion_timing import ClipTiming, validate_motion_timing
from despamo.data.motion_validation import validate_motion_package
from despamo.data.new_motion import (
    compact_motion_binding,
    load_new_motion_views,
    revalidate_motion_binding,
    validate_motion_reference,
)
from despamo.evaluation.signer_pilot import choose_checkpoint, score_pilot_batches
from despamo.models.aligned_fusion import (
    FUSION_ARMS,
    FUSION_CONFIGS,
    enable_aligned_fusion,
    fusion_frame_stats,
    interpolate_motion,
)
from despamo.models.connector_adapter import enable_connector_arm
from despamo.training.e1_budget import E1BudgetGuard, E1Ledger
from despamo.training.pilot_factory import PilotBaselineModule, shared_tensor_hash
from despamo.training.pilot_replay import capture_rng, restore_rng
from scripts.evaluate_e3_pilot import DECODER
from scripts.fusion_time_waiver import (
    compatible_sources,
    execution_fields,
    resume_ledger,
    time_waived,
    validate_waived_checkpoint,
)
from scripts.train_e3_pilot import E1_SHARED_HASH, POLICY
from scripts.train_projector_pilot import (
    _prepare_projector_inputs,
    build_seeded_projector_data,
    build_seeded_projector_model,
)
from scripts.train_signer_pilot import FullDevCheckpoints
from scripts.train_signer_pilot import (
    validate_e1_resume_checkpoint as _validate_e1_resume_checkpoint,
)
from tools.dinov3.storage import _fsync_directory, verify_location, writer

RUN_POLICY = "signer-pilot-time-aligned-fusion-v1"


def validate_e1_resume_checkpoint(path: Path, output: Path, identity: dict, spent: float) -> dict:
    if time_waived(identity):
        return validate_waived_checkpoint(path, output, identity, spent)
    return _validate_e1_resume_checkpoint(path, output, identity, spent)


INITIALIZATION_POLICY = "fusion-common-initialization-v1"
NEW_RUN_POLICY = "signer-pilot-time-aligned-fusion-new-motion-v1"
NEW_INITIALIZATION_POLICY = "fusion-common-initialization-new-motion-v1"
NEW_SOURCE_FILES = (
    "tools/motion/extractor.py", "tools/motion/records.py", "tools/motion/package.py",
    "tools/motion/__init__.py", "tools/motion/pyproject.toml", "tools/motion/uv.lock",
    "scripts/extract_motion_features.py", "src/despamo/data/new_motion.py",
    "scripts/run_motion_fusion_pipeline.py",
)
STEPS = 16000
WARMUP = 2000
DEV_STEPS = (5500, 10400, 16000)
MAX_CAP_SECONDS = 10800
FEATURE_KEY = "7122b42e35510611626c2a71d0731b79dd79b20de6e89bd49f62737c5c84a18e"
PROJECT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/train_fusion_study.py", "scripts/run_fusion_study.py",
    "scripts/fusion_time_waiver.py",
    "scripts/train_projector_pilot.py", "scripts/train_signer_pilot.py",
    "scripts/train_e3_pilot.py", "scripts/evaluate_e3_pilot.py",
    "scripts/prepare_motion_timing.py", "tools/dinov3/storage.py",
    "tools/dinov3/identity.py", "pyproject.toml", "uv.lock",
    "src/despamo/appearance/schema.py", "src/despamo/data/motion_timing.py",
    "src/despamo/models/aligned_fusion.py", "src/despamo/models/connector_adapter.py",
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


def _require_args(args) -> None:
    if (args.arm not in FUSION_ARMS or type(args.seed) is not int or args.seed not in (0, 1)
            or type(args.steps) is not int or args.steps != STEPS
            or type(args.gpu_cap_seconds) is not int
            or not 0 < args.gpu_cap_seconds <= MAX_CAP_SECONDS):
        raise ValueError("fusion arm/seed/horizon/cap invalid")


def fusion_code_hashes(project: Path, *, new_motion: bool = False) -> dict[str, str]:
    """Exact dependency closure; missing future controller blocks production profiles."""
    sources = {p for p in (project / "src").rglob("*") if p.is_file()
               and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    sources.update(project / name for name in SOURCE_FILES)
    if new_motion:
        sources.update(project / name for name in NEW_SOURCE_FILES)
    sources.update((project / "scripts").glob("evaluate_*.py"))
    sources.update(p for p in (project / "tools/dinov3").rglob("*.py")
                   if "tests" not in p.parts and "__pycache__" not in p.parts)
    if any(p.is_symlink() or not p.is_file() for p in sources):
        raise ValueError("NEEDS_CONTEXT: fusion source closure requires controller/regular files")
    actual = {str(p.relative_to(project)): file_hash(p) for p in sorted(sources)}
    return compatible_sources(actual, "runner")


def hardware_identity() -> dict:
    """Read driver/device facts without constructing CUDA tensors or a model."""
    fields = "name,compute_cap,memory.total,driver_version,uuid"
    lines = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True,
    ).strip().splitlines()
    if len(lines) != 1:
        raise ValueError("fusion requires one unambiguous physical GPU")
    name, capability, memory, driver, uuid = [v.strip() for v in lines[0].split(",")]
    if not name or not driver or not uuid or int(memory) <= 2048:
        raise ValueError("fusion hardware identity incomplete")
    return {"gpu_name": name, "capability": capability, "vram_mib": int(memory),
            "driver": driver, "gpu_uuid": uuid, "torch": torch.__version__,
            "cuda": torch_version.cuda}


def _safe_destination(path: Path, args) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unsafe fusion destination symlink")
    for name in ("baseline_run", "adapted_root", "dataset", "dino_root", "motion_root"):
        source = getattr(args, name, None)
        if source is not None and path.resolve().is_relative_to(source.resolve()):
            raise ValueError("fusion destination lies in prior run/input")
    package_root = getattr(args, "motion_package", None)
    if package_root is not None:
        package_identity = read_json(package_root / "complete/identity.json")
        protected = [package_root, Path(package_identity["inputs"]["frame_root"])]
        protected.append(Path(package_identity["inputs"]["frame_rows"]["path"]).parents[1])

        def protect_evidence(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    protected.append(Path(value["path"]).parent)
                else:
                    for item in value.values():
                        protect_evidence(item)
            elif isinstance(value, list):
                for item in value:
                    protect_evidence(item)

        protect_evidence(package_identity)
        if any(path.resolve().is_relative_to(p.resolve())
               or p.resolve().is_relative_to(path.resolve()) for p in protected):
            raise ValueError("fusion destination aliases new package/input/source roots")
    verify_location(path, PROJECT)


def motion_run_policy(identity: dict) -> str:
    """Accept new policy only with explicit, consistent actual-input graph binding."""
    policy = identity.get("run_policy", RUN_POLICY)
    if policy == RUN_POLICY and "motion_package" not in identity:
        return RUN_POLICY
    if policy != NEW_RUN_POLICY:
        raise ValueError("fusion motion policy/package mismatch")
    reference = identity.get("motion_package")
    if not isinstance(reference, dict):
        raise ValueError("fusion motion policy requires a package reference")
    validate_motion_reference(reference)
    config, bound_files = identity.get("config", {}), identity.get("bound_files", {})
    if not isinstance(config, dict) or not isinstance(bound_files, dict):
        raise ValueError("fusion new motion config/bound files must be mappings")
    data = config.get("data", identity.get("actual_motion", {}))
    if not isinstance(data, dict):
        raise ValueError("fusion new motion actual data must be a mapping")
    if (data.get("motion_root") != reference["root"]
            or data.get("motion_manifest") != reference["manifest_path"]
            or identity.get("timing_hash") != reference["timing_hash"]
            or bound_files.get(reference["completion"])
            != reference["completion_hash"]):
        raise ValueError("fusion new motion actual config/graph binding mismatch")
    return NEW_RUN_POLICY


def validate_new_motion_production(package: dict, frames: dict) -> None:
    """Production proof before models; no import of the dedicated GPU extractor."""
    import numpy as np

    from tools.motion.records import regular_path, window_rows

    clips = frames["clips"]
    if frames.get("encoder_key") != FEATURE_KEY or len(clips) != 7096:
        raise ValueError("new motion production cosine/7096 clip proof required")
    total = sum(row["frame_count"] for row in clips.values())
    windows = sum(len(window_rows(row["frame_count"])) for row in clips.values())
    if (total, windows) != (827354, 93232):
        raise ValueError("new motion production 827354/93232 source/window proof required")
    raw = np.load(regular_path(Path(package["annotation_path"])), allow_pickle=True).item()
    records = [raw[k] for k in sorted(k for k in raw if type(k) is int)]
    groups = {"train": [], "dev": [], "test": []}
    for row in records:
        signer = row["signer"]
        groups["dev" if signer == "Signer03" else "test" if signer == "Signer07"
               else "train"].append(row["fileid"])
    if (sorted(r["fileid"] for r in records) != package["clip_ids"]
            or {k: len(v) for k, v in groups.items()} != {"train": 5746, "dev": 582, "test": 768}
            or len({r["signer"] for r in records} - {"Signer03", "Signer07"}) != 7):
        raise ValueError("new motion production annotation/logical signer proof required")
    generator = package["identity"]
    model_files = generator["environment"].get("model_files")
    if not isinstance(model_files, list) or not model_files:
        raise ValueError("new motion generator actual model byte evidence required")
    for name, source in (("extractor", "tools/motion/extractor.py"),
                         ("helpers", "tools/motion/records.py")):
        if generator["code"].get(name, {}).get("path") != str(PROJECT / source):
            raise ValueError("new motion generator actor source label mismatch")
    source_paths = {item["path"]: item["sha256"] for item in generator["code"].values()}
    generator_sources = (*NEW_SOURCE_FILES, "src/despamo/appearance/provenance.py",
                         "src/despamo/appearance/schema.py", "src/despamo/data/manifest.py",
                         "src/despamo/data/motion_timing.py")
    for name in generator_sources:
        # Pipeline/admission are training actors, not historical generation actors.
        if name in ("scripts/run_motion_fusion_pipeline.py", "src/despamo/data/new_motion.py",
                    "tools/motion/uv.lock"):
            continue
        path = PROJECT / name
        if source_paths.get(str(path)) != _regular_hash(path):
            raise ValueError("new motion generator source differs from current producer")
    lock = generator["environment"]["lock"]
    if (lock["path"] != str(PROJECT / "tools/motion/uv.lock")
            or lock["sha256"] != _regular_hash(Path(lock["path"]))):
        raise ValueError("new motion generator environment lock mismatch")


def validate_motion_graph(identity: dict) -> None:
    """Full production graph closure at admission/recovery/adoption/report boundaries."""
    if motion_run_policy(identity) == NEW_RUN_POLICY:
        package = revalidate_motion_binding(identity["motion_package"])
        frames = read_json(Path(package["spatial_frame_rows_path"]))
        validate_new_motion_production(package, frames)
        _check_live(identity)


def validate_timing_join(timing: dict, frames: dict, manifest: dict,
                         receipts: dict) -> dict[str, ClipTiming]:
    """Task1 schema plus exact full-source adapted feature receipt join."""
    admitted = validate_motion_timing(timing, frames, manifest)
    if set(receipts) != set(admitted):
        raise ValueError("timing/receipt clip inventory mismatch")
    for clip_id, clip in admitted.items():
        row, receipt = frames["clips"][clip_id], receipts[clip_id]
        if any(receipt.get(k) != row[k] for k in
               ("paths", "source_indices", "source_hash", "feature_hash")):
            raise ValueError(f"full-source receipt mapping mismatch: {clip_id}")
        if len(row["paths"]) != clip.frame_count:
            raise ValueError("receipt rows do not cover full source")
    return admitted


def _regular_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"regular identity input required: {path}")
    return file_hash(path)


def _validate_extraction_sources(timing, hashes) -> None:
    # Task1 has raw extractor/helper hashes, not path fields. Resolve only through
    # actual current closure files or explicitly bound historical evidence files.
    actual = set(hashes.values())
    if any(timing["provenance"]["file_sha256"][name] not in actual
           for name in ("extractor", "helpers")):
        raise ValueError("NEEDS_CONTEXT: timing extractor/helper bytes need bound source files")


def _require_config(config) -> None:
    if (config.model.warm_up_steps != WARMUP or config.model.use_in_context
            or config.trainer.max_steps != STEPS
            or config.trainer.accumulate_grad_batches != 1
            or config.trainer.precision != "bf16" or config.trainer.gradient_clip_val != 1.0
            or config.data.batch_size != 4 or config.data.num_workers != 0
            or config.optimizer.weight_decay != 0.01):
        raise ValueError("fusion frozen recipe changed")


def validate_fusion_splits(protocol: dict, views: dict) -> None:
    for name, count in (("train", 5746), ("dev", 582), ("test", 768)):
        ids = tuple(protocol["split"]["groups"][name]["clip_ids"])
        if (len(ids) != count or len(set(ids)) != count
                or tuple(r["fileid"] for r in views[name].records) != ids
                or views[name].parent.spatial_crop_mode != "full"):
            raise ValueError("fusion full ordered logical split changed")


def _prepare_inputs(args):
    _require_args(args)
    controlled = argparse.Namespace(**vars(args), projector="linear")
    protocol, views, config, adapted, baseline_hash, _ = _prepare_projector_inputs(controlled)
    _require_config(config)
    if adapted["encoder_key"] != FEATURE_KEY or protocol.get("decoder") != DECODER:
        raise ValueError("fusion cosine feature/decoder identity changed")
    validate_fusion_splits(protocol, views)
    frame_path = args.adapted_root / "complete/frame_rows.json"
    frames, manifest = read_json(frame_path), read_json(args.motion_manifest)
    package_root = getattr(args, "motion_package", None)
    package = None
    motion_root, motion_manifest = args.motion_root, args.motion_manifest
    baseline_motion = None
    if package_root is not None:
        package = validate_motion_package(package_root, frames, verify_features=True)
        if (str(args.motion_timing) != package["timing_path"]
                or _regular_hash(args.motion_timing) != package["timing_hash"]
                or str(args.annotation) != package["annotation_path"]
                or _regular_hash(args.annotation) != package["annotation_hash"]):
            raise ValueError("new motion authoritative timing/annotation path/bytes mismatch")
        validate_new_motion_production(package, frames)
        baseline_motion = {"root": str(args.motion_root),
                           "manifest_path": str(args.motion_manifest),
                           "manifest_hash": _regular_hash(args.motion_manifest),
                           "protocol_identity_hash": digest(protocol)}
        views = load_new_motion_views(views, package, package_root)
        config = copy.deepcopy(config)
        motion_root, motion_manifest = package_root, Path(package["manifest_path"])
        config.data.motion_root = str(motion_root)
        config.data.motion_manifest = str(motion_manifest)
        manifest = read_json(motion_manifest)
    timing = read_json(args.motion_timing)
    receipts = {clip: read_json(args.adapted_root / "receipts/train" / f"{clip}.json")
                for clip in frames["clips"]}
    admitted = validate_timing_join(timing, frames, manifest, receipts)
    if set(admitted) != {r["fileid"] for v in views.values() for r in v.records}:
        raise ValueError("timing/view clip mapping mismatch")
    for view in views.values():
        for row in view.records:
            clip = admitted[row["fileid"]]
            record = view.parent.spatial_manifest.require("train", clip.clip_id)
            if record.length != clip.frame_count:
                raise ValueError("timing/view full spatial row count mismatch")
    files = timing["provenance"]["file_sha256"]
    if (files["frame_rows"] != _regular_hash(frame_path)
            or files["motion_manifest"] != _regular_hash(motion_manifest)
            or files["annotation"] != _regular_hash(args.annotation)):
        raise ValueError("timing raw input bytes mismatch")
    external = {item["path"]: item["sha256"] for item in timing["provenance"]["evidence"]}
    for path, expected in external.items():
        if _regular_hash(Path(path)) != expected:
            raise ValueError("chronology evidence bytes changed")
    # Freeze every supplied input file and every admitted feature/receipt. Neutral
    # preparation already validates encoder/source identities and finite arrays.
    paths = [args.protocol, args.text_manifest, args.motion_manifest, args.annotation,
             args.motion_timing, frame_path, args.adapted_root / "complete/manifest.json",
             args.adapted_root / "complete/identity.json"]
    for clip_id, row in timing["clips"].items():
        paths.extend((args.adapted_root / "receipts/train" / f"{clip_id}.json",
                      args.adapted_root / "train" / f"{clip_id}.npy",
                       motion_root / row["motion_record"]["path"]))
    if package is not None:
        paths.extend(Path(package[name]) for name in
                     ("completion", "manifest_path", "timing_path", "annotation_path",
                      "spatial_frame_rows_path"))
        paths.extend(Path(package["root"]) / "receipts/train" / f"{clip}.json" for clip in admitted)
        paths.extend(Path(item["path"]) for item in package["identity"]["code"].values())
        paths.append(Path(package["identity"]["environment"]["lock"]["path"]))
    paths.extend(args.baseline_run / name for name in
                 ("budget.json", "run-status.json", "selected-checkpoint.json"))
    selected = read_json(args.baseline_run / "selected-checkpoint.json")
    paths.append(args.baseline_run / f"step-{selected['step']}.ckpt")
    paths.extend(sorted(args.baseline_run.glob("dev-*.json")))
    bound_files = {str(p.resolve()): _regular_hash(p) for p in paths}
    bound_files.update(external)
    code = (fusion_code_hashes(PROJECT) if package is None
            else fusion_code_hashes(PROJECT, new_motion=True))
    _validate_extraction_sources(timing, {**bound_files, **code})
    dev_timing = [admitted[clip] for clip in protocol["split"]["groups"]["dev"]["clip_ids"]]
    dev_coverage = {"clip_count": len(dev_timing),
                    "real_frame_count": sum(clip.frame_count for clip in dev_timing),
                    "supported_frame_count": sum(clip.windows[-1][1] + 1 for clip in dev_timing)}
    binding = {"run_policy": RUN_POLICY if package is None else NEW_RUN_POLICY,
               "recipe": copy.deepcopy(RECIPE),
               "architectures": copy.deepcopy(FUSION_CONFIGS), "hardware": hardware_identity(),
               "code_hashes": code, "schema_hash": code["src/despamo/appearance/schema.py"],
               "protocol_hash": protocol["protocol_hash"], "split_hash": protocol["split_hash"],
               "protocol_identity_hash": digest(protocol),
               "baseline_checkpoint_hash": baseline_hash,
               "adapted_identity": adapted, "timing_hash": _regular_hash(args.motion_timing),
               "timing_payload_hash": digest(timing), "timing_coverage": timing["coverage"],
               "dev_frame_coverage": dev_coverage,
               "bound_files": bound_files}
    if package is not None:
        binding.update(motion_package=compact_motion_binding(package),
                       baseline_motion=baseline_motion,
                       actual_motion={"motion_root": str(motion_root),
                                      "motion_manifest": str(motion_manifest)})
    return protocol, views, config, binding


def _bindings(config, binding, cap):
    entries = {}
    for seed in (0, 1):
        resolved = copy.deepcopy(OmegaConf.to_container(config, resolve=True))
        if not isinstance(resolved, dict):
            raise ValueError("fusion config must resolve to a mapping")
        resolved["seed"] = seed
        entries[str(seed)] = {**copy.deepcopy(binding), "seed": seed,
                              "config": resolved, "gpu_cap_seconds": cap}
    return entries


def _validate_profile(profile, bindings) -> dict:
    if len({motion_run_policy(b) for b in bindings.values()}) != 1:
        raise ValueError("fusion common profile mixes motion mode policies")
    new_motion = all(b.get("run_policy") == NEW_RUN_POLICY for b in bindings.values())
    expected_policy = NEW_INITIALIZATION_POLICY if new_motion else INITIALIZATION_POLICY
    if (not isinstance(profile, dict) or set(profile) != {"profile_policy", "seeds", "profile_hash"}
            or profile["profile_policy"] != expected_policy
            or not isinstance(profile["seeds"], dict) or set(profile["seeds"]) != {"0", "1"}
            or profile["profile_hash"] != digest({k: v for k, v in profile.items()
                                                  if k != "profile_hash"})):
        raise ValueError("fusion common profile malformed/hash mismatch")
    for seed, binding in bindings.items():
        entry = profile["seeds"][seed]
        if (seed == "0" and isinstance(entry, dict)
                and entry.get("initial_base_hash") != E1_SHARED_HASH):
            raise ValueError("fusion profile seed0 base differs from verified E1")
        if (not isinstance(entry, dict)
                or set(entry) != set(binding) | {"initial_base_hash", "common_plain_hash"}
                or any(entry.get(k) != v for k, v in binding.items())
                or type(entry.get("seed")) is not int
                or any(not isinstance(entry.get(k), str)
                       or re.fullmatch(r"[0-9a-f]{64}", entry[k]) is None
                       for k in ("initial_base_hash", "common_plain_hash"))):
            raise ValueError("fusion profile seed/input/source/timing/hardware/recipe mismatch")
    return profile


def _read_profile(path, bindings):
    _regular_hash(path)
    return _validate_profile(read_json(path), bindings)


def _verify_seed0(model) -> None:
    if shared_tensor_hash(model) != E1_SHARED_HASH:
        raise ValueError("fusion seed0 fresh base differs from verified E1")


def _attach_plain(model, seed):
    model.visual_adapter = enable_connector_arm(model.visual_adapter, "plain_mlp", seed)
    return model


def common_plain_hash(model) -> str:
    """Same byte convention as shared_tensor_hash, excluding only new branch names."""
    sha = hashlib.sha256()
    count = 0
    for name, tensor in sorted(model.named_parameters()):
        if (not tensor.requires_grad
                or not name.startswith(("visual_adapter.", "language_model.", "vt_align."))
                or name.startswith(("visual_adapter.corrector.", "visual_adapter.gate."))):
            continue
        count += 1
        sha.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}\n".encode())
        sha.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if not count:
        raise ValueError("missing common plain tensors")
    return sha.hexdigest()


def initialize_fusion(args) -> dict:
    """CPU common base/plain attestations for both seeds; no study/optimizer initialization."""
    _require_args(args)
    path = args.initialization_profile
    if args.arm != "reference" or path is None or args.resume_checkpoint is not None:
        raise ValueError("initialize requires reference, profile and no resume")
    protocol, _, config, binding = _prepare_inputs(args)
    bindings = _bindings(config, binding, args.gpu_cap_seconds)
    _safe_destination(path, args)
    with writer(path.parent):
        if getattr(args, "motion_package", None) is not None:
            validate_motion_graph(bindings["0"])
        if path.exists():
            profile, status = _read_profile(path, bindings), "reused"
        else:
            entries = {}
            rng = capture_rng()
            try:
                for seed in (0, 1):
                    seeded = copy.deepcopy(config)
                    seeded.seed = seed
                    model = build_seeded_projector_model(seeded, STEPS, seed)
                    if any(t.device.type != "cpu" for t in
                           (*model.parameters(), *model.buffers())):
                        raise ValueError("fusion profile requires CPU tensors")
                    if seed == 0:
                        _verify_seed0(model)
                    base_hash = shared_tensor_hash(model)
                    _attach_plain(model, seed)
                    entries[str(seed)] = {**bindings[str(seed)], "initial_base_hash": base_hash,
                                          "common_plain_hash": common_plain_hash(model)}
                    del model
                    gc.collect()
            finally:
                restore_rng(rng)
            _check_live(binding)
            if getattr(args, "motion_package", None) is not None:
                validate_motion_graph(bindings["0"])
            profile = {"profile_policy": (NEW_INITIALIZATION_POLICY
                       if getattr(args, "motion_package", None) is not None
                       else INITIALIZATION_POLICY), "seeds": entries}
            profile["profile_hash"] = digest(profile)
            _validate_profile(profile, bindings)
            atomic_json(path, profile)
            status = "initialized"
    return {"status": status, "initialization_profile": str(path),
            "initialization_profile_hash": profile["profile_hash"],
            "protocol_hash": protocol["protocol_hash"]}


def _run_key(identity) -> str:
    payload = {k: v for k, v in identity.items() if k != "run_key"}
    prefix = "fusion-new-motion" if motion_run_policy(identity) == NEW_RUN_POLICY else "fusion"
    return f"{prefix}-{identity['arm']}-seed{identity['seed']}-16000-" + digest(payload)


def prepare_fusion(args) -> tuple[dict, Path, dict, dict, DictConfig]:
    """Read-only admission; no models, optimizers, output directories or writer locks."""
    _require_args(args)
    protocol, views, config, binding = _prepare_inputs(args)
    profile = _read_profile(args.initialization_profile,
                            _bindings(config, binding, args.gpu_cap_seconds))
    entry = profile["seeds"][str(args.seed)]
    identity = {**entry, "run_policy": (NEW_RUN_POLICY
                 if getattr(args, "motion_package", None) is not None else RUN_POLICY),
                 "arm": args.arm,
                "variant": f"E3_fusion_{args.arm}", "steps": STEPS,
                "architecture": copy.deepcopy(FUSION_CONFIGS[args.arm]),
                "dev_checkpoints": list(DEV_STEPS), "warmup_steps": WARMUP,
                "lr_warmup_steps": 1600, "resume_retention": 2, "snapshot_interval": 500,
                "initialization_profile_hash": profile["profile_hash"],
                "initialization_profile_file_hash": _regular_hash(args.initialization_profile),
                "initialization_profile_path": str(args.initialization_profile.resolve()),
                "supervision_policy": POLICY, "human_review_status": "not_assessed"}
    identity["run_key"] = _run_key(identity)
    output = args.output_base / identity["run_key"]
    _safe_destination(output, args)
    return identity, output, protocol, views, cast(DictConfig, config)


def build_fusion_model(config, arm: str, seed: int,
                       initial_base_hash: str) -> PilotBaselineModule:
    if arm not in FUSION_ARMS or type(seed) is not int or seed not in (0, 1):
        raise ValueError("fusion model arm/seed invalid")
    model = build_seeded_projector_model(config, STEPS, seed)
    if shared_tensor_hash(model) != initial_base_hash:
        raise ValueError("fusion attested original base hash changed")
    _attach_plain(model, seed)
    model.visual_adapter = enable_aligned_fusion(model.visual_adapter, arm, seed)
    return model


def validate_effective_optimizer(model) -> dict:
    configured = model.configure_optimizers()
    optimizer = configured.get("optimizer") if isinstance(configured, dict) else configured
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("fusion optimizer must be AdamW")
    expected = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            if name.startswith("language_model."):
                expected[id(p)] = "flan"
            elif name.startswith(("visual_adapter.", "vt_align.")):
                expected[id(p)] = "adapter"
            else:
                raise ValueError("unowned fusion optimizer parameter")
    seen, rates = set(), {}
    for group in optimizer.param_groups:
        labels = {expected.get(id(p)) for p in group["params"]}
        if len(labels) != 1 or None in labels:
            raise ValueError("fusion optimizer mixed/empty/unowned group")
        label = labels.pop()
        if (label in rates or group["lr"] != RECIPE[f"{label}_lr"]
                or group["betas"] != (0.9, 0.98) or group["eps"] != 1e-8
                or group["weight_decay"] != 0.01):
            raise ValueError("fusion effective optimizer recipe mismatch")
        rates[label] = group["lr"]
        for p in group["params"]:
            if id(p) in seen:
                raise ValueError("fusion duplicated optimizer parameter")
            seen.add(id(p))
    if seen != set(expected) or set(rates) != {"adapter", "flan"}:
        raise ValueError("fusion optimizer missing trainable parameter/group")
    return {"optimizer": "AdamW", "group_learning_rates": rates,
            "betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.01}


def _gradient_norm(parameters, name, positive=True) -> float:
    parameters = [p for p in parameters if p.requires_grad]
    grads = [p.grad for p in parameters]
    if not parameters or any(g is None or not torch.isfinite(g).all() for g in grads):
        raise RuntimeError(f"fusion missing/non-finite {name} gradient")
    norm = sum(float(g.detach().float().norm().cpu()) for g in grads)
    if not math.isfinite(norm) or (positive and norm <= 0):
        raise RuntimeError(f"fusion zero/non-finite {name} gradient")
    return norm


def fusion_gradient_smoke(model, batch, arm: str) -> dict:
    """Two joint backwards around one disposable AdamW step; restore even on failure."""
    if arm not in FUSION_ARMS:
        raise ValueError("unknown fusion smoke arm")
    rng, prompt = capture_rng(), model.rng.getstate()
    # Frozen FLAN tensors are never updated. Avoid cloning them into scarce VRAM.
    tensors = [(p, p.detach().clone()) for p in model.parameters() if p.requires_grad]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    grads = [(p, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    device = next(model.parameters()).device
    initial = shared_tensor_hash(model)
    started = time.monotonic()
    optimizer = None
    result = None
    try:
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        adapter = model.visual_adapter

        def backward():
            model.zero_grad(set_to_none=True)
            context = (torch.autocast("cuda", dtype=torch.bfloat16)
                       if device.type == "cuda" else nullcontext())
            with context:
                losses = model.compute_losses(batch)
                joint = losses["loss"] + model.vt_weight * losses["contra_loss"]
            if not torch.isfinite(joint):
                raise FloatingPointError("fusion joint smoke loss non-finite")
            joint.backward()
            return float(joint.detach().cpu())

        def norms(first=False):
            groups = {"spatial": adapter.spatial_projector.parameters(),
                      "motion": adapter.motion_projector.parameters(),
                      "output": adapter.multimodal_projector.parameters(),
                      "flan": model.language_model.parameters(),
                      "vt": model.vt_align.parameters()}
            result = {name: _gradient_norm(params, name) for name, params in groups.items()}
            if arm != "reference":
                result["corrector_last"] = _gradient_norm(adapter.corrector[-1].parameters(),
                                                          "corrector_last")
                result["corrector_early"] = _gradient_norm(adapter.corrector[0].parameters(),
                                                           "corrector_early", not first)
                if arm == "aligned_learned":
                    result["gate"] = _gradient_norm(adapter.gate.parameters(), "gate", not first)
                if first and any(result[k] != 0 for k in ("corrector_early",)
                                 + (("gate",) if arm == "aligned_learned" else ())):
                    raise RuntimeError("fusion zero-initial branch gradient unexpected")
            return result

        joint_loss = backward()
        first = norms(first=True)
        second = None
        if arm != "reference":
            # Explicit real adapter LR: no LambdaLR step-zero multiplier.
            optimizer = torch.optim.AdamW(
                [p for p in adapter.parameters() if p.requires_grad], lr=3e-4,
                betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01,
            )
            optimizer.step()
            backward()
            second = norms()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        limit = (torch.cuda.get_device_properties(device).total_memory - 2 * 1024**3
                 if device.type == "cuda" else None)
        if limit is not None and peak > limit:
            raise RuntimeError("fusion smoke exceeds actual VRAM minus 2 GiB")
        result = {"status": "passed", "objective": "joint-ce-vt", "optimizer_updates": 0,
                  "disposable_optimizer_updates": int(arm != "reference"),
                  "disposable_learning_rate": 3e-4 if arm != "reference" else None,
                  "joint_loss": joint_loss, "first_gradient_norms": first,
                  "second_gradient_norms": second, "cuda_peak_bytes": peak,
                  "fit_limit_bytes": limit, "cuda_fit": device.type == "cuda",
                  "runtime_seconds": time.monotonic() - started, "initial_model_hash": initial}
    finally:
        if optimizer is not None:
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for tensor, saved in (*tensors, *buffers):
                tensor.copy_(saved)
        for p, grad in grads:
            p.grad = grad
        for module, mode in modes:
            module.training = mode
        model.rng.setstate(prompt)
        restore_rng(rng)
    restored = shared_tensor_hash(model)
    if restored != initial:
        raise RuntimeError("fusion smoke altered initialized tensors")
    result["restored_model_hash"] = restored
    return result


def _check_live(identity) -> None:
    policy = motion_run_policy(identity)
    code = (fusion_code_hashes(PROJECT, new_motion=True) if policy == NEW_RUN_POLICY
            else fusion_code_hashes(PROJECT))
    if (code != identity["code_hashes"]
            or hardware_identity() != identity["hardware"]):
        raise ValueError("fusion source/hardware changed")
    for path, expected in identity["bound_files"].items():
        if _regular_hash(Path(path)) != expected:
            raise ValueError("fusion timing/source/input bytes changed")
    if "initialization_profile_path" in identity:
        if _regular_hash(Path(identity["initialization_profile_path"])) != identity[
                "initialization_profile_file_hash"]:
            raise ValueError("fusion profile bytes changed")


def report_binding(identity: dict) -> dict:
    return {"run_policy": motion_run_policy(identity), "run_identity_hash": digest(identity),
            "model_variant": identity["variant"], "protocol_hash": identity["protocol_hash"],
            "timing_hash": identity["timing_hash"],
            "arm": identity["arm"], "seed": identity["seed"],
            "supervision_policy": POLICY, "human_review_status": "not_assessed"}


def _validate_stats(stats, arm, clips=582):
    keys = {"supported_frame_count", "gate_count", "gate_sum", "gate_sumsq", "gate_min",
            "gate_max", "gate_below_0_05", "gate_above_0_95", "correction_norm",
            "real_frame_count", "clip_count"}
    if not isinstance(stats, dict) or set(stats) != keys or stats["clip_count"] != clips:
        raise ValueError("fusion full-dev statistics inventory invalid")
    for k in ("supported_frame_count", "gate_count", "real_frame_count", "clip_count"):
        if type(stats[k]) is not int or stats[k] < 0:
            raise ValueError("fusion statistics count invalid")
    count = stats["supported_frame_count"]
    if not 0 < count <= stats["real_frame_count"]:
        raise ValueError("fusion coverage invalid")
    numeric = keys - {"supported_frame_count", "gate_count", "real_frame_count", "clip_count"}
    if arm == "reference":
        if stats["gate_count"] != 0 or any(stats[k] is not None for k in numeric):
            raise ValueError("reference gate statistics not applicable")
    else:
        if stats["gate_count"] != count or any(
                type(stats[k]) not in (float, int) or not math.isfinite(stats[k])
                or stats[k] < 0 for k in numeric):
            raise ValueError("aligned fusion statistics invalid")
        if (not 0 <= stats["gate_min"] <= stats["gate_max"] <= 1
                or stats["gate_sum"] > count or stats["gate_sumsq"] > stats["gate_sum"]
                or stats["gate_below_0_05"] + stats["gate_above_0_95"] > count):
            raise ValueError("fusion gate statistics impossible")


def fusion_initialization(model, identity: dict) -> dict:
    """Attest the fresh variant and enumerate every unique trainable parameter."""
    if common_plain_hash(model) != identity["common_plain_hash"]:
        raise ValueError("fusion initialization common-plain hash mismatch")
    counts = {"adapter": 0, "vt": 0, "flan": 0}
    inventory = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = _parameter_group(name)
        counts[group] += parameter.numel()
        inventory[name] = {"shape": list(parameter.shape), "numel": parameter.numel()}
    counts["adapter_vt"] = counts["adapter"] + counts["vt"]
    counts["total"] = counts["adapter_vt"] + counts["flan"]
    receipt = {"schema_version": 1, "kind": "fusion-initialization",
               "run_policy": motion_run_policy(identity), "run_key": identity["run_key"],
               "run_identity_hash": digest(identity),
               "initial_base_hash": identity["initial_base_hash"],
               "common_plain_hash": identity["common_plain_hash"],
               "initial_variant_hash": shared_tensor_hash(model),
               "trainable_parameters": counts, "trainable_inventory": inventory}
    receipt["content_hash"] = digest(receipt)
    return receipt


def _parameter_group(name: str) -> str:
    for prefix, group in (("visual_adapter.", "adapter"), ("vt_align.", "vt"),
                          ("language_model.", "flan")):
        if name.startswith(prefix):
            return group
    raise ValueError(f"unowned fusion initialization parameter: {name}")


def validate_fusion_initialization(run: Path, identity: dict) -> dict:
    path = run / "initialization.json"
    _regular_hash(path)
    receipt = read_json(path)
    keys = {"schema_version", "kind", "run_policy", "run_key", "run_identity_hash",
            "initial_base_hash", "common_plain_hash", "initial_variant_hash",
            "trainable_parameters", "trainable_inventory", "content_hash"}
    if (set(receipt) != keys or type(receipt["schema_version"]) is not int
            or receipt["schema_version"] != 1 or receipt["kind"] != "fusion-initialization"
            or receipt["run_policy"] != motion_run_policy(identity)
            or receipt["run_key"] != identity["run_key"]
            or receipt["run_identity_hash"] != digest(identity)
            or any(receipt[k] != identity[k] for k in ("initial_base_hash", "common_plain_hash"))
            or not isinstance(receipt["initial_variant_hash"], str)
            or re.fullmatch(r"[0-9a-f]{64}", receipt["initial_variant_hash"]) is None
            or receipt["content_hash"] != digest({k: v for k, v in receipt.items()
                                                  if k != "content_hash"})):
        raise ValueError("fusion initialization identity/content mismatch")
    counts = {"adapter": 0, "vt": 0, "flan": 0}
    inventory = receipt["trainable_inventory"]
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("fusion initialization inventory missing")
    for name, item in inventory.items():
        if (not isinstance(item, dict) or set(item) != {"shape", "numel"}
                or not isinstance(item["shape"], list)
                or any(type(n) is not int or n <= 0 for n in item["shape"])
                or type(item["numel"]) is not int or item["numel"] != math.prod(item["shape"])):
            raise ValueError("fusion initialization inventory shape/count mismatch")
        counts[_parameter_group(name)] += item["numel"]
    counts["adapter_vt"] = counts["adapter"] + counts["vt"]
    counts["total"] = counts["adapter_vt"] + counts["flan"]
    if (receipt["trainable_parameters"] != counts
            or any(type(n) is not int or n <= 0 for n in receipt["trainable_parameters"].values())
            or (identity["arm"] == "reference"
                and receipt["initial_variant_hash"] != receipt["common_plain_hash"])):
        raise ValueError("fusion initialization parameter coverage mismatch")
    return receipt


def ensure_fusion_initialization(run: Path, model, identity: dict, *, existing: bool) -> dict:
    expected = fusion_initialization(model, identity)
    path = run / "initialization.json"
    if existing:
        receipt = validate_fusion_initialization(run, identity)
        if receipt != expected:
            raise ValueError("fusion initialization immutable fresh-model receipt mismatch")
    else:
        if path.exists() or path.is_symlink():
            raise ValueError("fusion initialization exists; refusing overwrite")
        atomic_json(path, expected)
    return expected


def _empty_fusion_telemetry() -> dict:
    peak = {"device_type": None, "peak_allocated_bytes": None, "peak_reserved_bytes": None,
            "sample_count": 0, "evidence_scope": "not_observed"}
    return {"schema_version": 1, "training": dict(peak), "dev": dict(peak)}


def _validate_fusion_peak(peak) -> None:
    keys = {"device_type", "peak_allocated_bytes", "peak_reserved_bytes", "sample_count",
            "evidence_scope"}
    if not isinstance(peak, dict) or set(peak) != keys or type(peak["sample_count"]) is not int:
        raise ValueError("fusion telemetry peak schema invalid")
    device = peak["device_type"]
    if device is None:
        if peak != _empty_fusion_telemetry()["training"]:
            raise ValueError("fusion telemetry unobserved phase must be explicit null")
        return
    scope = {"cuda": "process_allocator_high_water_sampled_during_phase",
             "cpu": "cpu_no_cuda_measurement"}
    if (device not in scope or peak["evidence_scope"] != scope[device]
            or peak["sample_count"] <= 0
            or any(type(peak[k]) is not int or peak[k] < 0
                   for k in ("peak_allocated_bytes", "peak_reserved_bytes"))
            or peak["peak_reserved_bytes"] < peak["peak_allocated_bytes"]
            or (device == "cpu" and (peak["peak_allocated_bytes"] or peak["peak_reserved_bytes"]))):
        raise ValueError("fusion telemetry peak evidence invalid")


def _validate_fusion_telemetry(telemetry) -> None:
    if (not isinstance(telemetry, dict) or set(telemetry) != {"schema_version", "training", "dev"}
            or type(telemetry["schema_version"]) is not int or telemetry["schema_version"] != 1):
        raise ValueError("fusion telemetry schema invalid")
    for phase in ("training", "dev"):
        _validate_fusion_peak(telemetry[phase])


def merge_fusion_telemetry(durable: dict, checkpoint: dict) -> dict:
    """Componentwise maxima, not sums: checkpoint observations may already be durable."""
    _validate_fusion_telemetry(durable)
    _validate_fusion_telemetry(checkpoint)
    result = copy.deepcopy(durable)
    for phase in ("training", "dev"):
        old, saved = result[phase], checkpoint[phase]
        if saved["device_type"] is None:
            continue
        if old["device_type"] is None:
            result[phase] = dict(saved)
        elif old["device_type"] != saved["device_type"]:
            raise ValueError("fusion telemetry recovery device changed")
        else:
            for key in ("peak_allocated_bytes", "peak_reserved_bytes", "sample_count"):
                old[key] = max(old[key], saved[key])
    return result


def capture_fusion_peak(model, phase: str, previous: dict | None = None) -> dict:
    """No allocator reset: conservative process high-water includes resident/earlier phases."""
    if phase not in ("training", "dev"):
        raise ValueError("unknown fusion telemetry phase")
    previous = _empty_fusion_telemetry() if previous is None else previous
    _validate_fusion_telemetry(previous)
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    measured = _empty_fusion_telemetry()
    measured[phase] = {
        "device_type": device.type,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else 0,
        "sample_count": previous[phase]["sample_count"] + 1,
        "evidence_scope": ("process_allocator_high_water_sampled_during_phase" if cuda
                           else "cpu_no_cuda_measurement"),
    }
    return merge_fusion_telemetry(previous, measured)


def fusion_checkpoint_sizes(run: Path) -> dict[str, int]:
    """Logical saved sizes from actual regular owned files; no smoke checkpoint estimate."""
    sizes = {}
    for path in sorted(run.glob("*.ckpt")):
        match = re.fullmatch(r"(?:resume|step)-(\d+)\.ckpt", path.name)
        if match is None or not 0 < int(match.group(1)) <= STEPS:
            raise ValueError("unowned fusion checkpoint size inventory")
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe fusion checkpoint size inventory")
        sizes[path.name] = path.stat().st_size
    return sizes


def fusion_visual_token_lengths(model, batch) -> list[dict]:
    """Measure an eval diagnostic forward's actual output mask; no structural prediction."""
    device = next(model.parameters()).device
    context = (torch.autocast("cuda", dtype=torch.bfloat16)
               if device.type == "cuda" else nullcontext())
    with torch.inference_mode(), context:
        _, mask = model.visual_adapter(batch.spatial.to(device), batch.spatial_mask.to(device),
                                       batch.motion.to(device), batch.motion_mask.to(device))
        if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[0] != len(batch.clip_ids):
            raise ValueError("fusion visual output mask invalid")
        lengths = mask.sum(dim=1).cpu().tolist()
    return [{"clip_id": clip, "valid_tokens": int(length), "padded_tokens": mask.shape[1]}
            for clip, length in zip(batch.clip_ids, lengths, strict=True)]


def _validate_dev_telemetry(report, dev_ids) -> None:
    rows = report.get("visual_token_lengths")
    if (not isinstance(rows, list) or len(rows) != len(dev_ids)
            or report.get("visual_token_length_scope")
            != "eval_diagnostic_forward_actual_output_mask"
            or any(not isinstance(row, dict) or set(row) != {
                "clip_id", "valid_tokens", "padded_tokens"} for row in rows)
            or tuple(row["clip_id"] for row in rows) != dev_ids
            or any(type(row[k]) is not int for row in rows
                   for k in ("valid_tokens", "padded_tokens"))
            or any(not 0 < row["valid_tokens"] <= row["padded_tokens"] for row in rows)):
        raise ValueError("fusion measured visual token lengths/ordered IDs invalid")
    _validate_fusion_peak(report.get("cuda_memory"))
    if report["cuda_memory"]["device_type"] is None:
        raise ValueError("fusion dev telemetry phase unobserved")


def _validate_report(run, report, step, identity, dev_ids):
    checkpoint = run / f"step-{step}.ckpt"
    if report.get("content_hash") != digest({k: v for k, v in report.items()
                                              if k != "content_hash"}):
        raise ValueError("fusion dev content hash mismatch")
    if (report.get("checkpoint_step") != step or report.get("split") != "dev"
            or report.get("split_hash") != identity["split_hash"]
            or report.get("decoding") != DECODER
            or tuple(item.get("clip_id") for item in report.get("items", ())) != dev_ids
            or any(report.get(k) != v for k, v in report_binding(identity).items())
            or "bleu4" not in report.get("metrics", {})
            or any(type(v) not in (float, int) or not math.isfinite(v)
                   for v in report["metrics"].values())
            or report.get("checkpoint_hash") != _regular_hash(checkpoint)):
        raise ValueError("fusion ordered full dev/recipe/checkpoint mismatch")
    _validate_stats(report.get("fusion_stats"), identity["arm"])
    if any(report["fusion_stats"][key] != value
           for key, value in identity["dev_frame_coverage"].items()):
        raise ValueError("fusion dev coverage differs from admitted timing")
    validate_fusion_initialization(run, identity)
    if (report.get("initialization_hash") != _regular_hash(run / "initialization.json")
            or type(report.get("checkpoint_size_bytes")) is not int
            or report["checkpoint_size_bytes"] != checkpoint.stat().st_size):
        raise ValueError("fusion initialization/checkpoint size telemetry mismatch")
    _validate_dev_telemetry(report, dev_ids)
    return checkpoint


def _dev_report_hashes(run: Path) -> dict:
    return {f"dev-{step}.json": _regular_hash(run / f"dev-{step}.json") for step in DEV_STEPS}


def validate_fusion_selection(run: Path, protocol: dict, dev_ids: tuple[str, ...],
                              expected_identity: dict | None = None) -> dict:
    if (run.is_symlink() or not run.is_dir() or len(dev_ids) != 582
            or len(set(dev_ids)) != 582
            or dev_ids != tuple(protocol["split"]["groups"]["dev"]["clip_ids"])):
        raise ValueError("fusion completion requires exact ordered 582 dev IDs")
    for name in ("budget.json", "run-status.json", "selected-checkpoint.json"):
        _regular_hash(run / name)
    budget = read_json(run / "budget.json")
    status = read_json(run / "run-status.json")
    selected = read_json(run / "selected-checkpoint.json")
    identity = budget.get("identity", {})
    cap, spent = identity.get("gpu_cap_seconds"), budget.get("gpu_seconds")
    validate_motion_graph(identity)
    if (identity.get("run_policy") != motion_run_policy(identity)
            or identity.get("arm") not in FUSION_ARMS
            or type(identity.get("seed")) is not int or identity["seed"] not in (0, 1)
            or identity.get("variant") != f"E3_fusion_{identity['arm']}"
            or identity.get("steps") != STEPS or identity.get("recipe") != RECIPE
            or identity.get("architecture") != FUSION_CONFIGS[identity["arm"]]
            or identity.get("dev_checkpoints") != list(DEV_STEPS)
            or identity.get("run_key") != run.name or _run_key(identity) != run.name
            or (expected_identity is not None and expected_identity != identity)
            or identity.get("protocol_hash") != protocol["protocol_hash"]
            or identity.get("split_hash") != protocol["split_hash"]
            or identity.get("supervision_policy") != POLICY
            or identity.get("human_review_status") != "not_assessed"
            or type(cap) is not int or not 0 < cap <= MAX_CAP_SECONDS
            or type(spent) not in (int, float) or not math.isfinite(spent) or spent < 0
            or (not time_waived(identity) and spent >= cap)
            or budget.get("ceiling_seconds") != cap or budget.get("status") != "closed"
            or budget.get("optimizer_step") != STEPS or status.get("status") != "complete"
            or status.get("global_step") != STEPS or status.get("run_key") != run.name):
        raise ValueError("fusion completion identity/budget/step mismatch")
    _check_live(identity)
    initialization = validate_fusion_initialization(run, identity)
    initialization_hash = _regular_hash(run / "initialization.json")
    telemetry = budget.get("fusion_telemetry")
    _validate_fusion_telemetry(telemetry)
    for phase in ("training", "dev"):
        if telemetry[phase]["device_type"] is None:
            raise ValueError(f"fusion completion has unobserved {phase} telemetry")
    if (budget.get("initialization_hash") != initialization_hash
            or selected.get("initialization_hash") != initialization_hash):
        raise ValueError("fusion initialization receipt byte binding mismatch")
    if (selected.get("fusion_telemetry") != telemetry
            or selected.get("fusion_telemetry_hash") != digest(telemetry)
            or status.get("fusion_telemetry_hash") != digest(telemetry)
            or status.get("initialization_hash") != initialization_hash
            or selected.get("checkpoint_sizes_bytes") != fusion_checkpoint_sizes(run)
            or budget.get("checkpoint_sizes_bytes") != fusion_checkpoint_sizes(run)):
        raise ValueError("fusion completion telemetry/checkpoint size binding mismatch")
    if selected.get("dev_report_hashes") != _dev_report_hashes(run):
        raise ValueError("fusion selection all dev byte hashes mismatch")
    reports = []
    for step in DEV_STEPS:
        report = read_json(run / f"dev-{step}.json")
        checkpoint = torch.load(_validate_report(run, report, step, identity, dev_ids),
                                map_location="cpu", weights_only=False)
        if (checkpoint.get("run_metadata") != identity
                or checkpoint.get("pilot_identity") != identity
                or type(checkpoint.get("global_step")) is not int
                or checkpoint["global_step"] != step):
            raise ValueError("fusion checkpoint full identity mismatch")
        _validate_checkpoint_telemetry(checkpoint, initialization_hash)
        if merge_fusion_telemetry(telemetry, checkpoint["fusion_telemetry"]) != telemetry:
            raise ValueError("fusion checkpoint telemetry exceeds durable completion maxima")
        dev_peak = {**_empty_fusion_telemetry(), "dev": report["cuda_memory"]}
        if merge_fusion_telemetry(telemetry, dev_peak) != telemetry:
            raise ValueError("fusion dev telemetry exceeds durable completion maxima")
        reports.append(report)
    best = choose_checkpoint(reports, dev_ids)
    if (selected.get("step") != best["checkpoint_step"]
            or selected.get("checkpoint_hash") != best["checkpoint_hash"]
            or selected.get("metrics") != best["metrics"]
            or any(selected.get(k) != v for k, v in report_binding(identity).items())):
        raise ValueError("fusion selected checkpoint/tie/identity mismatch")
    return {"checkpoint": run / f"step-{selected['step']}.ckpt", "identity": identity,
            "checkpoint_hash": selected["checkpoint_hash"], "step": selected["step"],
            "metrics": selected["metrics"], "initialization": initialization,
            "initialization_hash": initialization_hash, "fusion_telemetry": telemetry,
            "checkpoint_sizes_bytes": selected["checkpoint_sizes_bytes"]}


def _validate_checkpoint_telemetry(checkpoint, initialization_hash) -> None:
    if checkpoint.get("initialization_hash") != initialization_hash:
        raise ValueError("fusion checkpoint initialization binding mismatch")
    _validate_fusion_telemetry(checkpoint.get("fusion_telemetry"))


class FusionLedger(E1Ledger):
    def __init__(self, root, identity, charged, step, monotonic, wall):
        super().__init__(root, identity, charged, step, monotonic, wall)
        self.initialization_hash = None
        self.telemetry = _empty_fusion_telemetry()
        if (root / "budget.json").exists():
            budget = read_json(root / "budget.json")
            validate_fusion_initialization(root, identity)
            self.initialization_hash = _regular_hash(root / "initialization.json")
            if budget.get("initialization_hash") != self.initialization_hash:
                raise ValueError("fusion resume initialization byte binding mismatch")
            _validate_fusion_telemetry(budget.get("fusion_telemetry"))
            self.telemetry = copy.deepcopy(budget["fusion_telemetry"])

    def _write(self, status: str, phase: str | None = None) -> None:
        atomic_json(self.root / "budget.json", {
            "identity": self.identity, "gpu_seconds": self.last, "optimizer_step": self.step,
            "pid": os.getpid(), "phase": phase, "status": status, "wall_timestamp": self.wall(),
            "ceiling_seconds": self.identity["gpu_cap_seconds"],
            "initialization_hash": self.initialization_hash,
            "fusion_telemetry": self.telemetry,
            "checkpoint_sizes_bytes": fusion_checkpoint_sizes(self.root),
            **execution_fields(self.identity),
        })

    @classmethod
    def resume(cls, root, identity, checkpoint_seconds, *, checkpoint_step,
               monotonic=time.monotonic, wall=time.time):
        if not time_waived(identity):
            return super().resume(root, identity, checkpoint_seconds,
                                  checkpoint_step=checkpoint_step, monotonic=monotonic, wall=wall)
        return resume_ledger(cls, root, identity, checkpoint_seconds, checkpoint_step,
                             monotonic, wall)

    def charge(self, *, step: int, phase: str) -> float:
        if time_waived(self.identity):
            if type(step) is not int or step < self.step or not phase:
                raise ValueError("waived ledger step/phase moved backwards")
            elapsed = self.monotonic() - self.started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("waived ledger monotonic clock moved backwards")
            self.last = max(self.last, self.charged + elapsed)
            self.step = step
            self._write("running", phase)
            return self.last
        spent = super().charge(step=step, phase=phase)
        if spent >= self.identity["gpu_cap_seconds"]:
            raise RuntimeError("fusion run cap exhausted")
        return spent

    def require_remaining(self, reserve_seconds: float, *, step: int, phase: str) -> float:
        if not math.isfinite(reserve_seconds) or reserve_seconds < 0:
            raise ValueError("fusion reserve invalid")
        spent = self.charge(step=step, phase=phase)
        if (not time_waived(self.identity)
                and spent + reserve_seconds >= self.identity["gpu_cap_seconds"]):
            raise RuntimeError("fusion cap lacks time reserve")
        return spent


def _refuse_completed(output):
    status, selected = output / "run-status.json", output / "selected-checkpoint.json"
    if selected.is_symlink() or status.is_symlink():
        raise ValueError("fusion run record unsafe")
    if status.is_file():
        record = read_json(status)
        if record.get("status") == "complete":
            raise ValueError("fusion run already complete")
        if (record.get("status") not in ("partial", "smoke_passed")
                or record.get("run_key") != output.name
                or type(record.get("global_step")) is not int
                or not 0 <= record["global_step"] <= STEPS):
            raise ValueError("unknown fusion status or step/root mismatch")


@contextmanager
def owned_fusion_ledger(args, identity, output):
    validate_motion_graph(identity)
    expected_policy = NEW_RUN_POLICY if getattr(args, "motion_package", None) is not None else (
        RUN_POLICY)
    if (motion_run_policy(identity) != expected_policy
            or (expected_policy == NEW_RUN_POLICY
                and identity["motion_package"]["root"] != str(args.motion_package))
            or identity.get("run_key") != output.name
            or _run_key(identity) != output.name):
        raise ValueError("fusion ledger requires owned policy/root")
    _refuse_completed(output)
    adopt_smoke = args.resume_checkpoint is None and output.exists()
    if args.resume_checkpoint is None:
        if output.is_symlink() or (adopt_smoke and not output.is_dir()):
            raise ValueError("unsafe fusion output")
        if not adopt_smoke:
            output.mkdir()
            _fsync_directory(output.parent)
    elif output.is_symlink() or not output.is_dir():
        raise ValueError("fusion resume root unsafe/missing")
    with writer(output):
        _refuse_completed(output)
        if adopt_smoke:
            _regular_hash(output / "run-status.json")
            _regular_hash(output / "budget.json")
            status = read_json(output / "run-status.json")
            budget = read_json(output / "budget.json")
            if (status.get("status") != "smoke_passed" or status.get("global_step") != 0
                    or status.get("run_key") != output.name or budget.get("identity") != identity
                    or budget.get("status") != "closed" or budget.get("optimizer_step") != 0
                    or budget.get("ceiling_seconds") != identity["gpu_cap_seconds"]
                    or type(budget.get("gpu_seconds")) not in (float, int)
                    or not math.isfinite(budget["gpu_seconds"])
                    or budget["gpu_seconds"] < 0
                    or (not time_waived(identity)
                        and budget["gpu_seconds"] >= identity["gpu_cap_seconds"] - 300)
                    or any(output.glob("*.ckpt"))
                    or (output / "selected-checkpoint.json").exists()):
                raise ValueError("fusion output exists without owned smoke-only closed ledger")
            ledger = cast(FusionLedger, FusionLedger.resume(output, identity, 0, checkpoint_step=0))
        elif args.resume_checkpoint is None:
            ledger = FusionLedger(output, identity, 0, 0, time.monotonic, time.time)
            ledger._write("created")
        else:
            _regular_hash(output / "budget.json")
            budget = read_json(output / "budget.json")
            if (budget.get("identity") != identity
                    or budget.get("ceiling_seconds") != identity["gpu_cap_seconds"]
                    or budget.get("status") not in ("closed", "paused", "running", "resumed")):
                raise ValueError("fusion resume budget identity/status mismatch")
            spent = budget.get("gpu_seconds")
            if type(spent) not in (float, int) or not math.isfinite(spent) or spent < 0:
                raise ValueError("fusion resume seconds invalid")
            effective = spent
            if budget["status"] not in ("closed", "paused"):
                elapsed = time.time() - budget["wall_timestamp"]
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("fusion resume wall clock invalid")
                effective += elapsed
            if not time_waived(identity) and effective >= identity["gpu_cap_seconds"] - 300:
                raise ValueError("fusion resume cap lacks startup reserve")
            checkpoint = validate_e1_resume_checkpoint(
                args.resume_checkpoint, output, identity, spent,
            )
            if (type(checkpoint["global_step"]) is not int
                    or not 0 < checkpoint["global_step"] <= STEPS
                    or not math.isfinite(checkpoint["gpu_seconds_cumulative"])):
                raise ValueError("fusion resume checkpoint step/seconds invalid")
            validate_fusion_initialization(output, identity)
            initialization_hash = _regular_hash(output / "initialization.json")
            _validate_checkpoint_telemetry(checkpoint, initialization_hash)
            ledger = cast(FusionLedger, FusionLedger.resume(
                output, identity, checkpoint["gpu_seconds_cumulative"],
                checkpoint_step=checkpoint["global_step"],
            ))
            ledger.telemetry = merge_fusion_telemetry(
                ledger.telemetry, checkpoint["fusion_telemetry"],
            )
        try:
            ledger.require_remaining(300, step=ledger.step, phase="startup")
            yield ledger
        except BaseException as error:
            atomic_json(output / "run-status.json", {
                        "status": "partial", "global_step": ledger.step,
                        "run_key": output.name, "error": type(error).__name__})
            raise
        finally:
            ledger.close()


def _durable_checkpoint(trainer, target):
    pending = target.with_name(f".{target.name}.{os.getpid()}.pending")
    if any(p.exists() or p.is_symlink() for p in (target, pending)):
        raise ValueError("fusion checkpoint exists; refusing overwrite")
    try:
        trainer.save_checkpoint(pending)
        _regular_hash(pending)
        with pending.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(pending, target)
        _fsync_directory(target.parent)
    finally:
        pending.unlink(missing_ok=True)


class FusionResumeSnapshots(pl.Callback):
    def __init__(self, output, ledger):
        super().__init__()
        if (output != ledger.root
                or ledger.identity.get("run_policy") != motion_run_policy(ledger.identity)):
            raise ValueError("fusion snapshots require owned root")
        self.output, self.ledger = output, ledger
        self.completed = set()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step <= 0 or step % 500 or step in self.completed:
            return
        snapshots = []
        for path in self.output.glob("resume-*.ckpt"):
            match = re.fullmatch(r"resume-(\d+)\.ckpt", path.name)
            if match is None:
                raise ValueError("unsafe fusion snapshot name")
            _regular_hash(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if (saved.get("pilot_identity") != self.ledger.identity
                    or saved.get("run_metadata") != self.ledger.identity
                    or saved.get("global_step") != int(match.group(1))):
                raise ValueError("unowned fusion snapshot")
            snapshots.append((int(match.group(1)), path))
        self.ledger.require_remaining(300, step=step, phase="before_resume_snapshot")
        target = self.output / f"resume-{step}.ckpt"
        _durable_checkpoint(trainer, target)
        self.ledger.charge(step=step, phase="resume_snapshot_saved")
        self.completed.add(step)
        snapshots.append((step, target))
        for old_step, path in sorted(snapshots)[:-2]:
            path.unlink()
            self.completed.discard(old_step)
        _fsync_directory(self.output)


def _batch_frame_stats(model, batch, arm):
    """Explicit diagnostic recomputation in eval; generation forward remains unchanged."""
    adapter = model.visual_adapter
    device = next(model.parameters()).device
    spatial, smask = batch.spatial.to(device), batch.spatial_mask.to(device)
    motion, mmask = batch.motion.to(device), batch.motion_mask.to(device)
    totals = None
    context = (torch.autocast("cuda", dtype=torch.bfloat16)
               if device.type == "cuda" else nullcontext())
    with torch.no_grad(), context:
        svalues = adapter.spatial_projector(spatial.masked_fill(~smask[..., None], 0))
        mvalues = adapter.motion_projector(motion.masked_fill(~mmask[..., None], 0))
        for i in range(spatial.shape[0]):
            s, m = svalues[i, smask[i]], mvalues[i, mmask[i]]
            starts = torch.arange(m.shape[0], device=device) * 8
            aligned, supported = interpolate_motion(m, torch.arange(s.shape[0], device=device),
                                                     starts, (starts + 15).clamp_max(len(s) - 1))
            gates = corrections = None
            if arm != "reference":
                g = (torch.full_like(s[:, :1], 0.5) if adapter.gate is None
                     else adapter.gate(torch.cat((s, aligned), dim=-1)).sigmoid())
                corrections = (g * adapter.corrector(aligned)).masked_fill(~supported[:, None], 0)
                gates = g.squeeze(-1)
            stats = fusion_frame_stats(supported, gates, corrections, arm=arm)
            stats.update(real_frame_count=len(s), clip_count=1)
            totals = _merge_stats(totals, stats)
    return totals


def _merge_stats(total, stats):
    if total is None:
        return dict(stats)
    result = dict(total)
    for k, value in stats.items():
        old = result[k]
        if k in ("gate_min", "gate_max"):
            values = [v for v in (old, value) if v is not None]
            result[k] = (min(values) if k == "gate_min" else max(values)) if values else None
        else:
            result[k] = None if value is None else (old or 0) + value
    return result


class FusionDevCheckpoints(FullDevCheckpoints):
    def on_train_start(self, trainer, pl_module):
        ledger = self.ledger
        assert ledger is not None
        for step in self.steps:
            target, path = self.output / f"step-{step}.ckpt", self.output / f"dev-{step}.json"
            if path.exists() or path.is_symlink():
                _regular_hash(path)
                _validate_report(self.output, read_json(path), step,
                                 ledger.identity, self.expected)
                self.completed.add(step)
            elif step < trainer.global_step:
                raise ValueError("fusion missing earlier dev report requires matching checkpoint")
            elif target.exists() or target.is_symlink():
                if step != trainer.global_step:
                    raise ValueError("fusion incomplete dev requires matching step")
                self._score_saved(pl_module, step, target)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        ledger = self.ledger
        assert ledger is not None
        step = trainer.global_step
        if step in self.steps and step not in self.completed:
            ledger.require_remaining(300, step=step, phase="before_dev_checkpoint")
            target = self.output / f"step-{step}.ckpt"
            _durable_checkpoint(trainer, target)
            self._score_saved(pl_module, step, target)

    def _score_saved(self, pl_module, step, target):
        model, ledger, generator = pl_module, self.ledger, self.loader.generator
        assert ledger is not None and generator is not None
        ledger = cast(FusionLedger, ledger)
        _check_live(ledger.identity)
        ledger.require_remaining(300, step=step, phase="before_dev")
        modes = [(m, m.training) for m in model.modules()]
        rng, prompt = capture_rng(), model.rng.getstate()
        loader_rng = generator.get_state()
        stats = None
        lengths = []
        seen = 0

        def batches():
            nonlocal stats
            for batch in self.loader:
                stats = _merge_stats(
                    stats, _batch_frame_stats(model, batch, ledger.identity["arm"]),
                )
                lengths.extend(fusion_visual_token_lengths(model, batch))
                yield batch

        def charged(count):
            nonlocal seen
            seen += count
            ledger.require_remaining(30, step=step, phase=f"dev-{seen}/582")

        try:
            model.eval()
            report = score_pilot_batches(model, batches(), self.expected, "dev", self.split_hash,
                                          on_batch_end=charged)
            ledger.telemetry = capture_fusion_peak(model, "dev", getattr(ledger, "telemetry", None))
            ledger.charge(step=step, phase="dev_telemetry_measured")
            report.update(checkpoint_step=step, checkpoint_hash=_regular_hash(target),
                          checkpoint_size_bytes=target.stat().st_size,
                          initialization_hash=_regular_hash(self.output / "initialization.json"),
                          visual_token_lengths=lengths,
                          visual_token_length_scope="eval_diagnostic_forward_actual_output_mask",
                          cuda_memory=copy.deepcopy(ledger.telemetry["dev"]),
                          fusion_stats=stats, **report_binding(ledger.identity))
            report["content_hash"] = digest(report)
            _validate_report(self.output, report, step, ledger.identity, self.expected)
            _check_live(ledger.identity)
            atomic_json(self.output / f"dev-{step}.json", report)
            self.completed.add(step)
        finally:
            try:
                ledger.telemetry = capture_fusion_peak(
                    model, "dev", getattr(ledger, "telemetry", None),
                )
                ledger.charge(step=step, phase="dev_telemetry_finalized")
            finally:
                for module, mode in modes:
                    module.training = mode
                model.rng.setstate(prompt)
                restore_rng(rng)
                generator.set_state(loader_rng)


def _recover_terminal_dev(model, data, dev, resume: Path, output: Path,
                           identity: dict, ledger) -> None:
    """16000 recovery: strict tensors/sampler/RNG reload and evaluation only."""
    target = output / f"step-{STEPS}.ckpt"
    source = target if target.exists() or target.is_symlink() else resume
    checkpoint = validate_e1_resume_checkpoint(source, output, identity, ledger.last)
    if checkpoint["global_step"] != STEPS or "prompt_rng_state" not in checkpoint:
        raise ValueError("fusion terminal recovery requires full step/RNG metadata")
    validate_fusion_initialization(output, identity)
    initialization_hash = _regular_hash(output / "initialization.json")
    _validate_checkpoint_telemetry(checkpoint, initialization_hash)
    ledger.telemetry = merge_fusion_telemetry(
        getattr(ledger, "telemetry", _empty_fusion_telemetry()), checkpoint["fusion_telemetry"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    data.load_state_dict(checkpoint["pilot_sampler"])
    model.rng.setstate(checkpoint["prompt_rng_state"])
    restore_rng(checkpoint["pilot_rng"])
    ledger.charge(step=STEPS, phase="terminal_checkpoint_loaded")
    if not target.exists():
        ledger.require_remaining(300, step=STEPS, phase="before_terminal_dev_checkpoint")
        copy_only = SimpleNamespace(save_checkpoint=lambda p: shutil.copyfile(source, p))
        _durable_checkpoint(copy_only, target)
    dev.on_train_start(SimpleNamespace(global_step=STEPS), model)


class FusionRuntimeGuard(E1BudgetGuard):
    ledger: FusionLedger

    def __init__(self, ledger: FusionLedger):
        if not time_waived(ledger.identity):
            super().__init__(ledger)
        else:
            pl.Callback.__init__(self)
            self.ledger = ledger
            self.identity = dict(ledger.identity)
            self.spent_gpu_seconds = ledger.last
            self.started = time.monotonic()

    def on_train_start(self, trainer, pl_module):
        _check_live(self.ledger.identity)
        self.ledger.telemetry = capture_fusion_peak(pl_module, "training", self.ledger.telemetry)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ledger.telemetry = capture_fusion_peak(pl_module, "training", self.ledger.telemetry)
        # Hardware/source admission throughout fit, not only at its boundaries.
        if trainer.global_step % 500 == 0:
            _check_live(self.ledger.identity)
            require_storage(self.ledger.root)
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_end(self, trainer, pl_module):
        self.ledger.telemetry = capture_fusion_peak(pl_module, "training", self.ledger.telemetry)
        self.ledger.charge(step=trainer.global_step, phase="training_telemetry_finalized")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.ledger.telemetry = capture_fusion_peak(pl_module, "training", self.ledger.telemetry)
        if not time_waived(self.ledger.identity):
            super().on_save_checkpoint(trainer, pl_module, checkpoint)
        else:
            checkpoint["pilot_identity"] = self.identity
            checkpoint["pilot_sampler"] = trainer.datamodule.state_dict()
            checkpoint["pilot_rng"] = capture_rng()
            checkpoint["gpu_seconds_cumulative"] = self.ledger.charge(
                step=trainer.global_step, phase="checkpoint")
            checkpoint.update(**execution_fields(self.ledger.identity))
        checkpoint["initialization_hash"] = self.ledger.initialization_hash
        checkpoint["fusion_telemetry"] = copy.deepcopy(self.ledger.telemetry)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        _validate_checkpoint_telemetry(checkpoint, self.ledger.initialization_hash)
        self.ledger.telemetry = merge_fusion_telemetry(
            self.ledger.telemetry, checkpoint["fusion_telemetry"],
        )
        if not time_waived(self.ledger.identity):
            super().on_load_checkpoint(trainer, pl_module, checkpoint)
        else:
            previous = checkpoint.get("gpu_seconds_cumulative")
            if (checkpoint.get("pilot_identity") != self.identity
                    or type(previous) not in (int, float) or not math.isfinite(previous)
                    or not 0 <= previous <= self.ledger.last):
                raise ValueError("waived checkpoint identity/accounting mismatch")
            trainer.datamodule.load_state_dict(checkpoint["pilot_sampler"])
            restore_rng(checkpoint["pilot_rng"])

    def on_exception(self, trainer, pl_module, exception):
        self.ledger.telemetry = capture_fusion_peak(pl_module, "training", self.ledger.telemetry)
        super().on_exception(trainer, pl_module, exception)


def require_storage(path: Path) -> dict:
    linux_free = shutil.disk_usage(path).free
    windows_free = None
    if Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists():
        windows_free = int(subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command",
             "(Get-PSDrive -Name C).Free"], text=True,
        ).strip())
    if linux_free < 50 * 1024**3 or (windows_free is not None and windows_free < 50 * 1024**3):
        raise ValueError("fusion Linux/Windows storage below 50 GiB reserve")
    return {"linux_free_bytes": linux_free, "windows_free_bytes": windows_free}


def _smoke_batches(model, views, ledger, identity):
    records = views["train"].records
    frame_path = Path(identity["config"]["data"]["spatial_root"]) / "complete/frame_rows.json"
    frames = read_json(frame_path)
    long_indices = sorted(range(len(records)), key=lambda i:
                          frames["clips"][records[i]["fileid"]]["frame_count"], reverse=True)[:4]
    results = {}
    rng = capture_rng()
    try:
        for label, indices in (("ordinary", range(4)), ("long", long_indices)):
            ledger.require_remaining(300, step=ledger.step, phase=f"before_smoke_{label}")
            physical = collate_phoenix([views["train"][i] for i in indices])
            result = fusion_gradient_smoke(model, physical, identity["arm"])
            if not result["cuda_fit"]:
                raise RuntimeError("fusion smoke requires real CUDA fit")
            result["clip_ids_hash"] = digest(physical.clip_ids)
            results[label] = result
            ledger.charge(step=ledger.step, phase=f"smoke_{label}")
    finally:
        restore_rng(rng)
    return results


def _execute(args, *, smoke_only=False):
    if smoke_only and args.resume_checkpoint is not None:
        raise ValueError("fusion smoke mode cannot relabel a production resume")
    if not args.authorize_fusion_run:
        raise ValueError("fusion GPU modes require --authorize-fusion-run")
    identity, output, protocol, views, config = prepare_fusion(args)
    require_storage(args.output_base)
    if not torch.cuda.is_available():
        raise RuntimeError("fusion runtime requires CUDA")
    with owned_fusion_ledger(args, identity, output) as ledger:
        old_alarm = signal.getsignal(signal.SIGALRM)

        def timeout(signum, frame):
            raise TimeoutError("fusion run cap expired")

        try:
            signal.signal(signal.SIGALRM, timeout)
            if not time_waived(identity):
                signal.setitimer(signal.ITIMER_REAL,
                                 max(0.001, identity["gpu_cap_seconds"] - ledger.last - 1))
            _check_live(identity)
            model = build_fusion_model(config, args.arm, args.seed, identity["initial_base_hash"])
            if common_plain_hash(model) != identity["common_plain_hash"]:
                raise ValueError("fusion common plain parameter hash changed")
            ensure_fusion_initialization(
                output, model, identity,
                existing=(args.resume_checkpoint is not None
                          or ledger.initialization_hash is not None),
            )
            ledger.initialization_hash = _regular_hash(output / "initialization.json")
            model.run_metadata = identity
            terminal = args.resume_checkpoint is not None and ledger.step == STEPS
            if not terminal:
                validate_effective_optimizer(model)
            model.to("cuda")
            ledger.charge(step=ledger.step, phase="model_loaded")
            if not terminal:
                smokes = _smoke_batches(model, views, ledger, identity)
                atomic_json(output / "gradient-smoke.json", smokes)
            if smoke_only:
                if terminal:
                    raise ValueError("terminal recovery requires run mode")
                result = {"status": "smoke_passed", "global_step": ledger.step,
                          "run_key": output.name}
                atomic_json(output / "run-status.json", result)
                return result
            data = build_seeded_projector_data(views["train"], STEPS, args.seed)
            ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
            dev = FusionDevCheckpoints(views["dev"], ids, protocol["split_hash"], output,
                                       STEPS, WARMUP, 4, identity["variant"], ledger=ledger,
                                       checkpoint_schedule=DEV_STEPS)
            _check_live(identity)
            if terminal:
                assert args.resume_checkpoint is not None
                _recover_terminal_dev(model, data, dev, args.resume_checkpoint,
                                      output, identity, ledger)
            else:
                trainer = pl.Trainer(**dict(config.trainer), callbacks=[FusionRuntimeGuard(ledger),
                                     dev, FusionResumeSnapshots(output, ledger)])
                checkpoint_path = str(args.resume_checkpoint) if args.resume_checkpoint else None
                trainer.fit(model, datamodule=data, ckpt_path=checkpoint_path)
                if trainer.global_step != STEPS:
                    raise RuntimeError("fusion run stopped before horizon")
            _check_live(identity)
            reports = [read_json(output / f"dev-{s}.json") for s in DEV_STEPS]
            for step, report in zip(DEV_STEPS, reports, strict=True):
                _validate_report(output, report, step, identity, ids)
            best = choose_checkpoint(reports, ids)
            ledger.charge(step=STEPS, phase="selection_complete")
            selection = {
                "step": best["checkpoint_step"], "checkpoint_hash": best["checkpoint_hash"],
                "metrics": best["metrics"], **report_binding(identity),
                "dev_report_hashes": _dev_report_hashes(output),
                "initialization_hash": ledger.initialization_hash,
                "fusion_telemetry": copy.deepcopy(ledger.telemetry),
                "fusion_telemetry_hash": digest(ledger.telemetry),
                "checkpoint_sizes_bytes": fusion_checkpoint_sizes(output),
                **execution_fields(identity),
            }
            selection_path = output / "selected-checkpoint.json"
            if selection_path.exists():
                _regular_hash(selection_path)
                if read_json(selection_path) != selection:
                    raise ValueError("fusion partial selection differs from validated reports")
            else:
                atomic_json(selection_path, selection)
            result = {"status": "complete", "global_step": STEPS, "run_key": output.name,
                      "selected_step": best["checkpoint_step"], "arm": args.arm, "seed": args.seed,
                      "initialization_hash": ledger.initialization_hash,
                       "fusion_telemetry_hash": digest(ledger.telemetry),
                       **execution_fields(identity)}
            atomic_json(output / "run-status.json", result)
            return result
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)


def run_fusion(args) -> dict:
    return _execute(args)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("initialize", "preflight", "smoke", "run"),
                        required=True)
    for name in ("protocol", "dataset", "text-manifest", "dino-root", "motion-root",
                 "motion-manifest", "annotation", "adapted-root", "baseline-run", "output-base",
                 "motion-timing", "initialization-profile"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=FUSION_ARMS, required=True)
    parser.add_argument("--seed", choices=(0, 1), type=int, default=0)
    parser.add_argument("--steps", choices=(STEPS,), type=int, default=STEPS)
    parser.add_argument("--gpu-cap-seconds", type=int, default=MAX_CAP_SECONDS)
    parser.add_argument("--authorize-fusion-run", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--motion-package", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "initialize":
        result = initialize_fusion(args)
    elif args.mode == "preflight":
        identity, output, *_ = prepare_fusion(args)
        result = {"status": "ready", "identity": identity, "run_key": identity["run_key"],
                  "output": str(output)}
    else:
        result = _execute(args, smoke_only=args.mode == "smoke")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
