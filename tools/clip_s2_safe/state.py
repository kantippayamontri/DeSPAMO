"""Original tensor identity plus immutable new-runtime and CE-guard sidecars."""

import argparse
import json
import math
from pathlib import Path

import torch

from tools.benchmark.compact_checkpoint import load_snapshot, restore_rng, save_snapshot
from tools.clip_plain.generation import guard_device
from tools.clip_s2_safe.policy import evaluation_policy, load_config, source_hashes, verify_original
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_archive import _copy_file, require_archive_storage
from tools.dinov3.storage import _atomic, atomic_json, writer


def make_binding(config: dict, original: dict) -> dict:
    return dict(
        schema_version=1,
        config=json.loads(json.dumps(config)),
        runtime_policy="clip-s2-safe-resume-v1",
        original_identity=original["identity"],
        original_identity_hash=digest(original["identity"]),
        anchor=config["anchor"],
        anchor_sha256=config["anchor_sha256"],
        anchor_receipt_sha256=config["anchor_receipt_sha256"],
        anchor_step=config["anchor_step"],
        initial_budget_sha256=config["initial_budget_sha256"],
        initial_hold_sha256=config["initial_hold_sha256"],
        initial_charged_seconds=config["initial_charged_seconds"],
        sources=source_hashes(),
        config_hash=digest(config),
        evaluation_policy=evaluation_policy(),
        evaluation_policy_hash=digest(evaluation_policy()),
        run_root=config["run_root"],
        runtime_root=config["runtime_root"],
        archive_root=config["archive_root"],
        spatial_width=2048,
        recipe=original["identity"]["recipe"],
    )


def verify_binding(binding: dict) -> None:
    if (
        binding.get("schema_version") != 1
        or binding.get("runtime_policy") != "clip-s2-safe-resume-v1"
        or binding.get("spatial_width") != 2048
        or binding["original_identity"]["seed"] != 0
        or binding["original_identity"]["policy"] != "official-frozen-clip-s2-reference-spamo200-v1"
    ):
        raise ValueError("S² original/runtime identity mismatch")
    if (
        binding["sources"] != source_hashes()
        or binding["original_identity_hash"] != digest(binding["original_identity"])
        or binding["evaluation_policy"] != evaluation_policy()
        or binding["evaluation_policy_hash"] != digest(evaluation_policy())
    ):
        raise ValueError("Runtime source/evaluation binding drift")


def publish_migration(config: dict, binding: dict) -> dict:
    verify_binding(binding)
    if binding["config_hash"] != digest(config):
        raise ValueError("Migration config changed")
    root = Path(config["runtime_root"])
    with writer(root / "migration"):
        path = root / "migration.json"
        if path.exists():
            if json.loads(path.read_text()) != binding:
                raise ValueError("Immutable migration differs")
            return binding
        original = verify_original(config)
        if make_binding(config, original) != binding:
            raise ValueError("Original migration evidence changed")
        budget = Path(config["run_root"]) / "budget.json"
        _atomic(root / "original-budget.json", lambda handle: handle.write(budget.read_bytes()))
        if sha256_file(root / "original-budget.json") != config["initial_budget_sha256"]:
            raise ValueError("Initial budget copy differs")
        atomic_json(path, binding)
        return binding


def read_binding(config: dict) -> dict:
    b = json.loads((Path(config["runtime_root"]) / "migration.json").read_text())
    verify_binding(b)
    if b["config_hash"] != digest(config):
        raise ValueError("Runtime config drift")
    return b


def audit_native_schema(config: dict) -> dict:
    """Read trusted native training schema on CPU, without model/device/RNG restore."""
    original = verify_original(config)
    state = torch.load(config["anchor"], map_location="cpu", weights_only=False)
    if (
        state["identity"] != original["identity"]
        or state["step"] != config["anchor_step"]
        or state["sampler"]["batches_yielded"] != state["step"]
    ):
        raise ValueError("Anchor internal identity/sampler schema changed")
    if not {"weights", "optimizer", "sampler", "rng", "prompt_rng"}.issubset(state):
        raise ValueError("Full anchor state missing")
    for value in state["weights"].values():
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid CPU mutable tensor")
    for value in state["rng"]["cuda"]:
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.uint8
            or value.ndim != 1
        ):
            raise ValueError("Invalid CUDA RNG byte schema")
    groups = state["optimizer"]["param_groups"]
    if len({p for group in groups for p in group["params"]}) != sum(
        len(group["params"]) for group in groups
    ):
        raise ValueError("Duplicate optimizer parameter IDs")
    return dict(
        step=state["step"],
        weights_count=len(state["weights"]),
        optimizer_state_count=len(state["optimizer"]["state"]),
        optimizer_group_names=[g.get("name") for g in groups],
        sampler_batches_yielded=state["sampler"]["batches_yielded"],
        sampler_cursor=state["sampler"]["cursor"],
        cuda_rng_byte_lengths=[len(x) for x in state["rng"]["cuda"]],
        cuda_rng_restored=False,
    )


def validate_guard_state(state: dict, *, checkpoint_sha256: str, binding: dict, step: int) -> dict:
    values = state.get("ce_values", [])
    expected = min(200, max(0, step - binding["recipe"]["vt_warmup_steps"]))
    if (
        state.get("schema_version") != 1
        or state.get("checkpoint_sha256") != checkpoint_sha256
        or state.get("binding_hash") != digest(binding)
        or state.get("last_update") != step
        or len(values) != expected
        or any(type(v) not in [int, float] or not math.isfinite(v) for v in values)
    ):
        raise ValueError("CE guard sidecar checkpoint/history mismatch")
    return state


def checkpoint_bundle(path: Path, binding: dict) -> dict:
    verify_binding(binding)
    if path.is_symlink():
        raise ValueError("Native regular snapshot required")
    receipt = json.loads(path.with_suffix(".json").read_text())
    if (
        receipt["identity"] != binding["original_identity"]
        or sha256_file(path) != receipt["sha256"]
        or path.stat().st_size != receipt["bytes"]
    ):
        raise ValueError("Original snapshot receipt drift")
    if path == Path(binding["anchor"]):
        if (
            receipt["sha256"] != binding["anchor_sha256"]
            or sha256_file(path.with_suffix(".json")) != binding["anchor_receipt_sha256"]
            or receipt["step"] != binding["anchor_step"]
        ):
            raise ValueError("Allowlisted anchor changed")
        guard = dict(
            schema_version=1,
            checkpoint_sha256=receipt["sha256"],
            binding_hash=digest(binding),
            last_update=receipt["step"],
            ce_values=[],
        )
    else:
        root = Path(binding["runtime_root"]) / "checkpoints"
        if root not in path.parents:
            raise ValueError("Foreign runtime snapshot")
        side = path.with_suffix(".runtime.json")
        gp = path.with_suffix(".guard.json")
        if not side.is_file() or not gp.is_file() or side.is_symlink() or gp.is_symlink():
            raise ValueError("Complete runtime/guard sidecars required")
        runtime = json.loads(side.read_text())
        guard = json.loads(gp.read_text())
        if runtime.get("mode") != path.parent.name or runtime.get("mode") not in [
            "production",
            "handoff_probe",
        ]:
            raise ValueError("Snapshot runtime mode/directory mismatch")
        adoption = runtime.get("adoption_approval_sha256")
        if adoption is not None:
            certificate = Path(binding["runtime_root"]) / "adoptions" / f"{receipt['sha256']}.json"
            if not certificate.is_file():
                raise ValueError("Adoption has not committed to authoritative journal")
            entry = json.loads(certificate.read_text())
            if (
                entry.get("binding_hash") != digest(binding)
                or entry.get("checkpoint_sha256") != receipt["sha256"]
                or entry.get("approval_sha256") != adoption
            ):
                raise ValueError("Adoption certificate mismatch")
        elif (
            runtime["mode"] == "production"
            and receipt["step"] % binding["recipe"]["steps_per_epoch"]
        ):
            raise ValueError("Partial production step needs explicit adoption proof")
        if (
            runtime["binding_hash"] != digest(binding)
            or runtime["checkpoint_sha256"] != receipt["sha256"]
            or runtime["guard_sha256"] != sha256_file(gp)
            or runtime["step"] != receipt["step"]
        ):
            raise ValueError("Runtime sidecar mismatch")
    validate_guard_state(
        guard, checkpoint_sha256=receipt["sha256"], binding=binding, step=receipt["step"]
    )
    return dict(receipt=receipt, guard=guard)


def load_weights(path: Path, model, binding: dict, *, base_hash: str) -> dict:
    bundle = checkpoint_bundle(path, binding)
    state = load_snapshot(
        path, model, None, binding["original_identity"], base_hash=base_hash, replay=False
    )
    state["safe_guard"] = bundle["guard"]
    return state


def restore_training(
    state: dict, model, optimizer, sampler, *, device: torch.device, scope: dict | None = None
) -> None:
    if not {"optimizer", "sampler", "rng", "prompt_rng"}.issubset(state):
        raise ValueError("Full training snapshot required")
    if state["step"] != state["sampler"]["batches_yielded"]:
        raise ValueError("Sampler/update mismatch")
    if device.type == "cpu" and state["rng"]["cuda"]:
        raise RuntimeError("Native CUDA RNG cannot be restored on CPU; validate bytes only")
    saved = state["optimizer"]
    if len(saved["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Optimizer group count mismatch")
    for old, new in zip(saved["param_groups"], optimizer.param_groups, strict=True):
        if old.get("name") != new.get("name") or len(old["params"]) != len(new["params"]):
            raise ValueError("Optimizer parameter group mismatch")
        for key, param in zip(old["params"], new["params"], strict=True):
            slot = saved["state"].get(key)
            if slot is None:
                continue  # LoRA states may be absent during VT-only warmup.
            for field in ["exp_avg", "exp_avg_sq"]:
                value = slot.get(field)
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or value.shape != param.shape
                    or value.dtype != param.dtype
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError("Optimizer moment tensor mismatch")
                if field == "exp_avg_sq" and (value < 0).any():
                    raise ValueError("Negative AdamW second moment")
            step = slot.get("step")
            if isinstance(step, torch.Tensor):
                if step.device.type != "cpu" or step.numel() != 1:
                    raise ValueError("Optimizer step schema mismatch")
                step = float(step)
            if (
                not isinstance(step, (int, float))
                or not math.isfinite(step)
                or not 0 <= step <= state["step"]
            ):
                raise ValueError("Invalid optimizer step count")
    from tools.dinov3.official_sampler import OfficialBatchSampler

    validated_sampler = OfficialBatchSampler(sampler.ids, seed=0, batch_size=sampler.batch_size)
    validated_sampler.load_state_dict(state["sampler"])
    guard_device(device, scope)
    optimizer.load_state_dict(state["optimizer"])
    sampler.load_state_dict(state["sampler"])
    restore_rng(state["rng"])
    model.rng.setstate(state["prompt_rng"])


def _sidecars(path: Path, receipt: dict, guard: dict, binding: dict, mode: str) -> None:
    g = dict(
        schema_version=1,
        checkpoint_sha256=receipt["sha256"],
        binding_hash=digest(binding),
        last_update=receipt["step"],
        ce_values=list(guard["ce_values"]),
    )
    validate_guard_state(
        g, checkpoint_sha256=receipt["sha256"], binding=binding, step=receipt["step"]
    )
    atomic_json(path.with_suffix(".guard.json"), g)
    atomic_json(
        path.with_suffix(".runtime.json"),
        dict(
            schema_version=1,
            binding_hash=digest(binding),
            checkpoint_sha256=receipt["sha256"],
            guard_sha256=sha256_file(path.with_suffix(".guard.json")),
            step=receipt["step"],
            mode=mode,
        ),
    )


def publish_checkpoint(config: dict, snapshot: dict, guard: dict, *, mode: str) -> dict:
    b = read_binding(config)
    if (
        mode not in ["production", "handoff_probe"]
        or snapshot["identity"] != b["original_identity"]
        or snapshot["step"] <= b["anchor_step"]
        or snapshot["step"] != snapshot["sampler"]["batches_yielded"]
        or guard["last_update"] != snapshot["step"]
    ):
        raise ValueError("Owned snapshot identity/mode/update mismatch")
    root = Path(config["runtime_root"]) / "checkpoints" / mode
    path = root / f"resume-step-{snapshot['step']}.pt"
    # Validate history before writing any tensor bytes; crash-partial bundles remain blocked.
    validate_guard_state(
        dict(
            schema_version=1,
            checkpoint_sha256="pending",
            binding_hash=digest(b),
            last_update=guard["last_update"],
            ce_values=guard["ce_values"],
        ),
        checkpoint_sha256="pending",
        binding=b,
        step=snapshot["step"],
    )
    receipt = save_snapshot(path, snapshot)
    _sidecars(path, receipt, guard, b, mode)
    result = dict(checkpoint=str(path), checkpoint_sha256=receipt["sha256"], step=snapshot["step"])
    epoch = snapshot["step"] // b["recipe"]["steps_per_epoch"]
    if (
        mode == "handoff_probe"
        or snapshot["step"] % b["recipe"]["steps_per_epoch"] == 0
        and epoch in b["recipe"]["dev_epochs"]
    ):
        candidate = root / f"candidate-step-{snapshot['step']}.pt"
        slim = {
            k: snapshot[k]
            for k in [
                "schema_version",
                "identity",
                "step",
                "weights",
                "frozen_fingerprint",
                "prompt_rng",
            ]
        }
        row = save_snapshot(candidate, slim)
        _sidecars(candidate, row, guard, b, mode)
        result.update(candidate=str(candidate), candidate_sha256=row["sha256"])
    return result


def make_anchor_candidate(config: dict, binding: dict) -> Path:
    root = Path(config["runtime_root"]) / "checkpoints/production"
    path = root / f"candidate-step-{binding['anchor_step']}.pt"
    if path.exists():
        checkpoint_bundle(path, binding)
        return path
    import torch

    state = torch.load(binding["anchor"], map_location="cpu", weights_only=False)
    if state["identity"] != binding["original_identity"]:
        raise ValueError("Anchor internal identity changed")
    slim = {
        k: state[k]
        for k in [
            "schema_version",
            "identity",
            "step",
            "weights",
            "frozen_fingerprint",
            "prompt_rng",
        ]
    }
    receipt = save_snapshot(path, slim)
    _sidecars(path, receipt, dict(ce_values=[]), binding, "production")
    return path


def archive_candidate(config: dict, candidate: Path, binding: dict) -> dict:
    checkpoint_bundle(candidate, binding)
    root = Path(config["runtime_root"]) / "checkpoints"
    relative = candidate.relative_to(root)
    target = Path(config["archive_root"]) / "candidates" / relative
    require_archive_storage(Path(config["archive_root"]), candidate.stat().st_size + 16384)
    for suffix in [".pt", ".json", ".guard.json", ".runtime.json"]:
        source = candidate.with_suffix(suffix)
        dest = target.with_suffix(suffix)
        if dest.exists():
            if dest.is_symlink() or sha256_file(dest) != sha256_file(source):
                raise ValueError("Existing archive bytes differ")
        else:
            _copy_file(source, dest)
    return dict(path=str(target), sha256=sha256_file(target))


def prune_owned_resumes(config: dict, binding: dict, *, allowed: bool) -> list[str]:
    if not allowed:
        return []
    root = Path(config["runtime_root"]) / "checkpoints/production"
    paths = sorted(root.glob("resume-step-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    for path in paths:
        checkpoint_bundle(path, binding)
    for candidate in root.glob("candidate-step-*.pt"):
        archive_candidate(config, candidate, binding)
    removed = []
    for path in paths[:-2]:
        for suffix in [".pt", ".json", ".guard.json", ".runtime.json"]:
            path.with_suffix(suffix).unlink()
        removed.append(path.name)
    return removed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--authorize-preparation", action="store_true")
    a = p.parse_args()
    c = load_config(a.config)
    original = verify_original(c)
    b = make_binding(c, original)
    if a.prepare:
        if not a.authorize_preparation:
            raise RuntimeError("Separate CPU preparation approval required")
        publish_migration(c, b)
    print(
        json.dumps(dict(binding_hash=digest(b), prepared=a.prepare, anchor_step=b["anchor_step"]))
    )


if __name__ == "__main__":
    main()
