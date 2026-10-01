"""Explicitly capped train-only DINO LoRA adaptation for frozen E3 pilot."""

import argparse
import math
import os
import random
import shutil
import signal
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from tools.dinov3.e3_data import load_e3_batch, validate_e3_sources
from tools.dinov3.e3_model import (
    FACTORS,
    NUISANCE,
    DINOFactorHeads,
    attach_late_lora,
    relational_structure_loss,
    scale_reference_loss,
)
from tools.dinov3.identity import MODEL, digest, sha256_file
from tools.dinov3.storage import atomic_json, verify_location, writer

BASE_SHA = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
PROTOCOL_HASH = "0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7"
DATASET_KEY = "b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a"
POLICY = "qwen-schema98-unreviewed-v1"
BATCH_SIZES = (4, 8, 16)


def cosine_warmup_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to zero, as DIFFER uses (paper section 4).

    The original E3 adaptation ran at a constant LoRA LR with no decay, which applied
    roughly 42x more cumulative update pressure than DIFFER despite fewer epochs. That
    is the suspected cause of handshape erosion at 25 epochs.
    """
    if total_steps < 1 or warmup_steps < 0 or warmup_steps > total_steps or step < 0:
        raise ValueError("E3 LR schedule window invalid")
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def make_e3_optimizer(lora_params, head_params, *, lora_lr: float, head_lr: float,
                      weight_decay: float | None, warmup_steps: int, total_steps: int):
    """Build the adaptation optimizer and, when scheduled, its cosine LR scheduler.

    `total_steps == 0` selects the legacy constant-LR path so the completed
    no-decay runs stay exactly reproducible.
    """
    groups = [{"params": list(lora_params), "lr": lora_lr},
              {"params": list(head_params), "lr": head_lr}]
    extra = {} if weight_decay is None else {"weight_decay": weight_decay}
    optimizer = torch.optim.AdamW(groups, **extra)
    if total_steps < 1:
        return optimizer, None
    schedule = partial(cosine_warmup_factor, warmup_steps=warmup_steps,
                       total_steps=total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def e3_run_path(output_base: Path, identity: dict) -> Path:
    if not identity.get("run_key", "").startswith("e3-adapt-"):
        raise ValueError("E3 adaptation run identity invalid")
    return output_base / identity["run_key"]


def grl_alpha(step: int, calibration_steps: int, adapt_steps: int) -> float:
    if not 0 <= calibration_steps < adapt_steps or not 1 <= step <= adapt_steps:
        raise ValueError("E3 adaptation step schedule invalid")
    if step <= calibration_steps:
        return 0.0
    return min(1.0, (step - calibration_steps) /
               max(1, math.ceil(0.1 * (adapt_steps - calibration_steps))))


def step_e3(model, heads, batch: dict, *, step: int, calibration_steps: int,
            adapt_steps: int, relational_weight: float = 0.0) -> dict[str, torch.Tensor]:
    if not 1 <= step <= adapt_steps or relational_weight < 0:
        raise ValueError("E3 optimizer step or relational weight out of range")
    images224, images448 = batch["images224"], batch["images448"]
    if images224.shape[:2] != images448.shape[:2] or images224.shape[1] != 5:
        raise ValueError("E3 sampled frame batch mismatch")
    count = images224.shape[0]
    with torch.set_grad_enabled(step > calibration_steps):
        first = model(pixel_values=images224.reshape(-1, *images224.shape[2:]))
        second = model(pixel_values=images448.reshape(-1, *images448.shape[2:]))
        adapted = torch.cat((first.last_hidden_state[:, 0].float(),
                             second.last_hidden_state[:, 0].float()), dim=-1)
    teacher = batch["teacher"].reshape(-1, adapted.shape[-1])
    factors = heads(adapted.reshape(count, 5, -1), batch["vectors"],
                    batch["masks"], batch["labels"],
                    grl_alpha(step, calibration_steps, adapt_steps))
    reference = scale_reference_loss(adapted, teacher)
    relational = relational_structure_loss(adapted, teacher)
    combined = sum((0.05 if name in NUISANCE else 0.10) * factors[name]
                   for name in FACTORS)
    if step > calibration_steps:
        # Structure preservation only matters once DINO itself can move.
        combined = combined + 0.10 * reference
        if relational_weight:
            combined = combined + relational_weight * (relational["pattern"]
                                                       + relational["spread"])
    result = {"combined_loss": combined, "reference_loss": reference,
              "relational_pattern": relational["pattern"],
              "relational_spread": relational["spread"], **factors}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("E3 adaptation loss non-finite")
    return result


class E3BatchSampler:
    """Seeded clip order whose cursor/RNG are owned by each snapshot."""

    def __init__(self, ids: tuple[str, ...], *, seed: int, batch_size: int = 4):
        if (batch_size not in BATCH_SIZES or len(ids) < batch_size
                or len(set(ids)) != len(ids)):
            raise ValueError("E3 sampler requires unique train clips and batch 4, 8 or 16")
        self.ids = ids
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.order = list(ids)
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.batches_yielded = 0

    def next_ids(self) -> tuple[str, ...]:
        if self.cursor + self.batch_size > len(self.order):
            self.order = list(self.ids)
            self.rng.shuffle(self.order)
            self.cursor = 0
        selected = tuple(self.order[self.cursor:self.cursor + self.batch_size])
        self.cursor += self.batch_size
        self.batches_yielded += 1
        return selected

    def state_dict(self) -> dict:
        return {"order": list(self.order), "cursor": self.cursor,
                "batches_yielded": self.batches_yielded, "rng": self.rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        if (len(state.get("order", ())) != len(self.ids)
                or set(state["order"]) != set(self.ids)
                or not 0 <= state["cursor"] <= len(self.ids)
                or type(state["batches_yielded"]) is not int
                or state["batches_yielded"] < 0):
            raise ValueError("E3 sampler snapshot invalid")
        self.order = list(state["order"])
        self.cursor = state["cursor"]
        self.batches_yielded = state["batches_yielded"]
        self.rng.setstate(state["rng"])


def validate_run_args(args) -> None:
    if (not getattr(args, "authorize_e3_run", False)
            or type(args.gpu_cap_seconds) is not int or args.gpu_cap_seconds < 1):
        raise ValueError("E3 training requires explicit authorization and GPU budget")
    if (type(args.adapt_steps) is not int or type(args.calibration_steps) is not int
            or not 0 < args.calibration_steps < args.adapt_steps):
        raise ValueError("E3 calibration/adaptation steps invalid")
    if args.resume is None and args.output.exists():
        raise ValueError("E3 training output already exists")
    if args.resume is not None and not args.output.is_dir():
        raise ValueError("E3 resume output directory missing")


def require_e3_storage(parent: Path) -> None:
    if not parent.is_dir() or shutil.disk_usage(parent).free < 4 * 1024**3:
        raise ValueError("E3 adaptation snapshot storage below 4 GiB")


def resume_spend(budget: dict, *, checkpoint_seconds: float,
                 now: float, cap: float) -> float:
    recorded = budget.get("gpu_seconds")
    timestamp = budget.get("wall_timestamp")
    if (budget.get("status") not in ("paused", "running")
            or not isinstance(recorded, (int, float)) or not isinstance(timestamp, (int, float))
            or not isinstance(checkpoint_seconds, (int, float))
            or not all(math.isfinite(value) for value in (recorded, timestamp,
                                                          checkpoint_seconds, now, cap))
            or min(recorded, checkpoint_seconds, timestamp) < 0):
        raise ValueError("E3 resume budget/spend invalid")
    if now < timestamp:
        raise ValueError("E3 resume wall clock moved backwards")
    spent = max(recorded, checkpoint_seconds)
    if budget["status"] == "running":
        spent += now - timestamp  # conservative accounting for unclean stop
    if spent >= cap:
        raise ValueError("E3 resume GPU cap exhausted")
    return spent


def _budget(identity: dict, status: str, spent: float, step: int) -> dict:
    return {"identity": identity, "status": status, "gpu_seconds": spent,
            "wall_timestamp": time.time(), "pid": os.getpid(), "optimizer_step": step}


def validate_e3_resume(path: Path, identity: dict, spent: float) -> dict:
    if (path.parent != path.resolve().parent or path.parent.name != identity["run_key"]
            or not path.name.startswith("resume-") or path.suffix != ".pt"):
        raise ValueError("E3 resume identity/path mismatch")
    snapshots = sorted(path.parent.glob("resume-*.pt"),
                       key=lambda item: int(item.stem.split("-")[1]))
    if not snapshots or snapshots[-1] != path:
        raise ValueError("E3 resume must use latest snapshot")
    receipt = path.with_suffix(".json")
    if (not receipt.is_file() or
            __import__("json").loads(receipt.read_text()).get("sha256") != sha256_file(path)):
        raise ValueError("E3 resume checkpoint receipt/hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)  # owned run snapshot
    if (checkpoint.get("identity") != identity
            or checkpoint.get("step") != int(path.stem.split("-")[1])
            or checkpoint.get("sampler", {}).get("batches_yielded") != checkpoint.get("step")
            or any(key not in checkpoint for key in ("optimizer", "model", "heads", "rng"))):
        raise ValueError("E3 resume identity/step/state mismatch")
    seconds = checkpoint.get("gpu_seconds")
    if not isinstance(seconds, (int, float)) or not 0 <= seconds <= spent:
        raise ValueError("E3 resume spend would rewind or reset")
    return checkpoint


def _lora_state(model) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if name.endswith(("down.weight", "up.weight")) and parameter.requires_grad}


def _load_lora_state(model, state: dict) -> None:
    expected = {name: parameter for name, parameter in model.named_parameters()
                if name.endswith(("down.weight", "up.weight")) and parameter.requires_grad}
    if set(state) != set(expected) or len(expected) != 16:
        raise ValueError("E3 checkpoint Q/V LoRA state mismatch")
    with torch.no_grad():
        for name, parameter in expected.items():
            if state[name].shape != parameter.shape or not torch.isfinite(state[name]).all():
                raise ValueError(f"E3 LoRA tensor invalid: {name}")
            parameter.copy_(state[name].to(parameter.device))


def _snapshot(output: Path, step: int, identity: dict, spent: float, model,
              heads, optimizer, sampler: E3BatchSampler, scheduler=None) -> None:
    target = output / f"resume-{step}.pt"
    if target.exists() or target.with_suffix(".json").exists():
        raise ValueError("E3 snapshot already exists")
    pending = output / f".resume-{step}.{os.getpid()}.pending"
    if pending.exists():
        raise ValueError("E3 pending snapshot already exists")
    state = {"identity": identity, "step": step, "gpu_seconds": spent,
             "model": _lora_state(model), "heads": heads.state_dict(),
             "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
             "scheduler": scheduler.state_dict() if scheduler is not None else None,
             "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                     "python": random.getstate(), "numpy": np.random.get_state()}}
    try:
        torch.save(state, pending)
        os.replace(pending, target)
        atomic_json(target.with_suffix(".json"), {"sha256": sha256_file(target), "step": step})
    finally:
        pending.unlink(missing_ok=True)


def _inputs(args) -> tuple[dict, dict, dict, dict, tuple[str, ...]]:
    if args.protocol.parent.name != PROTOCOL_HASH or args.dataset.name != DATASET_KEY:
        raise ValueError("E3 frozen dataset/protocol identity mismatch")
    protocol = __import__("json").loads(args.protocol.read_text())
    source = __import__("json").loads((args.dataset / "source.json").read_text())
    text = __import__("json").loads(args.text_manifest.read_text())
    rows_path = args.dino_root / "complete/frame_rows.json"
    if (protocol.get("protocol_hash") != PROTOCOL_HASH
            or sha256_file(args.text_manifest) != protocol.get("text_manifest_hash")
            or sha256_file(rows_path) != protocol.get("frame_rows_hash")
            or sha256_file(args.dino_root / "complete/manifest.json")
            != protocol.get("spatial_manifest_hash")):
        raise ValueError("E3 frozen source manifest mismatch")
    ids = validate_e3_sources(protocol, source, text)
    if len(ids) != 5746 or len(source["clips"]) != 7096:
        raise ValueError("E3 frozen train/physical inventory mismatch")
    rows = __import__("json").loads(rows_path.read_text())["clips"]
    return protocol, source, text, rows, ids


def execute(args) -> dict:
    protocol, source, text, rows, ids = _inputs(args)
    root = Path(__file__).resolve().parent
    identity = {"run_policy": "signer-pilot-e3-dino-lora-v1",
                "protocol_hash": PROTOCOL_HASH, "split_hash": protocol["split_hash"],
                "source_hash": protocol["source_hash"],
                "text_manifest_hash": protocol["text_manifest_hash"],
                "frame_rows_hash": protocol["frame_rows_hash"],
                "base_revision": BASE_SHA, "seed": 0, "batch_size": args.batch_size,
                "relational_weight": args.relational_weight,
                "lora_lr": args.lora_lr, "head_lr": args.head_lr,
                "weight_decay": args.weight_decay,
                "cosine_decay": bool(args.cosine_decay),
                "lr_warmup_steps": args.lr_warmup_steps,
                "relational_terms": ["pattern", "spread"],
                "adapt_steps": args.adapt_steps,
                "calibration_steps": args.calibration_steps,
                "gpu_cap_seconds": args.gpu_cap_seconds,
                "supervision_policy": POLICY, "human_review_status": "not_assessed",
                "code_hashes": {name: sha256_file(root / name) for name in
                                ("e3_train.py", "e3_model.py", "e3_data.py", "frames.py")},
                "lock_hash": sha256_file(root / "uv.lock")}
    identity["run_key"] = "e3-adapt-" + digest(identity)
    args.output = e3_run_path(args.output_base, identity)
    verify_location(args.output, root.parents[1])
    if args.mode == "preflight":
        return {"status": "ready", "run_key": identity["run_key"], "train": len(ids)}
    validate_run_args(args)
    require_e3_storage(args.output.parent)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("E3 adaptation needs at least 12 GiB free CUDA memory")
    with writer(args.output):
        spent = 0.0
        checkpoint = None
        if args.resume is not None:
            budget = __import__("json").loads((args.output / "budget.json").read_text())
            if (budget.get("identity") != identity
                    or budget.get("status") not in ("paused", "running")):
                raise ValueError("E3 resume budget identity/status mismatch")
            if (budget["status"] == "running" and isinstance(budget.get("pid"), int)
                    and Path(f"/proc/{budget['pid']}").exists()):
                raise ValueError("E3 previous adaptation process still exists")
            candidate = torch.load(args.resume, map_location="cpu", weights_only=False)
            checkpoint = validate_e3_resume(
                args.resume, identity,
                max(budget["gpu_seconds"], candidate.get("gpu_seconds", 0)),
            )
            spent = resume_spend(budget, checkpoint_seconds=checkpoint["gpu_seconds"],
                                 now=time.time(), cap=args.gpu_cap_seconds)
        else:
            atomic_json(args.output / "budget.json", _budget(identity, "running", 0.0, 0))
        started = time.monotonic()
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        model = AutoModel.from_pretrained(
            MODEL, revision=BASE_SHA, cache_dir=args.cache, local_files_only=True,
            trust_remote_code=False, use_safetensors=True,
        ).eval().to("cuda")
        if (model.config.hidden_size != 1024 or model.config.patch_size != 16
                or model.config.num_register_tokens != 4):
            raise ValueError("E3 pinned DINOv3 model mismatch")
        attach_late_lora(model)
        heads = DINOFactorHeads().to("cuda")
        lora = [p for p in model.parameters() if p.requires_grad]
        optimizer, scheduler = make_e3_optimizer(
            lora, heads.parameters(),
            lora_lr=args.lora_lr, head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.lr_warmup_steps,
            total_steps=args.adapt_steps if args.cosine_decay else 0,
        )
        sampler = E3BatchSampler(ids, seed=0, batch_size=args.batch_size)
        initial = 0
        if checkpoint is not None:
            _load_lora_state(model, checkpoint["model"])
            heads.load_state_dict(checkpoint["heads"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            sampler.load_state_dict(checkpoint["sampler"])
            torch.set_rng_state(checkpoint["rng"]["torch"])
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
            random.setstate(checkpoint["rng"]["python"])
            np.random.set_state(checkpoint["rng"]["numpy"])
            initial = checkpoint["step"]
            del checkpoint
        if initial >= args.adapt_steps:
            raise ValueError("E3 adaptation already completed")
        try:
            for step in range(initial + 1, args.adapt_steps + 1):
                elapsed = spent + time.monotonic() - started
                if elapsed >= args.gpu_cap_seconds - 30:
                    raise TimeoutError("E3 adaptation GPU cap near; stop before checkpoint")
                clip_ids = sampler.next_ids()
                batch = load_e3_batch(clip_ids, source, text, rows, frames=args.frames,
                                      dino_root=args.dino_root, text_manifest=args.text_manifest)
                batch = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value
                         for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = step_e3(model, heads, batch, step=step,
                                     calibration_steps=args.calibration_steps,
                                     adapt_steps=args.adapt_steps,
                                     relational_weight=args.relational_weight)
                losses["combined_loss"].backward()
                torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
                if step > args.calibration_steps:
                    torch.nn.utils.clip_grad_norm_(lora, 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if step % 500 == 0 or step == args.adapt_steps:
                    torch.cuda.synchronize()
                    elapsed = spent + time.monotonic() - started
                    if elapsed >= args.gpu_cap_seconds:
                        raise TimeoutError("E3 adaptation GPU cap exceeded")
                    _snapshot(
                        args.output, step, identity, elapsed, model, heads, optimizer,
                        sampler, scheduler
                    )
                    atomic_json(args.output / "budget.json",
                                _budget(identity, "running", elapsed, step))
            final = args.output / f"resume-{args.adapt_steps}.pt"
            result = {"status": "complete", "checkpoint_hash": sha256_file(final),
                      "checkpoint": final.name, "step": args.adapt_steps,
                      "gpu_seconds": spent + time.monotonic() - started, "identity": identity}
            atomic_json(args.output / "selected-checkpoint.json", result)
            atomic_json(args.output / "budget.json",
                        _budget(identity, "closed", result["gpu_seconds"], args.adapt_steps))
            return result
        except BaseException as error:
            torch.cuda.synchronize()
            atomic_json(args.output / "budget.json",
                        _budget(identity, "paused" if isinstance(error, KeyboardInterrupt)
                                else "running", spent + time.monotonic() - started,
                                sampler.batches_yielded))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "run"))
    for name in ("protocol", "dataset", "text-manifest", "dino-root", "frames",
                 "cache", "output-base"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--adapt-steps", required=True, type=int)
    parser.add_argument("--calibration-steps", required=True, type=int)
    parser.add_argument("--gpu-cap-seconds", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=4, choices=BATCH_SIZES)
    parser.add_argument("--relational-weight", type=float, default=0.0)
    # DIFFER-faithful schedule: paper uses SGD 2e-6 with cosine decay over 60 epochs.
    # Defaults below preserve the original constant-LR behaviour exactly.
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--cosine-decay", action="store_true")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--authorize-e3-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.mode == "run":
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("E3 cap")))
        signal.alarm(args.gpu_cap_seconds)
    try:
        print(__import__("json").dumps(execute(args), sort_keys=True))
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
