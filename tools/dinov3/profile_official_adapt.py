"""Disposable real joint-training profiles; never publish production checkpoints."""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from transformers import AutoModel, DINOv3ViTModel

from tools.dinov3.e3_model import DINOFactorHeads, attach_late_lora
from tools.dinov3.e3_train import BASE_SHA, make_e3_optimizer, step_e3
from tools.dinov3.identity import MODEL, digest, sha256_file
from tools.dinov3.official_data import admit_official_data, load_official_batch
from tools.dinov3.storage import atomic_json, writer


def profile_ids(admission: dict, batch_size: int) -> tuple[str, ...]:
    if batch_size not in (16, 24, 32) or len(admission["train_ids"]) < batch_size:
        raise ValueError("only requested profiles 16/24/32 with sufficient clips are allowed")
    groups = {}
    for clip_id in admission["train_ids"]:
        groups.setdefault(admission["clips"][clip_id]["signer"], []).append(clip_id)
    ids = []
    for offset in range(batch_size):
        for signer in sorted(groups):
            if offset < len(groups[signer]):
                ids.append(groups[signer][offset])
                if len(ids) == batch_size:
                    return tuple(ids)
    raise ValueError("profile batch cannot be assembled")


def optimizer_state_report(optimizer) -> dict:
    expected = [p for group in optimizer.param_groups for p in group["params"]]
    initialized = [p for p in expected if {"exp_avg", "exp_avg_sq", "step"}
                   <= optimizer.state.get(p, {}).keys()]
    size = sum(v.numel() * v.element_size() for state in optimizer.state.values()
               for v in state.values() if isinstance(v, torch.Tensor))
    return dict(expected_parameters=len(expected), initialized_parameters=len(initialized),
                tensor_bytes=size)


def joint_update(model, heads, optimizer, batch: dict) -> dict:
    optimizer.zero_grad(set_to_none=True)
    # Late-joint helper index makes GRL=1. Optimizer uses conservative peak LRs,
    # not the zero-LR endpoint. This is a memory probe, not an epoch simulation.
    with torch.autocast(device_type=batch["images224"].device.type, dtype=torch.bfloat16):
        losses = step_e3(model, heads, batch, step=17720, calibration_steps=2215,
                         adapt_steps=17720, relational_weight=.10)
    losses["combined_loss"].backward()
    lora = [p for p in model.parameters() if p.requires_grad]
    head_params = list(heads.parameters())
    if any(p.grad is not None and not torch.isfinite(p.grad).all()
           for p in lora + head_params):
        raise FloatingPointError("non-finite joint profile gradient")
    lora_norm = torch.nn.utils.clip_grad_norm_(lora, 1.0)
    head_norm = torch.nn.utils.clip_grad_norm_(head_params, 1.0)
    if not torch.isfinite(lora_norm) or not torch.isfinite(head_norm):
        raise FloatingPointError("non-finite joint profile gradient norm")
    optimizer.step()
    return {**{k: float(v.detach()) for k, v in losses.items()},
            "lora_grad_norm": float(lora_norm), "head_grad_norm": float(head_norm)}


def _other_gpu_pids() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=10)
    return [int(line.strip()) for line in output.splitlines()
            if line.strip().isdigit() and int(line.strip()) != os.getpid()]


def profile_identity(batch_size: int, data_identity: dict, ids: tuple[str, ...]) -> dict:
    return dict(policy="official-dino-joint-profile-v1", batch_size=batch_size,
                data_identity=data_identity, clip_ids=list(ids), model=MODEL, revision=BASE_SHA,
                lora_rank=8, lora_alpha=8, lora_blocks=[20, 21, 22, 23], seed=0,
                precision="bf16", warmup_updates=2,
                measured_updates=5 if batch_size == 24 else 3, grl=1.,
                lora_lr=1e-5, head_lr=3e-5, relational_weight=.1, weight_decay=.05,
                code_hashes={n: sha256_file(Path(__file__).with_name(n)) for n in
                             ("profile_official_adapt.py", "official_data.py", "e3_train.py",
                              "e3_model.py", "frames.py")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(16, 24, 32), required=True)
    for name in ("train-annotation", "dev-annotation", "test-annotation", "source",
                 "merged-manifest", "teacher-root", "frames-root", "cache", "output-base"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorize-profile", action="store_true")
    args = parser.parse_args(argv)
    if not args.authorize_profile:
        raise ValueError("explicit profile authorization required")
    if _other_gpu_pids():
        raise RuntimeError("another GPU compute process owns the device")
    a = admit_official_data(args.train_annotation, args.dev_annotation, args.test_annotation,
                           args.source, args.merged_manifest, args.teacher_root, args.frames_root)
    ids = profile_ids(a, args.batch_size)
    batch = load_official_batch(ids, a)
    identity = profile_identity(args.batch_size, a["identity"], ids)
    root = args.output_base / f"batch{args.batch_size}-{digest(identity)}"
    with writer(root):
        if (root / "profile.json").exists():
            raise ValueError("completed profile exists; preserve measured artifact")
        if _other_gpu_pids():
            raise RuntimeError("GPU became owned during CPU admission")
        atomic_json(root / "identity.json", identity)
        atomic_json(root / "owner.json", dict(pid=os.getpid(),
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(
                        ") ", 1)[1].split()[19]))
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        report: dict = dict(identity=identity, status="running")
        try:
            model = cast(DINOv3ViTModel, AutoModel.from_pretrained(
                MODEL, revision=BASE_SHA, cache_dir=str(args.cache), local_files_only=True,
                trust_remote_code=False, use_safetensors=True)).eval()
            torch.nn.Module.to(model, device=torch.device("cuda"))
            attach_late_lora(model)
            heads = DINOFactorHeads().to("cuda")
            lora = [p for p in model.parameters() if p.requires_grad]
            optimizer, _ = make_e3_optimizer(lora, heads.parameters(), lora_lr=1e-5,
                                             head_lr=3e-5, weight_decay=.05,
                                             warmup_steps=0, total_steps=0)
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            losses: dict = {}
            for _ in range(2):
                losses = joint_update(model, heads, optimizer, batch)
            state = optimizer_state_report(optimizer)
            if state["initialized_parameters"] != state["expected_parameters"]:
                raise RuntimeError("profile optimizer states not fully initialized")
            warm_peak = dict(allocated=torch.cuda.max_memory_allocated(),
                             reserved=torch.cuda.max_memory_reserved())
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(identity["measured_updates"]):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                losses = joint_update(model, heads, optimizer, batch)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - begin)
            free, total = torch.cuda.mem_get_info()
            report.update(status="complete", optimizer_state=state, warmup_peak=warm_peak,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          step_seconds=durations, mean_step_seconds=statistics.mean(durations),
                          clips_per_second=args.batch_size/statistics.mean(durations),
                          gpu_free_bytes=free, gpu_total_bytes=total, losses=losses,
                          trainable_parameters=sum(p.numel() for p in lora)+sum(
                              p.numel() for p in heads.parameters()),
                          gradient_checkpointing=bool(model.is_gradient_checkpointing),
                          attention_implementation=model.config._attn_implementation,
                          torch_version=torch.__version__,
                          gpu_name=torch.cuda.get_device_name())
        except torch.cuda.OutOfMemoryError as error:
            report.update(status="oom", error=str(error),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        except BaseException as error:
            report.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            report["elapsed_seconds"] = time.perf_counter() - started
            report["content_hash"] = digest(report)
            atomic_json(root / "profile.json", report)
        print(json.dumps(dict(status=report["status"], report=str(root / "profile.json"))))


if __name__ == "__main__":
    main()
