"""Owned official-split 40-epoch DINO adaptation; explicit smoke/run authorization."""

import argparse
import json
import math
import random
import signal
import subprocess
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np
import torch
from transformers import AutoModel, DINOv3ViTModel

from tools.dinov3.e3_model import (
    FACTORS,
    NUISANCE,
    DINOFactorHeads,
    attach_late_lora,
    relational_structure_loss,
    scale_reference_loss,
)
from tools.dinov3.e3_train import BASE_SHA, _load_lora_state, _lora_state
from tools.dinov3.identity import MODEL, digest, sha256_file
from tools.dinov3.official_checkpoints import (
    POLICY,
    _receipt,
    prune_owned_resumes,
    publish_milestone,
    validate_official_resume,
    validate_state,
)
from tools.dinov3.official_data import admit_official_data, load_official_batch
from tools.dinov3.official_runtime import (
    OfficialLedger,
    require_official_storage,
)
from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.dinov3.official_schedule import AdaptSchedule, grl_factor, head_factor, lora_factor
from tools.dinov3.profile_official_adapt import _other_gpu_pids, optimizer_state_report, profile_ids
from tools.dinov3.storage import _atomic, atomic_json, writer


def capture_rng() -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else [])


def restore_rng(rng: dict) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])


def make_optimizer(model, heads):
    return torch.optim.AdamW([
        dict(name="lora", params=[p for p in model.parameters() if p.requires_grad], lr=1e-5),
        dict(name="heads", params=heads.parameters(), lr=3e-5)],
        betas=(.9, .999), eps=1e-8, weight_decay=.05)


def compute_official_losses(model, heads, batch: dict, k: int,
                            schedule: AdaptSchedule, relational_weight: float = .10) -> dict:
    alpha = grl_factor(k, schedule)
    count = batch["images224"].shape[0]
    with torch.set_grad_enabled(k > schedule.calibration_steps):
        first = model(pixel_values=batch["images224"].flatten(0, 1))
        second = model(pixel_values=batch["images448"].flatten(0, 1))
        adapted = torch.cat((first.last_hidden_state[:, 0].float(),
                             second.last_hidden_state[:, 0].float()), -1)
    with torch.autocast(batch["images224"].device.type, enabled=False):
        teacher = batch["teacher"].float().reshape_as(adapted)
        factors = heads(adapted.reshape(count, 5, -1), batch["vectors"].float(), batch["masks"],
                        batch["labels"], alpha)
        reference = scale_reference_loss(adapted, teacher)
        relational = relational_structure_loss(adapted, teacher)
        combined = torch.stack([(.05 if n in NUISANCE else .10)*factors[n] for n in FACTORS]).sum()
        if k > schedule.calibration_steps:
            combined = combined + .10*reference + relational_weight*(
                relational["pattern"]+relational["spread"])
    losses = dict(combined_loss=combined, reference_loss=reference,
                  relational_pattern=relational["pattern"], relational_spread=relational["spread"],
                  **factors)
    if not all(torch.isfinite(v).all() for v in losses.values()):
        raise FloatingPointError("official adaptation non-finite loss")
    return losses


def train_official_update(model, heads, optimizer, batch: dict, k: int,
                          schedule: AdaptSchedule, relational_weight: float = .10) -> dict:
    hf, lf, grl = head_factor(k, schedule), lora_factor(k, schedule), grl_factor(k, schedule)
    if {g.get("name") for g in optimizer.param_groups} != {"lora", "heads"}:
        raise ValueError("explicit official optimizer group names required")
    for group in optimizer.param_groups:
        group["lr"] = 1e-5*lf if group["name"] == "lora" else 3e-5*hf
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(batch["images224"].device.type, dtype=torch.bfloat16):
        losses = compute_official_losses(model, heads, batch, k, schedule, relational_weight)
    losses["combined_loss"].backward()
    lora = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is not None and not torch.isfinite(p.grad).all()
           for p in lora+list(heads.parameters())):
        raise FloatingPointError("official adaptation non-finite gradient")
    ln = torch.nn.utils.clip_grad_norm_(lora, 1.)
    hn = torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.)
    if not torch.isfinite(ln) or not torch.isfinite(hn):
        raise FloatingPointError("official adaptation non-finite gradient norm")
    optimizer.step()
    return dict(**{n: float(v.detach()) for n, v in losses.items()},
                lora_grad_norm=float(ln), head_grad_norm=float(hn),
                head_lr=3e-5*hf, lora_lr=1e-5*lf, grl=grl, applied_update=k)


def validate_config(config: dict) -> None:
    if (config.get("schema_version") != 1 or config.get("policy") != POLICY
            or config.get("seed") != 0 or config.get("schedule") != {
                "train_clips": 7096, "batch_size": 24, "epochs": 40, "calibration_epochs": 5,
                "lora_warmup_epochs": 2, "grl_ramp_epochs": 2, "checkpoint_epochs": 10}
            or config.get("optimizer") != dict(name="AdamW", lora_lr=1e-5, head_lr=3e-5,
                betas=[.9,.999], epsilon=1e-8, weight_decay=.05, gradient_clip_norm=1.)
            or config.get("losses") != dict(temperature=.07, nuisance_factor_weight=.05,
                articulator_factor_weight=.10, reference_weight=.10, relational_weight=.10)
            or config.get("model") != dict(name=MODEL, revision=BASE_SHA, lora_rank=8,
                lora_alpha=8, lora_blocks=[20,21,22,23], lora_targets=["q","v"],
                sampled_frames=5, resolutions=[224,448])
            or config.get("precision") != "bf16" or config.get("loss_precision") != "float32"
            or config.get("drop_last") is not True or config.get("resume_retention") != 2
            or config.get("duration_limit_seconds") is not None
            or "duration_limit_seconds" not in config
            or config.get("reserve_bytes") != 50*1024**3
            or config.get("supervision_policy") !=
                "qwen-stable-v1-crop-articulators-v2-unreviewed-v1"
            or config.get("human_review_status") != "not_assessed"):
        raise ValueError("production config differs from approved recipe")


def _code() -> dict:
    base = Path(__file__).resolve().parent
    return {str(p): sha256_file(p) for p in sorted(base.glob("*.py"))} | {
        str(base/"uv.lock"): sha256_file(base/"uv.lock"),
        str(base/"pyproject.toml"): sha256_file(base/"pyproject.toml")}


def _recheck_code(identity: dict):
    if _code() != identity["code_hashes"]:
        raise ValueError("official adaptation source/lock changed")


def _model(config: dict):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = cast(DINOv3ViTModel, AutoModel.from_pretrained(
        MODEL, revision=BASE_SHA, cache_dir=config["inputs"]["model_cache"],
        local_files_only=True, trust_remote_code=False, use_safetensors=True)).eval()
    torch.nn.Module.to(model, device=torch.device("cuda"))
    attach_late_lora(model)
    heads = DINOFactorHeads().to("cuda")
    return model, heads, make_optimizer(model, heads)


def _state(model, heads, optimizer, sampler, schedule, identity, charged) -> dict:
    k = sampler.batches_yielded
    return dict(schema_version=1, identity=identity, epoch=k//schedule.steps_per_epoch,
                completed_updates=k, applied_update=k, model=_lora_state(model),
                heads={n:v.detach().cpu().clone() for n,v in heads.state_dict().items()},
                optimizer=optimizer.state_dict(), schedule=schedule.state_dict(k),
                sampler=sampler.state_dict(), rng=capture_rng(), charged_seconds=charged)


def _cuda_batch(batch):
    return {k:v.to("cuda") if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def smoke_setup_charge(config: dict) -> float:
    total=0.
    for p in (Path(config["working_root"])/"smoke-attempts"/digest(config)).glob("*.json"):
        record=json.loads(p.read_text())
        charge=record.get("charged_seconds")
        if (record.get("config_hash")!=digest(config) or type(charge) not in (float,int)
                or not math.isfinite(charge) or charge<0):
            raise ValueError("smoke attempt/accounting invalid")
        total+=charge
    return total


def run_official_adaptation(config: dict, mode: str, resume: Path | None = None) -> dict:
    validate_config(config)
    if mode not in {"preflight", "smoke", "run"}:
        raise ValueError("invalid execution mode")
    start = time.monotonic()
    inputs = config["inputs"]
    manifest = Path(inputs["merged_manifest"])
    if sha256_file(manifest) != inputs["merged_manifest_sha256"]:
        raise ValueError("approved merged manifest changed")
    admission = admit_official_data(*[Path(inputs[k]) for k in (
        "train_annotation", "dev_annotation", "test_annotation", "source", "merged_manifest",
        "teacher_root", "frames_root")])
    base = Path(config["working_root"])
    model_file = (Path(inputs["model_cache"])/f"models--{MODEL.replace('/', '--')}"
                  /"snapshots"/BASE_SHA)
    identity: dict = dict(policy=POLICY, config=config, data_identity=admission["identity"],
                    code_hashes=_code(), runtime={n:version(n) for n in (
                        "torch","transformers","numpy","pillow")},
                    model_files={p.name:sha256_file(p) for p in model_file.iterdir()},
                    working_root=str(base.resolve()), hardware=subprocess.check_output([
                        "nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                        "--format=csv,noheader,nounits"],text=True,timeout=10).strip())
    key = f"official-dino-adapt-{digest(identity)}"
    root = base/"runs"/key
    identity.update(run_key=key, run_root=str(root.resolve()))
    schedule = AdaptSchedule(**config["schedule"])
    if mode == "preflight":
        return dict(status="ready",run_root=str(root),total_steps=schedule.total_steps,
                    storage=require_official_storage(base, 1024**3), identity=identity)
    require_official_storage(base,1024**3)
    if _other_gpu_pids():
        raise RuntimeError("GPU owned by another compute process")
    if mode == "smoke":
        smoke_root = base/"smokes"/key
        with writer(smoke_root):
            report_path = smoke_root/"smoke.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report["identity"] != identity:
                    raise ValueError("smoke identity mismatch")
                return report
            model, heads, optimizer = _model(config)
            batch = _cuda_batch(load_official_batch(profile_ids(admission,24),admission))
            before = _lora_state(model)
            cal = train_official_update(model,heads,optimizer,batch,
                                        schedule.calibration_steps,schedule)
            if any(not torch.equal(v,before[n]) for n,v in _lora_state(model).items()):
                raise ValueError("calibration changed LoRA weights")
            joint = train_official_update(model,heads,optimizer,batch,
                                           schedule.calibration_steps+1,schedule)
            joint = train_official_update(model,heads,optimizer,batch,
                schedule.calibration_steps+schedule.lora_warmup_steps,schedule)
            if optimizer_state_report(optimizer)["initialized_parameters"] != 30:
                raise ValueError("smoke AdamW state incomplete")
            sampler = OfficialBatchSampler(admission["train_ids"],batch_size=24)
            for _ in range(schedule.milestone_steps[0]):
                sampler.next_ids()
            state = _state(model,heads,optimizer,sampler,schedule,identity,0.)
            compact = {k:state[k] for k in ("schema_version","identity","epoch",
                       "completed_updates","applied_update","model","heads","charged_seconds")}
            with tempfile.TemporaryDirectory(dir=smoke_root) as scratch:
                full_path,model_path=Path(scratch)/"full.pt",Path(scratch)/"model.pt"
                _atomic(full_path,lambda h:torch.save(state,h))
                _atomic(model_path,lambda h:torch.save(compact,h))
                full_bytes,model_bytes=full_path.stat().st_size,model_path.stat().st_size
                restored=torch.load(full_path,map_location="cpu",weights_only=False)
                validate_state(restored)
                for n,v in state["model"].items():
                    if not torch.equal(v,restored["model"][n]):
                        raise ValueError("smoke save/load mismatch")
                optimizer.load_state_dict(restored["optimizer"])
                heads.load_state_dict(restored["heads"],strict=True)
                _load_lora_state(model,restored["model"])
                sampler.load_state_dict(restored["sampler"])
                restore_rng(restored["rng"])
                # Prove the loaded CUDA optimizer can execute a subsequent joint step.
                train_official_update(model,heads,optimizer,batch,
                                      schedule.milestone_steps[0]+1,schedule)
            projected = 5*model_bytes+4*full_bytes
            report = dict(status="smoke_complete",identity=identity,calibration=cal,joint=joint,
                          full_snapshot_bytes=full_bytes, model_bytes=model_bytes,
                          projected_bytes=projected,optimizer_state=optimizer_state_report(optimizer),
                          charged_seconds=time.monotonic()-start,
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            atomic_json(report_path,report)
            return report
    smoke_path = base/"smokes"/key/"smoke.json"
    smoke = json.loads(smoke_path.read_text())
    if smoke.get("status") != "smoke_complete" or smoke["identity"] != identity:
        raise ValueError("matching completed smoke required")
    require_official_storage(base,smoke["projected_bytes"])
    with writer(root):
        if (root/"completion.json").exists():
            raise ValueError("adaptation already complete")
        atomic_json(root/"identity.json",identity)
        atomic_json(root/"inputs.json",admission["identity"])
        for path,sha in identity["code_hashes"].items():
            p=Path(path)
            if sha256_file(p)!=sha:
                raise ValueError("source changed before sealing")
            dest=root/"producer"/p.name
            if not dest.exists():
                _atomic(dest,lambda h,src=p:h.write(src.read_bytes()))
            if sha256_file(dest)!=sha:
                raise ValueError("sealed code mismatch")
        inherited=smoke_setup_charge(config)
        if inherited<smoke["charged_seconds"]:
            raise ValueError("smoke attempt accounting missing")
        ledger = OfficialLedger(root,identity,inherited,setup_seconds=time.monotonic()-start)
        try:
            model,heads,optimizer=_model(config)
            sampler=OfficialBatchSampler(admission["train_ids"],batch_size=24)
            snapshots=sorted(root.glob("resume-epoch-*.pt"),key=lambda p:int(p.stem.split('-')[-1]))
            chosen=resume or (snapshots[-1] if snapshots else None)
            if chosen is not None:
                state=validate_official_resume(chosen,identity,json.loads(ledger.path.read_text()))
                _load_lora_state(model,state["model"])
                heads.load_state_dict(state["heads"],strict=True)
                optimizer.load_state_dict(state["optimizer"])
                sampler.load_state_dict(state["sampler"])
                restore_rng(state["rng"])
            ledger.charge(sampler.batches_yielded,"durable_replay",replay=True)
            last=time.monotonic()
            for k in range(sampler.batches_yielded+1,schedule.total_steps+1):
                batch=_cuda_batch(load_official_batch(sampler.next_ids(),admission))
                values=train_official_update(model,heads,optimizer,batch,k,schedule)
                ledger.completed=k
                if k%25==0:
                    print(json.dumps(dict(step=k,epoch=k/schedule.steps_per_epoch,**values)),flush=True)
                if time.monotonic()-last>=10:
                    ledger.charge(k,"train")
                    last=time.monotonic()
                if k%schedule.steps_per_epoch==0:
                    _recheck_code(identity)
                    require_official_storage(root,smoke["projected_bytes"])
                if k in schedule.milestone_steps:
                    charged=ledger.charge(k,"checkpoint")
                    state=_state(model,heads,optimizer,sampler,schedule,identity,charged)
                    publish_milestone(root,state,state["epoch"])
                    del state
                    prune_owned_resumes(root)
                    ledger.charge(k,"checkpoint_complete")
            _recheck_code(identity)
            for epoch in (10,20,30,40):
                _receipt(root/f"model-epoch-{epoch}.pt")
            receipt=_receipt(root/"model-epoch-40.pt")
            atomic_json(root/"selected-checkpoint.json",dict(status="complete",epoch=40,
                checkpoint="model-epoch-40.pt",checkpoint_hash=receipt["sha256"],identity=identity))
            ledger.close("complete")
            payload={str(p.relative_to(root)):dict(sha256=sha256_file(p),bytes=p.stat().st_size)
                     for p in root.rglob("*") if p.is_file() and p.name!=".writer.lock"}
            completion=dict(status="complete",identity=identity,payload_files=payload)
            atomic_json(root/"completion.json",completion)
            print(json.dumps(dict(status="complete",run_root=str(root))),flush=True)
        except BaseException:
            ledger.close("interrupted")
            raise
    from tools.dinov3.official_archive import archive_completed_run
    try:
        archive=archive_completed_run(root,Path(config["archive_root"]))
        print(json.dumps(dict(status="archived",manifest=str(archive))),flush=True)
    except Exception as error:
        atomic_json(root/"archive-receipts"/"failure.json",dict(error=str(error)))
        print(json.dumps(dict(status="archive_failed",error=str(error))),flush=True)
    return completion


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--mode",choices=("preflight","smoke","run"),required=True)
    parser.add_argument("--resume-checkpoint",type=Path)
    parser.add_argument("--authorize-adaptation",action="store_true")
    args=parser.parse_args(argv)
    if args.mode!="preflight" and not args.authorize_adaptation:
        raise ValueError("explicit adaptation authorization required")
    def stop(signum,frame):
        raise KeyboardInterrupt(f"signal{signum}")
    signal.signal(signal.SIGTERM,stop)
    config=json.loads(args.config.read_text())
    attempt_start=time.monotonic()
    status="failed"
    try:
        result=run_official_adaptation(config,args.mode,args.resume_checkpoint)
        status="complete"
    finally:
        if args.mode=="smoke":
            path=Path(config["working_root"])/"smoke-attempts"/digest(config)/f"{time.time_ns()}.json"
            atomic_json(path,dict(config_hash=digest(config),status=status,
                        charged_seconds=time.monotonic()-attempt_start))
    print(json.dumps({k:v for k,v in result.items() if k in (
        "status","run_root","total_steps","model_bytes","full_snapshot_bytes",
        "projected_bytes","charged_seconds")}),flush=True)


if __name__=="__main__":
    main()
