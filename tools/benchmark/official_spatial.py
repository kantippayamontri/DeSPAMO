"""Full-video official train/dev/test DINO spatial features from epoch40 LoRA."""

import argparse
import json
import signal
import time
from importlib.metadata import version
from pathlib import Path
from typing import cast

import torch
from transformers import AutoModel, DINOv3ViTModel

from tools.benchmark.feature_package import (
    finish_package,
    official_inventory,
    source_rows,
    verify_clip,
    verify_package,
    write_clip,
)
from tools.dinov3.e3_model import attach_late_lora
from tools.dinov3.e3_train import BASE_SHA, _load_lora_state
from tools.dinov3.encoder import extract
from tools.dinov3.identity import MODEL, digest, sha256_file
from tools.dinov3.official_runtime import OfficialLedger, require_official_storage
from tools.dinov3.profile_official_adapt import _other_gpu_pids
from tools.dinov3.storage import atomic_json, writer


def admit_epoch40(run: Path) -> dict:
    selected=json.loads((run/"selected-checkpoint.json").read_text())
    if selected.get("status")!="complete" or selected.get("epoch")!=40:
        raise ValueError("predeclared completed epoch40 required")
    completion=json.loads((run/"completion.json").read_text())
    checkpoint=run/selected["checkpoint"]
    if checkpoint.name!="model-epoch-40.pt" or checkpoint.is_symlink():
        raise ValueError("exact native epoch40 model required")
    receipt=json.loads(checkpoint.with_suffix(".json").read_text())
    sha=sha256_file(checkpoint)
    if (completion.get("status")!="complete" or completion["identity"]!=selected["identity"]
            or sha!=selected["checkpoint_hash"] or sha!=receipt["sha256"]
            or checkpoint.stat().st_size!=receipt["bytes"]):
        raise ValueError("epoch40 weights/completion/receipt changed")
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if (state["identity"]!=selected["identity"] or state["epoch"]!=40
            or state["completed_updates"]!=11800
            or state["identity"]["config"]["model"]["revision"]!=BASE_SHA):
        raise ValueError("adaptation checkpoint tensor/epoch/base identity mismatch")
    return dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha,
                selection_sha256=sha256_file(run/"selected-checkpoint.json"),
                completion_sha256=sha256_file(run/"completion.json"),state=state)


def _code() -> dict:
    root=Path(__file__).resolve().parents[2]
    paths=[Path(__file__),Path(__file__).with_name("feature_package.py")]
    paths += [root/"tools/dinov3"/name for name in (
        "encoder.py","frames.py","e3_model.py","e3_train.py","identity.py","storage.py",
        "official_runtime.py","uv.lock","pyproject.toml")]
    return {str(p):sha256_file(p) for p in paths}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("preflight","smoke","run","verify"),required=True)
    for name in ("adapt-run","annotation-root","row-map","frames-root","cache","output-base"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--authorize-extraction",action="store_true")
    args=parser.parse_args(argv)
    if args.batch_size!=32:
        raise ValueError("declared frame inference batch32 required")
    if args.mode in {"smoke","run"} and not args.authorize_extraction:
        raise ValueError("GPU extraction authorization required")
    def stop(signum,frame):
        raise KeyboardInterrupt(f"signal{signum}")
    signal.signal(signal.SIGTERM,stop)
    adapted=admit_epoch40(args.adapt_run)
    state=adapted.pop("state")
    inventory=official_inventory(args.annotation_root,args.row_map)
    for split,clips in inventory.items():
        for clip,row in clips.items():
            row["paths"]=[str(args.frames_root/p) for p in row["paths"]]
            actual=sorted((args.frames_root/split/clip).glob("*.png"))
            if [str(p) for p in actual]!=row["paths"]:
                raise ValueError("source frame inventory differs from pinned row-map")
    model_snapshot=args.cache/f"models--{MODEL.replace('/', '--')}"/"snapshots"/BASE_SHA
    identity=dict(policy="official-phoenix14t-epoch40-spatial-v1",kind="spatial",width=2048,
        adaptation=adapted,row_map_sha256=sha256_file(args.row_map),frames_root=str(args.frames_root),
        annotations={s:sha256_file(args.annotation_root/f"{s}_info_ml.npy")
                     for s in ("train","dev","test")},model=MODEL,revision=BASE_SHA,
        model_files={p.name:sha256_file(p) for p in model_snapshot.iterdir()},
        scales=[224,448],tokens="concatenated_cls",dtype="float32",batch_size=32,
        code_hashes=_code(),runtime={n:version(n) for n in (
            "torch","transformers","numpy","pillow")})
    key=digest(identity)
    root=args.output_base/f"spatial-{key}"
    total=sum(r["frame_count"] for c in inventory.values() for r in c.values())
    projected=total*2048*4+3*1024**3
    require_official_storage(args.output_base,projected)
    if args.mode=="preflight":
        print(json.dumps(dict(status="ready",root=str(root),clips=8257,frames=total,
                              projected_bytes=projected)),flush=True)
        return
    if args.mode=="verify":
        print(json.dumps(verify_package(root,identity,inventory,2048)),flush=True)
        return
    if _other_gpu_pids():
        raise RuntimeError("GPU already owned")
    destination=root if args.mode=="run" else args.output_base/f"spatial-smoke-{key}"
    with writer(destination):
        if (destination/"complete").exists():
            print(json.dumps(verify_package(destination,identity,inventory,2048)),flush=True)
            return
        atomic_json(destination/"identity.json",identity)
        ledger=OfficialLedger(destination,identity)
        ledger.charge(0,"verified_clip_replay",replay=True)
        try:
            model=cast(DINOv3ViTModel,AutoModel.from_pretrained(MODEL,revision=BASE_SHA,
                cache_dir=str(args.cache),local_files_only=True,use_safetensors=True,
                trust_remote_code=False)).eval()
            attach_late_lora(model)
            _load_lora_state(model,state["model"])
            model.requires_grad_(False).eval()
            torch.nn.Module.to(model,device=torch.device("cuda"))
            torch.backends.cuda.matmul.allow_tf32=False
            del state
            chosen=inventory
            if args.mode=="smoke":
                chosen={s:{list(c)[0]:next(iter(c.values()))} for s,c in inventory.items()}
                longest=max(inventory["train"],key=lambda i:inventory["train"][i]["frame_count"])
                chosen["train"][longest]=inventory["train"][longest]
            done=0
            last=time.monotonic()
            for split,clips in chosen.items():
                for clip,row in clips.items():
                    paths=[Path(p) for p in row["paths"]]
                    if (destination/"receipts"/split/f"{clip}.json").exists():
                        verify_clip(destination,split,clip,paths,identity)
                    else:
                        before=source_rows(paths)
                        values,resolutions=extract(paths,model,"cuda",batch_size=32)
                        write_clip(destination,split,clip,paths,values,identity,
                                   dict(observed_source_rows=before,resolutions=resolutions))
                    done+=1
                    if done%25==0 or args.mode=="smoke":
                        print(json.dumps(dict(split=split,clip=clip,clips=done,total_clips=8257,
                                             frames=row["frame_count"])),flush=True)
                    if time.monotonic()-last>=10:
                        ledger.charge(done,"extract")
                        last=time.monotonic()
                    if done%100==0:
                        if _code()!=identity["code_hashes"]:
                            raise ValueError("extraction source bytes changed")
                        require_official_storage(destination,max(0,projected-int(done/8257*projected)))
            if args.mode=="smoke":
                result=dict(status="smoke_complete",clips=done,
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
                atomic_json(destination/"smoke.json",result)
            else:
                ledger.charge(done,"validating_publication")
                result=finish_package(destination,identity,inventory,2048)
            ledger.close("complete")
            print(json.dumps({**result,"status":result.get("status","complete"),
                              "root":str(destination)}),flush=True)
        except BaseException:
            ledger.close("interrupted")
            raise


if __name__=="__main__":
    main()
