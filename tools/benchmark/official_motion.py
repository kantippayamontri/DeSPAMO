"""Official all-split motion coverage with unchanged VideoMAE generator."""

import argparse
import json
import signal
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from tools.benchmark.feature_package import (
    finish_package,
    official_inventory,
    other_gpu_pids,
    source_rows,
    verify_clip,
    write_clip,
)
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_runtime import OfficialLedger, require_official_storage
from tools.dinov3.storage import atomic_json, writer


def windows(count: int) -> list[dict]:
    if type(count)is not int or count<1:
        raise ValueError("positive frame count required")
    starts=[0] if count<16 else range(0,count-15,8)
    return [dict(row=i,input_source_indices=[min(start+j,count-1) for j in range(16)],
                 real_start=start,real_end=min(start+15,count-1),centre=start+7.5)
            for i,start in enumerate(starts)]


def reuse_train_clip(parent: Path,clip: str,paths: list[Path]) -> tuple[np.ndarray,dict]:
    receipt_path=parent/"receipts/train"/f"{clip}.json"
    array_path=parent/"train"/f"{clip}.npy"
    receipt=json.loads(receipt_path.read_text())
    before=source_rows(paths)
    if (receipt["source_rows"]!=before or receipt["frame_count"]!=len(paths)
            or receipt["windows"]!=windows(len(paths))
            or sha256_file(array_path)!=receipt["array"]["sha256"]):
        raise ValueError("parent training motion source/windows/bytes changed")
    values=np.load(array_path,allow_pickle=False)
    if (values.shape!=(len(windows(len(paths))),1024) or values.dtype!=np.float32
            or not np.isfinite(values).all()):
        raise ValueError("parent motion shape/dtype invalid")
    return values,dict(origin="verified_parent_copy",observed_source_rows=before,
                       parent_receipt=str(receipt_path),parent_receipt_sha256=sha256_file(receipt_path),
                       parent_array_sha256=receipt["array"]["sha256"],windows=receipt["windows"])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("smoke","run"),required=True)
    for name in ("annotation-root","row-map","frames-root","parent-motion","output-base"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--authorize-extraction",action="store_true")
    args=parser.parse_args(argv)
    if not args.authorize_extraction:
        raise ValueError("explicit motion extraction authorization required")
    from despamo.data.motion_validation import validate_motion_package
    from tools.motion.extractor import SNAPSHOT, extract_motion_clip, load_encoder
    from tools.motion.records import MODEL, PROCESSOR

    def stop(signum,frame):
        raise KeyboardInterrupt(f"signal{signum}")
    signal.signal(signal.SIGTERM,stop)
    parent_identity=json.loads((args.parent_motion/"complete/identity.json").read_text())
    parent_frames=json.loads(Path(parent_identity["inputs"]["frame_rows"]["path"]).read_text())
    validate_motion_package(args.parent_motion,parent_frames,verify_features=True)
    inventory=official_inventory(args.annotation_root,args.row_map)
    for clips in inventory.values():
        for row in clips.values():
            row["paths"]=[str(args.frames_root/p) for p in row["paths"]]
    import tools.motion.extractor as producer
    if producer.__file__ is None:
        raise ValueError("motion generator source path unavailable")
    code_paths=[Path(__file__),Path(__file__).with_name("feature_package.py"),
                Path(producer.__file__),Path(producer.__file__).with_name("records.py")]
    identity: dict=dict(policy="official-phoenix14t-mixed-origin-motion-v1",
        kind="motion",width=1024,
        model=MODEL,processor=PROCESSOR,frames_root=str(args.frames_root),
        row_map_sha256=sha256_file(args.row_map),
        parent_motion=str(args.parent_motion),
        parent_completion_sha256=sha256_file(args.parent_motion/"complete/completion.json"),
        parent_identity_sha256=sha256_file(args.parent_motion/"complete/identity.json"),
        code_hashes={str(p):sha256_file(p) for p in code_paths},batch_windows=8,
        runtime={n:version(n) for n in ("torch","transformers","numpy","pillow")})
    key=digest(identity)
    root=args.output_base/f"motion-{key}"
    destination=root if args.mode=="run" else args.output_base/f"motion-smoke-{key}"
    require_official_storage(args.output_base,2*1024**3)
    if other_gpu_pids():
        raise RuntimeError("GPU already owned")
    with writer(destination):
        if (destination/"complete").exists():
            from tools.benchmark.feature_package import verify_package
            print(json.dumps(verify_package(destination,identity,inventory,1024)),flush=True)
            return
        atomic_json(destination/"identity.json",identity)
        ledger=OfficialLedger(destination,identity)
        ledger.charge(0,"verified_clip_replay",replay=True)
        try:
            loaded,processor=load_encoder(SNAPSHOT)
            model=cast(Any,loaded)
            model.to("cuda")
            chosen=inventory
            if args.mode=="smoke":
                chosen={s:{list(c)[0]:next(iter(c.values()))} for s,c in inventory.items()}
            done=0
            last=time.monotonic()
            for split,clips in chosen.items():
                for clip,row in clips.items():
                    paths=[Path(p) for p in row["paths"]]
                    if (destination/"receipts"/split/f"{clip}.json").exists():
                        verify_clip(destination,split,clip,paths,identity)
                    else:
                        if split=="train":
                            values,extra=reuse_train_clip(args.parent_motion,clip,paths)
                        else:
                            values,sources,observed=extract_motion_clip(tuple(paths),model,processor,
                                                                       batch_size=8)
                            if list(observed)!=windows(len(paths)):
                                raise ValueError("actual VideoMAE window order differs")
                            extra=dict(origin="new_official_split_extraction",
                                       observed_source_rows=sources,windows=list(observed))
                        write_clip(destination,split,clip,paths,values,identity,extra)
                    done+=1
                    if done%25==0 or args.mode=="smoke":
                        print(json.dumps(dict(split=split,clip=clip,clips=done,total_clips=8257)),
                              flush=True)
                    if time.monotonic()-last>=10:
                        ledger.charge(done,"extract")
                        last=time.monotonic()
                    if done%100==0:
                        if any(sha256_file(Path(p))!=sha
                               for p,sha in identity["code_hashes"].items()):
                            raise ValueError("motion producer bytes changed")
                        require_official_storage(destination,2*1024**3)
            if args.mode=="smoke":
                result=dict(status="smoke_complete",clips=done,
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
                atomic_json(destination/"smoke.json",result)
            else:
                ledger.charge(done,"validating_publication")
                result=finish_package(destination,identity,inventory,1024)
            ledger.close("complete")
            print(json.dumps({**result,"root":str(destination)}),flush=True)
        except BaseException:
            ledger.close("interrupted")
            raise


if __name__=="__main__":
    main()
