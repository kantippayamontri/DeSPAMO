"""Full official split feature arrays, source receipts and manifest-last publication."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.storage import _atomic, _fsync_directory, atomic_json


def other_gpu_pids() -> list[int]:
    output=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits"],text=True,timeout=10)
    return [int(line) for line in output.splitlines() if line.strip().isdigit()
            and int(line)!=os.getpid()]


def official_inventory(annotation_root: Path,row_map: Path) -> dict:
    rows=json.loads(row_map.read_text())["clips"]
    inventory,seen={},set()
    for split,count in (("train",7096),("dev",519),("test",642)):
        raw=np.load(annotation_root/f"{split}_info_ml.npy",allow_pickle=True).item()
        records=[raw[k] for k in sorted(k for k in raw if type(k)is int)]
        ids=[r["fileid"] for r in records]
        if len(records)!=count or len(set(ids))!=count or seen.intersection(ids):
            raise ValueError("official split IDs/counts overlap or mismatch")
        if {r["signer"] for r in records}!={f"Signer{i:02}" for i in range(1,10)}:
            raise ValueError("official signer inventory mismatch")
        seen.update(ids)
        clips={}
        for record in records:
            clip=record["fileid"]
            if clip in {".",".."} or "/" in clip or "\\" in clip:
                raise ValueError("invalid official clip ID")
            row=rows[clip]
            paths=row["paths"]
            if (row["source_indices"]!=list(range(row["frame_count"]))
                    or len(paths)!=row["frame_count"] or paths!=sorted(set(paths))
                    or any(not p.startswith(f"{split}/{clip}/") or ".." in Path(p).parts
                           for p in paths)):
                raise ValueError("official full frame paths/order mismatch")
            clips[clip]=dict(paths=paths,frame_count=len(paths),signer=record["signer"])
        inventory[split]=clips
    if set(rows)!=seen:
        raise ValueError("row-map/official clip inventory mismatch")
    return inventory


def source_rows(paths: list[Path]) -> list[dict]:
    if not paths or len(set(paths))!=len(paths):
        raise ValueError("unique source frames required")
    result=[]
    for i,path in enumerate(paths):
        if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError("regular unaliased source required")
        result.append(dict(index=i,path=str(path.resolve()),sha256=sha256_file(path)))
    return result


def _validate(values: np.ndarray,identity: dict,frame_count: int):
    expected=frame_count if identity["kind"]=="spatial" else max(1,(frame_count-16)//8+1)
    if (values.shape!=(expected,identity["width"]) or values.dtype!=np.float32
            or not np.isfinite(values).all()):
        raise ValueError("feature shape/dtype/finiteness mismatch")


def verify_clip(root: Path,split: str,clip: str,paths: list[Path],identity: dict) -> dict:
    path=root/split/f"{clip}.npy"
    receipt=root/"receipts"/split/f"{clip}.json"
    if not path.is_file() or not receipt.is_file() or path.is_symlink() or receipt.is_symlink():
        raise ValueError("complete feature/receipt required")
    row=json.loads(receipt.read_text())
    observed=source_rows(paths)
    if (row.get("identity_hash")!=digest(identity) or row.get("split")!=split
            or row.get("clip_id")!=clip or row.get("source_rows")!=observed
            or row.get("feature_hash")!=sha256_file(path)):
        raise ValueError("feature/source/receipt identity changed")
    array=np.load(path,allow_pickle=False,mmap_mode="r")
    _validate(array,identity,len(paths))
    return row


def write_clip(root: Path,split: str,clip: str,paths: list[Path],values: np.ndarray,
               identity: dict,extra: dict) -> dict:
    if split not in {"train","dev","test"} or not clip or "/" in clip or "\\" in clip:
        raise ValueError("invalid feature clip/split")
    _validate(values,identity,len(paths))
    observed=source_rows(paths)
    if "observed_source_rows" in extra and extra["observed_source_rows"]!=observed:
        raise ValueError("source changed during actual inference")
    path=root/split/f"{clip}.npy"
    receipt=root/"receipts"/split/f"{clip}.json"
    if receipt.exists():
        return verify_clip(root,split,clip,paths,identity)
    if path.exists() or path.is_symlink():
        raise ValueError("unreceipted feature exists; preserve crash artifact")
    _atomic(path,lambda h:np.save(h,values,allow_pickle=False))
    row=dict(clip_id=clip,split=split,identity_hash=digest(identity),frame_count=len(paths),
             paths=[r["path"] for r in observed],source_rows=observed,
             source_hash=digest([(r["path"],r["sha256"]) for r in observed]),
             source_indices=list(range(len(paths))),feature_hash=sha256_file(path),
             feature_rows=len(values),width=values.shape[1],
             extra={k:v for k,v in extra.items() if k!="observed_source_rows"})
    atomic_json(receipt,row)
    return row


def verify_package(root: Path,identity: dict,inventory: dict,width: int,
                   complete: Path | None=None) -> dict:
    complete=root/"complete" if complete is None else complete
    actual=json.loads((complete/"identity.json").read_text())
    manifest=json.loads((complete/"manifest.json").read_text())
    if actual!=identity or manifest["expected_dim"]!=width or manifest["encoder"]!=digest(identity):
        raise ValueError("feature package identity mismatch")
    records=[]
    for split,clips in inventory.items():
        if {p.stem for p in (root/split).glob("*.npy")}!=set(clips):
            raise ValueError("extra/missing feature array")
        for clip,source in clips.items():
            row=verify_clip(root,split,clip,[Path(p) for p in source["paths"]],identity)
            records.append(dict(clip_id=clip,split=split,path=f"{split}/{clip}.npy",
                                length=row["feature_rows"],width=width,dtype="float32"))
    if manifest["records"]!=records:
        raise ValueError("complete feature manifest inventory mismatch")
    return dict(clips=len(records),manifest=str(root/"complete/manifest.json"),
                manifest_sha256=sha256_file(complete/"manifest.json"))


def finish_package(root: Path,identity: dict,inventory: dict,width: int) -> dict:
    if (root/"complete").exists():
        return verify_package(root,identity,inventory,width)
    stage=root/f".complete-{os.getpid()}"
    if stage.exists():
        raise ValueError("incomplete publication stage exists")
    records,rows=[],{}
    for split,clips in inventory.items():
        for clip,source in clips.items():
            row=verify_clip(root,split,clip,[Path(p) for p in source["paths"]],identity)
            rows[clip]=row
            records.append(dict(clip_id=clip,split=split,path=f"{split}/{clip}.npy",
                                length=row["feature_rows"],width=width,dtype="float32"))
    atomic_json(stage/"identity.json",identity)
    atomic_json(stage/"manifest.json",dict(schema_version=1,encoder=digest(identity),
                                         expected_dim=width,records=records))
    atomic_json(stage/"frame_rows.json",dict(encoder_key=digest(identity),clips=rows))
    verify_package(root,identity,inventory,width,complete=stage)
    os.replace(stage,root/"complete")
    _fsync_directory(root)
    return verify_package(root,identity,inventory,width)
