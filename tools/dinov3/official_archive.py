"""Verified cold B: archival; never remove native training originals."""

import argparse
import errno
import json
import os
import shutil
from pathlib import Path

from tools.dinov3.identity import sha256_file
from tools.dinov3.official_runtime import windows_free
from tools.dinov3.storage import atomic_json, writer


def require_archive_storage(archive_base: Path, needed_bytes: int,
                            reserve_bytes: int = 50*1024**3) -> dict:
    mount=Path("/mnt/b")
    mounts=Path("/proc/mounts").read_text().splitlines()
    actual=any(line.split()[1]==str(mount) and line.split()[0].lower().startswith("b:")
               for line in mounts)
    if (not actual or not archive_base.resolve().is_relative_to(mount)
            or any(p.is_symlink() for p in (archive_base,*archive_base.parents))):
        raise ValueError("archive requires actual B: mount without symlink fallback")
    free=windows_free("B")
    if needed_bytes<0 or reserve_bytes!=50*1024**3 or free<needed_bytes+reserve_bytes:
        raise ValueError("archive B: space below demand/reserve")
    return dict(windows_free_bytes=free,required_bytes=needed_bytes+reserve_bytes)


def _inside(root: Path,name: str) -> Path:
    path=root/name
    if (Path(name).is_absolute() or ".." in Path(name).parts
            or path.is_symlink() or root.resolve() not in path.resolve().parents):
        raise ValueError("archive payload path escapes owned root")
    return path


def _verify_files(root: Path,payload: dict) -> None:
    for name,receipt in payload.items():
        path=_inside(root,name)
        if (not path.is_file() or sha256_file(path)!=receipt["sha256"]
                or path.stat().st_size!=receipt["bytes"]):
            raise ValueError("archive payload missing/hash/size mismatch")


def _copy_file(source: Path,dest: Path) -> None:
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_name(dest.name+".partial")
    if dest.is_symlink() or temp.is_symlink():
        raise ValueError("unsafe archive destination")
    with source.open("rb") as src,temp.open("wb") as out:
        shutil.copyfileobj(src,out)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp,dest)


def _archive_json(path: Path,value: dict) -> bool:
    temp=path.with_suffix(".partial")
    if path.is_symlink() or temp.is_symlink():
        raise ValueError("unsafe archive marker")
    with temp.open("w") as handle:
        json.dump(value,handle,sort_keys=True,indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp,path)
    try:
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return True
    except OSError as error:
        if error.errno not in {errno.EINVAL,errno.ENOTSUP,errno.EACCES}:
            raise
        return False


def verify_official_archive(archive: Path,source_identity: dict) -> dict:
    path=archive/"archive-manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("complete archive manifest required")
    data=json.loads(path.read_text())
    if data["source_identity"]!=source_identity or data["status"]!="verified":
        raise ValueError("archive identity/status mismatch")
    _verify_files(archive,data["files"])
    actual={str(p.relative_to(archive)) for p in archive.rglob("*") if p.is_file()}
    if actual!=set(data["files"])|{"archive-manifest.json"}:
        raise ValueError("archive inventory contains extra/missing files")
    return data


def archive_completed_run(run: Path,archive_base: Path) -> Path:
    completion_path=run/"completion.json"
    if completion_path.is_symlink() or not completion_path.is_file():
        raise ValueError("completed native adaptation required")
    completion=json.loads(completion_path.read_text())
    if completion.get("status")!="complete" or completion["identity"]["run_key"]!=run.name:
        raise ValueError("native adaptation not complete/owned")
    files=dict(completion["payload_files"])
    _verify_files(run,files)
    files["completion.json"]=dict(sha256=sha256_file(completion_path),
                                  bytes=completion_path.stat().st_size)
    require_archive_storage(archive_base,sum(r["bytes"] for r in files.values()))
    archive=archive_base/run.name
    with writer(run):
        if archive.exists():
            verify_official_archive(archive,completion["identity"])
            return archive/"archive-manifest.json"
        stage=archive_base/(run.name+".partial")
        if stage.is_symlink():
            raise ValueError("unsafe archive stage")
        stage.mkdir(parents=True,exist_ok=True)
        for name in files:
            _copy_file(_inside(run,name),_inside(stage,name))
        _verify_files(run,files)
        _verify_files(stage,files)
        data: dict=dict(status="verified",source_identity=completion["identity"],
                  source_root=str(run),
                  archive_root=str(archive),files=files,durability="file-fsync-rehashed-copy")
        supported=_archive_json(stage/"archive-manifest.json",data)
        data["directory_fsync_supported"]=supported
        _archive_json(stage/"archive-manifest.json",data)
        verify_official_archive(stage,completion["identity"])
        os.replace(stage,archive)
        verify_official_archive(archive,completion["identity"])
        atomic_json(run/"archive-receipts/verified.json",dict(manifest=str(archive/"archive-manifest.json"),
                    manifest_sha256=sha256_file(archive/"archive-manifest.json"),**data))
    return archive/"archive-manifest.json"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--archive-base",type=Path,required=True)
    parser.add_argument("--mode",choices=("copy","verify"),required=True)
    args=parser.parse_args(argv)
    if args.mode=="copy":
        print(archive_completed_run(args.run,args.archive_base))
    else:
        identity=json.loads((args.run/"completion.json").read_text())["identity"]
        verify_official_archive(args.archive_base/args.run.name,identity)
        print("verified")


if __name__=="__main__":
    main()
