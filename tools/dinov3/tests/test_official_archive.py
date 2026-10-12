import importlib
import json

import pytest

from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json


def module():
    return importlib.import_module("tools.dinov3.official_archive")


def completed(tmp_path):
    run=tmp_path/"official-dino-adapt-test"
    run.mkdir()
    identity=dict(run_key=run.name,run_root=str(run))
    (run/"model-epoch-40.pt").write_bytes(b"test weights")
    atomic_json(run/"identity.json",identity)
    payload={p.name:dict(sha256=sha256_file(p),bytes=p.stat().st_size) for p in run.iterdir()}
    atomic_json(run/"completion.json",dict(status="complete",identity=identity,payload_files=payload))
    return run,identity


def test_archive_copy_verify_idempotence_and_native_bookkeeping(tmp_path,monkeypatch):
    run,identity=completed(tmp_path)
    monkeypatch.setattr(module(),"require_archive_storage",lambda *args,**kwargs:{})
    manifest=module().archive_completed_run(run,tmp_path/"archive")
    result=module().verify_official_archive(manifest.parent,identity)
    assert result["source_identity"]==identity
    assert (run/"model-epoch-40.pt").is_file()
    atomic_json(run/"archive-receipts/extra.json",dict(bookkeeping="later"))
    assert module().archive_completed_run(run,tmp_path/"archive")==manifest
    (manifest.parent/"model-epoch-40.pt").write_bytes(b"fake weights")
    with pytest.raises(ValueError):
        module().verify_official_archive(manifest.parent,identity)


def test_incomplete_run_and_extra_archive_files_rejected(tmp_path,monkeypatch):
    run,identity=completed(tmp_path)
    monkeypatch.setattr(module(),"require_archive_storage",lambda *args,**kwargs:{})
    manifest=module().archive_completed_run(run,tmp_path/"archive")
    (manifest.parent/"extra.pt").write_bytes(b"extra")
    with pytest.raises(ValueError):
        module().verify_official_archive(manifest.parent,identity)
    (run/"completion.json").unlink()
    with pytest.raises(ValueError):
        module().archive_completed_run(run,tmp_path/"other")


def test_interrupted_copy_has_no_final_archive(tmp_path,monkeypatch):
    run,_=completed(tmp_path)
    monkeypatch.setattr(module(),"require_archive_storage",lambda *args,**kwargs:{})
    def fail(src,dest):
        raise OSError("copy failed")
    monkeypatch.setattr(module(),"_copy_file",fail)
    with pytest.raises(OSError):
        module().archive_completed_run(run,tmp_path/"archive")
    assert not (tmp_path/"archive"/run.name/"archive-manifest.json").exists()


def test_archive_requires_actual_b_mount(tmp_path):
    with pytest.raises(ValueError):
        module().require_archive_storage(tmp_path,1)


def test_path_escape_rejected_from_sealed_payload(tmp_path,monkeypatch):
    run,_=completed(tmp_path)
    monkeypatch.setattr(module(),"require_archive_storage",lambda *args,**kwargs:{})
    completion=json.loads((run/"completion.json").read_text())
    completion["payload_files"]["../escape"]={"sha256":"wrong","bytes":1}
    atomic_json(run/"completion.json",completion)
    with pytest.raises(ValueError):
        module().archive_completed_run(run,tmp_path/"archive")


def test_unsupported_directory_fsync_is_recorded_honestly(tmp_path,monkeypatch):
    import errno
    import os
    import stat

    run,identity=completed(tmp_path)
    monkeypatch.setattr(module(),"require_archive_storage",lambda *args,**kwargs:{})
    real_fsync=os.fsync
    def unsupported(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL,"directory fsync unsupported")
        real_fsync(fd)
    # Only exercise NTFS-style marker publication, not native receipt fsync.
    path=tmp_path/"marker.json"
    monkeypatch.setattr(os,"fsync",unsupported)
    assert module()._archive_json(path,{"status":"verified"}) is False
    assert json.loads(path.read_text())["status"]=="verified"
