import importlib

import numpy as np
import pytest

from tools.benchmark.feature_package import source_rows
from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json


def module():
    return importlib.import_module("tools.benchmark.official_motion")


def test_verified_parent_train_array_preserves_bytes_and_source(tmp_path):
    paths=[]
    for i in range(16):
        p=tmp_path/"frames/train/a"/f"images{i+1:04}.png"
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(bytes([i]))
        paths.append(p)
    parent=tmp_path/"parent"
    (parent/"train").mkdir(parents=True)
    np.save(parent/"train/a.npy",np.ones((1,1024),np.float32))
    receipt=dict(source_rows=source_rows(paths),frame_count=16,
                 array=dict(sha256=sha256_file(parent/"train/a.npy"),shape=[1,1024]),
                 windows=[dict(row=0,input_source_indices=list(range(16)),
                               real_start=0,real_end=15,centre=7.5)])
    atomic_json(parent/"receipts/train/a.json",receipt)
    values,extra=module().reuse_train_clip(parent,"a",paths)
    assert values.shape==(1,1024) and extra["origin"]=="verified_parent_copy"
    assert extra["parent_array_sha256"]==sha256_file(parent/"train/a.npy")
    paths[0].write_bytes(b"changed")
    with pytest.raises(ValueError):
        module().reuse_train_clip(parent,"a",paths)


def test_windows_include_only_real_full_support_or_short_padding():
    assert module().windows(24)==[
        dict(row=0,input_source_indices=list(range(16)),real_start=0,real_end=15,centre=7.5),
        dict(row=1,input_source_indices=list(range(8,24)),real_start=8,real_end=23,centre=15.5)]
    assert module().windows(5)[0]["input_source_indices"]==[0,1,2,3,4]+[4]*11
