import importlib

import numpy as np
import pytest


def module():
    return importlib.import_module("tools.benchmark.feature_package")


def fixture(tmp_path):
    paths=[]
    for i in range(5):
        path=tmp_path/"frames/train/a"/f"images{i+1:04}.png"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(bytes([i]))
        paths.append(path)
    root=tmp_path/"features"
    identity={"key":"test","kind":"spatial","width":2048}
    return root,paths,identity


def test_clip_receipt_hashes_source_and_preserves_full_frame_order(tmp_path):
    root,paths,identity=fixture(tmp_path)
    values=np.ones((5,2048),np.float32)
    row=module().write_clip(root,"train","a",paths,values,identity,{})
    assert row["frame_count"]==5 and row["source_indices"]==[0,1,2,3,4]
    assert row["paths"]==[str(p) for p in paths]
    verified=module().verify_clip(root,"train","a",paths,identity)
    assert verified==row
    paths[0].write_bytes(b"changed")
    with pytest.raises(ValueError):
        module().verify_clip(root,"train","a",paths,identity)


def test_bad_float_shape_and_dtype_cannot_publish(tmp_path):
    root,paths,identity=fixture(tmp_path)
    for shape,dtype in [((5,1024),np.float32),((5,2048),np.float64)]:
        with pytest.raises(ValueError):
            module().write_clip(root,"train","a",paths,np.ones(shape,dtype),identity,{})
    assert not (root/"receipts/train/a.json").exists()


def test_resume_uses_identical_arrays_and_rejects_changed_bytes(tmp_path):
    root,paths,identity=fixture(tmp_path)
    values=np.ones((5,2048),np.float32)
    module().write_clip(root,"train","a",paths,values,identity,{})
    assert module().verify_clip(root,"train","a",paths,identity)
    np.save(root/"train/a.npy",np.zeros((5,2048),np.float32))
    with pytest.raises(ValueError):
        module().verify_clip(root,"train","a",paths,identity)


def test_final_manifest_uses_exact_split_and_feature_schema(tmp_path):
    root,paths,identity=fixture(tmp_path)
    module().write_clip(root,"train","a",paths,np.ones((5,2048),np.float32),identity,{})
    inventory={"train":{"a":{"paths":[str(p) for p in paths]}}}
    result=module().finish_package(root,identity,inventory,2048)
    import json
    manifest=json.loads((root/"complete/manifest.json").read_text())
    assert manifest["records"]==[dict(clip_id="a",split="train",path="train/a.npy",
                                     length=5,width=2048,dtype="float32")]
    assert result["clips"]==1
    assert module().verify_package(root,identity,inventory,2048)["clips"]==1


def test_manifest_does_not_allow_extra_array(tmp_path):
    root,paths,identity=fixture(tmp_path)
    module().write_clip(root,"train","a",paths,np.ones((5,2048),np.float32),identity,{})
    np.save(root/"train/extra.npy",np.ones((5,2048),np.float32))
    with pytest.raises(ValueError):
        module().finish_package(root,identity,
                                {"train":{"a":{"paths":[str(p) for p in paths]}}},2048)
