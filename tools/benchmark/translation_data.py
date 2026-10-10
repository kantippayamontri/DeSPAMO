"""Exact official benchmark feature/data admission and deterministic split batches."""

import json
from pathlib import Path

import numpy as np

from despamo.data.batch import collate_phoenix
from despamo.data.manifest import FeatureManifest
from despamo.data.phoenix14t import Phoenix14T
from tools.dinov3.identity import digest, sha256_file

DATA=Path("/home/kan/datasets/despamo")
SPATIAL=DATA/"official-phoenix14t-features-v1/spatial-a866a2a0bfcc45d7e6ff5555ee22bb928fd86f1f166af7382d1d05b233cac90b"
MOTION=DATA/"official-phoenix14t-features-v1/motion-1123ff445677bf310ac0b8c9b45cac4f690b9102f5dad1960aa2fef04937e82d"
ANNOTATIONS=Path("/home/kan/Research/SpaMo/preprocess/Phoenix14T")


def admit_data(verify_arrays: bool=True) -> dict:
    manifests={}
    bindings={}
    feature_hashes={}
    source_hashes={}
    for kind,root,width,sha in [
        ("spatial",SPATIAL,2048,"5769cbe92343356e0ffaf87f7cb3fda305b4422fdf597dbf82ad856f1265ae5d"),
        ("motion",MOTION,1024,"023a054fe0d3b9e04b5c0472ef77ebb8d031130e604841d67acfe7d81f66fd03")]:
        path=root/"complete/manifest.json"
        if sha256_file(path)!=sha:
            raise ValueError("declared feature manifest bytes changed")
        manifests[kind]=FeatureManifest.load(path)
        identity=json.loads((root/"complete/identity.json").read_text())
        if manifests[kind].encoder!=digest(identity) or manifests[kind].expected_dim!=width:
            raise ValueError("feature identity/dimension mismatch")
        bindings[str(path)]=sha
        bindings[str(root/"complete/identity.json")]=sha256_file(root/"complete/identity.json")
        hashes={}
        sources={}
        for record in manifests[kind].records:
            receipt=json.loads((root/"receipts"/record.split/f"{record.clip_id}.json").read_text())
            feature=root/record.path
            if receipt["identity_hash"]!=digest(identity):
                raise ValueError("feature receipt identity mismatch")
            if verify_arrays:
                if sha256_file(feature)!=receipt["feature_hash"]:
                    raise ValueError("feature array hash drift")
                a=np.load(feature,allow_pickle=False,mmap_mode="r")
                if (a.shape!=(record.length,width) or a.dtype!=np.float32
                        or not np.isfinite(a).all()):
                    raise ValueError("feature array contract mismatch")
            hashes[record.split,record.clip_id]=receipt["feature_hash"]
            sources[record.split,record.clip_id]=receipt["source_hash"]
        feature_hashes[kind]=hashes
        source_hashes[kind]=sources
    if source_hashes["spatial"]!=source_hashes["motion"]:
        raise ValueError("spatial/motion features originate from different actual frame bytes")
    datasets={}
    ids={}
    seen=set()
    for split,n in (("train",7096),("dev",519),("test",642)):
        annotation=ANNOTATIONS/f"{split}_info_ml.npy"
        dataset=Phoenix14T(annotation,split,SPATIAL,MOTION,manifests["spatial"],manifests["motion"],
                           spatial_crop_mode="full")
        clip_ids=tuple(r["fileid"] for r in dataset.records)
        if (len(clip_ids)!=n or len(set(clip_ids))!=n or seen.intersection(clip_ids)
                or {r["signer"] for r in dataset.records}!={f"Signer{i:02}" for i in range(1,10)}):
            raise ValueError("official split IDs/signers mismatch")
        for clip in clip_ids:
            s=manifests["spatial"].require(split,clip)
            m=manifests["motion"].require(split,clip)
            if m.length!=max(1,(s.length-16)//8+1):
                raise ValueError("spatial/motion full-source timing mismatch")
        seen.update(clip_ids)
        datasets[split],ids[split]=dataset,clip_ids
        bindings[str(annotation)]=sha256_file(annotation)
    return dict(datasets=datasets,ids=ids,bindings=bindings,feature_hashes=feature_hashes,
                identity=dict(bindings=bindings,ordered_ids={s:digest(v) for s,v in ids.items()},
                              feature_hashes_digest=digest({kind:{f"{s}/{i}":sha for (s,i),sha
                                    in h.items()} for kind,h in feature_hashes.items()})))


def batch_for(admission: dict,split: str,clip_ids: tuple[str,...]):
    ds=admission["datasets"][split]
    lookup={i:n for n,i in enumerate(admission["ids"][split])}
    samples=[]
    for clip in clip_ids:
        for kind,root in (("spatial",SPATIAL),("motion",MOTION)):
            if sha256_file(root/split/f"{clip}.npy")!=admission["feature_hashes"][kind][split,clip]:
                raise ValueError("feature changed during training load")
        samples.append(ds[lookup[clip]])
    return collate_phoenix(samples)


def eval_batches(admission: dict,split: str,batch_size: int=4):
    ids=admission["ids"][split]
    for offset in range(0,len(ids),batch_size):
        yield batch_for(admission,split,ids[offset:offset+batch_size])
