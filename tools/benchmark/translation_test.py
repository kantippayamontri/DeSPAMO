"""Once-only logical official-test scoring from six frozen dev selections."""

import argparse
import json
from pathlib import Path

import torch

from despamo.evaluation.metrics import evaluate_translations
from tools.benchmark.compact_checkpoint import load_snapshot
from tools.benchmark.official_translation import BASE, CACHE
from tools.benchmark.translation_data import admit_data, batch_for
from tools.benchmark.translation_model import build_model, cuda_batch, optimizer_for
from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json, writer


def score_run(root: Path) -> dict:
    selected=json.loads((root/"selected-checkpoint.json").read_text())
    complete=json.loads((root/"complete.json").read_text())
    frozen=json.loads((BASE/"controller/frozen-selections.json").read_text())
    expected=next((r for r in frozen if r["run_root"]==str(root)),None)
    if expected is None or expected["selection_hash"]!=sha256_file(root/"selected-checkpoint.json"):
        raise ValueError("test requires exact frozen dev selection")
    identity=selected["identity"]
    if complete["status"]!="complete" or complete["selected"]!=selected:
        raise ValueError("completed selected run required")
    admission=admit_data()
    if admission["identity"]!=identity["data"]:
        raise ValueError("test data identity changed")
    with writer(root/"official-test"):
        report_path=root/"official-test/report.json"
        claim=root/"official-test/claim.json"
        binding=dict(selection_hash=expected["selection_hash"],
                     checkpoint_hash=selected["checkpoint_hash"],identity=identity)
        if claim.exists():
            old=json.loads(claim.read_text())
            if old["binding"]!=binding:
                raise ValueError("once-only test claim conflicts with checkpoint")
            if old["status"]=="complete":
                if sha256_file(report_path)!=old["report_sha256"]:
                    raise ValueError("completed test report changed")
                return json.loads(report_path.read_text())
        else:
            atomic_json(claim,dict(binding=binding,status="in_progress"))
        model,base_hash=build_model(identity["seed"],identity["arm"],CACHE)
        optimizer=optimizer_for(model)
        load_snapshot(root/selected["checkpoint"],model,optimizer,identity,
                       base_hash=base_hash,replay=False)
        model.to("cuda").eval()
        ids=admission["ids"]["test"]
        items=[]
        for offset in range(0,len(ids),4):
            part=root/"official-test"/f"batch-{offset}.json"
            batch_ids=ids[offset:offset+4]
            source_batch=batch_for(admission,"test",batch_ids)
            if part.exists():
                rows=json.loads(part.read_text())
                if (tuple(r["clip_id"] for r in rows)!=batch_ids
                        or [r["reference"] for r in rows]!=[t.lower() for t in source_batch.texts]):
                    raise ValueError("test journal clip order changed")
            else:
                batch=cuda_batch(source_batch)
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    visual,mask=model.visual_adapter(batch.spatial,batch.spatial_mask,
                                                      batch.motion,batch.motion_mask)
                    predictions=model.language_model.generate_text(visual,mask,
                        [model.prompt_template.format("German")]*len(batch_ids),"deterministic",5,64)
                rows=[dict(clip_id=i,prediction=p,reference=r.lower())
                      for i,p,r in zip(batch_ids,predictions,batch.texts,strict=True)]
                atomic_json(part,rows)
            items.extend(rows)
        if tuple(r["clip_id"] for r in items)!=ids:
            raise ValueError("complete ordered642test clips required")
        metrics=evaluate_translations([r["prediction"] for r in items],
                                      [r["reference"] for r in items])
        report=dict(split="test",items=items,metrics=metrics,selection=selected,
                    decoder=dict(beam_size=5,max_length=64,in_context=False))
        atomic_json(report_path,report)
        atomic_json(claim,dict(binding=binding,status="complete",report_sha256=sha256_file(report_path)))
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--authorize-official-test",action="store_true")
    args=parser.parse_args(argv)
    if not args.authorize_official_test:
        raise ValueError("official test authorization required")
    report=score_run(args.run)
    print(json.dumps(dict(status="complete",metrics=report["metrics"])),flush=True)


if __name__=="__main__":
    main()
