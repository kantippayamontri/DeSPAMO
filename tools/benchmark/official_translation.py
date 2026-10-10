"""Matched official translation training, compact candidates/resumes and profiles."""

import argparse
import gc
import json
import math
import signal
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import torch

from despamo.evaluation.signer_pilot import score_pilot_batches
from tools.benchmark.compact_checkpoint import (
    capture_rng,
    capture_state,
    common_fingerprint,
    load_snapshot,
    restore_rng,
    save_snapshot,
)
from tools.benchmark.feature_package import other_gpu_pids
from tools.benchmark.translation_data import admit_data, batch_for, eval_batches
from tools.benchmark.translation_model import (
    FLAN_REVISION,
    build_model,
    cuda_batch,
    optimizer_for,
    schedule,
    update,
)
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_archive import _archive_json, _copy_file, require_archive_storage
from tools.dinov3.official_runtime import OfficialLedger, require_official_storage
from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.dinov3.storage import _atomic, atomic_json, writer

PROJECT=Path("/home/kan/Research/DeSpaMo")
BASE=Path("/home/kan/datasets/despamo/official-phoenix14t-translation-v1")
ARCHIVE=Path("/mnt/b/DeSpaMo/archives/official-phoenix14t-translation-v1")
CACHE="/home/kan/datasets/spamo/hf_cache"


def select_report(reports: list[dict]) -> dict:
    if not reports or any(type(r["epoch"])is not int or not math.isfinite(
            r["metrics"]["bleu4"]) for r in reports):
        raise ValueError("finite dev reports required")
    return min(reports,key=lambda r:(-r["metrics"]["bleu4"],r["epoch"]))


def code_hashes() -> dict:
    here=Path(__file__).resolve().parent
    paths=list(here.glob("*translation*.py"))+[here/"compact_checkpoint.py"]
    paths += list((PROJECT/"src").rglob("*.py"))
    paths += [PROJECT/"uv.lock",PROJECT/"pyproject.toml",here.parent/"dinov3/official_sampler.py",
              here.parent/"dinov3/official_runtime.py",here.parent/"dinov3/storage.py"]
    return {str(p):sha256_file(p) for p in sorted(set(paths))}


def model_files() -> dict:
    snap=Path(CACHE)/"models--google--flan-t5-xl/snapshots"/FLAN_REVISION
    return {str(p):sha256_file(p) for p in snap.iterdir() if p.is_file()}


def candidate_to_archive(root: Path,state: dict,epoch: int) -> Path:
    dest=ARCHIVE/root.name/f"model-epoch-{epoch}.pt"
    scratch=root/f"candidate-epoch-{epoch}.pt"
    if dest.exists():
        if not dest.with_suffix(".json").exists():
            if not scratch.is_file() or not scratch.with_suffix('.json').is_file():
                raise ValueError("unreceipted archive lacks owned native source")
            native=json.loads(scratch.with_suffix('.json').read_text())
            if (native["identity"]!=state["identity"] or native["sha256"]!=sha256_file(scratch)
                    or native["sha256"]!=sha256_file(dest)):
                raise ValueError("pending archive bytes differ from native candidate")
            _archive_json(dest.with_suffix('.json'),native)
        receipt=json.loads(dest.with_suffix(".json").read_text())
        if receipt["sha256"]!=sha256_file(dest) or receipt["identity"]!=state["identity"]:
            raise ValueError("existing dev checkpoint archive changed")
        return dest
    require_archive_storage(ARCHIVE,512*1024**2)
    if scratch.exists():
        prior=torch.load(scratch,map_location="cpu",weights_only=False)
        if (prior["identity"]!=state["identity"] or prior["step"]!=state["step"]
                or any(not torch.equal(prior["weights"][n],v) for n,v in state["weights"].items())):
            raise ValueError("pending native candidate differs from durable replay")
    else:
        save_snapshot(scratch,state)
    receipt=json.loads(scratch.with_suffix(".json").read_text())
    _copy_file(scratch,dest)
    if sha256_file(dest)!=receipt["sha256"]:
        raise ValueError("candidate archive copy failed hash verification")
    _archive_json(dest.with_suffix(".json"),receipt)
    scratch.unlink()
    scratch.with_suffix(".json").unlink()
    return dest


def prune_resumes(root: Path):
    paths=sorted(root.glob("resume-epoch-*.pt"),key=lambda p:int(p.stem.split('-')[-1]))
    for p in paths:
        receipt=json.loads(p.with_suffix(".json").read_text())
        if receipt["sha256"]!=sha256_file(p):
            raise ValueError("resume receipt hash mismatch")
    for p in paths[:-2]:
        p.unlink()
        p.with_suffix(".json").unlink()


def _profile(admission: dict,batch_size: int,started: float) -> dict:
    recipe=schedule(batch_size)
    model,base_hash=build_model(0,"aligned_learned",CACHE)
    optimizer=optimizer_for(model)
    model.to("cuda").train()
    ids=admission["ids"]["train"]
    ordered=sorted(ids,key=lambda i:admission["datasets"]["train"].spatial_manifest.require(
        "train",i).length,reverse=True)
    root=BASE/"profiles"/f"batch{batch_size}"
    with writer(root):
        identity=dict(purpose="profile",batch_size=batch_size,data=admission["identity"],
                      source=code_hashes(),frozen_fingerprint=base_hash)
        results={}
        batch=None
        report: dict={}
        try:
            batches=(("ordinary",ids[:batch_size]),("long",tuple(ordered[:batch_size])))
            for label,chosen in batches:
                batch=cuda_batch(batch_for(admission,"train",tuple(chosen)))
                torch.cuda.reset_peak_memory_stats()
                durations=[]
                values: dict={}
                for j in range(4):
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    values=update(model,optimizer,batch,recipe["lr_warmup_steps"]+j,recipe,
                                  force_joint=True)
                    torch.cuda.synchronize()
                    if j>0:
                        durations.append(time.perf_counter()-start)
                states=sum(bool(optimizer.state.get(p)) for g in optimizer.param_groups
                           for p in g["params"])
                expected=sum(len(g["params"]) for g in optimizer.param_groups)
                if states!=expected:
                    raise ValueError("profile optimizer not fully initialized")
                results[label]=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=durations,
                    clips_per_second=batch_size/statistics.mean(durations),
                    optimizer_initialized=states,expected_optimizer_parameters=expected,loss=values)
            # Native compact checkpoint and fresh-model reload; identical deterministic generation.
            model.eval()
            dev_ids=sorted(admission["ids"]["dev"],key=lambda i:
                admission["datasets"]["dev"].spatial_manifest.require("dev",i).length,reverse=True)
            small=cuda_batch(batch_for(admission,"dev",tuple(dev_ids[:4])))
            def predict():
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                    visual,mask=model.visual_adapter(small.spatial,small.spatial_mask,
                                                      small.motion,small.motion_mask)
                    return model.language_model.generate_text(visual,mask,
                        [model.prompt_template.format("German")]*len(small.clip_ids),
                        "deterministic",5,64)
            expected_text=predict()
            state=capture_state(model,optimizer,identity,4,{"profile":True},base_hash=base_hash)
            path=root/"reload.pt"
            if path.exists():
                raise ValueError("profile artifacts already exist")
            receipt=save_snapshot(path,state)
            mutable_bytes=sum(v.numel()*v.element_size() for v in state["weights"].values())
            del state,optimizer,model,batch
            gc.collect()
            torch.cuda.empty_cache()
            model,fresh_base_hash=build_model(0,"aligned_learned",CACHE)
            if fresh_base_hash!=base_hash:
                raise ValueError("fresh pinned base differs for compact reload")
            optimizer=optimizer_for(model)
            model.to("cuda").eval()
            load_snapshot(path,model,optimizer,identity,base_hash=base_hash)
            if predict()!=expected_text:
                raise ValueError("compact generation reload differs")
            report=dict(status="complete",identity=identity,batch_size=batch_size,results=results,
                        snapshot_bytes=receipt["bytes"],mutable_bytes=mutable_bytes,
                        generation_equal=True,fresh_base_reload=True,
                        gpu_free_bytes=torch.cuda.mem_get_info()[0])
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            report=dict(status="oom",identity=identity,batch_size=batch_size,error=str(error),
                        results=results,peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report["charged_seconds"]=time.monotonic()-started
        atomic_json(root/"profile.json",report)
        print(json.dumps({k:v for k,v in report.items() if k in (
            "status","batch_size","results","snapshot_bytes","mutable_bytes","gpu_free_bytes")}),
            flush=True)
    return report


def reconcile_selection(root: Path,identity: dict) -> dict:
    reports=[json.loads(p.read_text()) for p in root.glob("dev-epoch-*.json")]
    if any(r.get("run_identity_hash")!=digest(identity) for r in reports):
        raise ValueError("dev report identity changed")
    best=select_report(reports)
    candidate=Path(best["checkpoint_path"])
    receipt=json.loads(candidate.with_suffix(".json").read_text())
    if receipt["identity"]!=identity or sha256_file(candidate)!=best["checkpoint_hash"]:
        raise ValueError("selected archive/checkpoint receipt changed")
    selected=root/f"selected-epoch-{best['epoch']}.pt"
    if selected.exists():
        if sha256_file(selected)!=best["checkpoint_hash"]:
            raise ValueError("native selected weights changed")
    else:
        _atomic(selected,lambda h:h.write(candidate.read_bytes()))
    atomic_json(selected.with_suffix(".json"),receipt)
    pointer=dict(epoch=best["epoch"],checkpoint=selected.name,
                 checkpoint_hash=best["checkpoint_hash"],metrics=best["metrics"],identity=identity)
    atomic_json(root/"selected-checkpoint.json",pointer)
    for old in root.glob("selected-epoch-*.pt"):
        if old!=selected:
            old.unlink()
            old.with_suffix(".json").unlink()
    return best


def _score_dev(model,optimizer,admission,identity,root,epoch,base_hash,sampler,recipe):
    report_path=root/f"dev-epoch-{epoch}.json"
    if report_path.exists():
        reconcile_selection(root,identity)
        return json.loads(report_path.read_text())
    state=capture_state(model,optimizer,identity,epoch*recipe["steps_per_epoch"],
                         sampler.state_dict(),base_hash=base_hash,full=False)
    candidate=candidate_to_archive(root,state,epoch)
    modes=[(m,m.training) for m in model.modules()]
    rng=capture_rng()
    prompt_rng=model.rng.getstate()
    try:
        report=score_pilot_batches(model,eval_batches(admission,"dev",4),
            admission["ids"]["dev"],"dev",identity["data"]["ordered_ids"]["dev"])
    finally:
        for m,flag in modes:
            m.training=flag
        restore_rng(rng)
        model.rng.setstate(prompt_rng)
    report.update(epoch=epoch,checkpoint_step=epoch*recipe["steps_per_epoch"],
                  checkpoint_path=str(candidate),checkpoint_hash=sha256_file(candidate),
                  run_identity_hash=digest(identity))
    atomic_json(report_path,report)
    best=reconcile_selection(root,identity)
    print(json.dumps(dict(dev_epoch=epoch,bleu4=report["metrics"]["bleu4"],
                          selected_epoch=best["epoch"])),flush=True)
    return report


def run(seed: int,arm: str,batch_size: int,mode: str) -> dict:
    started=time.monotonic()
    admission=admit_data()
    if other_gpu_pids():
        raise RuntimeError("GPU already owned")
    require_official_storage(BASE,6*1024**3)
    if mode=="profile":
        return _profile(admission,batch_size,started)
    recipe=schedule(batch_size)
    model,base_hash=build_model(seed,arm,CACHE)
    common=common_fingerprint(model)
    initialization_path=BASE/"common-initialization"/f"seed{seed}.json"
    with writer(initialization_path.parent):
        expected=dict(seed=seed,common_mutable_fingerprint=common,frozen_fingerprint=base_hash)
        if initialization_path.exists():
            if json.loads(initialization_path.read_text())!=expected:
                raise ValueError("matched-arm common initialization changed")
        else:
            atomic_json(initialization_path,expected)
    identity=dict(policy="official-plain-mlp-fusion-translation-v1",seed=seed,arm=arm,
        recipe=recipe,decoder=dict(beam=5,max_length=64,in_context=False),data=admission["identity"],
        code_hashes=code_hashes(),frozen_fingerprint=base_hash,flan_revision=FLAN_REVISION,
        common_initialization=common,
        base_files=model_files(),runtime={n:version(n) for n in ("torch","transformers","peft")})
    key=f"translation-{arm}-seed{seed}-{digest(identity)}"
    root=BASE/"runs"/key
    optimizer=optimizer_for(model)
    model.to("cuda").train()
    with writer(root):
        if (root/"complete.json").exists():
            return json.loads((root/"complete.json").read_text())
        atomic_json(root/"identity.json",identity)
        profile_charge=0.
        if seed==0 and arm=="reference":
            profile_charge=sum(json.loads(p.read_text()).get("charged_seconds",0.)
                               for p in (BASE/"profiles").glob('*/profile.json'))
        ledger=OfficialLedger(root,identity,initial_seconds=profile_charge,
                              setup_seconds=time.monotonic()-started)
        sampler=OfficialBatchSampler(admission["ids"]["train"],seed=seed,batch_size=batch_size)
        try:
            resumes=sorted(root.glob("resume-epoch-*.pt"),key=lambda p:int(p.stem.split('-')[-1]))
            if resumes:
                restored=load_snapshot(resumes[-1],model,optimizer,identity,base_hash=base_hash)
                if (restored["step"]>ledger.completed
                        or restored["sampler"]["batches_yielded"]!=restored["step"]
                        or restored["step"]%recipe["steps_per_epoch"]):
                    raise ValueError("checkpoint ahead of durable ledger")
                sampler.load_state_dict(restored["sampler"])
            ledger.charge(sampler.batches_yielded,"durable_replay",replay=True)
            epoch_done=sampler.batches_yielded//recipe["steps_per_epoch"]
            if epoch_done and epoch_done%2==0:
                _score_dev(model,optimizer,admission,identity,root,epoch_done,base_hash,sampler,recipe)
            last=time.monotonic()
            for k in range(sampler.batches_yielded+1,recipe["total_steps"]+1):
                batch=cuda_batch(batch_for(admission,"train",sampler.next_ids()))
                values=update(model,optimizer,batch,k,recipe)
                ledger.completed=k
                if k%25==0:
                    print(json.dumps(dict(epoch=k/recipe["steps_per_epoch"],**values)),flush=True)
                if time.monotonic()-last>=10:
                    ledger.charge(k,"train")
                    last=time.monotonic()
                if k%recipe["steps_per_epoch"]==0:
                    epoch=k//recipe["steps_per_epoch"]
                    if code_hashes()!=identity["code_hashes"]:
                        raise ValueError("translation source changed")
                    if any(sha256_file(Path(p))!=sha for p,sha in admission["bindings"].items()):
                        raise ValueError("annotation/feature metadata changed")
                    require_official_storage(BASE,6*1024**3)
                    ledger.charge(k,"resume_publication")
                    state=capture_state(model,optimizer,identity,k,sampler.state_dict(),
                                         base_hash=base_hash)
                    save_snapshot(root/f"resume-epoch-{epoch}.pt",state)
                    del state
                    prune_resumes(root)
                    if epoch%2==0:
                        ledger.charge(k,"dev_generation")
                        _score_dev(model,optimizer,admission,identity,root,epoch,base_hash,sampler,recipe)
                        ledger.charge(k,"dev_complete")
            reports=[json.loads(p.read_text()) for p in root.glob("dev-epoch-*.json")]
            if sorted(r["epoch"] for r in reports)!=recipe["dev_epochs"]:
                raise ValueError("complete20 dev reports required")
            best=select_report(reports)
            selected=json.loads((root/"selected-checkpoint.json").read_text())
            if selected["epoch"]!=best["epoch"]:
                raise ValueError("best checkpoint pointer mismatch")
            ledger.close("complete")
            result=dict(status="complete",seed=seed,arm=arm,run_root=str(root),selected=selected,
                        dev_reports={p.name:sha256_file(p) for p in root.glob("dev-epoch-*.json")})
            atomic_json(root/"complete.json",result)
            print(json.dumps(dict(status="complete",run_root=str(root),selected_epoch=best["epoch"])),
                  flush=True)
            return result
        except BaseException:
            ledger.close("interrupted")
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("profile","run"),required=True)
    parser.add_argument("--batch-size",type=int,choices=(4,8),required=True)
    parser.add_argument("--seed",type=int,choices=(0,1),default=0)
    parser.add_argument("--arm",choices=("reference","aligned_fixed","aligned_learned"),
                        default="reference")
    parser.add_argument("--authorize-translation",action="store_true")
    args=parser.parse_args(argv)
    if not args.authorize_translation:
        raise ValueError("explicit translation authorization required")
    def stop(signum,frame):
        raise KeyboardInterrupt(f"signal{signum}")
    signal.signal(signal.SIGTERM,stop)
    run(args.seed,args.arm,args.batch_size,args.mode)


if __name__=="__main__":
    main()
