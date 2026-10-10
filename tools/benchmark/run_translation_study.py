"""Sequential six-run official40epoch benchmark with frozen once-only test claims."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from tools.benchmark.official_translation import ARCHIVE, BASE
from tools.dinov3.identity import sha256_file
from tools.dinov3.official_archive import _archive_json, _copy_file, require_archive_storage
from tools.dinov3.official_runtime import process_owner
from tools.dinov3.storage import atomic_json, writer

PROJECT=Path("/home/kan/Research/DeSpaMo")
WORKTREE=Path(__file__).resolve().parents[2]
ARMS=("reference","aligned_fixed","aligned_learned")


def choose_batch(profiles: dict) -> int:
    for batch in (8,4):
        p=profiles.get(batch,{})
        if (p.get("status")=="complete" and p.get("generation_equal") is True
                and p.get("gpu_free_bytes",0)>=2*1024**3
                and p["results"]["long"]["peak_reserved_bytes"]<=21*1024**3):
            return batch
    raise ValueError("neither profiled batch satisfies generation/memory margin")


def validate_selection_inventory(rows: list[dict]):
    if (len(rows)!=6 or {(r["seed"],r["arm"]) for r in rows}!={
            (s,a) for s in (0,1) for a in ARMS}
            or any(not r.get("selection_hash") for r in rows)):
        raise ValueError("exact six frozen dev selections required")


def archive_final_payload(run: Path) -> Path:
    if not (run/"complete.json").is_file() or not (run/"official-test/report.json").is_file():
        raise ValueError("completed training and test report required for final archival")
    files={str(p.relative_to(run)):dict(sha256=sha256_file(p),bytes=p.stat().st_size)
           for p in run.rglob("*") if p.is_file() and p.name!=".writer.lock"
           and "archive-receipts" not in p.parts}
    require_archive_storage(ARCHIVE,sum(int(r["bytes"]) for r in files.values()))
    dest=ARCHIVE/run.name/"final-payload"
    if not (dest/"archive-manifest.json").exists():
        for name in files:
            _copy_file(run/name,dest/name)
        for name,r in files.items():
            if sha256_file(dest/name)!=r["sha256"]:
                raise ValueError("final archive payload hash mismatch")
        _archive_json(dest/"archive-manifest.json",dict(files=files,source_root=str(run)))
    recorded=json.loads((dest/"archive-manifest.json").read_text())
    if recorded["files"]!=files:
        raise ValueError("final archive inventory changed")
    for name,r in files.items():
        if sha256_file(dest/name)!=r["sha256"]:
            raise ValueError("final archive bytes changed")
    atomic_json(run/"archive-receipts/final.json",dict(manifest=str(dest/"archive-manifest.json"),
                sha256=sha256_file(dest/"archive-manifest.json")))
    return dest/"archive-manifest.json"


def execute_stage(state: dict,path: Path,name: str,command: list[str],env: dict):
    old=state["stages"].get(name)
    if old:
        if old["command"]!=command:
            raise ValueError("durable study stage command changed")
        if old["status"]=="complete":
            return
        if old["status"]=="running" and process_owner(old["owner"]["pid"]) is not None:
            raise RuntimeError("prior owned study child still live")
    log=path.parent/f"{name}.log"
    with log.open("a") as handle:
        child=subprocess.Popen(command,cwd=WORKTREE,env=env,stdout=handle,
                                stderr=subprocess.STDOUT,start_new_session=True)
        owner=process_owner(child.pid)
        while owner is not None and owner["command"]!=command:
            time.sleep(.001)
            owner=process_owner(child.pid)
        if owner is None:
            raise RuntimeError("child exited before ownership publication")
        stage: dict=dict(status="running",owner=owner,command=command)
        state["stages"][name]=stage
        atomic_json(path,state)
        while child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            stage["last_wall"]=time.time()
            atomic_json(path,state)
    stage.update(status="complete" if child.returncode==0 else "failed",exit_code=child.returncode)
    atomic_json(path,state)
    if child.returncode:
        raise RuntimeError(f"stage {name} failed; inspect {log}")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size",type=int,choices=(4,8),required=True)
    parser.add_argument("--authorize-study",action="store_true")
    args=parser.parse_args(argv)
    if not args.authorize_study:
        raise ValueError("explicit six-run benchmark authorization required")
    python=str(PROJECT/".venv/bin/python")
    env={**os.environ,"PYTHONPATH":f"{PROJECT/'src'}:{WORKTREE}:{PROJECT}",
         "LD_LIBRARY_PATH":"/usr/lib/wsl/lib","HF_HUB_OFFLINE":"1","OMP_NUM_THREADS":"2",
         "PYTHONDONTWRITEBYTECODE":"1","CUDA_VISIBLE_DEVICES":"0"}
    root=BASE/"controller"
    profiles={b:json.loads((BASE/f"profiles/batch{b}/profile.json").read_text())
              for b in (4,8) if (BASE/f"profiles/batch{b}/profile.json").exists()}
    if choose_batch(profiles)!=args.batch_size:
        raise ValueError("study batch differs from accepted profile decision")
    with writer(root):
        path=root/"state.json"
        state: dict=json.loads(path.read_text()) if path.exists() else dict(stages={})
        if state.get("batch_size",args.batch_size)!=args.batch_size:
            raise ValueError("study batch identity changed")
        state.update(status="running",batch_size=args.batch_size,owner=process_owner(os.getpid()))
        atomic_json(path,state)
        try:
            for seed in (0,1):
                for arm in ARMS:
                    command=[python,"-B","-m","tools.benchmark.official_translation","--mode","run",
                             "--seed",str(seed),"--arm",arm,"--batch-size",str(args.batch_size),
                             "--authorize-translation"]
                    execute_stage(state,path,f"seed{seed}-{arm}",command,env)
            selections=[]
            for p in (BASE/"runs").glob('*/complete.json'):
                result=json.loads(p.read_text())
                identity=result["selected"]["identity"]
                selections.append(dict(seed=identity["seed"],arm=identity["arm"],
                    run_root=str(p.parent),selection_hash=sha256_file(p.parent/"selected-checkpoint.json")))
            validate_selection_inventory(selections)
            atomic_json(root/"frozen-selections.json",selections)
            for selected in selections:
                command=[python,"-B","-m","tools.benchmark.translation_test","--run",
                         selected["run_root"],"--authorize-official-test"]
                execute_stage(state,path,f"test-seed{selected['seed']}-{selected['arm']}",command,env)
            rows=[]
            for selected in selections:
                report=json.loads((Path(selected["run_root"])/"official-test/report.json").read_text())
                rows.append(dict(**selected,dev_metrics=report["selection"]["metrics"],
                                 test_metrics=report["metrics"],selected_epoch=report["selection"]["epoch"]))
            summary=dict(status="complete",protocol="official-mixed-signer",runs=rows,
                means={arm:{metric:sum(r["test_metrics"][metric] for r in rows if r['arm']==arm)/2
                            for metric in rows[0]["test_metrics"]} for arm in ARMS})
            atomic_json(BASE/"summary.json",summary)
            for selected in selections:
                archive_final_payload(Path(selected["run_root"]))
            state.update(status="complete",summary_sha256=sha256_file(BASE/"summary.json"))
            atomic_json(path,state)
        except BaseException as error:
            state.update(status="failed",error=str(error))
            atomic_json(path,state)
            raise


if __name__=="__main__":
    main()
