"""Sequential detached all-split preparation: epoch40 DINO, then matching motion."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from tools.dinov3.official_runtime import process_owner
from tools.dinov3.storage import atomic_json, writer

PROJECT=Path("/home/kan/Research/DeSpaMo")
WORKTREE=Path(__file__).resolve().parents[2]
DATA=Path("/home/kan/datasets/despamo")
ADAPT=DATA/"official-phoenix14t-dino-adaptation-v1/runs/official-dino-adapt-bfea762ea2f71c244c9f3e2ef5f6be1a0f10b48bde2a5c6f05dcc8ba0faf5f0d"
ANNOTATION=Path("/home/kan/Research/SpaMo/preprocess/Phoenix14T")
ROWMAP=DATA/"features/1e84e43d142abc6242cce75c3e65abe9e1e03a6c61e3770181d7be381f3bc31d/complete/frame_rows.json"
PARENT=DATA/"signer-pilot/motion-features-fb35379b1e2ea0c4d19a186cd290952b512eff270f5394540c902f344eae842c"
OUTPUT=DATA/"official-phoenix14t-features-v1"


def stage_command(kind: str,mode: str) -> list[str]:
    if kind not in {"spatial","motion"} or mode not in {"preflight","smoke","run","verify"}:
        raise ValueError("unknown preparation stage")
    python=PROJECT/("tools/dinov3/.venv/bin/python" if kind=="spatial"
                    else "tools/motion/.venv/bin/python")
    cmd=[str(python),"-B","-m",f"tools.benchmark.official_{kind}","--mode",mode,
         "--annotation-root",str(ANNOTATION),"--row-map",str(ROWMAP),
         "--frames-root",str(DATA/"phoenix-frames-210x260px"),"--output-base",str(OUTPUT)]
    if kind=="spatial":
        cmd += ["--adapt-run",str(ADAPT),"--cache",str(DATA/"hf_cache")]
    else:
        cmd += ["--parent-motion",str(PARENT)]
    if mode in {"smoke","run"}:
        cmd.append("--authorize-extraction")
    return cmd


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-preparation",action="store_true")
    args=parser.parse_args(argv)
    if not args.authorize_preparation:
        raise ValueError("explicit preparation authorization required")
    env={**os.environ,"PYTHONPATH":f"{PROJECT/'src'}:{WORKTREE}:{PROJECT}",
         "LD_LIBRARY_PATH":"/usr/lib/wsl/lib","HF_HUB_OFFLINE":"1","OMP_NUM_THREADS":"2",
         "PYTHONDONTWRITEBYTECODE":"1","CUDA_VISIBLE_DEVICES":"0"}
    controller=OUTPUT/"controller"
    with writer(controller):
        path=controller/"state.json"
        state: dict=(json.loads(path.read_text()) if path.exists()
                     else dict(status="created",stages={}))
        for _name,stage in state["stages"].items():
            if stage["status"]=="running" and process_owner(stage["owner"]["pid"]) is not None:
                raise RuntimeError("owned prior child still active; do not duplicate preparation")
        state.update(status="running",owner=process_owner(os.getpid()))
        atomic_json(path,state)
        try:
            for kind in ["spatial","motion"]:
                name=f"{kind}-run"
                old=state["stages"].get(name)
                if old and old["status"]=="complete":
                    continue
                command=stage_command(kind,"run")
                stage: dict=dict(status="running",command=command,started_wall=time.time())
                log=controller/f"{name}.log"
                with log.open("a") as handle:
                    child=subprocess.Popen(command,cwd=WORKTREE,env=env,stdout=handle,
                                            stderr=subprocess.STDOUT,start_new_session=True)
                    stage["owner"]=process_owner(child.pid)
                    state["stages"][name]=stage
                    atomic_json(path,state)
                    while child.poll() is None:
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                        stage["last_seen_wall"]=time.time()
                        atomic_json(path,state)
                stage.update(status="complete" if child.returncode==0 else "failed",
                             exit_code=child.returncode,ended_wall=time.time())
                atomic_json(path,state)
                if child.returncode:
                    raise RuntimeError(f"stage {name} failed; inspect {log}")
            state["status"]="complete"
            atomic_json(path,state)
            print(json.dumps(state),flush=True)
        except BaseException as error:
            state.update(status="failed",error=str(error))
            atomic_json(path,state)
            raise


if __name__=="__main__":
    main()
