"""Detached20-minute health checks and strictly verified transient recovery."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

PROJECT=Path("/home/kan/Research/DeSpaMo")
WORKTREE=PROJECT/".superpowers/sdd/2026-10-07-official-phoenix14t-dino-adaptation/worktree"
STUDY=Path("/home/kan/datasets/despamo/official-phoenix14t-translation-v1")
COMMAND=[str(PROJECT/".venv/bin/python"),"-B","-m","tools.benchmark.run_translation_study",
         "--batch-size","8","--authorize-study"]
HASH_CACHE={}


def read_owner(pid: int) -> dict | None:
    try:
        proc=Path('/proc')/str(pid)
        fields=(proc/'stat').read_text().rsplit(') ',1)[1].split()
        if fields[0]=='Z':
            return None
        return dict(pid=pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    start_ticks=fields[19],command=(proc/'cmdline').read_bytes().decode().rstrip(
                        '\0').split('\0'))
    except (OSError,UnicodeError):
        return None


def owner_matches(actual: dict,expected: dict,command: list[str] | None=None) -> bool:
    return all(actual.get(k)==expected.get(k) for k in ('pid','boot_id','start_ticks')) and (
        actual['command']==(command or expected['command']))


def tail_text(path: Path) -> str:
    if not path.is_file():
        return ''
    with path.open('rb') as handle:
        handle.seek(max(0,path.stat().st_size-65536))
        return handle.read().decode(errors='replace')


def inspect_study(root: Path,owner_reader=read_owner) -> dict:
    state=json.loads((root/'controller/state.json').read_text())
    result=dict(controller_status=state['status'],restart_allowed=False)
    if state.get('batch_size')!=8:
        return dict(**result,state='recipe_mismatch')
    if state['status']=='complete':
        return dict(**result,state='complete')
    expected=state.get('owner')
    actual=owner_reader(expected['pid']) if expected else None
    if actual is not None and not owner_matches(actual,expected):
        return dict(**result,state='owner_mismatch')
    workers=[]
    for name,stage in state.get('stages',{}).items():
        if stage['status']!='running' or stage.get('owner') is None:
            continue
        expected_worker=stage['owner']
        actual_worker=owner_reader(expected_worker['pid'])
        if actual_worker is not None:
            if not owner_matches(actual_worker,expected_worker,stage['command']):
                return dict(**result,state='owner_mismatch',stage=name)
            workers.append(name)
    result['active_stages']=workers
    if actual is not None:
        return dict(**result,state='running')
    if workers:
        return dict(**result,state='orphan_worker_active')
    if state['status']=='running':
        result['restart_allowed']=True
        return dict(**result,state='controller_missing')
    error=state.get('error','')
    for name,stage in state.get('stages',{}).items():
        if stage['status']=='failed':
            text=tail_text(root/'controller'/f'{name}.log')
            if 'Traceback' in text:
                text=text[text.rfind('Traceback'):]
            error+='\n'+text
    transient=re.search(r'KeyboardInterrupt|signal15|ConnectionResetError|BrokenPipeError|'
                        r'Input/output error|Errno 5|CUDA out of memory|CUDA driver error',error)
    if transient:
        result['restart_allowed']=True
        return dict(**result,state='transient_failure',error=error[-4000:])
    return dict(**result,state='needs_agent',error=error[-4000:])


def _sha(path: Path) -> str:
    st=path.stat()
    signature=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    previous=HASH_CACHE.get(str(path))
    if previous and previous[0]==signature:
        return previous[1]
    hasher=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            hasher.update(chunk)
    after=path.stat()
    if signature!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('input changed during watchdog verification')
    HASH_CACHE[str(path)]=(signature,hasher.hexdigest())
    return hasher.hexdigest()


def recovery_ready(root: Path) -> dict:
    state=json.loads((root/'controller/state.json').read_text())
    if state['batch_size']!=8:
        raise ValueError('batch/recipe cannot change in recovery')
    budgets=list((root/'runs').glob('*/budget.json'))
    if not budgets:
        raise ValueError('no owned run identity available; agent diagnosis required')
    for path in budgets:
        budget=json.loads(path.read_text())
        identity=budget['identity']
        if any(version(name)!=expected for name,expected in identity['runtime'].items()):
            raise ValueError('locked runtime versions changed; no automatic fresh identity')
        recipe=identity['recipe']
        if (recipe['batch_size']!=8 or recipe['epochs']!=40 or recipe['total_steps']!=35480
                or recipe['dev_epochs']!=list(range(2,41,2))):
            raise ValueError('owned recipe changed')
        namespaces=[identity['code_hashes'],identity['base_files'],identity['data']['bindings']]
        for namespace in namespaces:
            for name,sha in namespace.items():
                if _sha(Path(name))!=sha:
                    raise ValueError(f'identity bytes changed: {name}')
        resumes=sorted(path.parent.glob('resume-epoch-*.pt'),
                       key=lambda p:int(p.stem.split('-')[-1]))
        for candidate in resumes:
            receipt=json.loads(candidate.with_suffix('.json').read_text())
            if (receipt['sha256']!=_sha(candidate) or receipt['bytes']!=candidate.stat().st_size
                    or receipt['identity']!=identity
                    or receipt['step']>budget['completed_updates']):
                raise ValueError('durable resume receipt/step mismatch')
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                 '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    if pids:
        raise RuntimeError('GPU compute owner still active; recovery waits')
    with (root/'controller/.writer.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
    free_c=int(subprocess.check_output(['powershell.exe','-NoProfile','-Command',
               '(Get-PSDrive -Name C).Free'],text=True,timeout=15).strip())
    free_b=int(subprocess.check_output(['powershell.exe','-NoProfile','-Command',
               '(Get-PSDrive -Name B).Free'],text=True,timeout=15).strip())
    if min(free_c,free_b,shutil.disk_usage(root).free)<56*1024**3:
        raise ValueError('backing space below required reserve; no automatic cleanup')
    return dict(verified_runs=len(budgets),gpu_idle=True,source_data_base_exact=True)


def atomic(path: Path,value: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,mode='w',delete=False) as handle:
        temp=Path(handle.name)
        json.dump(value,handle,sort_keys=True,indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def observations(root: Path) -> list[dict]:
    rows=[]
    for p in (root/'runs').glob('*/budget.json'):
        b=json.loads(p.read_text())
        i=b['identity']
        devs=[]
        for d in p.parent.glob('dev-epoch-*.json'):
            v=json.loads(d.read_text())
            devs.append(dict(epoch=v['epoch'],bleu4=v['metrics']['bleu4']))
        rows.append(dict(seed=i['seed'],arm=i['arm'],status=b['status'],step=b['completed_updates'],
                         epoch=b['completed_updates']/i['recipe']['steps_per_epoch'],
                         phase=b.get('phase'),last_ledger_wall=b['wall_timestamp'],
                         dev=sorted(devs,key=lambda v:v['epoch'])))
    return sorted(rows,key=lambda r:(r['seed'],r['arm']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-watchdog',action='store_true')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    if not args.authorize_watchdog:
        raise ValueError('explicit unattended watchdog authorization required')
    root=STUDY
    watch=root/'watchdog'
    watch.mkdir(parents=True,exist_ok=True)
    with (watch/'.watch.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        status_path=watch/'status.json'
        prior=json.loads(status_path.read_text()) if status_path.exists() else {}
        recoveries=prior.get('recoveries',[])
        while True:
            record=dict(checked_wall=time.time(),monitor_owner=read_owner(os.getpid()),
                        interval_seconds=1200,recoveries=recoveries)
            try:
                health=inspect_study(root)
                record.update(health=health,runs=observations(root))
                if health['restart_allowed']:
                    if len(recoveries)>=3:
                        record['action']='needs_agent_retry_limit'
                    else:
                        ready=recovery_ready(root)
                        # Recheck after potentially slow byte verification, never race a live owner.
                        current=inspect_study(root)
                        if not current['restart_allowed']:
                            raise RuntimeError('ownership changed while verifying recovery')
                        env={**os.environ,'PYTHONPATH':f'{PROJECT}/src:{WORKTREE}:{PROJECT}',
                             'LD_LIBRARY_PATH':'/usr/lib/wsl/lib','HF_HUB_OFFLINE':'1',
                             'OMP_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'0',
                             'PYTHONDONTWRITEBYTECODE':'1'}
                        with (watch/'recovery-controller.log').open('a') as handle:
                            child=subprocess.Popen(COMMAND,cwd=WORKTREE,env=env,stdout=handle,
                                stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
                        owner=read_owner(child.pid)
                        while owner is not None and owner['command']!=COMMAND:
                            time.sleep(.01)
                            owner=read_owner(child.pid)
                        if owner is None:
                            raise RuntimeError('recovery exited before ownership publication')
                        recoveries.append(dict(wall=time.time(),owner=owner,verification=ready))
                        record['action']='restarted_same_controller'
                else:
                    record['action']='observe_only'
            except Exception as error:
                record.update(action='needs_agent',error_type=type(error).__name__,error=str(error))
            done=record.get('health',{}).get('state')=='complete'
            record['next_check_wall']=None if done else time.time()+1200
            atomic(status_path,record)
            with (watch/'history.jsonl').open('a') as history:
                history.write(json.dumps(record,sort_keys=True)+'\n')
                history.flush()
                os.fsync(history.fileno())
            print(json.dumps(record,sort_keys=True),flush=True)
            if args.once or record['next_check_wall'] is None:
                break
            time.sleep(1200)


if __name__=='__main__':
    main()
