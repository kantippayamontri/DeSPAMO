"""Durable, separately budgeted extraction-to-new-motion fusion coordinator."""

from __future__ import annotations

import argparse
import array
import ctypes
import fcntl
import json
import math
import os
import queue
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from scripts.run_projector_overnight import process_identity
from tools.motion.recovery import PIPELINE_NAME, pipeline_bootstrap, recovery_for

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/home/kan/datasets/despamo/signer-pilot")
DEFAULT_ROOT = BASE / "motion-fusion-pipeline-v1"
TRAINING_ROOT = BASE / "time-aligned-fusion-new-motion-study-v1"
POLICY = "motion-fusion-pipeline-v1"
NEW_POLICY = "time-aligned-fusion-new-motion-controller-v1"
CAPS = {"extraction": 18000, "training": 64800}
AUTHORIZATION = {"extraction_cap_seconds": 18000, "training_cap_seconds": 64800,
                 "cap_transfer": False}
GIB = 1024**3
RESULT_KEYS = frozenset("schema_version policy mode status featurekey identity root "
                       "bootstrap_root bootstrapkey binding_path input_observation_path "
                       "budget_path status_path profile_path spent_seconds remaining_seconds "
                       "authorization package profile error admission_seconds "
                       "terminal_admission".split())
IPC_MAX_FRAME = 65536
IPC_MAX_FDS = 16


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(cast(Any, value)) and cast(Any, value) >= 0


def safe(path: Path, *, file: bool = False) -> Path:
    if (not path.is_absolute() or path != path.absolute() or ".." in path.parts
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError(f"canonical non-aliased path required: {path}")
    if path.exists() and ((file and not path.is_file()) or path.stat().st_nlink > 1 and file):
        raise ValueError(f"unique regular file required: {path}")
    if file and not path.is_file():
        raise ValueError(f"missing file: {path}")
    return path


def pipeline_identity(args) -> dict:
    if not args.authorize_motion_fusion_pipeline:
        raise ValueError("--authorize-motion-fusion-pipeline required")
    paths = {name: str(safe(Path(getattr(args, name)))) for name in (
        "pipeline_root", "adapted_root", "annotation", "model_snapshot", "environment_lock")}
    paths.update(output_base=str(BASE), training_root=str(TRAINING_ROOT),
                 extractor_python=str(Path(args.extractor_python).absolute()))
    recovery = recovery_for(args)
    expected_root = BASE / PIPELINE_NAME if recovery is not None else DEFAULT_ROOT
    if (Path(paths["pipeline_root"]) != expected_root
            or Path(paths["adapted_root"]).parent != BASE
            or not Path(paths["adapted_root"]).name.startswith("e3-features-")
            or Path(paths["environment_lock"]) != ROOT / "tools/motion/uv.lock"
            or Path(args.extractor_python).absolute() != ROOT / "tools/motion/.venv/bin/python"):
        raise ValueError("fixed pipeline/input/environment roots required")
    # Source/environment setup evidence only. Dataset/model admission happens on
    # the charged CPU child, not in an unsupervised pre-clock parse.
    files = set((ROOT / "src").rglob("*.py")) | set((ROOT / "scripts").glob("*.py"))
    files.update((ROOT / "tools/motion").glob("*.py"))
    files.update(ROOT / n for n in ("pyproject.toml", "uv.lock", "tools/motion/pyproject.toml",
                                   "tools/motion/uv.lock"))
    if recovery is not None:
        files.add(Path(recovery["path"]))
    files.update((Path(sys.executable).resolve(), Path(paths["extractor_python"]).resolve()))
    sources = {str(safe(p, file=True)): file_hash(p) for p in sorted(files)}
    result = {"schema_version": 1, "policy": POLICY, "paths": paths, "sources": sources,
            "authorization": dict(AUTHORIZATION)}
    if recovery is not None:
        result["recovery"] = recovery
    return result


def stop_owned(receipt: dict, signum: int = signal.SIGKILL) -> None:
    current = process_identity(receipt["pid"])
    if current is None:
        return
    if receipt["boot_id"] != boot_id() or current != receipt["identity"]:
        raise ValueError("PID/boot/command ownership mismatch; refusing signal")
    # A session/group leader owns this separately sessioned subtree. For another
    # member, signal only its verified PID, never a foreign shared group.
    try:
        if current["pgid"] == current["session"] == receipt["pid"]:
            os.killpg(receipt["pid"], signum)
        else:
            os.kill(receipt["pid"], signum)
    except ProcessLookupError:
        pass


def descendants(parent: int) -> dict[int, dict]:
    rows = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            identity = process_identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if identity is not None:
            rows[int(path.name)] = (int(fields[1]), identity)
    owned = {parent}
    while True:
        added = {pid for pid, (ppid, _) in rows.items() if ppid in owned} - owned
        if not added:
            break
        owned.update(added)
    return {pid: {"pid": pid, "boot_id": boot_id(), "identity": rows[pid][1]}
            for pid in owned - {parent}}


def _budget_monitor(request: dict, control: int, ack: int) -> int:
    """Parent cumulative watchdog; deadline loop performs only pipe and /proc IO."""
    parent = request["parent"]
    soft, hard = request["soft_deadline"], request["hard_deadline"]
    os.write(ack, b"1")
    os.close(ack)
    pending = b""
    notified = False
    phase = "extraction"
    training_hard = None
    consumer = False
    worker = request.get("owned_worker")
    failure_writer = (ReceiptWriter(Path(request["failure_path"]))
                      if request.get("failure_path") else None)
    failure_sequence = 0
    seal_writer = (ReceiptWriter(Path(request["seal_path"]))
                   if request.get("seal_path") else None)
    seal_sequence = 0
    sealing = False
    releasing = False
    caller_disconnected = False
    while True:
        now = time.monotonic()
        if (caller_disconnected or parent["boot_id"] != boot_id()
                or process_identity(parent["pid"]) != parent["identity"]):
            if worker is not None:
                stop_owned(worker)
            if failure_writer is None:
                return 1
            if not failure_sequence:
                failure_sequence = failure_writer.submit({"status": "failed",
                    "nonce": request["nonce"], "reason": "caller disconnected",
                    "observed_monotonic": now, "boot_id": boot_id()})
            if failure_writer.persisted >= failure_sequence:
                return 1
            time.sleep(.01)
            continue
        if now >= soft and not notified:
            notified = True
            os.kill(parent["pid"], signal.SIGTERM)
        if now >= hard:
            if worker is not None:
                stop_owned(worker)
            if failure_writer is not None and not failure_sequence:
                failure_sequence = failure_writer.submit({"status": "capped",
                    "nonce": request["nonce"], "reason": "external closing/phase deadline",
                    "observed_monotonic": now, "boot_id": boot_id()})
            os.kill(parent["pid"], signal.SIGKILL)
            if failure_writer is None or failure_writer.persisted >= failure_sequence:
                return 1
        if select.select([control], [], [], min(.01, max(.001, soft - now)))[0]:
            data = os.read(control, 4096)
            if not data:
                # EOF may race the identity observation above. Custody follows
                # the same verified stop/durable-failure path as observed loss.
                caller_disconnected = True
                continue
            pending += data
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if line == b"A":
                    return 1 if notified or time.monotonic() >= hard else 0
                limit = json.loads(line)
                if limit.get("action") == "seal":
                    if (seal_writer is None or limit["proof"]["worker"] != worker
                            or limit["proof"]["worker_returncode"] != 0 or notified):
                        raise ValueError("external successful worker closure required")
                    seal_sequence = seal_writer.submit({"proof": limit["proof"],
                        "caller_cleanup_checked_monotonic": now, "boot_id": boot_id()})
                    sealing = True
                elif limit.get("action") == "release":
                    releasing = True
                elif limit.get("phase") == "training":
                    if phase != "extraction" or now >= hard:
                        raise ValueError("phase cap cannot reset or borrow")
                    if not finite(limit.get("extraction_seconds")) or limit[
                            "extraction_seconds"] >= CAPS["extraction"]:
                        raise ValueError("extraction phase exceeded authorization")
                    phase = "training"
                    hard = min(limit["hard_deadline"], now + CAPS["training"])
                    training_hard = hard
                    soft = min(limit["soft_deadline"], hard)
                elif limit.get("action") == "worker-exited":
                    if (consumer or phase != "training" or training_hard is None
                            or worker is None or process_identity(worker["pid"]) is not None
                            or limit["worker"] != worker or limit["returncode"] != 0
                            or now >= hard):
                        raise ValueError("consumer continuation requires bounded worker exit")
                    consumer = True
                    hard = min(training_hard, limit["hard_deadline"])
                    soft = min(hard, limit["soft_deadline"])
                else:
                    soft = min(soft, limit["soft_deadline"])
                    hard = min(hard, limit["hard_deadline"])
        if (releasing and sealing and seal_writer is not None
                and seal_writer.persisted >= seal_sequence):
            return 0 if not notified and time.monotonic() < hard else 1


class ReceiptWriter:
    """Persistence worker; never owns deadline or descendant termination decisions."""

    def __init__(self, path: Path):
        self.path = path
        self.pending: queue.SimpleQueue = queue.SimpleQueue()
        self.sequence = 0
        self.persisted = 0
        self.error: BaseException | None = None
        threading.Thread(target=self.write, daemon=True).start()

    def submit(self, value: dict) -> int:
        self.sequence += 1
        self.pending.put((self.sequence, json.loads(json.dumps(value))))
        return self.sequence

    def write(self) -> None:
        while True:
            sequence, value = self.pending.get()
            try:
                atomic_json(self.path, value)
                self.persisted = sequence
            except BaseException as error:
                self.error = error
                return


def guard_message(fd: int, command: str, value: dict | None = None) -> None:
    payload = command + (json.dumps(value) if value is not None else "") + "\n"
    os.write(fd, payload.encode())


def _ipc_ready(channel: socket.socket, deadline: float, *, writing: bool = False) -> None:
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("worker IPC frame deadline exceeded")
        read, write, _ = select.select([] if writing else [channel],
                                      [channel] if writing else [], [], min(.05, remaining))
        if (write if writing else read):
            return


def send_worker_frame(channel: socket.socket, packet: dict, deadline: float,
                      fds: list[int] | None = None) -> None:
    """One bounded stream frame; descriptor rights accompany its first header byte."""
    body = json.dumps(packet, allow_nan=False).encode()
    if not 0 < len(body) <= IPC_MAX_FRAME or len(fds or []) > IPC_MAX_FDS:
        raise ValueError("invalid worker IPC frame size/descriptor count")
    wire = struct.pack("!I", len(body)) + body
    ancillary = ([(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
                 if fds else [])
    offset = 0
    while offset < len(wire):
        _ipc_ready(channel, deadline, writing=True)
        try:
            written = channel.sendmsg([wire[offset:]], ancillary if offset == 0 else [],
                                      socket.MSG_DONTWAIT)
        except (BlockingIOError, InterruptedError):
            continue
        if written <= 0:
            raise ConnectionError("worker IPC stream closed during frame send")
        offset += written
    if time.monotonic() >= deadline:
        raise TimeoutError("worker IPC frame send exceeded deadline")


def receive_worker_frame(channel: socket.socket, deadline: float) -> tuple[dict, list[int]] | None:
    """EOF only between frames; every read collects ancillary data without overread."""
    rights: list[int] = []
    itemsize = array.array("i").itemsize

    def read_exact(size: int, *, header: bool = False) -> bytes | None:
        data = bytearray()
        while len(data) < size:
            _ipc_ready(channel, deadline)
            try:
                chunk, ancillary, flags, _ = channel.recvmsg(
                    size - len(data), socket.CMSG_SPACE(IPC_MAX_FDS * itemsize),
                    socket.MSG_DONTWAIT)
            except (BlockingIOError, InterruptedError):
                continue
            invalid = False
            for level, kind, value in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    integers = array.array("i")
                    integers.frombytes(value[:len(value) - len(value) % itemsize])
                    rights.extend(integers)
                    invalid = invalid or bool(len(value) % itemsize)
                else:
                    invalid = True
            if (invalid or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
                    or len(rights) > IPC_MAX_FDS
                    or ancillary and (not header or len(data) != 0)):
                raise ValueError("malformed/unexpected ancillary worker IPC frame data")
            if not chunk:
                if header and not data and not rights:
                    return None
                raise ValueError("truncated worker IPC header/payload frame")
            data.extend(chunk)
        return bytes(data)

    try:
        header = read_exact(4, header=True)
        if header is None:
            return None
        size = struct.unpack("!I", header)[0]
        if not 0 < size <= IPC_MAX_FRAME:
            raise ValueError("zero/oversize worker IPC frame")
        body = read_exact(size)
        if body is None:
            raise ValueError("truncated worker IPC payload frame")
        try:
            packet = json.loads(body)
        except (ValueError, UnicodeError) as error:
            raise ValueError("invalid JSON worker IPC frame") from error
        if not isinstance(packet, dict):
            raise ValueError("worker IPC frame must contain an object")
        if time.monotonic() >= deadline:
            raise TimeoutError("worker IPC frame receive exceeded deadline")
        result = packet, rights
        rights = []  # Ownership transfers only after a complete valid frame.
        return result
    finally:
        for fd in rights:
            os.close(fd)


def _watchdog(request_path: Path | dict, control: int, lock_fd: int) -> int:
    """Independent Linux subreaper: detached grandchildren remain owned children."""
    request = request_path if isinstance(request_path, dict) else read_json(request_path)
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    os.set_blocking(control, False)
    child = None
    known: dict[int, dict] = {}
    receipt_path = Path(request["receipt"])
    receipt = {"request_hash": digest(request), "boot_id": boot_id(),
               "watchdog": {"pid": os.getpid(), "boot_id": boot_id(),
                            "identity": process_identity(os.getpid())},
               "status": "armed", "returncode": None, "descendants": []}
    writer = ReceiptWriter(receipt_path)
    writer.submit(receipt)
    launches: queue.SimpleQueue = queue.SimpleQueue()
    soft, hard = request["soft_deadline"], request["hard_deadline"]
    pending = b""
    started = False
    timed_out = False
    notified = False
    accepted = False
    accepted_sequence = 0
    released = False
    stopping = False

    def launch():
        try:
            with Path(request["stdout"]).open("wb") as out, Path(
                    request["stderr"]).open("wb") as err:
                process = subprocess.Popen(request["command"], cwd=ROOT, env=request["env"],
                    stdout=out, stderr=err, start_new_session=True,
                    pass_fds=(lock_fd, *request.get("gpu_fds", []),
                              *request.get("handoff_fds", []), *request.get("training_fds", [])))
            launches.put((process, process_identity(process.pid), None))
        except BaseException as error:
            launches.put((None, None, error))

    try:
        while True:
            now = time.monotonic()
            parent = request["parent"]
            disconnected = (parent["boot_id"] != boot_id()
                            or process_identity(parent["pid"]) != parent["identity"])
            ready = select.select([control], [], [], .01)[0]
            if ready:
                data = os.read(control, 4096)
                disconnected = disconnected or not data
                pending += data
            while b"\n" in pending:
                message, pending = pending.split(b"\n", 1)
                if message == b"S" and not started and not disconnected:
                    started = True
                    threading.Thread(target=launch, daemon=True).start()
                elif message == b"X":
                    stopping = True
                elif message.startswith(b"D"):
                    limit = json.loads(message[1:])
                    soft = min(soft, limit["soft_deadline"])
                    hard = min(hard, limit["hard_deadline"])
                elif message == b"A":
                    if receipt["status"] != "returned" or receipt["returncode"] != 0:
                        stopping = True
                    else:
                        receipt.update(status="accepted", accepted_monotonic=time.monotonic())
                        accepted_sequence = writer.submit(receipt)
                        accepted = True
                elif message == b"R":
                    released = True
            if not launches.empty():
                child, identity, error = launches.get()
                if error is not None:
                    stopping = True
                    receipt["error"] = str(error)
                elif child is not None:
                    receipt.update(status="running", child_pid=child.pid,
                                   child={"pid": child.pid, "boot_id": boot_id(),
                                          "identity": identity})
                    writer.submit(receipt)
            known.update(descendants(os.getpid()))
            if child is not None and child.poll() is not None and receipt["returncode"] is None:
                # Reparented or separately sessioned descendants are not allowed
                # to keep computing after the direct command claims completion.
                live = {pid: r for pid, r in known.items() if process_identity(pid) is not None}
                for r in live.values():
                    stop_owned(r)
                receipt.update(returncode=child.returncode,
                               status="failed" if live else "returned",
                               descendants=list(known.values()),
                               returned_monotonic=time.monotonic())
                writer.submit(receipt)
            timed_out = timed_out or now >= soft
            if disconnected or stopping or timed_out or writer.error is not None:
                known.update(descendants(os.getpid()))
                for r in reversed(list(known.values())):
                    stop_owned(r)
                if receipt["status"] not in {"capped", "interrupted"}:
                    receipt.update(status="capped" if timed_out else "interrupted",
                                   returncode=None if child is None else child.poll(),
                                   descendants=list(known.values()))
                    writer.submit(receipt)
                live = descendants(os.getpid())
                if not live and (disconnected or stopping or writer.error is not None):
                    return 1
            if released and accepted and writer.persisted >= accepted_sequence:
                return 0 if not timed_out and time.monotonic() < hard else 1
            if timed_out and not notified:
                notified = True
                # Parent gets the entire termination reserve to reconcile and
                # abort. A stalled consumer is itself still deadline-bounded.
                if process_identity(parent["pid"]) == parent["identity"]:
                    os.kill(parent["pid"], signal.SIGTERM)
            if now >= hard:
                if process_identity(parent["pid"]) == parent["identity"]:
                    os.kill(parent["pid"], signal.SIGKILL)
                return 1
    finally:
        for r in reversed(list(descendants(os.getpid()).values())):
            stop_owned(r)
        os.close(control)


def delivered_result(path: Path) -> dict:
    # CLI may precede its final JSON with library diagnostics. Read the last
    # nonempty physical line; never hunt backward past a malformed last result.
    safe(path, file=True)
    last = None
    with path.open() as stream:
        for line in stream:
            if line.strip():
                last = line
    if last is None:
        raise ValueError("successful child delivered no JSON result")
    value = json.loads(last)
    if not isinstance(value, dict):
        raise ValueError("child result must be an object")
    return value


def require_gpu_idle() -> None:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                             "--format=csv,noheader,nounits"], check=True,
                            capture_output=True, text=True, timeout=5)
    if result.stdout.strip():
        raise ValueError("GPU has unowned computation")


def windows_free_bytes() -> int:
    command = ["powershell.exe", "-NoProfile", "-Command",
               "[Console]::Write((Get-PSDrive -Name C).Free)"]
    out = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    return int(out.stdout.strip())


def storage_admission(root: Path, runs: dict, *, linux_free: int | None = None,
                      windows_free: int | None = None, checkpoint_bytes: float = 5.6 * GIB,
                      extraction_bytes: float = 93232 * 1024 * 4,
                      environment_growth_bytes: float = 0) -> dict:
    from scripts.run_fusion_study import storage_projection

    projected = storage_projection(runs, checkpoint_bytes)
    if not finite(extraction_bytes) or not finite(environment_growth_bytes):
        raise ValueError("storage demand must be finite and nonnegative")
    required = (projected["required_free_bytes"] + extraction_bytes
                + 2 * GIB + environment_growth_bytes)
    linux = shutil.disk_usage(root).free if linux_free is None else linux_free
    windows = windows_free_bytes() if windows_free is None else windows_free
    if not finite(linux) or not finite(windows) or min(linux, windows) < required:
        raise ValueError("both Linux and Windows must fit projected demand plus reserves")
    return {"linux_free_bytes": linux, "windows_free_bytes": windows,
            "remaining_training_bytes": projected["remaining_demand_bytes"],
            "remaining_extraction_bytes": extraction_bytes,
            "environment_growth_bytes": environment_growth_bytes,
            "required_free_bytes": required}


def remaining_extraction_demand(root: Path, identity: dict, frames: dict) -> int:
    """Credit allocated, independently rebuilt owned features; never sparse lengths."""
    from tools.motion.records import make_clip_receipt

    demand = 93232 * 1024 * 4
    if not root.exists():
        return demand
    owner = read_json(safe(root / "owner.json", file=True))
    if owner.get("identity") != identity or owner.get("featurekey") != digest(identity):
        raise ValueError("foreign extraction allocation credit")
    credit = 0
    for receipt_path in (root / "receipts/train").glob("*.json"):
        receipt = read_json(safe(receipt_path, file=True))
        clip = receipt_path.stem
        if clip not in frames["clips"]:
            raise ValueError("foreign extraction receipt credit")
        array = safe(root / "train" / f"{clip}.npy", file=True)
        rebuilt = make_clip_receipt(clip, frames["clips"][clip], receipt["source_rows"],
                                    tuple(receipt["windows"]), array, identity)
        if rebuilt != receipt:
            raise ValueError("extraction allocation receipt/source/array drift")
        credit += min(len(receipt["windows"]) * 1024 * 4, array.stat().st_blocks * 512)
    return max(0, demand - credit)


# Executed only by the dedicated interpreter. Public producer validators remain
# authoritative; no copied clock validation and no mutating clock constructor.
CLOCK_QUERY = r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
from despamo.appearance.provenance import digest,file_hash,read_json
from tools.motion import extractor as e
request=read_json(Path(sys.argv[1]))
paths=request['identity']['paths']
args=SimpleNamespace(**{k:Path(paths[k]) for k in ('output_base','adapted_root','annotation',
    'model_snapshot','environment_lock')}, mode='run',gpu_cap_seconds=18000,
    authorize_motion_extraction=True,resume=True,
    recovery_record=request['identity'].get('recovery',{}).get('path'))
launch=e.bootstrap_launch_binding(args)
bootstrap=e.extraction_bootstrap_root(args)
result=read_json(Path(request['result_path']))
owner=read_json(bootstrap/'owner.json')
if owner != {'schema_version':1,'policy':'motion-extraction-bootstrap-owner-v1',
             'bootstrapkey':digest(launch),'launch':launch}:
    raise ValueError('bootstrap launch/source/interpreter mismatch')
raw=read_json(bootstrap/'budget.json')
if raw['policy']!='motion-extraction-clock-v2' or raw['authorization']!=e.AUTHORIZATION:
    raise ValueError('clock policy/cap mismatch')
outer=request['outer_receipt']; launched=request['outer_request']
if (outer['request_hash']!=digest(launched) or outer['returncode']!=0
    or outer['status']!='returned' or outer['boot_id']!=e.clocks()[2]
    or outer['child']['identity']['command']!=launched['command']):
    raise ValueError('successful matching bounded CLI launch required')
if result['terminal_admission'] is not None:
    child=outer['child']; process=child['identity']; active=raw['active']
    if (active['boot_id']!=child['boot_id']
        or active['owner']!={'pid':child['pid'],'start_ticks':process['start_ticks'],
            'pgid':process['pgid'],'command':'\x00'.join(process['command'])+'\x00'}):
        raise ValueError('terminal proof belongs to a foreign CLI process')
binding=read_json(bootstrap/'binding.json')
observation=read_json(bootstrap/'input-observation.json')
if observation != e.preparation_inputs(args):
    raise ValueError('input/model/lock/hardware observation drift')
if (binding != {'schema_version':1,'bootstrapkey':owner['bootstrapkey'],
                'featurekey':result['featurekey'],'root':result['root'],'identity':result['identity']}
    or binding['featurekey']!=digest(binding['identity'])
    or raw['bootstrapkey']!=owner['bootstrapkey']
    or raw['featurekey']!=binding['featurekey']
    or raw['binding']!={'featurekey':binding['featurekey'],'root':binding['root'],
                        'sha256':file_hash(bootstrap/'binding.json')}):
    raise ValueError('immutable binding mismatch')
root=Path(binding['root'])
if root != bootstrap.parent/('motion-features-'+binding['featurekey']):
    raise ValueError('foreign content root')
feature_owner=read_json(root/'owner.json')
if feature_owner != {'schema_version':1,'policy':'motion-extraction-owner-v2',
    'featurekey':binding['featurekey'],'identity':binding['identity'],
    'authorization':e.AUTHORIZATION,'bootstrap_root':str(bootstrap),
    'bootstrapkey':owner['bootstrapkey']}:
    raise ValueError('feature ownership mismatch')
terminal=raw['terminal_close']; nonce=terminal['nonce']
for key,name in (('candidate_path','terminal-candidate'),('receipt_path','terminal'),
                 ('admission_path','terminal-admission'),('failure_path','terminal-failure')):
    if Path(terminal[key]) != bootstrap/(name+'-'+nonce+'.json'):
        raise ValueError('foreign terminal path/nonce')
for key,name in (('bootstrap_root',''),('budget_path','budget.json'),
                 ('status_path','status.json'),('binding_path','binding.json'),
                 ('input_observation_path','input-observation.json')):
    if result[key] != str(bootstrap/name):
        raise ValueError('foreign result clock path')
if (result['bootstrapkey']!=owner['bootstrapkey']
    or result['profile_path']!=str(root/'profile.json')):
    raise ValueError('result bootstrap/profile mismatch')
if raw['status']!='terminal_pending':
    raise ValueError('raw complete without immutable admission is not success')
clock=e.ExtractionClock.__new__(e.ExtractionClock)
clock.path,clock.state=bootstrap/'budget.json',raw
receipt=e.validate_terminal_close(clock,allow_pending=True)
snapshot=e.admitted_terminal_snapshot(clock)
admission=read_json(Path(terminal['admission_path']))
if result['terminal_admission'] is not None and result['terminal_admission']!=admission:
    raise ValueError('returned admission proof differs from persisted proof')
if (snapshot['status']!=result['status'] or snapshot['spent_seconds']!=result['spent_seconds']
    or result['remaining_seconds']!=18000-snapshot['spent_seconds']):
    raise ValueError('returned projected clock mismatch')
e.verify_evidence(binding['identity'])
frames=read_json(Path(paths['adapted_root'])/'complete/frame_rows.json')
e.production_gate(frames,e.annotation_records(e.np.load(args.annotation,allow_pickle=True).item()))
profile=read_json(root/'profile.json')
if file_hash(root/'profile.json')!=raw['profile_sha256']:
    raise ValueError('profile hash mismatch')
samples=e.valid_profile_samples(root,binding['identity'],frames,required=e.profile_ids(frames))
if not samples:
    raise ValueError('durable profile evidence missing')
proof_paths=[clock.path,bootstrap/'status.json',bootstrap/'owner.json',
 bootstrap/'binding.json',bootstrap/'input-observation.json',root/'owner.json',
 root/'profile.json',Path(terminal['candidate_path']),Path(terminal['receipt_path']),
 Path(terminal['admission_path'])]
print(json.dumps({'status':snapshot['status'],'spent_seconds':snapshot['spent_seconds'],
 'featurekey':binding['featurekey'],'root':binding['root'],
 'bootstrapkey':owner['bootstrapkey'],'budget_sha256':file_hash(clock.path),
 'binding_sha256':file_hash(bootstrap/'binding.json'),
 'observation_sha256':file_hash(bootstrap/'input-observation.json'),
 'admission_sha256':file_hash(Path(terminal['admission_path'])),
 'terminal_sha256':file_hash(Path(terminal['receipt_path'])),
 'terminal_nonce':nonce,'hardware':observation['hardware'],
 'proof_bindings':{str(p):file_hash(p) for p in proof_paths},
 'failure_path':terminal['failure_path']}))
'''

PROJECTION_QUERY = r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
from despamo.appearance.provenance import digest,file_hash,read_json
from tools.motion import extractor as e
r=read_json(Path(sys.argv[1])); paths=r['identity']['paths']
args=SimpleNamespace(**{k:Path(paths[k]) for k in ('output_base','adapted_root','annotation',
    'model_snapshot','environment_lock')},mode='run',gpu_cap_seconds=18000,
    authorize_motion_extraction=True,resume=True,
    recovery_record=r['identity'].get('recovery',{}).get('path'))
launch=e.bootstrap_launch_binding(args)
b=e.extraction_bootstrap_root(args)
if not b.exists():
    print(json.dumps({'status':'absent','spent_seconds':0}))
    sys.exit(0)
owner=read_json(b/'owner.json')
if owner != {'schema_version':1,'policy':'motion-extraction-bootstrap-owner-v1',
             'bootstrapkey':digest(launch),'launch':launch}:
    raise ValueError('owned producer launch/source/interpreter mismatch')
raw=read_json(b/'budget.json')
if (raw['policy']!='motion-extraction-clock-v2' or raw['authorization']!=e.AUTHORIZATION
    or raw['bootstrapkey']!=digest(launch)):
    raise ValueError('producer clock cannot be replaced or reset')
if raw.get('binding') is not None:
    binding=read_json(b/'binding.json')
    identity=binding['identity']; key=digest(identity)
    if (identity!=r['expected_identity'] or binding['featurekey']!=key
        or binding['bootstrapkey']!=digest(launch)
        or binding['root']!=str(b.parent/('motion-features-'+key))
        or raw['binding']!={'featurekey':key,'root':binding['root'],
                           'sha256':file_hash(b/'binding.json')}):
        raise ValueError('producer feature/source binding drift')
    if read_json(b/'input-observation.json')!=e.preparation_inputs(args):
        raise ValueError('producer source/model/lock/hardware drift')
    e.verify_evidence(identity)
status=raw['status']; snapshot=raw
if status=='terminal_pending':
    t=raw['terminal_close']; nonce=t['nonce']
    for k,name in (('candidate_path','terminal-candidate'),('receipt_path','terminal'),
                   ('admission_path','terminal-admission'),('failure_path','terminal-failure')):
        if Path(t[k])!=b/(name+'-'+nonce+'.json'):
            raise ValueError('foreign producer terminal path')
    if Path(t['admission_path']).exists():
        clock=e.ExtractionClock.__new__(e.ExtractionClock)
        clock.path,clock.state=b/'budget.json',raw
        e.validate_terminal_close(clock,allow_pending=True)
        snapshot=e.admitted_terminal_snapshot(clock)
        status=snapshot['status']
    else:
        status='interrupted'
elif status in {'complete','profiled'}:
    raise ValueError('producer complete/profiled label lacks immutable admission')
spent=snapshot['spent_seconds']
if type(spent) not in (float,int) or not e.math.isfinite(spent) or spent<0:
    raise ValueError('invalid producer cumulative spend')
print(json.dumps({'status':status,'spent_seconds':spent,'bootstrapkey':digest(launch),
                  'featurekey':raw.get('featurekey'),'budget_sha256':file_hash(b/'budget.json')}))
'''


def bounded_cpu_query(python: str, code: str, request_path: Path, timeout: float = 300,
                      *, pass_fds: tuple[int, ...] = ()) -> dict:
    """Read-only helper inside an outer owned watchdog; never starts model work."""
    child = subprocess.Popen([python, "-c", code, str(request_path)], cwd=ROOT,
                             env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=True, pass_fds=pass_fds)
    receipt = {"pid": child.pid, "boot_id": boot_id(), "identity": process_identity(child.pid)}
    try:
        out, err = child.communicate(timeout=timeout)
    except BaseException:
        if receipt["identity"] is not None:
            stop_owned(receipt)
        child.wait(timeout=5)
        raise
    if child.returncode != 0:
        raise ValueError(f"actual producer CPU query rejected: {err[-2000:]}")
    value = json.loads(out.splitlines()[-1])
    if not isinstance(value, dict):
        raise ValueError("CPU query result object required")
    return value


def validate_extraction_handoff(cache_root: Path, expected: dict) -> dict:
    """Independent full byte admission, with actual producer clock query bounded by caller."""
    from scripts import train_fusion_study as runner
    from tools.motion.package import validate_motion_package

    identity = expected["identity"]
    result = read_json(Path(expected["result_path"]))
    if (set(result) != RESULT_KEYS or result["status"] != "complete"
            or result["mode"] != "run" or result["package"] is None
            or result["authorization"] != AUTHORIZATION or result["root"] != str(cache_root)):
        raise ValueError("complete exact extraction result required")
    query_path = safe(Path(expected["query_request_path"]), file=True)
    query_request = read_json(query_path)
    if (query_request.get("identity") != identity
            or query_request.get("result_path") != expected["result_path"]):
        raise ValueError("clock query must consume this exact result/launch identity")
    facts = bounded_cpu_query(identity["paths"]["extractor_python"], CLOCK_QUERY, query_path)
    if (facts["status"] != "complete" or facts["spent_seconds"] != result["spent_seconds"]
            or facts["featurekey"] != result["featurekey"] or facts["root"] != result["root"]
            or facts["bootstrapkey"] != result["bootstrapkey"]):
        raise ValueError("independently admitted clock required")
    frames = read_json(Path(identity["paths"]["adapted_root"]) / "complete/frame_rows.json")
    package = validate_motion_package(cache_root, frames, verify_features=True)
    if package != result["package"] or package["identity"] != result["identity"]:
        raise ValueError("claimed package differs from independent facts")
    runner.validate_new_motion_production(package, frames)
    for name, sha in facts["proof_bindings"].items():
        if file_hash(safe(Path(name), file=True)) != sha:
            raise ValueError("terminal proof changed during independent package admission")
    if facts.get("failure_path") and Path(facts["failure_path"]).exists():
        raise ValueError("terminal failure appeared during package admission")
    bootstrap = pipeline_bootstrap(identity)
    if (file_hash(bootstrap / "budget.json") != facts["budget_sha256"]
            or file_hash(bootstrap / "binding.json") != facts["binding_sha256"]
            or file_hash(bootstrap / "input-observation.json") != facts["observation_sha256"]):
        raise ValueError("clock/input binding changed during package consumer IO")
    return package


def admitted_pipeline_snapshot(root: Path, identity: dict, raw: dict) -> dict:
    """Read-only projection of a pending candidate and external worker-exit admission."""
    closing = raw["terminal_closure"]
    nonce = closing["nonce"]
    paths = {name: root / f"pipeline-{name}-{nonce}.json"
             for name in ("admission", "seal", "failure")}
    if (raw.get("status") != "closing_pending"
            or closing.get("policy") != "motion-fusion-pipeline-terminal-v2"
            or any(closing[name + "_path"] != str(path) for name, path in paths.items())
            or paths["failure"].exists()):
        raise ValueError("pipeline pending/admitted transaction ownership mismatch")
    proof = read_json(safe(paths["admission"], file=True))
    seal = read_json(safe(paths["seal"], file=True))
    observed = proof.get("worker_exited_monotonic")
    checked = proof.get("parent_checked_monotonic")
    upper = closing["upper_outer_seconds"]
    cleanup = seal.get("caller_cleanup_checked_monotonic")
    caller_deadline = proof.get("caller_deadline_monotonic")
    caller_upper = proof.get("training_outer_upper_seconds")
    if (proof != seal.get("proof") or proof.get("policy") != "motion-fusion-pipeline-admission-v2"
            or proof.get("nonce") != nonce or proof.get("status") != "complete"
            or proof.get("identity_hash") != digest(identity)
            or proof.get("candidate_sha256") != file_hash(root / "state.json")
            or proof.get("worker_returncode") != 0
            or proof.get("worker") != raw["external_worker"]
            or proof.get("boot_id") != closing["boot_id"]
            or not all(finite(v) for v in (observed, checked, upper, cleanup,
                                           caller_deadline, caller_upper))
            or seal.get("boot_id") != closing["boot_id"]
            or not closing["started_monotonic"] <= observed < closing["deadline_monotonic"]
            or not observed <= checked <= cleanup < caller_deadline
            or caller_upper >= CAPS["training"] or caller_upper < upper
            or upper >= CAPS["training"]
            or upper != raw["clocks"]["training"]["outer_closing_upper_bound_seconds"]
            or max(raw["clocks"]["extraction"]["outer_seconds"],
                   raw["clocks"]["extraction"]["inner_bound_seconds"]) >= CAPS["extraction"]):
        raise ValueError("successful bounded external worker-exit admission required")
    projected = json.loads(json.dumps(raw))
    projected["status"], projected["active"] = "complete", None
    projected["clocks"]["training"]["outer_seconds"] = max(
        projected["clocks"]["training"]["outer_seconds"], caller_upper)
    return projected


class MotionFusionPipeline:
    def __init__(self, root: Path, identity: dict):
        self.root, self.identity = safe(root), identity
        self.path = root / "state.json"
        self.lock: Any = None
        self.resume = False
        self.state: dict = {}
        self.guard_requests: list[Path] = []
        self.guard_controls: dict[Path, tuple[int, dict]] = {}
        self.gpu_fd: int | None = None
        self.budget_guard: Any = None
        self.budget_control: int | None = None
        self.previous_term: Any = None
        self.handoff_leases: list[Any] = []
        self.training_leases: list[Any] = []
        self.training_bindings: list[dict] = []
        self.delivery: Callable[[dict], None] | None = None
        self.worker_context = False
        self.worker_channel: socket.socket | None = None
        self.external_nonce: str | None = None
        self.worker_sequence = 0
        self.admitted = False

    def worker_event(self, event: dict, leases: list[Any] | None = None) -> None:
        if self.worker_channel is None:
            return
        self.worker_sequence += 1
        packet = {**event, "nonce": self.external_nonce, "boot_id": boot_id(),
                  "sequence": self.worker_sequence, "sent_monotonic": time.monotonic()}
        self.tick()
        active = self.state.get("active")
        deadline = time.monotonic() + (self.remaining(active["bucket"]) if active else 5.)
        closing = self.state.get("terminal_closure")
        if closing and closing.get("nonce") == self.external_nonce:
            deadline = min(deadline, closing["deadline_monotonic"])
        send_worker_frame(self.worker_channel, packet, deadline,
                          [lease.fileno() for lease in leases] if leases else None)
        if leases:
            reply = receive_worker_frame(self.worker_channel, deadline)
            if reply is None:
                raise ValueError("missing writer ownership worker IPC acknowledgment")
            acknowledgment, fds = reply
            try:
                if (fds or acknowledgment != {"kind": "owned", "nonce": self.external_nonce,
                    "sequence": self.worker_sequence, "boot_id": boot_id()}):
                    raise ValueError("external caller did not retain writer ownership")
            finally:
                for fd in fds:
                    os.close(fd)

    def record_failure(self, error: BaseException) -> None:
        self.tick()
        self.state.update(status="capped" if isinstance(error, TimeoutError) else
                          "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit))
                          else "failed", error=f"{type(error).__name__}: {error}")
        closure = self.state.get("terminal_closure")
        if closure and closure.get("failure_path"):
            atomic_json(Path(closure["failure_path"]), {"status": self.state["status"],
                "nonce": closure["nonce"], "observed_monotonic": time.monotonic(),
                "boot_id": boot_id()})
        self.save()

    def acquire_handoff_locks(self, cache_root: Path) -> None:
        if self.handoff_leases:
            raise ValueError("handoff writer leases already held")
        bootstrap = pipeline_bootstrap(self.identity)
        try:
            for root in (bootstrap, cache_root):
                lease = safe(root / ".writer.lock", file=True).open("r")
                self.handoff_leases.append(lease)
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.worker_event({"kind": "handoff-leases"}, self.handoff_leases)
        except BaseException:
            for lease in self.handoff_leases:
                lease.close()
            self.handoff_leases.clear()
            raise

    def acquire_training_leases(self) -> None:
        if self.training_leases:
            raise ValueError("training leases already held")
        root = Path(self.identity["paths"]["training_root"])
        paths = (root / ".fusion-controller.lock", root / "seed0/.study.lock",
                 root / "seed1/.study.lock")
        owner = {"pid": os.getpid(), "boot_id": boot_id(),
                 "identity": process_identity(os.getpid())}
        try:
            for path in paths:
                lease = safe(path, file=True).open("r")
                self.training_leases.append(lease)
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                st = os.fstat(lease.fileno())
                self.training_bindings.append({"path": str(path), "fd": lease.fileno(),
                    "device": st.st_dev, "inode": st.st_ino, "owner": owner,
                    "identity_hash": digest(self.identity),
                    "motion_package_hash": digest(self.state["motion_package"])})
            self.worker_event({"kind": "training-leases", "bindings": self.training_bindings,
                               "motion_package": self.state["motion_package"]},
                              self.training_leases)
        except BaseException:
            for lease in self.training_leases:
                lease.close()
            self.training_leases.clear()
            self.training_bindings.clear()
            raise

    def recheck_final_sources(self) -> None:
        self.check_sources()
        facts = self.state["summary"]
        for name, sha in facts.get("source_bindings", {}).items():
            if file_hash(safe(Path(name), file=True)) != sha:
                raise ValueError("final summary/source bytes changed before pipeline publication")
        path = Path(facts["path"])
        if path.exists() and file_hash(safe(path, file=True)) != facts["sha256"]:
            raise ValueError("final summary bytes changed before pipeline publication")

    @contextmanager
    def training_gpu_lock(self):
        path = Path(self.identity["paths"]["output_base"]) / ".motion-extraction-gpu.lock"
        safe(path, file=path.exists())
        with path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.gpu_fd = lock.fileno()
            try:
                yield
            finally:
                self.gpu_fd = None

    def __enter__(self):
        if (self.identity.get("policy") != POLICY
                or self.identity.get("authorization") != AUTHORIZATION
                or self.identity["paths"]["pipeline_root"] != str(self.root)):
            raise ValueError("pipeline immutable identity/caps invalid")
        self.root.mkdir(parents=True, exist_ok=True)
        safe(self.root / ".pipeline.lock")
        self.lock = (self.root / ".pipeline.lock").open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                self.state = read_json(safe(self.path, file=True))
                self.validate_state()
                if not self.resume:
                    raise ValueError("existing pipeline requires --resume")
                self.reconcile()
            else:
                if any(p.name != ".pipeline.lock" for p in self.root.iterdir()):
                    raise ValueError("missing state with existing work: allowance cannot reset")
                self.state = {"schema_version": 1, "policy": POLICY, "identity": self.identity,
                              "status": "created", "active": None, "stages": [],
                              "clocks": {b: {"cap_seconds": cap,
                                             "outer_seconds": self.identity.get(
                                                 "recovery", {}).get("credit", {}).get(b, 0.0),
                                             "inner_bound_seconds": self.identity.get(
                                                 "recovery", {}).get("credit", {}).get(b, 0.0)}
                                         for b, cap in CAPS.items()}}
            if not self.admitted:
                self.check_sources()
            if not self.admitted:
                self.save()
            self.previous_term = signal.getsignal(signal.SIGTERM)

            def terminated(signum, frame):
                raise TimeoutError("cumulative pipeline parent deadline/cancellation")

            signal.signal(signal.SIGTERM, terminated)
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, *args):
        if self.admitted:
            # Closed evidence is immutable even when read-only revalidation fails.
            if self.previous_term is not None:
                signal.signal(signal.SIGTERM, self.previous_term)
            for lease in [*self.handoff_leases, *self.training_leases]:
                lease.close()
            self.handoff_leases.clear()
            self.training_leases.clear()
            self.training_bindings.clear()
            if self.lock is not None:
                self.lock.close()
            return
        backup = (os.dup(self.lock.fileno()) if self.lock is not None
                  and not self.lock.closed else None)
        error = args[1] if len(args) > 1 else None
        try:
            if self.previous_term is not None:
                signal.signal(signal.SIGTERM, self.previous_term)
            for lease in [*self.handoff_leases, *self.training_leases]:
                lease.close()
            self.handoff_leases.clear()
            self.training_leases.clear()
            if self.lock is not None:
                self.lock.close()
            if backup is not None:
                self.lock = os.fdopen(backup, "r")
                backup = None
            self.tick()
            if error is not None:
                self.record_failure(error)
            if not self.worker_context:
                self.stop_budget_guard(require_success=False)
        except BaseException as failure:
            if self.lock is None or self.lock.closed:
                if backup is not None:
                    self.lock = os.fdopen(backup, "r")
                    backup = None
            self.record_failure(failure)
            if not self.worker_context:
                self.stop_budget_guard(require_success=False)
            raise
        finally:
            if backup is not None:
                os.close(backup)
            if self.lock is not None and not self.lock.closed:
                self.lock.close()

    def validate_state(self) -> None:
        if (self.state.get("schema_version") != 1 or self.state.get("policy") != POLICY
                or self.state.get("identity") != self.identity
                or not isinstance(self.state.get("stages"), list)
                or set(self.state.get("clocks", {})) != set(CAPS)):
            raise ValueError("incumbent pipeline identity/state mismatch")
        for b, cap in CAPS.items():
            clock = self.state["clocks"][b]
            inherited = self.identity.get("recovery", {}).get("credit", {}).get(b, 0)
            if (clock.get("cap_seconds") != cap
                    or not finite(clock.get("outer_seconds"))
                    or not finite(clock.get("inner_bound_seconds"))
                    or not finite(clock.get("outer_closing_upper_bound_seconds", 0))
                    or clock["outer_seconds"] < inherited
                    or clock["inner_bound_seconds"] < inherited):
                raise ValueError("corrupt pipeline cumulative clock")
        active = self.state.get("active")
        if active is not None:
            if (not isinstance(active, dict) or active.get("bucket") not in CAPS
                    or any(not finite(active.get(k)) for k in (
                        "monotonic", "wall", "started_outer"))
                    or not isinstance(active.get("boot_id"), str)
                    or not isinstance(active.get("owner"), dict)
                    or active["started_outer"] > self.state["clocks"][active["bucket"]][
                        "outer_seconds"]):
                raise ValueError("corrupt pipeline active interval/ownership anchor")
        for stage in self.state["stages"]:
            if (stage.get("bucket") not in CAPS or stage.get("signature") != digest(
                    stage.get("command"))):
                raise ValueError("stage command/signature/bucket corruption")
            if stage.get("status") in {"launching", "running"} and not stage.get("receipt"):
                raise ValueError("uncertain child ownership; no relaunch")
            end = stage.get("end_outer_seconds", 0)
            if (not finite(end) or end > self.state["clocks"][stage["bucket"]]["outer_seconds"]):
                raise ValueError("outer cumulative charge below retained stage evidence")

    def save(self) -> None:
        if self.lock is None or self.lock.closed:
            raise RuntimeError("controller lock required before state write")
        atomic_json(self.path, self.state)

    def check_sources(self) -> None:
        for name, sha in self.identity["sources"].items():
            if file_hash(safe(Path(name), file=True)) != sha:
                raise ValueError("pipeline source/environment/interpreter drift")

    def charged(self, bucket: str) -> float:
        c = self.state["clocks"][bucket]
        return max(c["outer_seconds"], c["inner_bound_seconds"],
                   c.get("outer_closing_upper_bound_seconds", 0))

    def remaining(self, bucket: str) -> float:
        return CAPS[bucket] - self.charged(bucket)

    def check_cap(self, bucket: str) -> None:
        if self.remaining(bucket) <= 0:
            raise TimeoutError(f"{bucket} cumulative cap exhausted")

    def observe_inner(self, bucket: str, value: float) -> None:
        if not finite(value) or value < self.state["clocks"][bucket]["inner_bound_seconds"]:
            raise ValueError("inner cumulative charge invalid or regressed")
        self.state["clocks"][bucket]["inner_bound_seconds"] = value
        self.check_cap(bucket)
        self.narrow_guards(bucket)

    def tick(self) -> None:
        active = self.state["active"]
        if active is None:
            return
        bucket = active["bucket"]
        if active["boot_id"] == boot_id():
            elapsed = time.monotonic() - active["monotonic"]
            if elapsed < 0:
                elapsed = CAPS[bucket]
        else:
            wall_elapsed = time.time() - active["wall"]
            elapsed = wall_elapsed if wall_elapsed >= 0 else CAPS[bucket]
        self.state["clocks"][bucket]["outer_seconds"] = max(
            self.state["clocks"][bucket]["outer_seconds"], active["started_outer"] + elapsed)

    def begin(self, bucket: str) -> None:
        self.tick()
        old = self.state.get("active")
        self.stop_budget_guard(require_success=old is not None and self.remaining(
            old["bucket"]) > 0)
        self.tick()
        if old is not None:
            self.check_cap(old["bucket"])
        self.state["active"] = {"bucket": bucket, "boot_id": boot_id(),
                                "monotonic": time.monotonic(), "wall": time.time(),
                                "started_outer": self.state["clocks"][bucket]["outer_seconds"],
                                "owner": {"pid": os.getpid(), "boot_id": boot_id(),
                                          "identity": process_identity(os.getpid())}}
        self.check_cap(bucket)
        # Arm before active state persistence, source IO or stage setup. Child
        # watchdogs own their descendant teardown; this one covers the gaps and
        # all coordinator consumer/state IO with the same cumulative deadline.
        available = self.remaining(bucket)
        reserve = min(30., available / 10)
        hard = time.monotonic() + available
        rd, wr = os.pipe()
        ack_rd, ack_wr = os.pipe()
        request = {"parent": self.state["active"]["owner"], "soft_deadline": hard - reserve,
                   "hard_deadline": hard, "bucket": bucket, "reserve": reserve}
        try:
            command = [sys.executable, str(Path(__file__)), "--budget-monitor",
                       json.dumps(request), str(rd), str(ack_wr)]
            self.budget_guard = subprocess.Popen(command, cwd=ROOT,
                env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
                start_new_session=True, pass_fds=(rd, ack_wr, self.lock.fileno(),
                                                 *[lease.fileno()
                                                   for lease in self.handoff_leases]))
            os.close(rd)
            rd = -1
            os.close(ack_wr)
            ack_wr = -1
            self.budget_control = wr
            if not select.select([ack_rd], [], [], min(5., available / 2))[0] or os.read(
                    ack_rd, 1) != b"1":
                raise RuntimeError("cumulative parent watchdog did not arm")
            self.state["budget_guard"] = {"pid": self.budget_guard.pid, "boot_id": boot_id(),
                "identity": process_identity(self.budget_guard.pid), "request": request,
                "command_signature": digest(command)}
        finally:
            for fd in (rd, ack_rd, ack_wr):
                if fd >= 0:
                    os.close(fd)
        self.save()
        if bucket == "training":
            self.worker_event({"kind": "phase", "phase": "training",
                "extraction_seconds": self.charged("extraction"),
                "hard_deadline": hard, "soft_deadline": hard - reserve})

    def stop_budget_guard(self, *, require_success: bool = False) -> None:
        if self.budget_guard is None:
            return
        try:
            if self.budget_control is not None:
                try:
                    os.write(self.budget_control, b"A\n")
                except BrokenPipeError:
                    pass
                os.close(self.budget_control)
                self.budget_control = None
            code = self.budget_guard.wait(timeout=5)
            self.tick()
            if require_success and code != 0:
                raise TimeoutError("cumulative parent watchdog did not close successfully")
        finally:
            self.budget_guard = None

    def narrow_guards(self, bucket: str) -> None:
        self.tick()
        if self.budget_control is not None:
            guard = self.state["budget_guard"]["request"]
            hard = time.monotonic() + max(0, self.remaining(bucket))
            os.write(self.budget_control, (json.dumps({"hard_deadline": hard,
                "soft_deadline": hard - guard["reserve"]}) + "\n").encode())
        for path in self.guard_requests:
            fd, request = self.guard_controls[path]
            if request["bucket"] == bucket:
                hard = time.monotonic() + max(0, self.remaining(bucket))
                guard_message(fd, "D", {"hard_deadline": hard,
                                       "soft_deadline": hard - request["reserve"]})

    def reconcile(self) -> None:
        closing = self.state.get("terminal_closure")
        if self.state.get("status") == "complete":
            raise ValueError("legacy early complete label lacks actual worker-exit proof")
        if (self.state.get("status") == "closing_pending" and closing
                and Path(closing.get("admission_path", "/missing-admission")).exists()
                and Path(closing.get("seal_path", "/missing-seal")).exists()
                and not Path(closing.get("failure_path", "/missing-failure")).exists()):
            self.state = admitted_pipeline_snapshot(self.root, self.identity, self.state)
            self.admitted = True
            return
        # Validate every receipt before touching incumbent bytes or signalling.
        receipts = []
        for stage in self.state["stages"]:
            if stage.get("status") in {"launching", "running"}:
                path = safe(Path(stage["receipt"]), file=True)
                receipt = read_json(path)
                request = read_json(Path(stage["request"]))
                if receipt["request_hash"] != digest(request):
                    raise ValueError("orphan receipt does not bind stage request")
                for r in [receipt["watchdog"], *receipt.get("descendants", [])]:
                    current = process_identity(r["pid"])
                    if current is not None and (r["boot_id"] != boot_id()
                                                or current != r["identity"]):
                        raise ValueError("orphan PID identity uncertain")
                receipts.append((stage, receipt))
        for stage, receipt in receipts:
            # Watchdog is a subreaper: parent disconnect makes it stop descendants.
            # Wait for that bounded cleanup rather than kill their ownership root.
            until = time.monotonic() + 30
            while process_identity(receipt["watchdog"]["pid"]) is not None:
                if time.monotonic() >= until:
                    raise ValueError("owned watchdog still active; no duplicate launch")
                time.sleep(.01)
            for r in receipt.get("descendants", []):
                stop_owned(r)
            stage["status"] = "interrupted"
        if self.state["active"] is not None:
            self.tick()
            self.state["active"] = None
            self.state["status"] = "interrupted"
            if any(self.remaining(b) <= 0 for b in CAPS):
                self.state["status"] = "capped"

    def execute(self, bucket: str, name: str, command: list[str],
                consume: Callable[[Path, dict], Any] | None = None,
                *, timeout: float | None = None):
        self.check_sources()
        if self.state["active"] is None or self.state["active"]["bucket"] != bucket:
            self.begin(bucket)
        self.tick()
        self.check_cap(bucket)
        available = self.remaining(bucket)
        if available <= 30:
            raise TimeoutError("30s termination/reconciliation reserve required")
        token = uuid.uuid4().hex
        stem = self.root / f"{len(self.state['stages']):04d}-{name}-{token}"
        request_path = stem.with_suffix(".request.json")
        receipt_path = stem.with_suffix(".receipt.json")
        stdout = stem.with_suffix(".stdout.jsonl")
        reserve = 30.0 if timeout is None else min(.3, available / 10)
        soft = time.monotonic() + (available - reserve if timeout is None else min(
            timeout, available - reserve))
        request = {"schema_version": 1, "bucket": bucket, "command": command,
                   "parent": {"pid": os.getpid(), "boot_id": boot_id(),
                              "identity": process_identity(os.getpid())},
                   "soft_deadline": soft, "hard_deadline": soft + reserve,
                   "reserve": reserve, "receipt": str(receipt_path), "stdout": str(stdout),
                   "stderr": str(stem.with_suffix(".stderr.log")),
                   "gpu_fds": [] if self.gpu_fd is None else [self.gpu_fd],
                   "handoff_fds": [lease.fileno() for lease in self.handoff_leases],
                   "training_fds": [lease.fileno() for lease in self.training_leases],
                   "env": {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
                           "OMP_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1"}}
        stage = {"name": name, "bucket": bucket, "status": "launching",
                 "command": command, "signature": digest(command), "request": str(request_path),
                 "receipt": str(receipt_path), "request_hash": digest(request)}
        atomic_json(request_path, request)
        self.state["stages"].append(stage)
        self.save()
        rd, wr = os.pipe()
        process = None
        self.guard_requests.append(request_path)
        self.guard_controls[request_path] = (wr, request)
        previous = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            raise TimeoutError("owned pipeline watchdog deadline/cancellation")

        signal.signal(signal.SIGTERM, terminate)
        try:
            process = subprocess.Popen([sys.executable, str(Path(__file__)), "--watchdog",
                                        json.dumps(request), str(rd), str(self.lock.fileno())],
                                       cwd=ROOT, env=request["env"], start_new_session=True,
                                       pass_fds=(rd, self.lock.fileno(),
                                                 *request["gpu_fds"], *request["handoff_fds"],
                                                 *request["training_fds"]))
            os.close(rd)
            rd = -1
            while not receipt_path.exists():
                if process.poll() is not None or time.monotonic() >= soft:
                    raise RuntimeError("watchdog failed before owned dispatch")
                time.sleep(.005)
            receipt = read_json(receipt_path)
            if receipt["request_hash"] != digest(request):
                raise ValueError("foreign watchdog receipt")
            stage.update(status="running", watchdog=receipt["watchdog"])
            self.save()
            guard_message(wr, "S")
            while True:
                receipt = read_json(receipt_path)
                self.tick()
                self.check_cap(bucket)
                if receipt["status"] in {"capped", "interrupted"}:
                    raise TimeoutError("owned child/CLI/consumer deadline")
                if receipt["returncode"] is not None:
                    break
                if process.poll() is not None:
                    raise RuntimeError("watchdog exited without bounded result receipt")
                time.sleep(.01)
            stage["returncode"] = receipt["returncode"]
            if receipt["status"] != "returned" or receipt["returncode"] != 0:
                raise RuntimeError(f"{name} child failed; retained stdout/stderr")
            value = consume(stdout, receipt) if consume is not None else None
            self.check_sources()
            self.tick()
            self.check_cap(bucket)
            stage["status"] = "complete"
            stage["end_outer_seconds"] = self.state["clocks"][bucket]["outer_seconds"]
            self.save()  # Guard is still armed across consumer and durable state IO.
            self.tick()
            self.check_cap(bucket)
            guard_message(wr, "A")
            while read_json(receipt_path)["status"] != "accepted":
                if process.poll() is not None:
                    raise RuntimeError("watchdog rejected acceptance")
                self.tick()
                self.check_cap(bucket)
                time.sleep(.005)
            # Final consumer/result/ledger IO remains watched until explicit release.
            self.tick()
            stage["end_outer_seconds"] = self.state["clocks"][bucket]["outer_seconds"]
            self.save()
            self.tick()
            self.check_cap(bucket)
            guard_message(wr, "R")
            process.wait(timeout=max(.01, soft + reserve - time.monotonic()))
            self.tick()
            self.check_cap(bucket)
            if process.returncode != 0:
                raise RuntimeError("watchdog acceptance failed")
            return value
        except BaseException as error:
            stage["status"] = "capped" if isinstance(error, TimeoutError) else "failed"
            if process is not None and process.poll() is None:
                try:
                    guard_message(wr, "X")
                except BrokenPipeError:
                    pass
                process.wait(timeout=max(.1, soft + reserve - time.monotonic()))
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)
            if rd >= 0:
                os.close(rd)
            os.close(wr)
            self.guard_requests.remove(request_path)
            self.guard_controls.pop(request_path)
            self.tick()
            if stage["status"] != "complete":
                stage["end_outer_seconds"] = self.state["clocks"][bucket]["outer_seconds"]
                self.save()

    def extraction_command(self, mode: str) -> list[str]:
        paths = self.identity["paths"]
        command = [str(ROOT / "tools/motion/.venv/bin/python"),
                   str(ROOT / "scripts/extract_motion_features.py"), "--mode", mode]
        for name in ("adapted_root", "annotation", "model_snapshot", "output_base",
                     "environment_lock"):
            command.extend(["--" + name.replace("_", "-"), paths[name]])
        command.extend(["--gpu-cap-seconds", "18000", "--authorize-motion-extraction"])
        recovery = self.identity.get("recovery")
        if recovery is not None:
            command.extend(["--recovery-record", recovery["path"]])
        bootstrap = pipeline_bootstrap(self.identity)
        if mode != "preflight" and bootstrap.exists():
            command.append("--resume")
        return command

    def query(self, bucket: str, name: str, python: str, code: str, request: dict) -> dict:
        path = self.root / f"query-{uuid.uuid4().hex}.json"
        atomic_json(path, request)
        result = self.execute(bucket, name, [python, "-c", code, str(path)],
                              lambda out, receipt: delivered_result(out))
        if not isinstance(result, dict):
            raise ValueError("query did not deliver an object")
        return result

    def admit_result(self, mode: str, out: Path, receipt: dict) -> dict:
        result = delivered_result(out)
        if (set(result) != RESULT_KEYS or type(result["schema_version"]) is not int
                or result["schema_version"] != 1
                or result["policy"] != "motion-extraction-result-v1" or result["mode"] != mode
                or result["authorization"] != AUTHORIZATION
                or any(not finite(result[k]) for k in (
                    "spent_seconds", "remaining_seconds", "admission_seconds"))):
            raise ValueError("exact22-key producer result required")
        # Failed prebinding results have nullable identity paths; they never enter
        # a success parser or dispatch training, even if a marker survived failure.
        want = {"preflight": "ready", "profile": "profiled", "run": "complete"}[mode]
        if result["status"] != want or result["error"] is not None:
            raise ValueError("unsuccessful extraction result; no handoff")
        if mode != "run" and result["package"] is not None:
            raise ValueError("unexpected package before completed run")
        if (not isinstance(result["identity"], dict)
                or digest(result["identity"]) != result["featurekey"]):
            raise ValueError("result feature identity mismatch")
        cache_root = Path(self.identity["paths"]["output_base"]) / (
            "motion-features-" + result["featurekey"])
        if result["root"] != str(cache_root):
            raise ValueError("result cache root must match authorized content key")
        prior = self.state.get("extraction_identity")
        if prior is not None and prior != result["identity"]:
            raise ValueError("preflight/profile/run source/hardware/content drift")
        self.state["extraction_identity"] = result["identity"]
        normalized = self.root / f"result-{uuid.uuid4().hex}.json"
        atomic_json(normalized, result)
        if mode == "preflight":
            if any(result[k] is not None for k in ("bootstrap_root", "bootstrapkey", "binding_path",
                   "input_observation_path", "budget_path", "status_path", "terminal_admission")):
                raise ValueError("preflight must not claim bootstrap/admission")
            return {"result_path": str(normalized), "status": "ready"}
        if mode == "run":
            self.acquire_handoff_locks(cache_root)
        facts = self.query("extraction", "clock-admission",
                           self.identity["paths"]["extractor_python"], CLOCK_QUERY,
                           {"identity": self.identity, "result_path": str(normalized),
                            "outer_receipt": receipt,
                            "outer_request": read_json(Path(self.state["stages"][-1]["request"]))})
        self.observe_inner("extraction", facts["spent_seconds"])
        self.state["clock_facts"] = facts
        self.state["extraction_result_path"] = str(normalized)
        self.state["extraction_outer_receipt"] = receipt
        # Nested query appended its own stage, so retain the CLI request by hash.
        self.state["extraction_outer_request"] = next(
            read_json(Path(s["request"])) for s in self.state["stages"]
            if s["request_hash"] == receipt["request_hash"])
        return {"result_path": str(normalized), "status": want, "clock_facts": facts}

    def gpu_admission(self, *, extraction: bool) -> None:
        code = """
import json,sys
from pathlib import Path
from despamo.appearance.provenance import read_json
from scripts.run_motion_fusion_pipeline import gpu_storage_query
print(json.dumps(gpu_storage_query(read_json(Path(sys.argv[1])))))
"""
        bucket = "extraction" if extraction else "training"
        self.query(bucket, "gpu-disk-admission", str(ROOT / ".venv/bin/python"), code,
                   {"identity": self.identity, "extraction": extraction,
                    "extraction_identity": self.state.get("extraction_identity")})

    def package_query(self) -> dict:
        request = {"identity": self.identity,
                   "result_path": self.state["extraction_result_path"],
                   "outer_receipt": self.state["extraction_outer_receipt"],
                   "outer_request": self.state["extraction_outer_request"]}
        code = """
import json,sys
from pathlib import Path
from despamo.appearance.provenance import read_json
from scripts.run_motion_fusion_pipeline import validate_extraction_handoff
from despamo.data.new_motion import compact_motion_binding
r=read_json(Path(sys.argv[1])); result=read_json(Path(r['result_path']))
package=validate_extraction_handoff(Path(result['root']),r)
print(json.dumps(compact_motion_binding(package)))
"""
        query_path = self.root / f"handoff-{uuid.uuid4().hex}.json"
        request["query_request_path"] = str(query_path)
        atomic_json(query_path, request)
        return self.query("extraction", "full-package-admission", str(ROOT / ".venv/bin/python"),
                          code, request)

    def extraction_modes(self) -> tuple[tuple[str, str], ...]:
        facts = self.query("extraction", "producer-projection",
                           self.identity["paths"]["extractor_python"], PROJECTION_QUERY,
                           {"identity": self.identity,
                            "expected_identity": self.state["extraction_identity"]})
        if facts["status"] in {"complete", "profiled"}:
            self.observe_inner("extraction", facts["spent_seconds"])
            return (("run", "extracting"),)
        if facts["status"] == "capped" or facts["spent_seconds"] >= CAPS["extraction"]:
            raise TimeoutError("owned producer extraction allowance exhausted")
        return (("profile", "profiling"), ("run", "extracting"))

    def run(self) -> dict:
        """External caller witnesses an actual coordinator-worker exit before admission."""
        self.__enter__()
        if self.admitted:
            try:
                self.validate_completed_resume()
                receipt = self.receipt()
                if self.delivery is not None:
                    self.delivery(receipt)
                return receipt
            finally:
                self.__exit__(None, None, None)
        caller, channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.external_nonce = uuid.uuid4().hex
        self.state["active"] = {"bucket": "extraction", "boot_id": boot_id(),
            "monotonic": time.monotonic(), "wall": time.time(),
            "started_outer": self.state["clocks"]["extraction"]["outer_seconds"],
            "owner": {"pid": os.getpid(), "boot_id": boot_id(),
                      "identity": process_identity(os.getpid())}}
        pid = os.fork()
        if pid == 0:
            caller.close()
            os.setsid()
            self.worker_context, self.worker_channel = True, channel
            backup = os.dup(self.lock.fileno())
            try:
                frame = receive_worker_frame(channel, time.monotonic() + 5.)
                if frame is None:
                    raise ValueError("missing initial worker IPC frame")
                initial, fds = frame
                try:
                    if (fds or initial.get("kind") != "start"
                            or initial.get("nonce") != self.external_nonce
                            or initial.get("boot_id") != boot_id()):
                        raise ValueError("invalid initial worker IPC frame")
                finally:
                    for fd in fds:
                        os.close(fd)
                self.state["external_worker"] = initial["worker"]
                self._run_stages()
                self.__exit__(None, None, None)
                self.worker_event({"kind": "worker-return-ready"})
                os._exit(0)
            except BaseException as error:
                self.lock = os.fdopen(backup, "r")
                self.record_failure(error)
                os._exit(1)
        channel.close()
        control = None
        waited = False
        successful = False
        worker = None
        owner_backup = os.dup(self.lock.fileno())
        try:
            until = time.monotonic() + 5
            current = process_identity(pid)
            while current is None or current["session"] != pid:
                if time.monotonic() >= until:
                    raise RuntimeError("coordinator worker ownership not established")
                time.sleep(.001)
                current = process_identity(pid)
            worker = {"pid": pid, "boot_id": boot_id(), "identity": current}
            self.state["external_worker"] = worker
            rd, control = os.pipe()
            ack_rd, ack_wr = os.pipe()
            available = self.remaining("extraction")
            hard = time.monotonic() + available
            failure_path = self.root / f"pipeline-failure-{self.external_nonce}.json"
            seal_path = self.root / f"pipeline-seal-{self.external_nonce}.json"
            request = {"parent": self.state["active"]["owner"], "owned_worker": worker,
                "hard_deadline": hard, "soft_deadline": hard - min(30., available / 10),
                "nonce": self.external_nonce, "failure_path": str(failure_path),
                "seal_path": str(seal_path)}
            self.budget_guard = subprocess.Popen([sys.executable, str(Path(__file__)),
                "--budget-monitor", json.dumps(request), str(rd), str(ack_wr)], cwd=ROOT,
                env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
                start_new_session=True, pass_fds=(rd, ack_wr, self.lock.fileno()))
            os.close(rd)
            os.close(ack_wr)
            if not select.select([ack_rd], [], [], 5)[0] or os.read(ack_rd, 1) != b"1":
                raise RuntimeError("external coordinator guardian did not arm")
            os.close(ack_rd)
            self.budget_control = control
            self.save()
            send_worker_frame(caller, {"kind": "start", "worker": worker,
                                      "nonce": self.external_nonce, "boot_id": boot_id()},
                              min(hard, time.monotonic() + 5.))
            returned = False
            sequence = 0
            previous_sent = 0.
            ipc_deadline = hard

            def receive_packet() -> bool:
                nonlocal returned, sequence, previous_sent, ipc_deadline
                frame = receive_worker_frame(caller, ipc_deadline)
                if frame is None:
                    return False
                event, fds = frame
                transferred = False
                try:
                    sent = event.get("sent_monotonic")
                    if (event.get("nonce") != self.external_nonce
                            or event.get("boot_id") != worker["boot_id"]
                            or type(event.get("sequence")) is not int
                            or event["sequence"] != sequence + 1 or not finite(sent)
                            or not previous_sent <= cast(Any, sent) <= time.monotonic()):
                        raise ValueError("worker IPC nonce/sequence/chronology mismatch")
                    sequence, previous_sent = event["sequence"], sent
                    kind = event["kind"]
                    if kind not in {"training-leases", "handoff-leases", "phase", "closing",
                                    "worker-return-ready"}:
                        raise ValueError("unknown worker IPC packet")
                    if fds and kind not in {"training-leases", "handoff-leases"}:
                        raise ValueError("unexpected descriptor-bearing worker IPC packet")
                    if kind in {"training-leases", "handoff-leases"}:
                        required = 3 if kind == "training-leases" else 2
                        if len(fds) != required:
                            raise ValueError("all owned worker IPC writer leases required")
                        leases = [os.fdopen(fd, "r") for fd in fds]
                        transferred = True
                        if kind == "training-leases":
                            self.training_leases.extend(leases)
                            self.state["motion_package"] = event["motion_package"]
                            self.training_bindings = event["bindings"]
                            for row, lease in zip(self.training_bindings, leases, strict=True):
                                st = os.fstat(lease.fileno())
                                if ((st.st_dev, st.st_ino) != (row["device"], row["inode"])
                                        or row["owner"] != worker):
                                    raise ValueError("foreign SCM_RIGHTS training ownership")
                                row["fd"] = lease.fileno()
                                row["owner"] = request["parent"]
                        else:
                            self.handoff_leases.extend(leases)
                        send_worker_frame(caller, {"kind": "owned", "nonce": self.external_nonce,
                                                  "sequence": sequence, "boot_id": boot_id()},
                                          ipc_deadline)
                    elif kind == "phase":
                        os.write(control, (json.dumps(event) + "\n").encode())
                        ipc_deadline = event["hard_deadline"]
                    elif kind == "closing":
                        os.write(control, (json.dumps(event["limit"]) + "\n").encode())
                        ipc_deadline = min(ipc_deadline, event["limit"]["hard_deadline"])
                    elif kind == "worker-return-ready":
                        if returned:
                            raise ValueError("duplicate worker IPC return-ready packet")
                        returned = True
                    return True
                finally:
                    if not transferred:
                        for fd in fds:
                            os.close(fd)

            while True:
                if select.select([caller], [], [], .01)[0]:
                    receive_packet()
                exited, status = os.waitpid(pid, os.WNOHANG)
                if exited:
                    waited = True
                    exit_observed = time.monotonic()
                    code = os.waitstatus_to_exitcode(status)
                    # waitpid can overtake queued frames. Drain to boundary EOF using
                    # the exact same validated parser before judging readiness.
                    while True:
                        remaining = ipc_deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("worker IPC post-exit drain exceeded deadline")
                        if select.select([caller], [], [], min(.01, remaining))[0]:
                            if not receive_packet():
                                break
                    if code != 0 or not returned:
                        raise RuntimeError(f"coordinator context/worker exit failed: {code}")
                    break
            self.state = read_json(safe(self.path, file=True))
            self.validate_state()
            closing = self.state["terminal_closure"]
            if exit_observed >= closing["deadline_monotonic"]:
                raise TimeoutError("coordinator worker exited after its closing deadline")
            self.tick()
            remaining = self.remaining("training")
            consumer_hard = time.monotonic() + remaining
            os.write(control, (json.dumps({"action": "worker-exited", "worker": worker,
                "returncode": code, "hard_deadline": consumer_hard,
                "soft_deadline": consumer_hard - min(30., remaining / 10)}) + "\n").encode())
            self.recheck_final_sources()
            self.tick()
            self.check_cap("training")
            if failure_path.exists():
                raise TimeoutError("external worker/consumer closing window exceeded")
            caller_deadline = time.monotonic() + 5.
            caller_upper = max(closing["upper_outer_seconds"],
                               self.state["clocks"]["training"]["outer_seconds"] + 5.)
            if caller_upper >= CAPS["training"]:
                raise TimeoutError("caller final publication cannot fit training cap")
            os.write(control, (json.dumps({"hard_deadline": caller_deadline,
                "soft_deadline": caller_deadline - .5}) + "\n").encode())
            proof = {"policy": "motion-fusion-pipeline-admission-v2", "status": "complete",
                "nonce": self.external_nonce, "identity_hash": digest(self.identity),
                "candidate_sha256": file_hash(self.path), "worker": worker,
                "worker_returncode": code, "worker_exited_monotonic": exit_observed,
                "parent_checked_monotonic": time.monotonic(), "boot_id": boot_id(),
                "caller_deadline_monotonic": caller_deadline,
                "training_outer_upper_seconds": caller_upper}
            atomic_json(Path(closing["admission_path"]), proof)
            result = self.receipt()
            result["training_seconds"] = max(result["training_seconds"], caller_upper)
            result["status"] = "complete"
            if self.delivery is not None:
                self.delivery(result)
            # Cleanup is covered by the external guardian. Root lock is retained
            # for failure reconciliation until the final no-IO return boundary.
            for lease in [*self.training_leases, *self.handoff_leases]:
                lease.close()
            self.training_leases.clear()
            self.handoff_leases.clear()
            if self.previous_term is not None:
                signal.signal(signal.SIGTERM, self.previous_term)
            self.tick()
            self.check_cap("training")
            if time.monotonic() >= caller_deadline:
                raise TimeoutError("caller cleanup exceeded closing deadline")
            os.write(control, (json.dumps({"action": "seal", "proof": proof}) + "\n").encode())
            while not seal_path.exists() or read_json(seal_path).get("proof") != proof:
                if time.monotonic() >= caller_deadline:
                    raise TimeoutError("external admission seal was not durable in time")
                time.sleep(.005)
            if failure_path.exists():
                raise TimeoutError("external guardian rejected closing")
            projected = admitted_pipeline_snapshot(self.root, self.identity, read_json(self.path))
            caller.close()
            self.lock.close()
            os.write(control, b'{"action":"release"}\n')
            self.budget_guard.wait(timeout=max(.01, caller_deadline
                                               - time.monotonic()))
            self.tick()
            if (self.budget_guard.returncode != 0
                    or time.monotonic() >= caller_deadline):
                raise TimeoutError("external guardian/caller closing failed")
            self.state = projected
            successful = True
            return result
        except BaseException as error:
            if self.lock is None or self.lock.closed:
                self.lock = os.fdopen(os.dup(owner_backup), "r")
            if not waited and worker is not None:
                stop_owned(worker)
                os.waitpid(pid, 0)
            if self.path.exists():
                self.state = read_json(self.path)
            self.record_failure(error)
            raise
        finally:
            caller.close()
            if control is not None:
                os.close(control)
            if not successful and self.budget_guard is not None:
                self.budget_guard.wait(timeout=1)
            for lease in [*self.training_leases, *self.handoff_leases]:
                if not lease.closed:
                    lease.close()
            if self.lock is not None and not self.lock.closed:
                self.lock.close()
            if not successful and self.previous_term is not None:
                signal.signal(signal.SIGTERM, self.previous_term)
            os.close(owner_backup)

    def validate_completed_resume(self) -> None:
        """Bounded read-only admission under retained producer/controller leases."""
        self.acquire_handoff_locks(Path(self.state["motion_package"]["root"]))
        self.acquire_training_leases()
        request = {"identity": self.identity, "candidate_path": str(self.path),
                   "candidate_sha256": file_hash(self.path),
                   "training_leases": self.training_bindings}
        path = self.root / f"resume-query-{uuid.uuid4().hex}.json"
        atomic_json(path, request)
        code = """
import json,sys
from pathlib import Path
from despamo.appearance.provenance import read_json
from scripts.run_motion_fusion_pipeline import validate_completed_pipeline
print(json.dumps(validate_completed_pipeline(read_json(Path(sys.argv[1])))))
"""
        facts = bounded_cpu_query(str(ROOT / ".venv/bin/python"), code, path, timeout=300,
            pass_fds=tuple(lease.fileno() for lease in [*self.handoff_leases,
                                                       *self.training_leases]))
        if facts != self.state["summary"] or file_hash(self.path) != request["candidate_sha256"]:
            raise ValueError("completed pipeline package/summary/source admission changed")

    def _run_stages(self) -> None:
        self.begin("extraction")
        self.state["status"] = "preflight"
        self.execute("extraction", "preflight", self.extraction_command("preflight"),
                     lambda p, r: self.admit_result("preflight", p, r))
        for mode, phase in self.extraction_modes():
            self.state["status"] = phase
            self.gpu_admission(extraction=True)
            self.execute("extraction", mode, self.extraction_command(mode),
                         lambda p, r, m=mode: self.admit_result(m, p, r))
        self.state["status"] = "validating"
        package = self.package_query()
        self.state["motion_package"] = package
        self.tick()
        self.check_cap("extraction")
        self.begin("training")
        self.state["status"] = "training"
        with self.training_gpu_lock():
            self.gpu_admission(extraction=False)
            command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_fusion_study.py"),
                       "--study-root", self.identity["paths"]["training_root"],
                       "--motion-package", package["root"],
                       "--motion-timing", package["timing_path"],
                       "--authorize-fusion-study"]
            self.execute("training", "new-fusion-controller", command, self.finish_training)
        self.tick()
        self.check_cap("training")

    def finish_training(self, out: Path, receipt: dict) -> dict:
        delivered = delivered_result(out)
        root = Path(self.identity["paths"]["training_root"])
        if delivered != {"status": "complete", "summary": str(root / "summary.json")}:
            raise ValueError("new controller successful summary delivery required")
        self.acquire_training_leases()
        request = {"identity": self.identity, "motion_package": self.state["motion_package"],
                   "extraction_proof_bindings": self.state["clock_facts"].get("proof_bindings", {}),
                   "extraction_failure_path": self.state["clock_facts"].get("failure_path"),
                   "training_leases": self.training_bindings}
        code = """
import json,sys
from pathlib import Path
from despamo.appearance.provenance import read_json
from scripts.run_motion_fusion_pipeline import validate_final_summary
print(json.dumps(validate_final_summary(read_json(Path(sys.argv[1])))))
"""
        facts = self.query("training", "final-six-run-admission", str(ROOT / ".venv/bin/python"),
                           code, request)
        self.observe_inner("training", facts["inner_seconds"])
        self.state["summary"] = facts
        self.recheck_final_sources()
        self.tick()
        closing_start = time.monotonic()
        upper = self.state["clocks"]["training"]["outer_seconds"] + 5.
        if upper >= CAPS["training"]:
            raise TimeoutError("final pipeline IO bound exceeds remaining training cap")
        outer_request = self.guard_requests[-1]
        stage = next(s for s in self.state["stages"] if s["request"] == str(outer_request))
        self.state["terminal_closure"] = {
            "policy": "motion-fusion-pipeline-terminal-v2", "boot_id": boot_id(),
            "nonce": self.external_nonce,
            "admission_path": str(self.root / f"pipeline-admission-{self.external_nonce}.json"),
            "seal_path": str(self.root / f"pipeline-seal-{self.external_nonce}.json"),
            "failure_path": str(self.root / f"pipeline-failure-{self.external_nonce}.json"),
            "stage_request_hash": stage["request_hash"], "started_monotonic": closing_start,
            "deadline_monotonic": closing_start + 5., "upper_outer_seconds": upper,
            "semantics": "observed_outer_plus_watchdog_bounded_final_io_reserve"}
        self.state["clocks"]["training"]["outer_closing_upper_bound_seconds"] = upper
        limit = {"hard_deadline": closing_start + 5., "soft_deadline": closing_start + 4.5}
        if self.budget_control is None:
            raise ValueError("cumulative parent watchdog required for final publication")
        os.write(self.budget_control, (json.dumps(limit) + "\n").encode())
        guard_message(self.guard_controls[outer_request][0], "D", limit)
        self.worker_event({"kind": "closing", "limit": limit})
        # The coordinator cannot know its own future context/process exit. Only
        # the external caller can project this candidate after waitpid exit0.
        self.state["status"] = "closing_pending"
        return facts

    def receipt(self) -> dict:
        return {"schema_version": 1, "policy": POLICY, "status": self.state["status"],
                "pipeline_root": str(self.root), "identity_hash": digest(self.identity),
                "extraction_seconds": self.charged("extraction"),
                "training_seconds": self.charged("training"),
                "authorization": dict(AUTHORIZATION), "summary": self.state.get("summary")}


def gpu_storage_query(request: dict) -> dict:
    from scripts import run_fusion_study as controller

    paths = request["identity"]["paths"]
    expected_hardware = (request.get("extraction_identity") or {}).get("hardware")
    if not isinstance(expected_hardware, dict):
        raise ValueError("pinned extraction hardware required before GPU admission")
    hardware = bounded_cpu_query(paths["extractor_python"],
        "import json; from tools.motion.extractor import hardware_identity; "
        "print(json.dumps(hardware_identity()))", Path(__file__))
    if hardware != expected_hardware:
        raise ValueError("hardware/GPU drift since pipeline preflight")
    runs = {}
    study = Path(paths["training_root"])
    for seed in (0, 1):
        path = study / f"seed{seed}/study-state.json"
        if path.exists():
            phase = read_json(safe(path, file=True))
            if phase.get("study_policy") != NEW_POLICY:
                raise ValueError("foreign training phase")
            for arm, row in phase["runs"].items():
                # Credit only independently admitted owned run inventory. A mere
                # complete label cannot waive future checkpoint demand.
                expected = row["identity"]
                controller.runner.validate_motion_graph(expected)
                run = Path(row["run"])
                if run.parent != study / "runs" or expected["run_key"] != run.name:
                    raise ValueError("foreign allocated checkpoint credit")
                complete = controller.owned_status(run, expected).get("status") == "complete"
                if complete:
                    query_phase = SimpleNamespace(root=study / f"seed{seed}", fusion_seed=seed,
                        active_arm=arm, motion_package=Path(expected["motion_package"]["root"]))
                    def selection_query(command, query_phase=query_phase):
                        controller.admit_fusion_query(command, query_phase)
                        selected = subprocess.run(command, cwd=ROOT, check=True, timeout=300,
                                                  capture_output=True, text=True)
                        return json.loads(selected.stdout.splitlines()[-1])
                    query_phase.cpu_query = selection_query
                    controller.completion_selection(query_phase, run, expected)
                sizes = controller.runner.fusion_checkpoint_sizes(run)
                runs[f"{seed}-{arm}"] = {**row, "complete": complete,
                                        "checkpoint_sizes_bytes": sizes}
    size = controller.checkpoint_size(runs) if runs else controller.CHECKPOINT_BYTES
    extraction_bytes = 93232 * 1024 * 4 if request["extraction"] else 0
    if request["extraction"] and request.get("extraction_identity") is not None:
        identity = request["extraction_identity"]
        cache = Path(paths["output_base"]) / ("motion-features-" + digest(identity))
        frames = read_json(Path(identity["inputs"]["frame_rows"]["path"]))
        extraction_bytes = remaining_extraction_demand(cache, identity, frames)
    require_gpu_idle()
    return storage_admission(Path(paths["pipeline_root"]), runs, checkpoint_bytes=size,
                             extraction_bytes=extraction_bytes)


def validate_final_summary(request: dict) -> dict:
    from scripts import run_fusion_study as controller

    root = Path(request["identity"]["paths"]["training_root"])
    state = read_json(safe(root / "state.json", file=True))
    path = safe(root / "summary.json", file=True)
    summary = read_json(path)
    package = request["motion_package"]
    def extraction_proof() -> None:
        for name, sha in request.get("extraction_proof_bindings", {}).items():
            if file_hash(safe(Path(name), file=True)) != sha:
                raise ValueError("extraction admission proof drift during training")
        failure = request.get("extraction_failure_path")
        if failure and Path(failure).exists():
            raise ValueError("extraction terminal failure marker appeared during training")
    extraction_proof()
    if (state.get("status") != "complete" or state.get("study_policy") != NEW_POLICY
            or state.get("budget_seconds") != 64800
            or state.get("per_seed_budget_seconds") != 32400
            or state.get("per_run_budget_seconds") != 10800
            or summary.get("study_policy") != NEW_POLICY
            or state.get("summary_sha256") != file_hash(path)
            or summary.get("motion_package") != package or state.get("motion_package") != package
            or summary.get("budget_seconds") != 64800
            or digest({k: v for k, v in summary.items() if k != "content_hash"}) != summary.get(
                "content_hash")):
        raise ValueError("final summary/controller/package byte binding invalid")
    # Reconstruct the six rows from current completion/selection/report sources;
    # do not trust a summary's copies of checkpoint or ordered-reference facts.
    from contextlib import ExitStack
    with ExitStack() as stack:
        results, accounting = [], []
        inherited = request.get("training_leases")
        expected_paths = [root / ".fusion-controller.lock", root / "seed0/.study.lock",
                          root / "seed1/.study.lock"]
        if inherited is not None:
            if [row["path"] for row in inherited] != [str(p) for p in expected_paths]:
                raise ValueError("all three exact parent-held training leases required")
            for row in inherited:
                st = os.fstat(row["fd"])
                current = safe(Path(row["path"]), file=True).stat()
                owner = row["owner"]
                if ((st.st_dev, st.st_ino) != (row["device"], row["inode"])
                        or (current.st_dev, current.st_ino) != (st.st_dev, st.st_ino)
                        or row["identity_hash"] != digest(request["identity"])
                        or row["motion_package_hash"] != digest(package)
                        or owner["boot_id"] != boot_id()
                        or process_identity(owner["pid"]) != owner["identity"]):
                    raise ValueError("inherited lease descriptor/identity ownership mismatch")
        else:
            for lease_path in expected_paths:
                lease = stack.enter_context(safe(lease_path, file=True).open("r"))
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in (0, 1):
            phase_root = root / f"seed{seed}"
            # Read-only projection: the controller's constructor/load and
            # cpu_query normally reconcile/write/charge its own phase clock.
            # Final consumer validation must not rewrite closed inner evidence.
            phase = controller.FusionPhase.__new__(controller.FusionPhase)
            phase.root, phase.fusion_seed = phase_root, seed
            phase.motion_package = Path(package["root"])
            phase.state = read_json(safe(phase_root / "study-state.json", file=True))
            phase.env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}

            def cpu_query(command, env=phase.env, query_phase=phase):
                controller.admit_fusion_query(command, query_phase)
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                        text=True, timeout=300, check=True)
                return json.loads(result.stdout.splitlines()[-1])

            phase.cpu_query = cpu_query
            if (phase.state.get("study_policy") != NEW_POLICY
                    or phase.state.get("fusion_seed") != seed
                    or phase.state.get("motion_package_root") != package["root"]
                    or phase.state.get("controller_sources") != state["controller_sources"]
                    or phase.state.get("budget_seconds") != 32400
                    or phase.state.get("run_cap_seconds") != 10800):
                raise ValueError("phase policy/source/cap ownership mismatch")
            if set(phase.state["runs"]) != set(controller.ARMS):
                raise ValueError("all three owned arms required in both phases")
            for arm in controller.ARMS:
                row = phase.state["runs"][arm]
                expected = row["identity"]
                controller.runner.validate_motion_graph(expected)
                if expected["motion_package"] != package:
                    raise ValueError("final run package graph differs from extraction handoff")
                phase.active_arm = arm
                results.append(controller.collect_result(phase, arm, Path(row["run"]), expected))
            spent = phase.state["spent_seconds"]
            if not finite(spent) or spent > 32400:
                raise ValueError("training phase cap overrun")
            stages = phase.state["stages"]
            if (any(not finite(s.get("elapsed_seconds")) for s in stages.values())
                    or not math.isclose(sum(s["elapsed_seconds"] for s in stages.values()), spent,
                                        abs_tol=1e-6)
                    or any(phase.run_spent(arm) > 10800 for arm in controller.ARMS)):
                raise ValueError("phase/per-run cumulative accounting mismatch")
            accounting.append({"seed": seed, "spent_seconds": spent,
                               "stages": phase.state["stages"]})
        controller.validate_summary_sources(results, root, root / "common-initialization.json",
                                            Path(package["timing_path"]), state)
        extraction_proof()
        rebuilt = controller.aggregate_fusion_results(results)
        if any(summary.get(k) != v for k, v in rebuilt.items() if k != "study_policy"):
            raise ValueError("summary differs from six independently admitted current runs")
        if summary["phase_accounting"] != accounting:
            raise ValueError("summary phase charge mismatch")
        inner = sum(a["spent_seconds"] for a in accounting)
        if inner > 64800:
            raise ValueError("training inner total overrun")
        if summary["spent_charged_seconds"] != inner:
            raise ValueError("summary cumulative inner charge mismatch")
        controller.validate_summary_sources(results, root, root / "common-initialization.json",
                                            Path(package["timing_path"]), state)
        if file_hash(path) != state["summary_sha256"]:
            raise ValueError("summary drift during final admission")
        extraction_proof()
        bindings = {str(path): file_hash(path),
                    str(root / "state.json"): file_hash(root / "state.json")}
        for anchor in (root / "common-initialization.json", root / "seed0/study-state.json",
                       root / "seed1/study-state.json", Path(package["timing_path"])):
            if anchor.exists():
                bindings[str(anchor)] = file_hash(safe(anchor, file=True))
        for row in results:
            bindings.update(row.get("source_files", {}))
            bindings.update(row["identity"].get("bound_files", {}))
        bindings.update(request.get("extraction_proof_bindings", {}))
        return {"path": str(path), "sha256": file_hash(path), "inner_seconds": inner,
                "motion_package": package, "controller_sources": state["controller_sources"],
                "source_bindings": bindings}


def validate_completed_pipeline(request: dict) -> dict:
    """Reconstruct closed success without refund, offline billing or evidence writes."""
    from despamo.data.new_motion import compact_motion_binding

    identity = request["identity"]
    path = safe(Path(request["candidate_path"]), file=True)
    if (path != Path(identity["paths"]["pipeline_root"]) / "state.json"
            or file_hash(path) != request["candidate_sha256"]):
        raise ValueError("completed pipeline candidate byte binding mismatch")
    state = admitted_pipeline_snapshot(path.parent, identity, read_json(path))
    for name, sha in identity["sources"].items():
        if file_hash(safe(Path(name), file=True)) != sha:
            raise ValueError("completed pipeline source/environment drift")
    package_request = {"identity": identity, "result_path": state["extraction_result_path"],
                       "outer_receipt": state["extraction_outer_receipt"],
                       "outer_request": state["extraction_outer_request"]}
    # Reuse the exact immutable clock-query request already retained in stage evidence.
    stage = next(s for s in reversed(state["stages"]) if s["name"] == "full-package-admission")
    original = read_json(Path(stage["command"][-1]))
    package_request["query_request_path"] = original["query_request_path"]
    package = validate_extraction_handoff(Path(state["motion_package"]["root"]), package_request)
    if compact_motion_binding(package) != state["motion_package"]:
        raise ValueError("completed pipeline extraction package/completion changed")
    facts = validate_final_summary({"identity": identity, "motion_package": state["motion_package"],
        "extraction_proof_bindings": state["clock_facts"].get("proof_bindings", {}),
        "extraction_failure_path": state["clock_facts"].get("failure_path"),
        "training_leases": request["training_leases"]})
    if facts != state["summary"]:
        raise ValueError("completed pipeline summary/source binding changed")
    # Repeat admission after all consumer IO, including source/candidate proof checks.
    if admitted_pipeline_snapshot(path.parent, identity, read_json(path)) != state:
        raise ValueError("completed pipeline terminal admission changed during validation")
    for name, sha in identity["sources"].items():
        if file_hash(safe(Path(name), file=True)) != sha:
            raise ValueError("completed pipeline source changed during validation")
    return facts


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--watchdog":
        raise SystemExit(_watchdog(json.loads(argv[1]), int(argv[2]), int(argv[3])))
    if argv and argv[0] == "--budget-monitor":
        raise SystemExit(_budget_monitor(json.loads(argv[1]), int(argv[2]), int(argv[3])))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_ROOT)
    for name in ("adapted-root", "annotation", "model-snapshot", "extractor-python",
                 "environment-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--authorize-motion-fusion-pipeline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recovery-record", type=Path)
    args = parser.parse_args(argv)
    identity = pipeline_identity(args)
    pipeline = MotionFusionPipeline(args.pipeline_root, identity)
    pipeline.resume = args.resume
    pipeline.delivery = lambda receipt: print(json.dumps(receipt, allow_nan=False), flush=True)
    pipeline.run()


if __name__ == "__main__":
    main()
