"""Owned accounting and native/backing-disk guards for official adaptation."""

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from tools.dinov3.storage import atomic_json, verify_location


def process_owner(pid: int) -> dict | None:
    try:
        root = Path(f"/proc/{pid}")
        fields = (root/"stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(pid=pid, boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    start_ticks=fields[19], command=(root/"cmdline").read_bytes().decode().rstrip(
                        "\0").split("\0"))
    except (FileNotFoundError, ProcessLookupError):
        return None


def windows_free(drive: str) -> int:
    if drive not in ("B", "C"):
        raise ValueError("owned backing drive must be B or C")
    return int(subprocess.check_output(["powershell.exe", "-NoProfile", "-Command",
               f"(Get-PSDrive -Name {drive}).Free"], text=True, timeout=15).strip())


def require_official_storage(root: Path, remaining_bytes: int,
                             reserve_bytes: int = 50*1024**3, windows_drive: str = "C") -> dict:
    if type(remaining_bytes) is not int or remaining_bytes < 0 or reserve_bytes != 50*1024**3:
        raise ValueError("invalid projected storage/reserve")
    verify_location(root, Path(__file__).resolve().parents[2])
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    linux, backing = shutil.disk_usage(ancestor).free, windows_free(windows_drive)
    needed = remaining_bytes + reserve_bytes
    if min(linux, backing) < needed:
        raise ValueError("native/backing free space below projected demand plus50GiB reserve")
    return dict(linux_free_bytes=linux, windows_free_bytes=backing,
                projected_bytes=remaining_bytes, reserve_bytes=reserve_bytes)


class OfficialLedger:
    def __init__(self, root: Path, identity: dict, initial_seconds: float = 0.,
                 setup_seconds: float = 0.):
        self.path = root/"budget.json"
        self.identity, self.root = identity, root
        self.completed, self.last = 0, initial_seconds+setup_seconds
        if self.path.exists():
            prior = json.loads(self.path.read_text())
            if prior["identity"] != identity or prior["status"] == "complete":
                raise ValueError("ledger identity/completion conflict")
            owner = prior.get("owner")
            live = process_owner(owner["pid"]) if owner else None
            if live is not None and (prior["status"] == "running" or live != owner):
                raise RuntimeError("previous ledger PID live; duplicate/reused owner rejected")
            self.last, self.completed = prior["charged_seconds"], prior["completed_updates"]
            if prior["status"] != "running":
                self.last+=setup_seconds
            if prior["status"] == "running":
                elapsed = time.time()-prior["wall_timestamp"]
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("ledger wall clock moved backwards")
                self.last += elapsed
        if type(self.last) not in (int, float) or not math.isfinite(self.last) or self.last < 0:
            raise ValueError("ledger charged time invalid")
        self.floor, self.started = self.last, time.monotonic()
        self.owner = process_owner(os.getpid())
        self._write("running", "startup")

    def _write(self, status: str, phase: str):
        atomic_json(self.path, dict(identity=self.identity, status=status, owner=self.owner,
                    charged_seconds=self.last, completed_updates=self.completed, phase=phase,
                    wall_timestamp=time.time(), duration_limit_seconds=None))

    def charge(self, completed: int, phase: str, *, replay: bool = False):
        if type(completed) is not int or completed < 0 or (
                completed < self.completed and not replay):
            raise ValueError("ledger completed-update rollback requires durable replay")
        self.last = max(self.last, self.floor+time.monotonic()-self.started)
        self.completed = completed
        self._write("running", phase)
        return self.last

    def close(self, status: str):
        self.charge(self.completed, "finalize")
        self._write(status, "finalized")
