"""Offline, byte-bound VideoMAE extraction with one durable cumulative clock.

The public CLI always enforces the full production inventory. Small fixtures use
direct helpers; they cannot select a relaxed CLI admission policy.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.version
from packaging.markers import Marker
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEModel

from despamo.appearance.provenance import digest, file_hash
from tools.motion.package import (
    publish_motion_package,
    resume_motion_package_publication,
    validate_motion_package,
)
from tools.motion.records import (
    MODEL,
    PROCESSOR,
    make_clip_receipt,
    make_motion_identity,
    regular_path,
    source_identity,
    window_rows,
)
from tools.motion.recovery import BOOTSTRAP_NAME, recovery_for

CAP = 18000
AUTHORIZATION = {"extraction_cap_seconds": CAP, "training_cap_seconds": 64800,
                 "cap_transfer": False}
PILOT = Path("/home/kan/datasets/despamo/signer-pilot")
FRAME_ROOT = Path("/home/kan/datasets/despamo/phoenix-frames-210x260px")
SNAPSHOT = (
    Path("/home/kan/datasets/spamo/hf_cache/models--MCG-NJU--videomae-large/snapshots")
    / str(MODEL["revision"]))
CHECKOUT = Path(__file__).resolve().parents[2]
VERSIONS = {"torch": "2.0.1", "transformers": "4.32.0", "numpy": "1.26.4",
            "pillow": "11.3.0", "safetensors": "0.8.0", "huggingface-hub": "0.36.2",
            "pydantic": "2.13.5", "pytest": "8.4.2"}
RUNTIME_FILES = {".writer.lock", "owner.json", "budget.json", "status.json", "profile.json",
                 "stage-request.json", "stage-result.json", "stage-error.json",
                 "worker-owner.json", "profile-samples", "generating-run.json",
                 "train", "receipts", "complete"}
_VERIFIED: dict[str, tuple] = {}
_MEASUREMENTS: ContextVar[dict | None] = ContextVar("motion_measurements", default=None)
CODE_FILES = (
    "tools/motion/extractor.py", "tools/motion/records.py", "tools/motion/package.py",
    "tools/motion/__init__.py", "scripts/extract_motion_features.py",
    "src/despamo/appearance/provenance.py", "src/despamo/appearance/schema.py",
    "src/despamo/data/manifest.py", "src/despamo/data/motion_timing.py",
    "tools/motion/pyproject.toml", "tools/motion/recovery.py",
)
BOOTSTRAP_FILES = {
    ".writer.lock", "owner.json", "budget.json", "status.json", "binding.json",
    "input-observation.json", "prepare-request.json", "prepared.json", "prepare-error.json",
    "parent-watchdog.json",
}
CLOSING_BOUND_SECONDS = 5.0


@contextmanager
def measure_clip() -> Iterator[dict]:
    stats = {key: 0 for key in ("source_bytes", "source_frames", "inference_batches",
                               "inference_windows", "input_frames")}
    token = _MEASUREMENTS.set(stats)
    try:
        yield stats
    finally:
        stats["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        _MEASUREMENTS.reset(token)


def measured(name: str, started: float, **counts) -> None:
    stats = _MEASUREMENTS.get()
    if stats is not None:
        stats[name] = stats.get(name, 0) + time.monotonic() - started
        for key, value in counts.items():
            stats[key] = stats.get(key, 0) + value


def read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(regular_path(path).read_bytes(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def safe_path(path: Path) -> Path:
    if not path.is_absolute() or path != path.resolve() or any(
        p.is_symlink() for p in (path, *path.parents)
    ):
        raise ValueError(f"canonical unaliased path required: {path}")
    if path.exists() and path.is_file():
        regular_path(path)
    return path


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(path: Path) -> None:
    safe_path(path)
    if not path.exists():
        durable_mkdir(path.parent)
        path.mkdir()
        fsync_directory(path.parent)
    if not path.is_dir():
        raise ValueError("directory required")


def atomic_write(path: Path, write: Callable) -> None:
    """Never remove crash debris; only replace this call's own deterministic temp."""
    safe_path(path)
    durable_mkdir(path.parent)
    tmp = path.parent / f".partial-{path.name}"
    # A prior incomplete atomic write is inspectable debris, not implicit permission.
    with tmp.open("xb") as handle:
        write(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)


def atomic_json(path: Path, value: dict) -> None:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    atomic_write(path, lambda f: f.write(raw))


@contextmanager
def file_lock(lock: Path) -> Iterator[int]:
    safe_path(lock)
    durable_mkdir(lock.parent)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        regular_path(lock)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another writer owns motion root") from None
        yield fd
    finally:
        os.close(fd)


@contextmanager
def exclusive_writer(root: Path) -> Iterator[int]:
    safe_path(root)
    with file_lock(root / ".writer.lock") as fd:
        yield fd


def make_processor() -> Any:
    return VideoMAEImageProcessor(
        do_resize=True, size={"shortest_edge": 224}, resample=Image.Resampling.BILINEAR,
        do_center_crop=True, crop_size={"height": 224, "width": 224},
        do_rescale=True, rescale_factor=1 / 255, do_normalize=True,
        image_mean=[0.5] * 3, image_std=[0.5] * 3)


def processor_tensor(processor: Any, videos: list) -> torch.Tensor:
    values = np.asarray(processor(videos, return_tensors="np")["pixel_values"])
    if values.shape != (len(videos), 16, 3, 224, 224) or values.dtype != np.float32:
        raise ValueError("processor shape/dtype mismatch")
    return torch.from_numpy(values)


def snapshot_files(snapshot: Path) -> list[dict]:
    safe_path(snapshot)
    if snapshot != SNAPSHOT or not snapshot.is_dir() or snapshot.is_symlink():
        raise ValueError("exact cached revision snapshot required")
    result = []
    for name in ("config.json", "model.safetensors"):
        p = snapshot / name
        resolved = regular_path(p.resolve(strict=True))
        if resolved.parent != snapshot.parents[1] / "blobs":
            raise ValueError("model snapshot must read regular cached blob bytes")
        result.append(evidence(resolved))
    config = json.loads((snapshot / "config.json").read_bytes())
    if any(config.get(k) != v for k, v in {
        "model_type": "videomae", "num_hidden_layers": 24, "hidden_size": 1024,
        "num_frames": 16, "num_channels": 3, "image_size": 224,
        "torch_dtype": "float32", "use_mean_pooling": False,
    }.items()):
        raise ValueError("cached VideoMAE config mismatch")
    return result


def load_encoder(snapshot: Path) -> tuple[object, object]:
    """Load on CPU; caller moves model to GPU inside the charged stage only."""
    before = snapshot_files(snapshot)
    loaded = VideoMAEModel.from_pretrained(
        str(snapshot), local_files_only=True, use_safetensors=True,
        trust_remote_code=False, torch_dtype=torch.float32, output_loading_info=True)
    model, info = cast(Any, loaded)
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
        raise ValueError(f"encoder weights incomplete: {info}")
    if snapshot_files(snapshot) != before:
        raise ValueError("cached model bytes changed while loading")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("FP32 model parameters required")
    model.requires_grad_(False).eval()
    return model, make_processor()


def extract_motion_clip(paths: tuple[Path, ...], model: object, processor: object, *,
                        batch_size: int = 16) -> tuple[np.ndarray, list[dict], tuple[dict, ...]]:
    if type(batch_size) is not int or not 1 <= batch_size <= 16 or not paths:
        raise ValueError("nonempty clip and batch size 1..16 required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = cast(Any, model)
    encoder.eval()
    params = tuple(encoder.parameters())
    if not params or any(p.dtype != torch.float32 for p in params):
        raise ValueError("FP32 model parameters required")
    device = params[0].device
    frames, sources = [], []
    for index, path in enumerate(paths):
        started = time.monotonic()
        raw = regular_path(path).read_bytes()
        measured("read_seconds", started, source_bytes=len(raw), source_frames=1)
        started = time.monotonic()
        sha = hashlib.sha256(raw).hexdigest()
        measured("hash_seconds", started)
        started = time.monotonic()
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (210, 260):
                raise ValueError("exact 210x260 source image required")
            frames.append((index, image.convert("RGB")))
        measured("decode_seconds", started)
        sources.append({"index": index, "path": path.as_posix(), "sha256": sha})
    plans = window_rows(len(frames))
    observed, outputs = [], []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for offset in range(0, len(plans), batch_size):
            batch = [[frames[i] for i in plan["input_source_indices"]]
                     for plan in plans[offset:offset + batch_size]]
            videos = [[image for _, image in video] for video in batch]
            started = time.monotonic()
            tensor = processor_tensor(processor, videos).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            measured("processor_transfer_seconds", started)
            # These ordinals describe the actual ordered video objects at invocation.
            actual_indices = [[index for index, _ in video] for video in batch]
            started = time.monotonic()
            out = encoder(pixel_values=tensor, output_hidden_states=True, return_dict=True)
            selected = out.hidden_states[-1][:, 0]
            if (selected.shape != (len(batch), 1024) or selected.dtype != torch.float32
                    or not torch.isfinite(selected).all()):
                raise ValueError("observed output shape/dtype/finiteness invalid")
            outputs.append(selected.cpu().numpy().copy())
            measured("inference_seconds", started, inference_batches=1,
                     inference_windows=len(batch), input_frames=len(batch) * 16)
            for indices in actual_indices:
                start = indices[0]
                observed.append({"row": len(observed), "input_source_indices": indices,
                                 "real_start": start, "real_end": max(indices),
                                 "centre": start + 7.5})
    return np.concatenate(outputs), sources, tuple(observed)


def evidence(path: Path) -> dict:
    return {"path": str(regular_path(path)), "sha256": file_hash(path)}


def verify_evidence(identity: dict, *, clip_check: bool = False) -> None:
    def walk(value):
        if isinstance(value, dict):
            if "path" in value or "sha256" in value:
                if set(value) != {"path", "sha256"}:
                    raise ValueError("invalid byte evidence")
                p = regular_path(Path(value["path"]))
                s = p.stat()
                signature = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
                             value["sha256"])
                if _VERIFIED.get(str(p)) == signature:
                    return
                if evidence(p) != value:
                    raise ValueError("source/code/environment/model evidence drift")
                _VERIFIED[str(p)] = signature
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
    walk(identity["code"])
    if clip_check:
        walk(identity["environment"]["lock"])
        walk(identity["environment"].get("model_files", []))
        # Actual preprocessing and model sources are monitored between clips;
        # the full copied environment is byte-checked at stage admission/publication.
        for item in identity["environment"].get("installed_files", []):
            if any(part in item["path"] for part in (
                "/PIL/", "/models/videomae/", "/image_transforms.py",
                "/image_utils.py", "/image_processing_utils.py", "/feature_extraction_utils.py",
            )):
                walk(item)
    else:
        walk(identity["environment"])
    walk(identity["inputs"]["frame_rows"])
    walk(identity["inputs"]["annotation"])


def generate_clip(root: Path, clip: str, row: dict, identity: dict, model: Any,
                  processor: Any) -> dict:
    if clip in {"", ".", ".."} or any(c in clip for c in "/\\"):
        raise ValueError("unsafe clip ID")
    safe_path(root)
    if root.name != f"motion-features-{digest(identity)}":
        raise ValueError("foreign feature root")
    array_path = safe_path(root / "train" / f"{clip}.npy")
    receipt_path = safe_path(root / "receipts/train" / f"{clip}.json")
    if receipt_path.exists():
        try:
            receipt = read_json(receipt_path)
            expected = make_clip_receipt(clip, row, receipt["source_rows"],
                                         tuple(receipt["windows"]), array_path, identity)
            if digest(receipt) != digest(expected):
                raise ValueError("completed clip receipt drift")
            return receipt
        except (OSError, KeyError, TypeError, EOFError) as exc:
            raise ValueError("corrupt completed clip") from exc
    paths = tuple(Path(identity["inputs"]["frame_root"]) / p for p in row["paths"])
    array, sources, windows = extract_motion_clip(paths, model, processor)
    # Preserve recorded relative/absolute spelling while hashing the consumed bytes.
    for source, spelling in zip(sources, row["paths"], strict=True):
        source["path"] = spelling
    source_hash = source_identity(tuple(Path(r["path"]) for r in sources),
                                  tuple(r["sha256"] for r in sources))
    if source_hash != row["source_hash"]:
        raise ValueError("consumed source differs from spatial source receipt")
    if digest(windows) != digest(window_rows(row["frame_count"])):
        raise ValueError("observed model window order mismatch")
    verify_evidence(identity, clip_check=True)
    # Recheck current bytes before committing either array or its receipt.
    if any(file_hash(regular_path(p)) != r["sha256"]
           for p, r in zip(paths, sources, strict=True)):
        raise ValueError("consumed source changed before clip commit")
    started = time.monotonic()
    atomic_write(array_path, lambda f: np.save(f, array, allow_pickle=False))
    measured("array_write_seconds", started)
    started = time.monotonic()
    receipt = make_clip_receipt(clip, row, sources, windows, array_path, identity)
    atomic_json(receipt_path, receipt)
    measured("receipt_validation_write_seconds", started)
    return receipt


def check_owned_inventory(root: Path, frames: dict) -> None:
    safe_path(root)
    if root.exists():
        for p in root.iterdir():
            safe_path(p)
            if p.name not in RUNTIME_FILES:
                raise ValueError(f"foreign/debris root entry: {p}")
    clips = frames["clips"]
    for relative, suffix in (("train", ".npy"), ("receipts/train", ".json")):
        directory = safe_path(root / relative)
        if directory.exists():
            if not directory.is_dir():
                raise ValueError("unsafe clip directory")
            allowed = {f"{clip}{suffix}" for clip in clips}
            for p in directory.iterdir():
                regular_path(p)
                if p.name not in allowed:
                    raise ValueError("foreign clip or interrupted temp debris")
    receipts = root / "receipts"
    if receipts.exists() and {p.name for p in receipts.iterdir()} - {"train"}:
        raise ValueError("foreign receipts directory")


def finish_package(root: Path, identity: dict, frames: dict, *,
                   before_commit: Callable[[], None] | None = None) -> dict:
    verify_evidence(identity)
    check_owned_inventory(root, frames)
    clips = {}
    for clip, row in sorted(frames["clips"].items()):
        p = root / "receipts/train" / f"{clip}.json"
        r = read_json(p)
        rebuilt = make_clip_receipt(clip, row, r["source_rows"], tuple(r["windows"]),
                                    root / "train" / f"{clip}.npy", identity)
        if digest(rebuilt) != digest(r):
            raise ValueError("actual generation receipt drift")
        clips[clip] = file_hash(p)
    run = {"schema_version": 1, "origin": "new_extraction_v1", "featurekey": digest(identity),
           "authorization": AUTHORIZATION, "code": identity["code"], "clips": clips}
    p = root / "generating-run.json"
    if p.exists():
        if read_json(p) != run:
            raise ValueError("immutable generating-run conflict")
    else:
        atomic_json(p, run)
    annotation = Path(identity["inputs"]["annotation"]["path"])
    if (root / "complete").exists():
        return resume_motion_package_publication(
            root, identity, frames, annotation, before_commit=before_commit)
    return publish_motion_package(root, identity, frames, annotation, before_commit=before_commit)


def clocks() -> tuple[float, float, str]:
    return (time.monotonic(), time.time(),
            Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def process_identity(pid: int) -> dict | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19], "pgid": int(fields[2]),
                "command": os.fsdecode(Path(f"/proc/{pid}/cmdline").read_bytes())}
    except FileNotFoundError:
        return None


class ExtractionClock:
    """Writer-held clock. Completed adoption never mutates the ledger."""

    def __init__(self, path: Path, featurekey: str | None, *, resume: bool,
                  bootstrapkey: str | None = None,
                  clocks: Callable = clocks, inherited_seconds: float = 0):
        if (type(inherited_seconds) not in (int, float)
                or not math.isfinite(inherited_seconds) or not 0 <= inherited_seconds < CAP):
            raise ValueError("invalid inherited extraction spend")
        self.path, self.clocks = path, clocks
        self.read_only = False
        self.admitted = False
        self.parent_guard: ParentDeadlineGuard | None = None
        policy = "motion-extraction-clock-v2" if bootstrapkey is not None else (
            "motion-extraction-clock-v1")
        if path.exists():
            self.state = read_json(path)
            if (self.state.get("policy") != policy
                    or (bootstrapkey is None and self.state.get("featurekey") != featurekey)
                    or self.state.get("bootstrapkey") != bootstrapkey
                    or digest(self.state.get("authorization")) != digest(AUTHORIZATION)):
                raise ValueError("clock identity/cap/authorization changed")
            if not resume:
                raise ValueError("existing clock requires --resume")
            self._validate()
            if self.state.get("inherited_seconds", 0) != inherited_seconds:
                raise ValueError("inherited extraction spend changed")
            if self.spent < inherited_seconds:
                raise ValueError("inherited extraction spend refunded")
            if self.state["status"] == "terminal_pending":
                admission_path = Path(self.state["terminal_close"]["admission_path"])
                if admission_path.exists():
                    self.state = admitted_terminal_snapshot(self)
                    self.admitted = True
                else:
                    active = self.state["active"]
                    owner = active["owner"]
                    if (owner is not None and clocks()[2] == active["boot_id"]
                            and process_identity(owner["pid"]) == owner):
                        raise RuntimeError("pending terminal owner still live")
                    self.finish("interrupted")
            elif self.state["status"] in {"profiled", "complete"}:
                self.admitted = True
            failure_observation = self.state.get("failure_close_observation")
            if (bootstrapkey is not None and self.state["status"] in {"failed", "capped"}
                    and isinstance(failure_observation, dict)):
                mono, wall, boot = clocks()
                delta = (mono - failure_observation["monotonic"]
                         if boot == failure_observation["boot_id"]
                         else wall - failure_observation["wall"])
                if not math.isfinite(delta) or delta < 0:
                    delta = CAP
                self.state["spent_seconds"] = max(
                    self.spent, failure_observation["spent_seconds"] + delta)
                if self.state["stages"]:
                    self.state["stages"][-1]["end_spent"] = self.spent
                self.state.pop("failure_close_observation", None)
                self.save()
            terminal = self.state.get("terminal_close")
            if (self.state["status"] in {"profiled", "complete"} and isinstance(terminal, dict)
                    and (Path(terminal["failure_path"]).exists()
                         or not Path(terminal["receipt_path"]).exists())):
                owner = self.state["stages"][-1]["owner"]
                if (owner is not None and clocks()[2] == terminal["clock_boot_id"]
                        and process_identity(owner["pid"]) == owner):
                    raise RuntimeError("unsealed closing owner still live")
                mono, wall, boot = clocks()
                delta = (mono - terminal["started_monotonic"] if boot == terminal["clock_boot_id"]
                         else wall - terminal["started_wall"])
                if not math.isfinite(delta) or delta < 0:
                    delta = CAP
                observed = terminal["observed_start_spend"] + delta
                self.state["spent_seconds"] = max(self.spent, observed)
                self.state["status"] = "capped" if self.spent >= CAP - 30 else "interrupted"
                self.state["stages"][-1].update(status=self.state["status"], end_spent=self.spent)
                self.save()
            if self.state["status"] == "running":
                mono, wall, boot = clocks()
                active = self.state["active"]
                for owner in (active["owner"], self.state["child"]):
                    if (owner is not None and boot == active["boot_id"]
                            and process_identity(owner["pid"]) == owner):
                        raise RuntimeError("owned process still live; refuse overlap/adoption")
                self.finish("interrupted")
        else:
            if resume:
                raise ValueError("resume requires existing owned clock")
            self.state = {"schema_version": 1, "policy": policy,
                          "featurekey": featurekey, "authorization": dict(AUTHORIZATION),
                          "spent_seconds": float(inherited_seconds), "status": "created",
                          "active": None,
                          "child": None, "stages": []}
            if inherited_seconds:
                self.state["inherited_seconds"] = float(inherited_seconds)
            self.state["watchdogs"] = []
            if bootstrapkey is not None:
                self.state.update(bootstrapkey=bootstrapkey, binding=None)

    def _validate(self):
        if self.state.get("schema_version") != 1:
            raise ValueError("unknown clock schema")
        spent = self.state["spent_seconds"]
        if isinstance(spent, bool) or not isinstance(spent, (float, int)) or not (
            math.isfinite(spent) and spent >= 0
        ):
            raise ValueError("invalid clock spend")
        if self.state["status"] not in {
            "created", "running", "terminal_pending", "profiled", "complete",
            "failed", "capped", "interrupted"
        }:
            raise ValueError("unknown clock status")
        if (self.state["status"] in {"running", "terminal_pending"}) != (
            self.state["active"] is not None
        ):
            raise ValueError("clock active/status mismatch")
        previous = 0.0
        for stage in self.state["stages"]:
            end = stage["end_spent"]
            if type(end) not in (int, float) or not math.isfinite(end) or not (
                previous <= end <= spent
            ):
                raise ValueError("clock spend reset below committed stage")
            previous = end
        if self.state["status"] == "complete" and spent >= CAP:
            raise ValueError("completed extraction exceeded cap")

    @property
    def spent(self) -> float:
        return float(self.state["spent_seconds"])

    def save(self):
        if not self.read_only:
            atomic_json(self.path, self.state)

    def start(self, mode: str, initial_seconds: float = 0, *, phase: str = "extracting"):
        if self.state["status"] in {"running", "terminal_pending", "complete"}:
            raise ValueError("clock cannot restart active/complete stage")
        if self.spent + initial_seconds >= CAP - 30:
            raise TimeoutError("extraction cap exhausted")
        mono, wall, boot = self.clocks()
        self.admitted = False
        self.state["spent_seconds"] += initial_seconds
        self.state.update(status="running", active={"mode": mode, "boot_id": boot,
                          "phase": phase,
                          "started_boot_id": boot,
                          "monotonic": mono, "wall": wall, "owner": process_identity(os.getpid()),
                          "started_spent": self.spent}, child=None)
        self.save()

    def bind_feature(self, identity: dict, root: Path):
        """One immutable late binding; clock path/spend never move or restart."""
        key = digest(identity)
        if root.name != f"motion-features-{key}":
            raise ValueError("feature root/identity mismatch")
        binding = {"schema_version": 1, "bootstrapkey": self.state["bootstrapkey"],
                   "featurekey": key, "root": str(safe_path(root)), "identity": identity}
        path = self.path.parent / "binding.json"
        if path.exists():
            if digest(read_json(path)) != digest(binding):
                raise ValueError("verified feature/input/code/hardware binding changed on resume")
        elif self.state.get("binding") is not None or self.read_only:
            raise ValueError("bound extraction lost immutable binding")
        else:
            atomic_json(path, binding)
        expected = {"featurekey": key, "root": str(root), "sha256": file_hash(path)}
        if self.state.get("binding") not in (None, expected):
            raise ValueError("clock/feature binding drift")
        if self.state["featurekey"] not in (None, key):
            raise ValueError("clock featurekey cannot change")
        self.tick()
        self.state.update(featurekey=key, binding=expected)
        if self.state["active"] is not None:
            active = self.state["active"]
            active["preparation_seconds"] = self.spent - active["started_spent"]
            active["phase"] = "extracting"
        self.save()

    def tick(self):
        active = self.state["active"]
        if active is None:
            return
        mono, wall, boot = self.clocks()
        if boot == active["boot_id"]:
            delta = mono - active["monotonic"]
        else:
            delta = wall - active["wall"]
        if not math.isfinite(delta) or delta < 0:
            # Unknown/rolled-back clock must not grant fresh allowance.
            delta = CAP
        self.state["spent_seconds"] += delta
        active.update(monotonic=mono, wall=wall, boot_id=boot)

    def check(self):
        self.tick()
        self.save()
        self.tick()  # A persist/fsync can block; the pre-write sample is not admission.
        if self.spent >= CAP - 30:
            raise TimeoutError("extraction cap reached")

    def finish(self, status: str):
        if self.admitted and self.state["status"] == "complete":
            return
        self.tick()
        successful = status in {"profiled", "complete"}
        if successful and self.spent >= CAP - 30:
            raise TimeoutError("extraction cannot close successfully without cleanup margin")
        active = self.state["active"]
        if successful and active is not None:
            active["phase"] = "closing"
            self.save()  # Persist an unclosed interval, never success before this IO.
            self.tick()
            if self.spent >= CAP - 30:
                raise TimeoutError("clock persist exhausted successful close allowance")
        anchor = self.clocks()
        observed = self.spent
        upper = observed
        if successful and self.parent_guard is not None:
            upper = self.parent_guard.closing_upper_bound
            if upper is None or observed > upper:
                raise TimeoutError("successful closure exceeded its reserved bound")
            self.state["terminal_close"] = self.parent_guard.binding()
            self.state["terminal_candidate"] = {"status": status, "spent_upper_bound": upper}
            self.state["spent_semantics"] = "observed_unclosed_terminal_interval"
            self.state["status"] = "terminal_pending"
            self.save()
            self.tick()
            if self.spent >= CAP - 30 or self.spent > upper:
                raise TimeoutError("terminal candidate persistence exceeded closing allowance")
            path = Path(self.state["terminal_close"]["candidate_path"])
            atomic_json(path, {"schema_version": 1, "clock": copy.deepcopy(self.state),
                               "budget_sha256": file_hash(self.path)})
            self.tick()
            if self.spent >= CAP - 30 or self.spent > upper:
                raise TimeoutError("terminal snapshot IO exceeded closing allowance")
            return
        for watchdog in self.state.get("watchdogs", []):
            if watchdog.get("ended_spent") is None:
                watchdog.update(ended_spent=self.spent, reconciliation=status)
        if self.state["active"] is not None:
            self.state["stages"].append({"mode": self.state["active"]["mode"],
                                        "phase": self.state["active"]["phase"],
                                        "boot_id": self.state["active"].get("started_boot_id"),
                                        "owner": self.state["active"]["owner"],
                                        "child": self.state["child"],
                                        "started_spent": self.state["active"]["started_spent"],
                                        "status": status, "end_spent": self.spent,
                                        "preparation_seconds": self.state["active"].get(
                                            "preparation_seconds", self.spent -
                                            self.state["active"]["started_spent"])})
        self.state.update(status=status, active=None, child=None)
        if not successful:
            self.state["spent_semantics"] = "observed_uncapped_lower_bound"
            self.state["failure_close_observation"] = {
                "monotonic": anchor[0], "wall": anchor[1], "boot_id": anchor[2],
                "spent_seconds": observed}
        self.save()
        mono, wall, boot = self.clocks()
        elapsed = mono - anchor[0] if boot == anchor[2] else wall - anchor[1]
        actual = observed + (elapsed if math.isfinite(elapsed) and elapsed >= 0 else CAP)
        if successful and (actual >= CAP - 30
                           or (self.parent_guard is not None and actual > upper)):
            self.state.update(status="capped", spent_seconds=actual,
                              spent_semantics="observed_uncapped_lower_bound")
            if self.state["stages"]:
                self.state["stages"][-1].update(status="capped", end_spent=actual)
            self.save()
            mono2, wall2, boot2 = self.clocks()
            extra = mono2 - mono if boot2 == boot else wall2 - wall
            self.state["spent_seconds"] += max(0, extra)
            raise TimeoutError("blocking final ledger IO exceeded successful close bound")
        if self.parent_guard is None or not successful:
            self.state["spent_seconds"] = actual
        self.state["last_observed_spend"] = actual

    def return_admission(self, guard: ParentDeadlineGuard) -> dict:
        """After all cleanup, admit in memory only; persistence is a caller/CLI tail."""
        if self.state["status"] != "terminal_pending":
            raise ValueError("pending terminal transaction required")
        guard.check_return()
        self.tick()
        upper = self.state["terminal_candidate"]["spent_upper_bound"]
        if self.spent > upper or self.spent >= CAP:
            raise TimeoutError("closing bound does not cover observed parent tail")
        proof = {"schema_version": 1, "policy": "motion-terminal-admission-v1",
                 "terminal_close": copy.deepcopy(self.state["terminal_close"]),
                 "candidate_sha256": guard.candidate_sha256,
                 "terminal_sha256": guard.terminal_sha256,
                 "budget_sha256": guard.budget_sha256,
                 "bootstrapkey": self.state.get("bootstrapkey"),
                 "featurekey": self.state["featurekey"],
                 "status": self.state["terminal_candidate"]["status"],
                 "spent_upper_bound": upper, "observed_spend_at_return_check": self.spent,
                 "parent_checked_monotonic": time.monotonic(), "boot_id": clocks()[2]}
        guard.check_return()
        return proof


class ParentDeadlineGuard:
    """Independent parent watchdog remains armed through terminal fsync and admission.

    Successful closure reserves a small truthful upper bound; the terminal receipt
    never claims to know the time of its own last fsync. Parent must check again.
    """

    def __init__(self, clock: ExtractionClock, lock_fd: int, *,
                 timeout_seconds: float | None = None):
        self.clock = clock
        self.bootstrap = clock.path.parent
        self.nonce = os.urandom(16).hex()
        self.receipt = self.bootstrap / f"terminal-{self.nonce}.json"
        self.candidate = self.bootstrap / f"terminal-candidate-{self.nonce}.json"
        self.admission = self.bootstrap / f"terminal-admission-{self.nonce}.json"
        self.failure = self.bootstrap / f"terminal-failure-{self.nonce}.json"
        self.closing_upper_bound: float | None = None
        self.closing_deadline: float | None = None
        self.closing_started: tuple[float, float, str] | None = None
        self.closing_start_spend: float | None = None
        self.closed = False
        self.candidate_sha256 = None
        self.terminal_sha256 = None
        self.budget_sha256 = None
        clock.tick()
        duration = CAP - clock.spent if timeout_seconds is None else timeout_seconds
        if duration <= 0:
            raise TimeoutError("no parent deadline allowance")
        self.deadline = time.monotonic() + duration
        read_fd, self.write_fd = os.pipe()
        self.ack_fd, ack_write = os.pipe()
        parent = process_identity(os.getpid())
        code = '''import hashlib,json,os,select,signal,sys,time
fd,ack,lock=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
parent=json.loads(sys.argv[4]); deadline=float(sys.argv[5]); failure=sys.argv[6]
def identity(pid):
    try:
        f=open('/proc/%s/stat'%pid).read().rsplit(')',1)[1].split()
        if f[0]=='Z': return None
        return {'pid':pid,'start_ticks':f[19],'pgid':int(f[2]),
                'command':os.fsdecode(open('/proc/%s/cmdline'%pid,'rb').read())}
    except FileNotFoundError: return None
def write(path,value):
    tmp=path+'.partial'
    with open(tmp,'xb') as h:
        h.write((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\\n').encode())
        h.flush();os.fsync(h.fileno())
    os.replace(tmp,path)
    d=os.open(os.path.dirname(path),os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(d)
    finally: os.close(d)
def stop(signum=None,frame=None):
    signal.signal(signal.SIGALRM,signal.SIG_IGN)
    if identity(parent['pid'])==parent:
        os.kill(parent['pid'],signal.SIGTERM)
        until=time.monotonic()+.5
        aborted=False
        while time.monotonic()<until:
            if select.select([fd],[],[],.02)[0]:
                token=os.read(fd,4096)
                if b'"op": "abort"' in token:
                    aborted=True;break
            if identity(parent['pid'])!=parent: break
        if not aborted and identity(parent['pid'])==parent:
            os.kill(parent['pid'],signal.SIGKILL)
    # Logging must not postpone deadline enforcement on a blocking filesystem.
    write(failure,{'policy':'motion-terminal-failure-v1','deadline':deadline,
                   'observed_monotonic':time.monotonic(),'parent':parent})
    sys.exit(124)
signal.signal(signal.SIGALRM,stop)
signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
with os.fdopen(fd,'r') as commands:
    for line in commands:
        command=json.loads(line)
        if command['op']=='closing':
            deadline=min(deadline,command['deadline'])
            signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
        elif command['op']=='seal':
            value=command['receipt']
            for name,path in command['files'].items():
                value[name]=hashlib.sha256(open(path,'rb').read()).hexdigest()
            value['observed_before_receipt_write']=time.monotonic()
            if time.monotonic()>=deadline: stop()
            write(command['path'],value)
            if time.monotonic()>=deadline: stop()
            os.write(ack,b'1')
        elif command['op']=='release':
            if time.monotonic()>=deadline: stop()
            signal.setitimer(signal.ITIMER_REAL,0)
            sys.exit(0)
sys.exit(125)
'''
        self.child = subprocess.Popen(
            [sys.executable, "-c", code, str(read_fd), str(ack_write), str(lock_fd),
             json.dumps(parent), str(self.deadline), str(self.failure)],
            pass_fds=(read_fd, ack_write, lock_fd), start_new_session=True)
        os.close(read_fd)
        os.close(ack_write)
        self.process = process_identity(self.child.pid)
        self.previous = {s: signal.signal(s, self.interrupted)
                         for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            atomic_json(self.bootstrap / "parent-watchdog.json", {
                "schema_version": 1, "policy": "motion-parent-watchdog-v1", "nonce": self.nonce,
                "boot_id": clocks()[2], "parent": parent, "watchdog": self.process,
                "deadline_monotonic": self.deadline})
        except BaseException:
            self.close()
            raise
        clock.parent_guard = self

    def interrupted(self, signum, frame):
        raise TimeoutError(f"owned parent deadline/signal {signum}")

    def send(self, value: dict):
        os.write(self.write_fd, (json.dumps(value, allow_nan=False) + "\n").encode())

    def begin_closing(self, bound_seconds: float = CLOSING_BOUND_SECONDS):
        self.clock.tick()
        self.closing_started = self.clock.clocks()
        self.closing_start_spend = self.clock.spent
        self.closing_upper_bound = self.clock.spent + bound_seconds
        self.closing_deadline = min(self.deadline, time.monotonic() + bound_seconds)
        if self.closing_upper_bound >= CAP - 30:
            raise TimeoutError("closing reserve does not fit remaining cap/margin")
        self.send({"op": "closing", "deadline": self.closing_deadline})

    def binding(self) -> dict:
        if self.closing_started is None:
            raise ValueError("closing interval not armed")
        return {"schema_version": 1, "policy": "motion-terminal-close-v1", "nonce": self.nonce,
                "receipt_path": str(self.receipt), "failure_path": str(self.failure),
                "candidate_path": str(self.candidate), "admission_path": str(self.admission),
                "spent_upper_bound": self.closing_upper_bound,
                "closing_deadline_monotonic": self.closing_deadline,
                "parent_deadline_monotonic": self.deadline, "boot_id": clocks()[2],
                "clock_boot_id": self.closing_started[2],
                "started_monotonic": self.closing_started[0],
                "started_wall": self.closing_started[1],
                "observed_start_spend": self.closing_start_spend}

    def seal(self, status_path: Path):
        self.send({"op": "seal", "path": str(self.receipt),
                   "files": {"candidate_sha256": self.binding()["candidate_path"],
                             "budget_sha256": str(self.clock.path),
                             "status_sha256": str(status_path)},
                   "receipt": {**self.binding(),
                               "bootstrapkey": self.clock.state.get("bootstrapkey"),
                               "featurekey": self.clock.state["featurekey"],
                               "status": self.clock.state["terminal_candidate"]["status"],
                               "clock_status": "terminal_pending"}})
        if os.read(self.ack_fd, 1) != b"1":
            raise TimeoutError("terminal receipt not acknowledged")
        if (self.closing_deadline is None or time.monotonic() >= self.closing_deadline
                or time.monotonic() >= self.deadline or self.failure.exists()):
            raise TimeoutError("parent independent closing cap check failed")
        receipt = validate_terminal_close(self.clock, allow_pending=True)
        self.candidate_sha256 = receipt["candidate_sha256"]
        self.budget_sha256 = receipt["budget_sha256"]
        self.terminal_sha256 = file_hash(self.receipt)
        self.check_return()
        self.send({"op": "release"})
        if self.child.wait(timeout=1) != 0:
            raise TimeoutError("parent terminal watchdog did not close successfully")
        self.check_return()

    def check_return(self):
        if (self.closing_deadline is None or time.monotonic() >= self.closing_deadline
                or time.monotonic() >= self.deadline or self.failure.exists()):
            raise TimeoutError("parent closing/global return deadline exceeded")
        # A filesystem existence check can itself block; sample after it too.
        if time.monotonic() >= self.closing_deadline or time.monotonic() >= self.deadline:
            raise TimeoutError("parent closing/global check IO exceeded deadline")

    def release_admission(self):
        if time.monotonic() >= self.deadline or self.failure.exists():
            raise TimeoutError("read-only parent admission deadline")
        self.send({"op": "release"})
        if self.child.wait(timeout=1) != 0 or time.monotonic() >= self.deadline:
            raise TimeoutError("read-only parent admission watchdog failed")

    def close(self):
        if self.closed:
            return
        if self.child.poll() is None:
            try:
                self.send({"op": "abort"})
            except BrokenPipeError:
                pass
            expired = time.monotonic() >= (self.closing_deadline or self.deadline)
            if expired:
                try:
                    self.child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait()
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)
        os.close(self.write_fd)
        os.close(self.ack_fd)
        if self.child.poll() is None:
            self.child.terminate()
            self.child.wait(timeout=2)
        self.clock.parent_guard = None
        self.closed = True


def validate_terminal_close(clock: ExtractionClock, *, allow_pending: bool = False) -> dict:
    terminal = clock.state.get("terminal_close")
    if not isinstance(terminal, dict) or Path(terminal["failure_path"]).exists():
        raise ValueError("successful bounded terminal receipt required")
    receipt = read_json(Path(terminal["receipt_path"]))
    candidate_path = Path(terminal["candidate_path"])
    candidate = read_json(candidate_path)
    pending_clock = candidate["clock"]
    requested = pending_clock["terminal_candidate"]["status"]
    current_active = clock.state.get("active")
    receipt_observation = receipt.get("observed_before_receipt_write")
    numbers = [terminal.get(k) for k in ("spent_upper_bound", "closing_deadline_monotonic",
                                       "parent_deadline_monotonic", "started_monotonic",
                                       "started_wall", "observed_start_spend")]
    if (any(type(v) not in (int, float) or not math.isfinite(cast(Any, v)) for v in numbers)
            or terminal.get("schema_version") != 1
            or terminal.get("policy") != "motion-terminal-close-v1"
            or terminal["closing_deadline_monotonic"] > terminal["parent_deadline_monotonic"]
            or terminal["started_monotonic"] < 0
            or type(receipt_observation) not in (int, float)
            or not math.isfinite(cast(Any, receipt_observation))
            or terminal["observed_start_spend"] < 0
            or pending_clock["featurekey"] != clock.state["featurekey"]
            or pending_clock.get("bootstrapkey") != clock.state.get("bootstrapkey")
            or (isinstance(current_active, dict)
                and pending_clock["active"]["owner"] != current_active["owner"])
            or pending_clock["active"]["started_spent"] > pending_clock["spent_seconds"]):
        raise ValueError("terminal candidate status/anchor invariant failed")
    if clock.state["status"] == "terminal_pending" and not allow_pending:
        raise ValueError("sealed candidate is not admitted complete")
    if (any(receipt.get(k) != v for k, v in terminal.items())
            or receipt.get("candidate_sha256") != file_hash(candidate_path)
            or receipt.get("budget_sha256") != candidate["budget_sha256"]
            or (allow_pending and receipt.get("budget_sha256") != file_hash(clock.path))
            or receipt.get("status_sha256") != file_hash(clock.path.parent / "status.json")
            or receipt.get("status") != requested or requested not in {"profiled", "complete"}
            or receipt.get("clock_status") != "terminal_pending"
            or pending_clock["status"] != "terminal_pending"
            or pending_clock["active"] is None
            or pending_clock["active"]["phase"] != "closing"
            or pending_clock["terminal_close"] != terminal
            or receipt.get("featurekey") != clock.state["featurekey"]
            or receipt.get("bootstrapkey") != clock.state.get("bootstrapkey")
            or not terminal["started_monotonic"] <= cast(Any, receipt_observation) < min(
                terminal["closing_deadline_monotonic"], terminal["parent_deadline_monotonic"])
            or terminal["spent_upper_bound"] != pending_clock["terminal_candidate"][
                "spent_upper_bound"]
            or pending_clock["spent_seconds"] > terminal["spent_upper_bound"]
            or terminal["observed_start_spend"] > terminal["spent_upper_bound"]
            or terminal["spent_upper_bound"] >= CAP):
        raise ValueError("terminal close receipt/budget/cap mismatch")
    return receipt


def admitted_terminal_snapshot(clock: ExtractionClock, *, admission: dict | None = None) -> dict:
    """Project immutable admitted state, never rewrite the pending budget snapshot."""
    terminal = clock.state["terminal_close"]
    if admission is None:
        admission = read_json(Path(terminal["admission_path"]))
    receipt = validate_terminal_close(clock, allow_pending=True)
    observed = admission.get("observed_spend_at_return_check")
    checked = admission.get("parent_checked_monotonic")
    if (admission.get("schema_version") != 1
            or admission.get("policy") != "motion-terminal-admission-v1"
            or admission.get("terminal_close") != terminal
            or admission.get("budget_sha256") != file_hash(clock.path)
            or admission.get("candidate_sha256") != receipt["candidate_sha256"]
            or admission.get("terminal_sha256") != file_hash(Path(terminal["receipt_path"]))
            or admission.get("status") != receipt["status"]
            or admission.get("featurekey") != clock.state["featurekey"]
            or admission.get("bootstrapkey") != clock.state.get("bootstrapkey")
            or admission.get("spent_upper_bound") != terminal["spent_upper_bound"]
            or type(observed) not in (int, float) or not math.isfinite(cast(Any, observed))
            or not terminal["observed_start_spend"] <= cast(Any, observed) <= terminal[
                "spent_upper_bound"]
            or type(checked) not in (int, float) or not math.isfinite(cast(Any, checked))
            or not receipt["observed_before_receipt_write"] <= cast(Any, checked) < min(
                terminal["closing_deadline_monotonic"], terminal["parent_deadline_monotonic"])
            or admission.get("boot_id") != terminal["boot_id"]):
        raise ValueError("terminal admission status/anchor/hash mismatch")
    state = copy.deepcopy(clock.state)
    active = state["active"]
    state["stages"].append({"mode": active["mode"], "phase": "closing",
                            "boot_id": active["started_boot_id"], "owner": active["owner"],
                            "child": state["child"], "started_spent": active["started_spent"],
                            "status": admission["status"],
                            "end_spent": terminal["spent_upper_bound"],
                            "preparation_seconds": active.get("preparation_seconds", 0)})
    state.update(status=admission["status"], active=None, child=None,
                 spent_seconds=terminal["spent_upper_bound"],
                 spent_semantics="observed_work_plus_conservative_closing_bound")
    return state


def record_terminal_admission(result: dict) -> None:
    """Caller/CLI tail, after extractor return. Outer Task4 watchdog must cover this IO."""
    proof = result.get("terminal_admission")
    if proof is None:
        return  # already-admitted read-only consumer admission
    if result["status"] not in {"profiled", "complete"} or proof["status"] != result["status"]:
        raise ValueError("unsuccessful result cannot admit terminal transaction")
    bootstrap = Path(result["bootstrap_root"])
    with exclusive_writer(bootstrap):
        path = Path(proof["terminal_close"]["admission_path"])
        raw = read_json(Path(result["budget_path"]))
        clock = ExtractionClock.__new__(ExtractionClock)
        clock.path, clock.state = Path(result["budget_path"]), raw
        validate_terminal_close(clock, allow_pending=True)
        if file_hash(clock.path) != proof["budget_sha256"]:
            raise ValueError("pending budget changed before caller admission")
        admitted_terminal_snapshot(clock, admission=proof)
        if path.exists():
            if read_json(path) != proof:
                raise ValueError("immutable terminal admission conflict")
        else:
            atomic_json(path, proof)
        admitted_terminal_snapshot(clock)


def supervise_command(command: list[str], clock: ExtractionClock, *,
                      lock_fd: int | tuple[int, ...] | None = None,
                      deadline_seconds: float | None = None) -> int:
    """Independent watchdog survives caller loss; EOF gate forbids unrecorded launch.

    Root flock is inherited by the watchdog and worker, so caller SIGKILL cannot
    allow a second writer while GPU work remains. No unowned PID is signalled.
    """
    clock.tick()
    available = CAP - clock.spent - 30 if deadline_seconds is None else deadline_seconds
    if available <= 0:
        raise TimeoutError("no watchdog allowance")
    gate_r, gate_w = os.pipe()
    lock_fds = () if lock_fd is None else (lock_fd,) if isinstance(lock_fd, int) else lock_fd
    # No GNU dependency: host timeout is uutils. Watchdog is this private copied
    # Python interpreter, in its own process group, with a hard group kill.
    code = '''import json, os, signal, subprocess, sys, time
fd, seconds = int(sys.argv[1]), float(sys.argv[2])
grace, locks = float(sys.argv[3]), tuple(json.loads(sys.argv[4]))
started = time.monotonic()
def stop(signum=None, frame=None):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.killpg(os.getpgrp(), signal.SIGTERM)
    time.sleep(grace)
    os.killpg(os.getpgrp(), signal.SIGKILL)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGALRM, stop)
signal.setitimer(signal.ITIMER_REAL, seconds)
token = os.read(fd, 1)
os.close(fd)
if token != b'1':
    sys.exit(125)
p = subprocess.Popen(sys.argv[5:], pass_fds=locks)
try:
    sys.exit(p.wait(timeout=max(.001, seconds - (time.monotonic() - started))))
except subprocess.TimeoutExpired:
    stop()
'''
    grace = min(5.0, available / 10)
    child = None
    launch_record = None
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", code, str(gate_r), str(available), str(grace),
             json.dumps(lock_fds), *command],
            start_new_session=True, pass_fds=(gate_r, *lock_fds))
        clock.state["child"] = process_identity(child.pid)
        if clock.state["child"] is None:
            raise RuntimeError("watchdog disappeared before ownership")
        launch_record = {"phase": clock.state["active"]["phase"],
                         "boot_id": clock.state["active"]["boot_id"],
                         "process": clock.state["child"], "started_spent": clock.spent,
                         "ended_spent": None, "returncode": None}
        clock.state.setdefault("watchdogs", []).append(launch_record)
        clock.save()
        os.write(gate_w, b"1")
        while child.poll() is None:
            time.sleep(0.05)
            clock.tick()
            clock.save()
        return child.wait()
    finally:
        os.close(gate_r)
        os.close(gate_w)
        if child is not None and child.poll() is None:
            # Popen object is our newly owned child; PGID equals its PID.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        clock.tick()
        if launch_record is not None:
            launch_record.update(ended_spent=clock.spent,
                                 returncode=None if child is None else child.returncode)
        clock.state["child"] = None
        clock.save()


def verify_distribution_lock(locked: dict, installed: dict[str, str]) -> None:
    packages = {p["name"]: p for p in locked["package"] if "registry" in p["source"]}
    for name, version in installed.items():
        if name not in packages or packages[name]["version"] != version:
            raise ValueError(f"installed distribution differs from lock: {name}")
        for dep in packages[name].get("dependencies", ()):
            if "marker" not in dep or Marker(dep["marker"]).evaluate():
                if dep["name"] not in installed:
                    raise ValueError(f"missing active locked dependency: {dep['name']}")
    if any(installed.get(name) != version for name, version in VERSIONS.items()):
        raise ValueError("missing or changed pinned direct dependency")


def environment_identity(lock: Path) -> dict:
    if sys.version_info[:2] != (3, 11) or Path(sys.prefix) != CHECKOUT / "tools/motion/.venv":
        raise ValueError("dedicated Python3.11 extractor environment required")
    if lock != CHECKOUT / "tools/motion/uv.lock":
        raise ValueError("dedicated UV lock required")
    locked = tomllib.loads(regular_path(lock).read_text())
    installed = {d.metadata["Name"].lower().replace("_", "-"): d for d in
                 importlib.metadata.distributions()}
    verify_distribution_lock(locked, {k: d.version for k, d in installed.items()})
    if any(importlib.metadata.version(k) != v for k, v in VERSIONS.items()) or (
        torch.__version__ != "2.0.1+cu117" or torch.version.cuda != "11.7"
    ):
        raise ValueError("pinned library/runtime mismatch")
    # Bind all installed files, including native processor/tensor libraries and
    # distribution metadata. UV copy mode makes these unique regular evidence.
    files = {}
    for d in installed.values():
        for f in d.files or ():
            if f.suffix == ".pyc":
                continue
            p = Path(os.path.abspath(str(d.locate_file(f))))
            if p.is_file():
                p = regular_path(p)
                files[str(p)] = evidence(p)
    return {"lock": evidence(lock), "versions": {k: d.version for k, d in installed.items()},
            "python": sys.version, "torch_runtime": torch.__version__, "cuda": torch.version.cuda,
            "installed_files": list(files.values()), "processor_config": make_processor().to_dict()}


def hardware_identity() -> dict:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version,compute_cap,memory.total",
         "--format=csv,noheader,nounits"], text=True, timeout=15).strip().splitlines()
    if len(text) != 1:
        raise ValueError("single GPU required")
    uuid, name, driver, cc, memory = (s.strip() for s in text[0].split(","))
    return {"uuid": uuid, "name": name, "driver": driver,
            "compute_capability": [int(x) for x in cc.split(".")], "memory_mib": int(memory)}


def require_gpu_idle() -> None:
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip()
    if active:
        raise RuntimeError("GPU process already active; no overlap authorized")


def require_storage(root: Path, remaining_windows: int) -> dict:
    # Array payload plus conservative receipt/metadata/temporary duplication.
    demand = remaining_windows * 1024 * 4 * 2 + 2 * 1024**3
    required = demand + 50 * 1024**3
    linux = shutil.disk_usage(root).free
    windows = int(subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command", "(Get-PSDrive -Name C).Free"],
        text=True, timeout=30).strip())
    if min(linux, windows) < required:
        raise ValueError("Linux/Windows demand + 50GiB reserve insufficient")
    return {"linux_free_bytes": linux, "windows_free_bytes": windows,
            "remaining_demand_bytes": demand, "reserve_bytes": 50 * 1024**3}


def annotation_records(raw: dict) -> dict:
    """Read legacy Phoenix integer records, using the existing dataset contract."""
    if not isinstance(raw, dict):
        raise ValueError("annotation dictionary required")
    records = {}
    for key in sorted(k for k in raw if type(k) is int):
        item = raw[key]
        if not isinstance(item, dict) or any(
            not isinstance(item.get(f), str) or not item[f].strip() for f in
            ("fileid", "signer", "gloss", "text", "en_text", "es_text", "fr_text")
        ):
            raise ValueError("invalid Phoenix annotation record")
        clip = item["fileid"]
        if clip in records or any(c in clip for c in "/\\"):
            raise ValueError("duplicate/unsafe annotation clip ID")
        records[clip] = item
    if not records:
        raise ValueError("annotation has no integer records")
    return records


def production_gate(frames: dict, annotation: dict) -> None:
    clips = frames.get("clips", {})
    if len(clips) != 7096 or set(clips) != set(annotation):
        raise ValueError("production physical train clip inventory mismatch")
    total, windows = 0, 0
    for clip, row in clips.items():
        if clip in {"", ".", ".."} or any(c in clip for c in "/\\"):
            raise ValueError("production unsafe clip ID")
        n = row["frame_count"]
        if type(n) is not int or n < 1 or row["source_indices"] != list(range(n)):
            raise ValueError("production frame ordinals invalid")
        if len(row["paths"]) != n or row["paths"] != sorted(set(row["paths"])):
            raise ValueError("production frame order invalid")
        for spelling in row["paths"]:
            p = Path(spelling)
            if (p.parent != FRAME_ROOT / "train" / clip or ".." in p.parts
                    or p.as_posix() != spelling):
                raise ValueError("production frame path outside exact physical source")
            regular_path(p)
        total += n
        windows += len(window_rows(n))
    groups = {"train": 0, "dev": 0, "test": 0}
    signers = set()
    for clip, item in annotation.items():
        if item["fileid"] != clip or not item["signer"]:
            raise ValueError("production annotation ID/signer invalid")
        signer = item["signer"]
        signers.add(signer)
        groups["dev" if signer == "Signer03" else "test" if signer == "Signer07"
               else "train"] += 1
    if (total, windows) != (827354, 93232) or groups != {
        "train": 5746, "dev": 582, "test": 768
    } or len(signers - {"Signer03", "Signer07"}) != 7:
        raise ValueError("production source/window/frozen logical split mismatch")


def prepare(args) -> tuple[dict, dict, Path]:
    if (args.mode not in {"preflight", "profile", "run"}
            or type(args.gpu_cap_seconds) is not int or args.gpu_cap_seconds != CAP
    ):
        raise ValueError("fixed extraction cap 18000 required")
    safe_path(args.adapted_root)
    if args.adapted_root.parent != PILOT or not args.adapted_root.name.startswith("e3-features-"):
        raise ValueError("canonical external adapted root required")
    if safe_path(args.output_base) != PILOT:
        raise ValueError("motion output must use approved external pilot root")
    frame_path = args.adapted_root / "complete/frame_rows.json"
    frames = read_json(frame_path)
    adapted = read_json(args.adapted_root / "complete/identity.json")
    if (adapted.get("protocol_hash") !=
            "0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7"
            or args.adapted_root.name != f"e3-features-{digest(adapted)}"
            or frames.get("encoder_key") != digest(adapted)):
        raise ValueError("adapted physical source lineage mismatch")
    # Annotation is a trusted, byte-pinned legacy NPY dictionary, never output data.
    regular_path(args.annotation)
    annotation = annotation_records(np.load(args.annotation, allow_pickle=True).item())
    production_gate(frames, annotation)
    environment = environment_identity(args.environment_lock)
    environment["model_files"] = snapshot_files(args.model_snapshot)
    environment["adapted_identity"] = evidence(args.adapted_root / "complete/identity.json")
    code = {"extractor" if i == 0 else "helpers" if i == 1 else f"closure_{i}":
            evidence(CHECKOUT / p) for i, p in enumerate(CODE_FILES)}
    identity = make_motion_identity(
        {"frame_root": str(FRAME_ROOT), "frame_rows": evidence(frame_path),
         "annotation": evidence(args.annotation)}, MODEL, PROCESSOR, environment, code,
        hardware_identity())
    return identity, frames, args.output_base / f"motion-features-{digest(identity)}"


def profile_ids(frames: dict) -> list[str]:
    ids = sorted(frames["clips"], key=lambda c: (frames["clips"][c]["frame_count"], c))
    return sorted({ids[round(q * (len(ids) - 1))] for q in (0, .1, .25, .5, .75, .9, 1)})


def tensor_parity(processor: Any) -> list[dict]:
    result = []
    for batch in (1, 8, 16):
        videos = [[Image.new("RGB", shape, (29, 71, 133)) for _ in range(16)] for shape in
                  [(210, 260), (260, 210), (9, 17), (224, 224)] * 4][:batch]
        optimized = processor_tensor(processor, videos)
        standard = processor(videos, return_tensors="pt")["pixel_values"]
        if not torch.equal(optimized, standard):
            raise ValueError("processor tensor bit parity failed")
        result.append({"batch_size": batch, "shape": list(optimized.shape), "bit_equal": True,
                       "sha256": hashlib.sha256(optimized.numpy().tobytes()).hexdigest()})
    return result


def _deadline(request: dict, cleanup_margin: float = 30):
    if (clocks()[2] != request["boot_id"]
            or time.monotonic() + cleanup_margin >= request["deadline_monotonic"]):
        raise TimeoutError("cooperative extraction deadline")


def record_profile_sample(root: Path, identity: dict, receipt: dict, sample: dict) -> dict:
    """Commit contemporaneous inference observation; never time a reused array as inference."""
    clip = receipt["clip_id"]
    metrics = sample["measurements"]
    seconds = sample["seconds"]
    inference = metrics.get("inference_seconds")
    if (sample.get("reused") is not False or sample["clip_id"] != clip
            or type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0
            or type(inference) not in (float, int) or not math.isfinite(inference)
            or not 0 < inference <= seconds
            or metrics.get("inference_windows") != len(receipt["windows"])
            or metrics.get("source_frames") != receipt["frame_count"]
            or metrics.get("input_frames") != 16 * len(receipt["windows"])
            or type(metrics.get("inference_batches")) is not int
            or metrics["inference_batches"] < 1):
        raise ValueError("actual fresh inference observation required")
    path = safe_path(root / "profile-samples" / f"{clip}.json")
    if path.exists():
        raise ValueError("immutable profile observation already exists")
    receipt_path = root / "receipts/train" / f"{clip}.json"
    if read_json(receipt_path) != receipt:
        raise ValueError("profile receipt changed before sample commit")
    observation = {"schema_version": 1, "policy": "motion-profile-observation-v1",
                   "origin": "actual_fresh_inference", "featurekey": digest(identity),
                   "clip_id": clip, "receipt_sha256": file_hash(receipt_path),
                   "array_sha256": receipt["array"]["sha256"],
                   "source_hash": receipt["source_hash"], "frames": receipt["frame_count"],
                   "windows": len(receipt["windows"]), "seconds": seconds,
                   "measurements": metrics, "boot_id": clocks()[2],
                   "observer": process_identity(os.getpid())}
    atomic_json(path, observation)
    return observation


def valid_profile_samples(root: Path, identity: dict, frames: dict, *,
                          required: list[str] | None = None) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    directory = safe_path(root / "profile-samples")
    allowed = {f"{clip}.json" for clip in frames["clips"]}
    if directory.exists():
        for path in directory.iterdir():
            regular_path(path)
            if path.name not in allowed:
                raise ValueError("foreign/partial profile observation")
            sample = read_json(path)
            clip = path.stem
            receipt_path = root / "receipts/train" / f"{clip}.json"
            receipt = read_json(receipt_path)
            rebuilt = make_clip_receipt(clip, frames["clips"][clip], receipt["source_rows"],
                                        tuple(receipt["windows"]), root / "train" / f"{clip}.npy",
                                        identity)
            seconds: Any = sample.get("seconds")
            metrics = sample.get("measurements", {})
            inference: Any = metrics.get("inference_seconds")
            if (digest(rebuilt) != digest(receipt) or sample.get("schema_version") != 1
                    or sample.get("policy") != "motion-profile-observation-v1"
                    or sample.get("origin") != "actual_fresh_inference"
                    or sample.get("featurekey") != digest(identity) or sample.get("clip_id") != clip
                    or sample.get("receipt_sha256") != file_hash(receipt_path)
                    or sample.get("array_sha256") != receipt["array"]["sha256"]
                    or sample.get("source_hash") != receipt["source_hash"]
                    or sample.get("frames") != receipt["frame_count"]
                    or sample.get("windows") != len(receipt["windows"])
                    or type(seconds) not in (float, int) or not math.isfinite(seconds)
                    or type(inference) not in (float, int) or not math.isfinite(inference)
                    or not 0 < inference <= seconds
                    or metrics.get("inference_windows") != len(receipt["windows"])
                    or metrics.get("source_frames") != receipt["frame_count"]
                    or metrics.get("input_frames") != len(receipt["windows"]) * 16
                    or type(metrics.get("inference_batches")) is not int
                    or metrics["inference_batches"] < 1):
                raise ValueError("profile observation receipt/source/feature/timing drift")
            samples[clip] = sample
    if any(clip not in samples for clip in required or []):
        raise ValueError("reused profile clip missing durable inference observation")
    return samples


def profile_rate(samples: dict[str, dict], prior: dict | None) -> float:
    if not samples:
        raise ValueError("profile requires actual inference observations")
    rate = max(s["seconds"] / s["windows"] for s in samples.values()) * 1.25
    if prior is not None:
        old = prior["seconds_per_window"]
        if type(old) not in (int, float) or not math.isfinite(old) or old <= 0:
            raise ValueError("invalid prior conservative profile rate")
        rate = max(rate, old)
    return rate


def validate_measured_profile(profile: dict, identity: dict) -> None:
    """Admit measured smoke evidence without predicting full-run duration."""
    parity = profile.get("parity")
    if (profile.get("schema_version") != 3
            or profile.get("admission_policy") != "measured-smoke-hard-cap-v1"
            or profile.get("featurekey") != digest(identity)
            or not isinstance(parity, list)
            or {p.get("batch_size") for p in parity if isinstance(p, dict)} != {1, 8, 16}
            or any(not isinstance(p, dict) or p.get("bit_equal") is not True for p in parity)
            or not isinstance(profile.get("samples"), list) or not profile["samples"]
            or not isinstance(profile.get("observation_sha256"), dict)
            or not profile["observation_sha256"]):
        raise ValueError("measured profile identity/parity/observations invalid")


def _worker(request: dict) -> dict:
    root = Path(request["root"])
    identity = request["identity"]
    frames = read_json(Path(identity["inputs"]["frame_rows"]["path"]))
    _deadline(request)
    verify_evidence(identity)
    if hardware_identity() != identity["hardware"]:
        raise ValueError("hardware drift")
    check_owned_inventory(root, frames)
    if (root / "complete/completion.json").exists():
        package = validate_motion_package(root, frames)
        return {"package": package, "profile": None}
    samples: dict[str, dict] = {}
    if request["mode"] == "profile":
        selected = profile_ids(frames)
        reused_ids = [c for c in selected if (root / "receipts/train" / f"{c}.json").exists()]
        samples = valid_profile_samples(root, identity, frames, required=reused_ids)
    require_gpu_idle()
    storage = require_storage(root, sum(len(window_rows(r["frame_count"]))
                                        for r in frames["clips"].values()))
    _deadline(request)
    torch.set_num_threads(2)
    load_started = time.monotonic()
    snapshot = Path(request["model_snapshot"])
    if snapshot_files(snapshot) != identity["environment"]["model_files"]:
        raise ValueError("snapshot bytes differ from admitted model evidence")
    model, processor = load_encoder(snapshot)
    if snapshot_files(snapshot) != identity["environment"]["model_files"]:
        raise ValueError("snapshot bytes changed during admitted load")
    model = model.to(device="cuda", dtype=torch.float32)  # type: ignore[attr-defined]
    load_seconds = time.monotonic() - load_started
    torch.cuda.reset_peak_memory_stats()
    parity = tensor_parity(processor)
    _deadline(request)
    ids = profile_ids(frames) if request["mode"] == "profile" else sorted(frames["clips"])
    measurements = []
    for clip in ids:
        _deadline(request)
        verify_evidence(identity, clip_check=True)
        if hardware_identity() != identity["hardware"]:
            raise ValueError("hardware drift")
        started = time.monotonic()
        reused = (root / "receipts/train" / f"{clip}.json").exists()
        with measure_clip() as stats:
            receipt = generate_clip(root, clip, frames["clips"][clip], identity, model, processor)
        torch.cuda.synchronize()
        sample = {"clip_id": clip, "seconds": time.monotonic() - started,
                  "reused": reused, "frames": receipt["frame_count"],
                  "windows": len(receipt["windows"]),
                  "array_bytes": (root / "train" / f"{clip}.npy").stat().st_size,
                  "measurements": stats}
        measurements.append(sample)
        if request["mode"] == "profile" and not reused:
            samples[clip] = record_profile_sample(root, identity, receipt, sample)
    profile = None
    if request["mode"] == "profile":
        # Reused sample timings are validation-only; never substitute them for inference costs.
        old = read_json(root / "profile.json") if (root / "profile.json").exists() else None
        samples = valid_profile_samples(root, identity, frames, required=ids)
        if old is not None:
            if (old.get("schema_version") != 3
                    or not isinstance(old.get("observation_sha256"), dict)
                    or old.get("featurekey") != digest(identity) or any(
                clip not in samples or file_hash(root / "profile-samples" / f"{clip}.json") != sha
                for clip, sha in old.get("observation_sha256", {}).items()
            )):
                raise ValueError("prior profile inference observation drift")
        complete_windows = sum(len(read_json(p)["windows"])
                                for p in (root / "receipts/train").glob("*.json"))
        remaining = 93232 - complete_windows
        profile = {"schema_version": 3, "admission_policy": "measured-smoke-hard-cap-v1",
                   "featurekey": digest(identity), "parity": parity,
                   "samples": [samples[c] for c in sorted(samples)],
                   "attempt_measurements": measurements,
                   "observation_sha256": {c: file_hash(root / "profile-samples" / f"{c}.json")
                                          for c in sorted(samples)},
                   "remaining_windows": remaining,
                   "model_load_seconds": load_seconds,
                   "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "storage": storage}
        atomic_json(root / "profile.json", profile)
    else:
        _deadline(request)
        package = finish_package(root, identity, frames, before_commit=lambda: _deadline(request))
        _deadline(request)
        return {"package": package, "profile": None}
    return {"package": None, "profile": profile}


def _worker_entry(request_path: str):
    request = read_json(Path(request_path))
    root = Path(request["root"])
    def interrupted(signum, frame):
        raise InterruptedError(f"extraction signal {signum}")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    try:
        atomic_json(root / "worker-owner.json", {
            "schema_version": 1, "request_hash": digest(request), "boot_id": clocks()[2],
            "process": process_identity(os.getpid())})
        result = _worker(request)
        result["request_hash"] = digest(request)
        atomic_json(root / "stage-result.json", result)
    except BaseException as exc:
        atomic_json(root / "stage-error.json", {"request_hash": digest(request),
                                               "error": f"{type(exc).__name__}: {exc}"})
        raise


def bootstrap_launch_binding(args) -> dict:
    """Cheap authorized launch contract; data/model/environment hashing stays supervised."""
    if (args.mode not in {"preflight", "profile", "run"}
            or type(args.gpu_cap_seconds) is not int or args.gpu_cap_seconds != CAP):
        raise ValueError("fixed extraction cap 18000 required")
    paths = {name: str(safe_path(getattr(args, name))) for name in (
        "output_base", "adapted_root", "annotation", "model_snapshot", "environment_lock")}
    if (Path(paths["output_base"]) != PILOT
            or Path(paths["adapted_root"]).parent != PILOT
            or not Path(paths["adapted_root"]).name.startswith("e3-features-")
            or Path(paths["model_snapshot"]) != SNAPSHOT
            or Path(paths["environment_lock"]) != CHECKOUT / "tools/motion/uv.lock"):
        raise ValueError("authorized normalized extraction paths required")
    result = {"schema_version": 1, "policy": "motion-extraction-launch-v1", "paths": paths,
            "authorization": dict(AUTHORIZATION),
            "code": {name: evidence(CHECKOUT / name) for name in CODE_FILES},
            "interpreter": evidence(Path(sys.executable).resolve())}
    recovery = recovery_for(args)
    if recovery is not None:
        result["recovery"] = recovery
    return result


def extraction_bootstrap_root(args, launch: dict | None = None) -> Path:
    name = BOOTSTRAP_NAME if recovery_for(args) is not None else "motion-extraction-bootstrap-v1"
    base = launch["paths"]["output_base"] if launch is not None else args.output_base
    return safe_path(Path(base) / name)


def preparation_inputs(args) -> dict:
    """First factual source observation is inside the charged preparation worker."""
    files = [args.adapted_root / "complete/frame_rows.json",
             args.adapted_root / "complete/identity.json", args.annotation, args.environment_lock]
    return {"schema_version": 1, "files": [evidence(p) for p in files],
            "model_files": snapshot_files(args.model_snapshot), "hardware": hardware_identity()}


def _prepare_entry(request_path: str):
    request = read_json(Path(request_path))
    bootstrap = Path(request_path).parent
    def interrupted(signum, frame):
        raise InterruptedError(f"preparation signal {signum}")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    try:
        _deadline(request)
        args = SimpleNamespace(**{k: Path(v) for k, v in request["launch"]["paths"].items()},
                               mode=request["mode"], resume=request["resume"],
                               gpu_cap_seconds=CAP, authorize_motion_extraction=True,
                               recovery_record=request["launch"].get("recovery", {}).get("path"))
        if digest(bootstrap_launch_binding(args)) != digest(request["launch"]):
            raise ValueError("preparation launch code/interpreter drift")
        observed = preparation_inputs(args)
        observation_path = bootstrap / "input-observation.json"
        if observation_path.exists():
            if digest(read_json(observation_path)) != digest(observed):
                raise ValueError("preparation input/model/lock/hardware changed on resume")
        else:
            atomic_json(observation_path, observed)
        _deadline(request)
        identity, frames, root = prepare(args)
        _deadline(request)
        if digest(preparation_inputs(args)) != digest(observed):
            raise ValueError("inputs changed during preparation")
        binding_path = bootstrap / "binding.json"
        if binding_path.exists():
            old = read_json(binding_path)
            if old["featurekey"] != digest(identity) or old["root"] != str(root):
                raise ValueError("verified feature/input/code/hardware changed on resume")
        package = None
        if request["adopt_complete"]:
            with exclusive_writer(root):
                package = validate_motion_package(root, frames)
            _deadline(request)
        atomic_json(bootstrap / "prepared.json", {
            "schema_version": 1, "request_hash": digest(request), "identity": identity,
            "root": str(root), "clip_ids": sorted(frames["clips"]), "package": package,
            "worker": process_identity(os.getpid()), "boot_id": clocks()[2],
            "input_observation_hash": file_hash(observation_path)})
    except BaseException as exc:
        atomic_json(bootstrap / "prepare-error.json", {
            "request_hash": digest(request), "error": f"{type(exc).__name__}: {exc}"})
        raise


def prepare_supervised(args, bootstrap: Path, launch: dict, clock: ExtractionClock, *,
                       lock_fd: int, adopt_complete: bool = False) -> tuple[dict, dict, Path]:
    clock.check()
    request = {"schema_version": 1, "launch": launch, "mode": args.mode, "resume": args.resume,
               "boot_id": clocks()[2], "deadline_monotonic": time.monotonic() + CAP - clock.spent,
               "adopt_complete": adopt_complete, "nonce": os.urandom(16).hex()}
    path = bootstrap / "prepare-request.json"
    atomic_json(path, request)
    rc = supervise_command(
        [sys.executable, "-c", "from tools.motion.extractor import _prepare_entry; "
         "import sys; _prepare_entry(sys.argv[1])", str(path)], clock, lock_fd=lock_fd)
    if rc != 0:
        error_path = bootstrap / "prepare-error.json"
        error = read_json(error_path) if error_path.exists() else {}
        message = error.get("error") if error.get("request_hash") == digest(request) else None
        if clock.spent >= CAP - 30:
            raise TimeoutError(f"preparation watchdog capped: {message or rc}")
        raise RuntimeError(f"preparation worker failed: {message or rc}")
    result = read_json(bootstrap / "prepared.json")
    if (result["request_hash"] != digest(request) or result["boot_id"] != request["boot_id"]
            or result["input_observation_hash"] != file_hash(bootstrap / "input-observation.json")):
        raise ValueError("stale/foreign preparation result")
    _deadline(request)
    return result["identity"], {"clips": dict.fromkeys(result["clip_ids"])}, Path(result["root"])


def _result(mode: str, status: str, identity: dict | None, root: Path | None, clock=None,
            package=None, profile=None, error=None, *, bootstrap: Path | None = None,
            admission_seconds: float = 0) -> dict:
    return {"schema_version": 1, "policy": "motion-extraction-result-v1", "mode": mode,
            "status": status, "featurekey": (clock.state["featurekey"] if identity is None
                                              and clock is not None else
                                              None if identity is None else digest(identity)),
            "identity": identity, "root": None if root is None else str(root),
            "bootstrap_root": None if bootstrap is None else str(bootstrap),
            "bootstrapkey": None if clock is None else clock.state.get("bootstrapkey"),
            "binding_path": None if bootstrap is None else str(bootstrap / "binding.json"),
            "input_observation_path": None if bootstrap is None else str(
                bootstrap / "input-observation.json"),
            "budget_path": None if clock is None else str(clock.path),
            "status_path": None if bootstrap is None else str(bootstrap / "status.json"),
            "profile_path": None if root is None else str(root / "profile.json"),
            "spent_seconds": 0.0 if clock is None else clock.spent,
            "remaining_seconds": CAP if clock is None else max(0.0, CAP - clock.spent),
            "authorization": dict(AUTHORIZATION), "package": package, "profile": profile,
            "error": error, "admission_seconds": admission_seconds,
            "terminal_admission": None}


def run_motion_extraction(args) -> dict:
    result = _run_motion_owned(args)
    clock = result.pop("_return_clock", None)
    guard = result.pop("_return_guard", None)
    # _run_motion_owned has released all local locks/FDs and cleaned its watchdog.
    # This is the last extractor-owned check, immediately before API return.
    if guard is not None and result["status"] in {"profiled", "complete"}:
        try:
            if clock.state["status"] == "terminal_pending":
                proof = clock.return_admission(guard)
                result["terminal_admission"] = proof
                result["spent_seconds"] = proof["spent_upper_bound"]
                result["remaining_seconds"] = CAP - proof["spent_upper_bound"]
            else:
                if time.monotonic() >= guard.deadline:
                    raise TimeoutError("read-only admission cleanup exceeded global deadline")
            if time.monotonic() >= guard.deadline or (
                guard.closing_deadline is not None and time.monotonic() >= guard.closing_deadline
            ):
                raise TimeoutError("final extractor return exceeded closing/global deadline")
        except BaseException as exc:
            with exclusive_writer(Path(result["bootstrap_root"])):
                if not clock.admitted:
                    clock.finish("capped" if isinstance(exc, TimeoutError) else "failed")
                    result["spent_seconds"] = clock.spent
                    result["remaining_seconds"] = max(0, CAP - clock.spent)
                result.update(status="capped" if isinstance(exc, TimeoutError) else "failed",
                              package=None, profile=None, terminal_admission=None,
                              error=f"{type(exc).__name__}: {exc}")
                atomic_json(Path(result["status_path"]), result)
    return result


def _run_motion_owned(args) -> dict:
    if args.mode != "preflight" and not args.authorize_motion_extraction:
        raise ValueError("explicit motion extraction authorization required")
    if args.mode == "preflight":
        identity, frames, root = prepare(args)
        return _result(args.mode, "ready", identity, root)
    launch = bootstrap_launch_binding(args)
    # Fixed root, deliberately NOT keyed by code/data/content hash. Changes cannot
    # mint another allowance. Only the launch contract may precede this clock.
    bootstrap = extraction_bootstrap_root(args, launch)
    if bootstrap.exists():
        names = {p.name for p in bootstrap.iterdir()}
        allowed_terminals = {n for n in names if re.fullmatch(
            r"terminal-(?:(?:failure|candidate|admission)-)?[0-9a-f]{32}\.json", n)}
        if (names - BOOTSTRAP_FILES - allowed_terminals
                or ("owner.json" not in names and names - {".writer.lock"})):
            raise ValueError("foreign/unowned bootstrap root")
        for p in bootstrap.iterdir():
            regular_path(p)
    with exclusive_writer(bootstrap) as bootstrap_fd:
        owner = {"schema_version": 1, "policy": "motion-extraction-bootstrap-owner-v1",
                 "bootstrapkey": digest(launch), "launch": launch}
        owner_path = bootstrap / "owner.json"
        if owner_path.exists():
            if digest(read_json(owner_path)) != digest(owner) or not args.resume:
                raise ValueError("bootstrap launch conflict or missing --resume")
        else:
            atomic_json(owner_path, owner)
        budget_path = bootstrap / "budget.json"
        initialized_only = not budget_path.exists() and {
            p.name for p in bootstrap.iterdir()
        } <= {".writer.lock", "owner.json"}
        clock = ExtractionClock(budget_path, None, bootstrapkey=digest(launch),
                                resume=args.resume and not initialized_only,
                                inherited_seconds=launch.get("recovery", {}).get(
                                    "credit", {}).get("extraction", 0))
        if clock.state["status"] in {"profiled", "complete"}:
            validate_terminal_close(clock)
        identity, root = None, None
        if clock.state.get("binding") is not None:
            root = Path(clock.state["binding"]["root"])
        if clock.state["status"] == "complete":
            # Closed extraction never relaunches GPU or rewrites its ledger. A
            # read-only consumer admission is still independently watchdog-bounded.
            admission = copy.copy(clock)
            admission.state = copy.deepcopy(clock.state)
            admission.read_only = True
            admission.state["status"] = "created"
            guard = None
            try:
                admission.start(args.mode, phase="preparing")
                guard = ParentDeadlineGuard(admission, bootstrap_fd)
                current, _, current_root = prepare_supervised(
                    args, bootstrap, launch, admission, lock_fd=bootstrap_fd, adopt_complete=True)
                admission.bind_feature(current, current_root)
                package = read_json(bootstrap / "prepared.json")["package"]
                admission.check()
                if package is None:
                    raise ValueError("complete adoption requires independent admission")
                guard.release_admission()
                result = _result(args.mode, "complete", current, current_root, clock, package,
                                 bootstrap=bootstrap,
                                 admission_seconds=admission.spent - clock.spent)
                result.update(_return_clock=clock, _return_guard=guard)
            except BaseException as exc:
                admission.tick()
                result = _result(args.mode, "failed", identity, root, clock, bootstrap=bootstrap,
                                 error=f"{type(exc).__name__}: {exc}",
                                 admission_seconds=admission.spent - clock.spent)
            finally:
                if guard is not None:
                    guard.close()
            return result
        guard = None
        try:
            clock.start(args.mode, phase="preparing")
            guard = ParentDeadlineGuard(clock, bootstrap_fd)
            candidate, frames, candidate_root = prepare_supervised(
                args, bootstrap, launch, clock, lock_fd=bootstrap_fd)
            # Compare immutable binding before creating any new feature root.
            if clock.state["featurekey"] is not None and (
                digest(candidate) != clock.state["featurekey"] or candidate_root != root
            ):
                raise ValueError("verified feature/input/code/hardware changed on resume")
            clock.bind_feature(candidate, candidate_root)
            identity, root = candidate, candidate_root
            status, package, profile = _run_prepared(
                args, identity, frames, root, bootstrap, clock, bootstrap_fd)
            guard.begin_closing()
            result = _result(args.mode, status, identity, root, clock, package, profile,
                             bootstrap=bootstrap)
            # Status serialization/fsync is inside the active interval. Success
            # requires the final ledger close; a stale provisional status is never
            # sufficient handoff evidence while the durable budget is running.
            atomic_json(bootstrap / "status.json", result)
            clock.check()
            clock.finish(status)
            guard.seal(bootstrap / "status.json")
            result["spent_seconds"] = clock.spent
            result["remaining_seconds"] = max(0.0, CAP - clock.spent)
        except BaseException as exc:
            clock.tick()
            status = "capped" if isinstance(exc, TimeoutError) or clock.spent >= CAP - 30 else (
                "failed")
            result = _result(args.mode, status, identity, root, clock,
                             error=f"{type(exc).__name__}: {exc}", bootstrap=bootstrap)
            try:
                atomic_json(bootstrap / "status.json", result)
            finally:
                clock.finish(status)
            result["spent_seconds"] = clock.spent
            result["remaining_seconds"] = max(0.0, CAP - clock.spent)
        if guard is not None:
            try:
                guard.close()
                if result["status"] in {"profiled", "complete"}:
                    guard.check_return()
                    result.update(_return_clock=clock, _return_guard=guard)
            except BaseException as exc:
                clock.finish("capped" if isinstance(exc, TimeoutError) else "failed")
                result = _result(args.mode, clock.state["status"], identity, root, clock,
                                 error=f"{type(exc).__name__}: {exc}", bootstrap=bootstrap)
                atomic_json(bootstrap / "status.json", result)
        return result


def _run_prepared(args, identity: dict, frames: dict, root: Path, bootstrap: Path,
                  clock: ExtractionClock, bootstrap_fd: int
                  ) -> tuple[str, dict | None, dict | None]:
    clock.check()
    owner = {"schema_version": 1, "policy": "motion-extraction-owner-v2",
             "featurekey": digest(identity), "identity": identity,
             "authorization": dict(AUTHORIZATION), "bootstrap_root": str(bootstrap),
             "bootstrapkey": clock.state["bootstrapkey"]}
    if root.exists() and not (root / ".writer.lock").exists():
        raise ValueError("foreign feature root without writer ownership")
    with exclusive_writer(root) as lock_fd:
        owner_path = root / "owner.json"
        if owner_path.exists():
            if digest(read_json(owner_path)) != digest(owner):
                raise ValueError("foreign feature root owner")
        elif {p.name for p in root.iterdir()} != {".writer.lock"}:
            raise ValueError("unowned preexisting feature artifacts")
        else:
            atomic_json(owner_path, owner)
        check_owned_inventory(root, frames)
        if (root / "budget.json").exists():
            raise ValueError("legacy feature clock cannot reset bootstrap allowance")
        if clock.state.get("profile_sha256") is not None and file_hash(
            regular_path(root / "profile.json")
        ) != clock.state["profile_sha256"]:
            raise ValueError("same-clock profile evidence drift")
        if args.mode == "run":
            profile_path = root / "profile.json"
            if (not profile_path.exists()
                    or clock.state.get("profile_sha256") != file_hash(profile_path)):
                raise ValueError("run requires verified same-clock profile")
            prior_profile = read_json(profile_path)
            validate_measured_profile(prior_profile, identity)
        request = {"schema_version": 1, "root": str(root), "identity": identity,
                   "mode": args.mode, "model_snapshot": str(args.model_snapshot),
                   "boot_id": clocks()[2],
                   "deadline_monotonic": time.monotonic() + CAP - clock.spent,
                   "nonce": os.urandom(16).hex()}
        atomic_json(root / "stage-request.json", request)
        command = [sys.executable, "-c", "from tools.motion.extractor import _worker_entry; "
                   "import sys; _worker_entry(sys.argv[1])", str(root / "stage-request.json")]
        with file_lock(root.parent / ".motion-extraction-gpu.lock") as gpu_fd:
            rc = supervise_command(command, clock, lock_fd=(bootstrap_fd, lock_fd, gpu_fd))
        if rc != 0:
            raise RuntimeError(f"extractor worker exited {rc}")
        outcome = read_json(root / "stage-result.json")
        if outcome["request_hash"] != digest(request):
            raise ValueError("stale/foreign stage result")
        clock.check()
        profile, package = outcome["profile"], outcome["package"]
        if package is not None:
            if (package["identity_hash"] != digest(identity)
                    or file_hash(Path(package["completion"])) != package["completion_hash"]):
                raise ValueError("worker package completion drift")
            return "complete", package, profile
        if profile is not None:
            if read_json(root / "profile.json") != profile:
                raise ValueError("worker profile evidence drift")
            clock.state["profile_sha256"] = file_hash(root / "profile.json")
            validate_measured_profile(profile, identity)
            clock.check()
            return "profiled", None, profile
        raise ValueError("worker returned no verified stage result")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "profile", "run"), required=True)
    for name in ("adapted-root", "annotation", "model-snapshot", "output-base", "environment-lock"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--gpu-cap-seconds", type=int, default=CAP)
    p.add_argument("--authorize-motion-extraction", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--recovery-record", type=Path)
    return p.parse_args(argv)
