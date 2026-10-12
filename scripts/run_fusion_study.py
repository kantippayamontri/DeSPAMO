"""Sequential, restart-safe six-run fusion controller; dev-only and timing-gated."""

import argparse
import fcntl
import json
import math
import os
import re
import shutil
import signal
import stat
import statistics
import subprocess
import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from scripts import train_fusion_study as runner
from scripts.fusion_time_waiver import compatible_sources, execution_fields, time_waived
from scripts.run_connector_study import prediction_diagnostics
from scripts.run_projector_overnight import BASE, INPUTS, PYTHON, REFERENCE, ROOT, Study

POLICY = "time-aligned-fusion-controller-v1"
NEW_POLICY = "time-aligned-fusion-new-motion-controller-v1"
ARMS = runner.FUSION_ARMS
PHASE_CAP = 32400
RUN_CAP = 10800
DEFAULT_ROOT = BASE / "time-aligned-fusion-study-v1"
NEW_DEFAULT_ROOT = BASE / "time-aligned-fusion-new-motion-study-v1"
GIB = 1024**3
CHECKPOINT_BYTES = 5.6 * GIB
RESERVE_BYTES = 50 * GIB
TOKEN_SCOPE = "eval_diagnostic_forward_actual_output_mask"
PEAK_SCOPE = "process_allocator_high_water_sampled_during_phase"
QUERY_POLICY = "fusion-owned-cpu-query-v1"


def fusion_query_command(phase, kind: str, run: Path, expected: dict,
                         checkpoint: Path | None = None) -> list[str]:
    """Retain full provenance in a unique immutable request, never in argv."""
    root = Path(getattr(phase, "root", run.parent))
    if (not root.is_absolute() or root != root.resolve()
            or any(p.is_symlink() for p in (root, *root.parents))
            or root.stat().st_uid != os.getuid()):
        raise ValueError("fusion query owned canonical root required")
    nonce = uuid.uuid4().hex
    path = root / f"fusion-query-{nonce}.json"
    request = {"policy": QUERY_POLICY, "nonce": nonce, "query_root": str(root),
               "kind": kind, "run": str(run), "protocol": str(INPUTS["protocol"]),
               "checkpoint": str(checkpoint) if checkpoint is not None else None,
               "expected_identity": expected, "expected_identity_digest": digest(expected),
               "controller_sources": controller_sources(
                   new_motion=runner.motion_run_policy(expected) == runner.NEW_RUN_POLICY)}
    atomic_json(path, request)
    path.chmod(0o400)
    return [str(PYTHON), "-c", selection_query_code() if kind == "selection"
            else recovery_query_code(), str(path), file_hash(path)]


def read_fusion_query(path: Path, raw_digest: str) -> dict:
    """Same admission in controller and CPU child; no caller-supplied code."""
    if (not path.is_absolute() or path != path.resolve()
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("fusion query regular owned canonical path required")
    regular(path)
    st = path.stat()
    if (st.st_uid != os.getuid() or path.parent.stat().st_uid != os.getuid() or st.st_nlink != 1
            or stat.S_IMODE(st.st_mode) != 0o400 or file_hash(path) != raw_digest):
        raise ValueError("fusion query owner/immutable/raw digest mismatch")
    request = read_json(path)
    expected = request["expected_identity"]
    nonce = request.get("nonce")
    if (request.get("policy") != QUERY_POLICY or not isinstance(nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", nonce) is None
            or path.name != f"fusion-query-{nonce}.json"
            or request.get("query_root") != str(path.parent)
            or request.get("expected_identity_digest") != digest(expected)
            or request.get("kind") not in {"selection", "recovery"}
            or Path(request["run"]).name != expected["run_key"]
            or expected.get("arm") not in ARMS or type(expected.get("seed")) is not int
            or expected["seed"] not in (0, 1)
            or request.get("protocol") != str(INPUTS["protocol"])
            or (request["kind"] == "selection") != (request.get("checkpoint") is None)
            or request.get("controller_sources") != controller_sources(
                new_motion=runner.motion_run_policy(expected) == runner.NEW_RUN_POLICY)):
        raise ValueError("fusion query nonce/identity/stage/source binding mismatch")
    if request["kind"] == "recovery":
        checkpoint = Path(request["checkpoint"])
        if (checkpoint.parent != Path(request["run"])
                or re.fullmatch(r"(?:step|resume)-\d+\.ckpt", checkpoint.name) is None):
            raise ValueError("fusion query recovery checkpoint ownership mismatch")
    runner.validate_motion_graph(expected)
    if file_hash(path) != raw_digest:
        raise ValueError("fusion query bytes changed during admission")
    return request


def admit_fusion_query(command: list[str], phase) -> dict:
    if (len(command) != 5 or command[:2] != [str(PYTHON), "-c"]
            or command[2] not in (selection_query_code(), recovery_query_code())):
        raise ValueError("fusion CPU query exact request command required")
    request = read_fusion_query(Path(command[3]), command[4])
    expected = request["expected_identity"]
    kind = "selection" if command[2] == selection_query_code() else "recovery"
    package = getattr(phase, "motion_package", None)
    if (request["kind"] != kind or request["query_root"] != str(phase.root.resolve())
            or expected["arm"] != phase.active_arm or expected["seed"] != phase.fusion_seed
            or runner.motion_run_policy(expected) != (
                runner.RUN_POLICY if package is None else runner.NEW_RUN_POLICY)
            or (package is not None and expected["motion_package"]["root"] != str(package))):
        raise ValueError("fusion CPU query phase/arm/seed/mode ownership mismatch")
    return request


def regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"regular fusion identity file required: {path}")


def finite_nonnegative(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def controller_sources(*, new_motion: bool = False) -> dict[str, str]:
    paths = [ROOT / "scripts" / name for name in
             ("run_fusion_study.py", "run_projector_overnight.py", "run_connector_study.py")]
    if new_motion:
        paths.extend(ROOT / name for name in runner.NEW_SOURCE_FILES)
    for path in paths:
        regular(path)
    actual = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    return compatible_sources(actual, "controller")


class FusionPhase(Study):
    """Neutral watchdog machinery, with local policy and cross-stage run caps."""

    def __init__(self, root: Path, seed: int, *, motion_package: Path | None = None,
                 inherited_seconds: float = 0):
        if not finite_nonnegative(inherited_seconds) or inherited_seconds >= RUN_CAP:
            raise ValueError("invalid inherited setup charge")
        if type(seed) is not int or seed not in (0, 1):
            raise ValueError("fusion phase seed invalid")
        if (any(p.is_symlink() for p in (root, *root.parents))
                or (root / ".study.lock").is_symlink()):
            raise ValueError("fusion phase root/lock unsafe")
        self.fusion_seed = seed
        self.motion_package = motion_package
        self.inherited_seconds = float(inherited_seconds)
        self.policy = POLICY if motion_package is None else NEW_POLICY
        self.active_arm = "reference"
        super().__init__(root, PHASE_CAP)

    def _load(self, budget: int):
        # Study acquires its lock before dispatching here. Policy conflicts must
        # be rejected before inherited recovery or any status publication.
        self.path = self.root / "study-state.json"
        if self.path.exists():
            regular(self.path)
            self.state = read_json(self.path)
        else:
            self.state = {"study_policy": self.policy, "fusion_seed": self.fusion_seed,
                          "controller_sources": self._sources(),
                          "budget_seconds": budget, "run_cap_seconds": RUN_CAP,
                          "spent_seconds": 0.0, "status": "created", "stages": {}, "runs": {}}
            if self.motion_package is not None:
                self.state["motion_package_root"] = str(self.motion_package)
            if self.inherited_seconds:
                self.state["inherited_seconds"] = self.inherited_seconds
                self.state["stages"]["previous-authorized-setup"] = {
                    "owned_arm": "reference", "status": "complete", "cpu_only": True,
                    "elapsed_seconds": self.inherited_seconds,
                    "origin": "validated-previous-training-recovery"}
                self.state["spent_seconds"] = self.inherited_seconds
        if (self.state.get("study_policy") != self.policy
                or self.state.get("motion_package_root") != (
                    str(self.motion_package) if self.motion_package is not None else None)
                or self.state.get("fusion_seed") != self.fusion_seed
                or self.state.get("budget_seconds") != PHASE_CAP
                or self.state.get("run_cap_seconds") != RUN_CAP
                or not finite_nonnegative(self.state.get("spent_seconds"))
                or not isinstance(self.state.get("stages"), dict)):
            raise ValueError("fusion phase policy/seed/budget mismatch")
        if self.state.get("inherited_seconds", 0) != self.inherited_seconds:
            raise ValueError("inherited setup charge changed")
        if self.state.get("controller_sources") != self._sources():
            raise ValueError("fusion controller helper source mismatch")
        for record in self.state["stages"].values():
            if (record.get("owned_arm") not in ARMS
                    or not finite_nonnegative(record.get("elapsed_seconds"))
                    or record.get("status") not in
                    {"pending", "launching", "running", "complete", "failed", "invalid",
                     "interrupted"}):
                raise ValueError("fusion phase stage ownership/accounting invalid")
        charged = sum(r["elapsed_seconds"] for r in self.state["stages"].values())
        if not math.isclose(charged, self.state["spent_seconds"], abs_tol=1e-6):
            raise ValueError("fusion phase accounting differs from durable stages")
        self.env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
                    "OMP_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
                    "LD_LIBRARY_PATH": "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        for name, record in self.state["stages"].items():
            if record["status"] in {"launching", "running"}:
                self._recover(name, record)
        self.save()

    def _sources(self) -> dict[str, str]:
        return (controller_sources() if self.motion_package is None
                else controller_sources(new_motion=True))

    def run_spent(self, arm: str) -> float:
        return sum(r["elapsed_seconds"] for r in self.state["stages"].values()
                   if r["owned_arm"] == arm)

    def bind_run(self, arm: str, run: Path, expected: dict) -> None:
        if expected.get("arm") != arm or expected.get("seed") != self.fusion_seed:
            raise ValueError("fusion accounting seed/arm mismatch")
        policy = runner.motion_run_policy(expected)
        if (policy != (runner.RUN_POLICY if self.motion_package is None else runner.NEW_RUN_POLICY)
                or (self.motion_package is not None
                    and expected["motion_package"]["root"] != str(self.motion_package))):
            raise ValueError("fusion phase motion policy/package mismatch")
        if not owned_status(run, expected):
            return
        budget = read_json(run / "budget.json")
        spent = budget["gpu_seconds"]
        if budget["status"] in {"running", "resumed"}:
            elapsed = time.time() - budget["wall_timestamp"]
            if not finite_nonnegative(elapsed):
                raise ValueError("fusion runner ledger wall clock invalid")
            spent += elapsed
        # Controller wall-time usually dominates runner GPU ledger. Missing
        # controller stages must never make an adopted run's spend disappear.
        overlapping = sum(record["elapsed_seconds"]
                          for name, record in self.state["stages"].items()
                          if name in {f"{arm}-smoke", f"{arm}-train",
                                      f"{arm}-imported-run-accounting"})
        missing = max(0, spent - overlapping)
        if missing:
            name = f"{arm}-imported-run-accounting"
            record = self.state["stages"].setdefault(name, {
                "status": "complete", "owned_arm": arm, "elapsed_seconds": 0.0,
                "attempts": 0, "stage_cap_seconds": RUN_CAP, "cpu_only": False,
                "command_hash": digest({"runner_identity": expected}),
                "accounting_evidence": "owned runner ledger floor; no child launched"})
            record["elapsed_seconds"] += missing
            self.state["spent_seconds"] += missing
            self.save()

    def cpu_query(self, command: list[str]) -> dict:
        if len(command) > 2 and command[1] == "-c":
            admit_fusion_query(command, self)
            allowed = True
        else:
            allowed = False
            if "--initialization-profile" in command and "--motion-timing" in command:
                profile = Path(command[command.index("--initialization-profile") + 1])
                timing = Path(command[command.index("--motion-timing") + 1])
                allowed = command == stage_command(self.active_arm, self.fusion_seed,
                                                     profile, timing, "preflight",
                                                     **({"motion_package": self.motion_package}
                                                        if self.motion_package is not None
                                                        else {}))
        if not allowed:
            raise ValueError("fusion CPU query cannot hide a GPU or initialization stage")
        return super().cpu_query(command)

    def run_stage(self, name: str, command: list[str], cap: int,
                  completion_validator: Callable[[], bool] | None = None):
        if time_waived():
            from scripts.fusion_time_waiver import run_without_deadline

            return run_without_deadline(self, name, command, completion_validator)
        if self.active_arm not in ARMS or not 0 < cap <= RUN_CAP:
            raise ValueError("fusion stage owner/cap invalid")
        if self.state["controller_sources"] != self._sources():
            raise ValueError("fusion controller helper source changed before stage")
        record = self.state["stages"].get(name)
        if record is not None and record["owned_arm"] != self.active_arm:
            raise ValueError("fusion stage owner changed")
        first_complete = completion_validator() if completion_validator is not None else None
        first_call = True

        def validated():
            nonlocal first_call
            if first_call:
                first_call = False
                return bool(first_complete)
            assert completion_validator is not None
            return completion_validator()

        elapsed = record["elapsed_seconds"] if record else 0
        remaining = min(PHASE_CAP - self.state["spent_seconds"],
                        RUN_CAP - self.run_spent(self.active_arm))
        # An integer conservative deadline is accepted by Study without changing
        # immutable runner identity/cap. Its own watchdog charges uncapped time.
        effective = min(cap, math.floor(elapsed + remaining))
        if effective <= elapsed:
            raise RuntimeError("fusion phase/run budget exhausted")
        if record is None:
            from scripts.run_projector_overnight import command_signature

            self.state["stages"][name] = {
                "status": "pending", "elapsed_seconds": 0.0, "attempts": 0,
                "command_hash": command_signature(command), "owned_arm": self.active_arm,
                "stage_cap_seconds": cap, "cpu_only": name.startswith("preflight-")}
        elif record.get("stage_cap_seconds", cap) != cap:
            raise ValueError("fusion stage cap changed")
        super().run_stage(name, command, effective,
                          validated if completion_validator is not None else None)


def stage_command(arm: str, seed: int, profile: Path, timing: Path, mode: str, *,
                  motion_package: Path | None = None) -> list[str]:
    if (arm not in ARMS or type(seed) is not int or seed not in (0, 1)
            or mode not in ("initialize", "preflight", "smoke", "run")
            or (mode == "initialize" and arm != "reference")):
        raise ValueError("fusion command arm/seed/mode invalid")
    return [str(PYTHON), str(ROOT / "scripts/train_fusion_study.py"),
            *[v for k, p in INPUTS.items() for v in ("--" + k, str(p))],
            "--mode", mode, "--arm", arm, "--seed", str(seed), "--steps", "16000",
            "--gpu-cap-seconds", str(RUN_CAP), "--initialization-profile", str(profile),
            "--motion-timing", str(timing), "--baseline-run", str(REFERENCE),
            "--output-base", str(profile.parent / "runs"), "--authorize-fusion-run",
            *(["--motion-package", str(motion_package)] if motion_package is not None else [])]


def validate_live(identity: dict, timing: Path, profile: Path) -> None:
    new_motion = runner.motion_run_policy(identity) == runner.NEW_RUN_POLICY
    runner.validate_motion_graph(identity)
    regular(timing)
    regular(profile)
    code = (runner.fusion_code_hashes(ROOT, new_motion=True) if new_motion
            else runner.fusion_code_hashes(ROOT))
    if (identity["code_hashes"] != code
            or identity["hardware"] != runner.hardware_identity()
            or identity["timing_hash"] != file_hash(timing)
            or identity["initialization_profile_file_hash"] != file_hash(profile)
            or identity["initialization_profile_path"] != str(profile.resolve())):
        raise ValueError("fusion source/hardware/profile/timing identity changed")
    for path, expected in identity["bound_files"].items():
        regular(Path(path))
        if file_hash(Path(path)) != expected:
            raise ValueError("fusion bound input bytes changed")


def owned_status(run: Path, expected: dict) -> dict:
    runner.validate_motion_graph(expected)
    if not run.exists():
        if run.is_symlink():
            raise ValueError("fusion run symlink")
        return {}
    if run.is_symlink() or not run.is_dir() or run.name != expected["run_key"]:
        raise ValueError("fusion run root identity mismatch")
    for name in ("budget.json", "run-status.json"):
        regular(run / name)
    budget, status = read_json(run / "budget.json"), read_json(run / "run-status.json")
    if (budget.get("identity") != expected or budget.get("ceiling_seconds") != RUN_CAP
            or not finite_nonnegative(budget.get("gpu_seconds"))
            or (not time_waived(expected) and budget["gpu_seconds"] >= RUN_CAP)
            or budget.get("status") not in ("closed", "paused", "running", "resumed")
            or status.get("run_key") != run.name
            or status.get("status") not in ("complete", "partial", "smoke_passed")
            or type(status.get("global_step")) is not int
            or not 0 <= status["global_step"] <= 16000):
        raise ValueError("fusion existing run budget/status identity mismatch")
    selected_path = run / "selected-checkpoint.json"
    if selected_path.exists() or selected_path.is_symlink():
        regular(selected_path)
        selected = read_json(selected_path)
        if (selected.get("run_identity_hash") != digest(expected)
                or any(selected.get(k) != expected[k] for k in ("seed", "arm", "timing_hash"))
                or selected.get("step") not in runner.DEV_STEPS):
            raise ValueError("fusion partial selected identity mismatch")
        checkpoint = run / f"step-{selected['step']}.ckpt"
        regular(checkpoint)
        if file_hash(checkpoint) != selected.get("checkpoint_hash"):
            raise ValueError("fusion partial selected checkpoint bytes mismatch")
    runner.validate_fusion_initialization(run, expected)
    if budget.get("initialization_hash") != file_hash(run / "initialization.json"):
        raise ValueError("fusion existing run initialization receipt byte binding mismatch")
    require_production_telemetry(budget.get("fusion_telemetry"), complete=False)
    return status


def selection_query_code() -> str:
    return """import json,sys
from pathlib import Path
from despamo.appearance.provenance import read_json,file_hash
from scripts.train_fusion_study import validate_fusion_selection
from scripts.run_fusion_study import read_fusion_query
q=read_fusion_query(Path(sys.argv[1]),sys.argv[2]); r=Path(q['run'])
p=read_json(Path(q['protocol']))
s=validate_fusion_selection(r,p,
    tuple(p['split']['groups']['dev']['clip_ids']),q['expected_identity'])
s['checkpoint']=str(s['checkpoint'])
names=['budget.json','run-status.json','selected-checkpoint.json','initialization.json',
       'gradient-smoke.json',*[f'dev-{step}.json' for step in (5500,10400,16000)],
        *s['checkpoint_sizes_bytes']]
s['source_files']={name:file_hash(r/name) for name in names}
print(json.dumps({'status':'complete',**s}))
"""


def completion_selection(phase, run: Path, expected: dict) -> dict:
    if owned_status(run, expected).get("status") != "complete":
        raise ValueError("fusion result requires validated completion")
    result = phase.cpu_query(fusion_query_command(phase, "selection", run, expected))
    if result.get("status") != "complete":
        raise ValueError("fusion completion selection validation failed")
    require_production_telemetry(result["fusion_telemetry"])
    return result


def validate_completion(phase, run: Path, expected: dict) -> bool:
    if owned_status(run, expected).get("status") != "complete":
        # A matching partial selected receipt must not suppress owned recovery.
        return False
    completion_selection(phase, run, expected)
    return True


def require_production_telemetry(telemetry: dict, *, complete: bool = True) -> None:
    runner.merge_fusion_telemetry(telemetry, telemetry)  # Public schema/device validator.
    for phase in ("training", "dev"):
        peak = telemetry[phase]
        if not complete and peak["device_type"] is None:
            continue
        if peak["device_type"] != "cuda" or peak["evidence_scope"] != PEAK_SCOPE:
            raise ValueError("fusion production requires observed conservative CUDA phase peaks")


def validate_report_telemetry(report: dict, dev_ids: tuple[str, ...], initialization_hash: str,
                              sizes: dict, telemetry: dict) -> None:
    lengths = report.get("visual_token_lengths")
    if (report.get("initialization_hash") != initialization_hash
            or type(report.get("checkpoint_size_bytes")) is not int
            or report["checkpoint_size_bytes"] != sizes.get(
                f"step-{report.get('checkpoint_step')}.ckpt")
            or report.get("visual_token_length_scope") != TOKEN_SCOPE
            or not isinstance(lengths, list) or len(lengths) != 582
            or any(not isinstance(row, dict)
                   or set(row) != {"clip_id", "valid_tokens", "padded_tokens"} for row in lengths)
            or tuple(row["clip_id"] for row in lengths) != dev_ids
            or any(type(row[k]) is not int for row in lengths
                   for k in ("valid_tokens", "padded_tokens"))
            or any(not 0 < row["valid_tokens"] <= row["padded_tokens"] for row in lengths)):
        raise ValueError("fusion bound measured diagnostic lengths/receipt/checkpoint size "
                         "mismatch")
    measured = {**telemetry, "dev": report.get("cuda_memory")}
    require_production_telemetry(measured, complete=False)
    if measured["dev"]["device_type"] != "cuda":
        raise ValueError("fusion dev report requires observed CUDA diagnostic peak evidence")
    if runner.merge_fusion_telemetry(telemetry, measured) != telemetry:
        raise ValueError("fusion report peak exceeds persisted conservative maxima")


def recovery_query_code() -> str:
    return """import json,sys
from pathlib import Path
from despamo.appearance.provenance import read_json
from scripts.train_signer_pilot import validate_e1_resume_checkpoint
from scripts.train_fusion_study import validate_fusion_initialization,merge_fusion_telemetry
from scripts.train_fusion_study import fusion_checkpoint_sizes,validate_motion_graph
from scripts.run_fusion_study import require_production_telemetry,validate_report_telemetry
from scripts.run_fusion_study import read_fusion_query
from despamo.appearance.provenance import file_hash,digest
q=read_fusion_query(Path(sys.argv[1]),sys.argv[2])
r=Path(q['run']); expected=q['expected_identity']; b=read_json(r/'budget.json')
validate_motion_graph(expected)
validate_fusion_initialization(r,expected)
initialization_hash=file_hash(r/'initialization.json')
if b.get('initialization_hash')!=initialization_hash:
    raise ValueError('fusion recovery initialization budget binding mismatch')
c=validate_e1_resume_checkpoint(Path(q['checkpoint']),r,expected,b['gpu_seconds'])
if c.get('initialization_hash')!=initialization_hash:
    raise ValueError('fusion recovery checkpoint initialization binding mismatch')
telemetry=merge_fusion_telemetry(b['fusion_telemetry'],c['fusion_telemetry'])
require_production_telemetry(telemetry,complete=False)
sizes=fusion_checkpoint_sizes(r)
from scripts.train_fusion_study import _validate_report,DEV_STEPS
from scripts.run_fusion_study import regular
import torch
p=read_json(Path(q['protocol'])); ids=tuple(p['split']['groups']['dev']['clip_ids'])
if (len(ids)!=582 or len(set(ids))!=582 or p['protocol_hash']!=expected['protocol_hash']
        or p['split_hash']!=expected['split_hash']):
    raise ValueError('partial fusion protocol/ordered dev identity mismatch')
allowed={f'dev-{step}.json' for step in DEV_STEPS}
if any(path.name not in allowed for path in r.glob('dev-*.json')):
    raise ValueError('partial fusion unknown durable dev report')
reports=[]; hashes={}
for step in DEV_STEPS:
    report_path=r/f'dev-{step}.json'
    if not report_path.exists() and not report_path.is_symlink():
        continue
    regular(report_path)
    report=read_json(report_path)
    path=_validate_report(r,report,step,expected,ids)
    validate_report_telemetry(report,ids,initialization_hash,sizes,telemetry)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if (checkpoint.get('run_metadata')!=expected or checkpoint.get('pilot_identity')!=expected
            or type(checkpoint.get('global_step')) is not int or checkpoint['global_step']!=step
            or checkpoint.get('initialization_hash')!=initialization_hash
            or merge_fusion_telemetry(telemetry,checkpoint['fusion_telemetry'])!=telemetry):
        raise ValueError('partial fusion dev checkpoint identity mismatch')
    reports.append(report); hashes[report_path.name]=file_hash(report_path)
selected=r/'selected-checkpoint.json'
if selected.exists():
    from scripts.train_fusion_study import report_binding
    from despamo.evaluation.signer_pilot import choose_checkpoint
    s=read_json(selected)
    if any(s.get(k)!=v for k,v in report_binding(expected).items()):
        raise ValueError('partial fusion selection binding mismatch')
    if (s.get('initialization_hash')!=initialization_hash
            or s.get('fusion_telemetry_hash')!=digest(s.get('fusion_telemetry'))
            or s.get('checkpoint_sizes_bytes')!=sizes
            or merge_fusion_telemetry(telemetry,s['fusion_telemetry'])!=telemetry):
        raise ValueError('partial fusion selected telemetry binding mismatch')
    if len(reports)!=len(DEV_STEPS):
        raise ValueError('partial fusion selection requires all three durable dev reports')
    best=choose_checkpoint(reports,ids)
    if (s.get('dev_report_hashes')!=hashes or s.get('step')!=best['checkpoint_step']
            or s.get('metrics')!=best['metrics']):
        raise ValueError('partial fusion selection reports/tie mismatch')
print(json.dumps({'step':c['global_step'],'checkpoint':q['checkpoint']}))
"""


def recovery_checkpoint(phase, run: Path, expected: dict) -> Path | None:
    status = owned_status(run, expected)
    if not status:
        return None
    candidates = []
    for p in run.glob("*.ckpt"):
        match = re.fullmatch(r"(step|resume)-(\d+)\.ckpt", p.name)
        if match is None:
            raise ValueError("fusion unexpected checkpoint name")
        regular(p)
        candidates.append((int(match[2]), match[1] == "step", p))
    if not candidates:
        if status["status"] == "smoke_passed" and status["global_step"] == 0:
            budget = read_json(run / "budget.json")
            if (budget["status"] != "closed" or budget.get("optimizer_step") != 0
                    or (run / "selected-checkpoint.json").exists()):
                raise ValueError("fusion smoke ledger is not closed step0")
            return None
        raise ValueError("fusion partial run has no durable recovery checkpoint")
    step, _, checkpoint = max(candidates)
    if not 0 < step <= 16000:
        raise ValueError("fusion recovery checkpoint step outside horizon")
    # Public neutral validator checks sampler/RNG/ownership on CPU. The runner
    # repeats admission under its run writer lock, including terminal recovery.
    result = phase.cpu_query(fusion_query_command(phase, "recovery", run, expected, checkpoint))
    if result.get("step") != step or result.get("checkpoint") != str(checkpoint):
        raise ValueError("fusion durable recovery validation failed")
    return checkpoint


def windows_free_bytes() -> int:
    # Study contract requires independent backing-volume evidence. Missing
    # PowerShell is an admission failure, not an implicit Windows waiver.
    return int(subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command", "(Get-PSDrive -Name C).Free"],
        text=True, timeout=30).strip())


def storage_projection(runs: dict, checkpoint_bytes: float = CHECKPOINT_BYTES) -> dict:
    if not finite_nonnegative(checkpoint_bytes) or checkpoint_bytes == 0:
        raise ValueError("fusion checkpoint size invalid")
    keys = {f"{seed}-{arm}" for seed in (0, 1) for arm in ARMS}
    if not set(runs) <= keys:
        raise ValueError("fusion storage unknown run key")
    unfinished = 6 - sum(record.get("complete") is True for record in runs.values())
    slots = unfinished * 5 + int(unfinished > 0)
    allocated = 0
    for record in runs.values():
        if record.get("complete"):
            continue  # Complete files are already reflected in free space.
        run = Path(record["run"])
        if run.is_symlink():
            raise ValueError("fusion storage run symlink")
        checkpoints = [p for p in run.glob("*.ckpt")
                       if re.fullmatch(r"(?:step|resume)-\d+\.ckpt", p.name)]
        for p in checkpoints:
            regular(p)
        # Credit allocated bytes, not sparse logical lengths or foreign files.
        # Never credit more than this run's five retained slots.
        allocated += min(5 * checkpoint_bytes, sum(p.stat().st_blocks * 512 for p in checkpoints))
    demand = max(0, slots * checkpoint_bytes - allocated)
    return {"checkpoint_slots": slots, "checkpoint_bytes": checkpoint_bytes,
            "allocated_owned_bytes": allocated, "remaining_demand_bytes": demand,
            "reserve_bytes": RESERVE_BYTES, "required_free_bytes": demand + RESERVE_BYTES}


def require_study_storage(root: Path, runs: dict, checkpoint_bytes: float) -> dict:
    projection = storage_projection(runs, checkpoint_bytes)
    linux, windows = shutil.disk_usage(root).free, windows_free_bytes()
    required = projection["required_free_bytes"]
    if linux < required:
        raise ValueError(f"fusion linux storage cannot fit demand+50 GiB: {linux} < {required}")
    if windows < required:
        raise ValueError(f"fusion windows storage cannot fit demand+50 GiB: {windows} < {required}")
    return {**projection, "linux_free_bytes": linux, "windows_free_bytes": windows}


def collect_result(phase: FusionPhase, arm: str, run: Path, expected: dict) -> dict:
    selected = completion_selection(phase, run, expected)
    paths = {str(run / name): sha for name, sha in selected["source_files"].items()}
    for path, sha in paths.items():
        regular(Path(path))
        if file_hash(Path(path)) != sha:
            raise ValueError("fusion validated telemetry artifact bytes changed before collection")
    reports = [read_json(run / f"dev-{step}.json") for step in runner.DEV_STEPS]
    report = next(r for r in reports if r["checkpoint_step"] == selected["step"])
    tokens = report["visual_token_lengths"]
    ids = tuple(i["clip_id"] for i in report["items"])
    for trajectory in reports:
        validate_report_telemetry(trajectory, ids, selected["initialization_hash"],
                                  selected["checkpoint_sizes_bytes"], selected["fusion_telemetry"])
        if trajectory["visual_token_lengths"] != tokens:
            raise ValueError("fusion measured dev token lengths changed across checkpoints")
    smoke_path = run / "gradient-smoke.json"
    regular(smoke_path)
    smoke = read_json(smoke_path)
    for path, sha in paths.items():
        if file_hash(Path(path)) != sha:
            raise ValueError("fusion validated telemetry artifact bytes changed during collection")
    return {"seed": phase.fusion_seed, "arm": arm, "run": str(run), "run_key": run.name,
            "identity": expected, "selected_step": selected["step"],
            "checkpoint_hash": selected["checkpoint_hash"], "metrics": selected["metrics"],
            "items": report["items"],
            "prediction_diagnostics": prediction_diagnostics(report["items"]),
            "dev_trajectory": reports, "fusion_stats": report["fusion_stats"],
            "hardware": expected["hardware"],
            "token_lengths": [row["valid_tokens"] for row in tokens],
            "visual_token_lengths": tokens, "token_lengths_evidence": TOKEN_SCOPE,
            "parameters": selected["initialization"]["trainable_parameters"],
            "initialization": selected["initialization"],
            "initialization_hash": selected["initialization_hash"],
            "checkpoint_sizes_bytes": selected["checkpoint_sizes_bytes"],
            "fusion_telemetry": selected["fusion_telemetry"],
            "gpu_seconds": read_json(run / "budget.json")["gpu_seconds"],
            "controller_charged_seconds": phase.run_spent(arm), "gradient_smoke": smoke,
            "smoke_cuda_peak_bytes": max(r["cuda_peak_bytes"] for r in smoke.values()),
            "memory_evidence": PEAK_SCOPE,
            "source_files": paths}


def aggregate_fusion_results(results: list[dict]) -> dict:
    policies = {runner.motion_run_policy(row.get("identity", {})) for row in results}
    if len(policies) != 1:
        raise ValueError("fusion summary mixes motion mode policies")
    policy = NEW_POLICY if policies == {runner.NEW_RUN_POLICY} else POLICY
    motion_metadata = {}
    if policy == NEW_POLICY:
        reference = results[0]["identity"]["motion_package"]
        if any(row["identity"]["motion_package"] != reference for row in results):
            raise ValueError("fusion summary new motion package mismatch")
        motion_metadata["motion_package"] = reference
    pairs = {}
    for row in results:
        key = (row["seed"], row["arm"])
        if key in pairs or type(row["seed"]) is not int:
            raise ValueError("fusion summary duplicate/invalid seed-arm key")
        pairs[key] = row
    if len(results) != 6 or set(pairs) != {(s, a) for s in (0, 1) for a in ARMS}:
        raise ValueError("fusion summary requires exactly six unique seed-arm rows")
    first = pairs[0, "reference"]
    references = [(i["clip_id"], i["reference"]) for i in first["items"]]
    if len(references) != 582 or len({i[0] for i in references}) != 582:
        raise ValueError("fusion summary requires full ordered 582 references")
    metrics = set(first["metrics"])
    if not {"bleu1", "bleu2", "bleu3", "bleu4", "rougeL_precision",
            "rougeL_recall", "rougeL_f1"} <= metrics:
        raise ValueError("fusion summary metric inventory missing BLEU1-4/ROUGE-L")
    for row in results:
        if (set(row["metrics"]) != metrics
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in row["metrics"].values())
                or [(i["clip_id"], i["reference"]) for i in row["items"]] != references
                or row["token_lengths"] != first["token_lengths"]
                or len(row["token_lengths"]) != 582
                or any(type(n) is not int or n <= 0 for n in row["token_lengths"])
                or row["hardware"] != first["hardware"]
                or not finite_nonnegative(row["gpu_seconds"])):
            raise ValueError("fusion summary metrics/references/tokens/hardware/accounting "
                             "mismatch")
        counts = row.get("parameters")
        if (not isinstance(counts, dict)
                or set(counts) != {"total", "adapter", "vt", "adapter_vt", "flan"}
                or any(type(n) is not int or n <= 0 for n in counts.values())
                or counts != row.get("initialization", {}).get("trainable_parameters")
                or counts["adapter_vt"] != counts["adapter"] + counts["vt"]
                or counts["total"] != counts["adapter_vt"] + counts["flan"]):
            raise ValueError("fusion actual immutable trainable counts mismatch")
        telemetry = row.get("fusion_telemetry")
        if not isinstance(telemetry, dict):
            raise ValueError("fusion persisted production telemetry missing")
        require_production_telemetry(telemetry)
        sizes = row.get("checkpoint_sizes_bytes")
        if (not isinstance(sizes, dict)
                or any(re.fullmatch(r"(?:step|resume)-\d+\.ckpt", name) is None
                       or type(size) is not int or size <= 0 for name, size in sizes.items())):
            raise ValueError("fusion actual checkpoint size inventory invalid")
        reports = row.get("dev_trajectory")
        if (not isinstance(reports, list)
                or [r.get("checkpoint_step") for r in reports] != list(runner.DEV_STEPS)):
            raise ValueError("fusion measured trajectory requires all three ordered dev reports")
        if (row.get("token_lengths_evidence") != TOKEN_SCOPE
                or row.get("visual_token_lengths") != first.get("visual_token_lengths")):
            raise ValueError("fusion diagnostic measured token lengths/scope mismatch")
        for report in reports:
            validate_report_telemetry(report, tuple(i[0] for i in references),
                                      row["initialization_hash"], sizes, telemetry)
            if (report["visual_token_lengths"] != row["visual_token_lengths"]
                    or [r["valid_tokens"] for r in report["visual_token_lengths"]]
                    != row["token_lengths"]):
                raise ValueError("fusion measured token lengths differ across dev checkpoints")
    contrasts = {}
    for label, candidate, control in (
            ("fixed-reference", "aligned_fixed", "reference"),
            ("learned-fixed", "aligned_learned", "aligned_fixed"),
            ("learned-reference", "aligned_learned", "reference")):
        contrasts[label] = {}
        for metric in sorted(metrics):
            deltas = [pairs[s, candidate]["metrics"][metric] - pairs[s, control]["metrics"][metric]
                      for s in (0, 1)]
            contrasts[label][metric] = {"per_seed": deltas, "mean": statistics.mean(deltas),
                                         "sample_std": statistics.stdev(deltas)}
    arms = {arm: {metric: {"per_seed": [pairs[s, arm]["metrics"][metric] for s in (0, 1)],
                            "mean": statistics.mean(pairs[s, arm]["metrics"][metric]
                                                    for s in (0, 1)),
                            "sample_std": statistics.stdev(pairs[s, arm]["metrics"][metric]
                                                          for s in (0, 1))}
                  for metric in sorted(metrics)} for arm in ARMS}
    maxima = results[0]["fusion_telemetry"]
    for row in results[1:]:
        maxima = runner.merge_fusion_telemetry(maxima, row["fusion_telemetry"])
    parameter_deltas = {arm: [pairs[s, arm]["parameters"]["total"]
                              - pairs[s, "reference"]["parameters"]["total"] for s in (0, 1)]
                        for arm in ARMS}
    return {"study_policy": policy, **motion_metadata,
            "status": "complete", "comparison_scope": "dev_only",
            "signer07_scored": False, "training_seeds": [0, 1], "contrasts": contrasts,
            "arms": arms, "results": results, "hardware": first["hardware"],
            "runner_gpu_seconds_total": sum(r["gpu_seconds"] for r in results),
            "ordered_reference_hash": digest(references), "token_lengths_matched": True,
            "visual_token_length_scope": TOKEN_SCOPE,
            "production_cuda_high_water_maxima": maxima,
            "actual_total_parameter_deltas_vs_reference": parameter_deltas,
            "limitations": ["Two seeds and one dev signer; no signer-invariance claim.",
                            "Extra corrector capacity confounds fixed-reference timing "
                            "interpretation.",
                            "Earlier reference variability limits two-seed conclusions.",
                            "Token masks were measured during eval diagnostic forwards, "
                            "not captured from generation forwards.",
                            "CUDA peaks are conservative process high-water observations "
                            "sampled during phases, not exclusive phase footprints; "
                            "maxima across runs/phases must not be summed as memory demand."]}


def run_phase(phase: FusionPhase, profile: Path, timing: Path, runs: dict) -> list[dict]:
    motion_package = getattr(phase, "motion_package", None)
    motion_kwargs = {"motion_package": motion_package} if motion_package is not None else {}
    if phase.fusion_seed == 0:
        phase.active_arm = "reference"
        phase.run_stage("common-initialization",
                         stage_command("reference", 0, profile, timing, "initialize",
                                       **motion_kwargs), RUN_CAP)
    regular(profile)
    results = []
    for arm in ARMS:
        phase.active_arm = arm
        ready = phase.cpu_query(stage_command(arm, phase.fusion_seed, profile, timing, "preflight",
                                              **motion_kwargs))
        if ready.get("status") != "ready":
            raise ValueError("fusion preflight did not return ready")
        expected, run = ready["identity"], Path(ready["output"])
        if motion_package is not None:
            if (runner.motion_run_policy(expected) != runner.NEW_RUN_POLICY
                    or expected["motion_package"]["root"] != str(motion_package)):
                raise ValueError("fusion preflight new motion mode/package mismatch")
        elif "motion_package" in expected or expected.get("run_policy") == runner.NEW_RUN_POLICY:
            raise ValueError("fusion legacy phase cannot adopt new motion run")
        if (expected.get("arm") != arm or expected.get("seed") != phase.fusion_seed
                or expected.get("gpu_cap_seconds") != RUN_CAP
                or expected.get("recipe") != runner.RECIPE
                or run.parent != profile.parent / "runs" or run.name != expected.get("run_key")):
            raise ValueError("fusion preflight seed/arm/recipe/cap/output mismatch")
        validate_live(expected, timing, profile)
        key = f"{phase.fusion_seed}-{arm}"
        previous = runs.get(key)
        if previous and previous["identity"] != expected:
            raise ValueError("fusion controller durable run identity changed")
        phase.bind_run(arm, run, expected)
        complete = validate_completion(phase, run, expected)
        checkpoint = None if complete else recovery_checkpoint(phase, run, expected)
        record = {"run": str(run), "identity": expected, "complete": complete,
                  "checkpoint_sizes_bytes": runner.fusion_checkpoint_sizes(run)}
        runs[key] = record
        phase.state["runs"][arm] = record
        phase.save()
        command = stage_command(arm, phase.fusion_seed, profile, timing, "run", **motion_kwargs)
        if not complete:
            # An owned production recovery never runs smoke CLI. Fresh smoke uses
            # runner's step0 same-cap ledger; run adopts it without reset.
            if not run.exists():
                storage = require_study_storage(profile.parent, runs, checkpoint_size(runs))
                phase.state["last_storage_admission"] = storage
                phase.save()
                phase.run_stage(f"{arm}-smoke",
                                 stage_command(arm, phase.fusion_seed, profile, timing, "smoke",
                                               **motion_kwargs),
                                RUN_CAP)
            if checkpoint is not None:
                command.extend(("--resume-checkpoint", str(checkpoint)))
            validate_live(expected, timing, profile)
            phase.state["last_storage_admission"] = require_study_storage(
                profile.parent, runs, checkpoint_size(runs))
            phase.save()
        phase.run_stage(f"{arm}-train", command, RUN_CAP,
                        completion_validator=lambda run=run, expected=expected:
                        validate_completion(phase, run, expected))
        result = collect_result(phase, arm, run, expected)
        record["complete"] = True
        record["checkpoint_sizes_bytes"] = result["checkpoint_sizes_bytes"]
        results.append(result)
        atomic_json(phase.root / f"{arm}-result.json", result)
        phase.save()
    phase.state["status"] = "complete"
    phase.save()
    return results


def checkpoint_size(runs: dict) -> float:
    sizes = []
    for record in runs.values():
        actual = runner.fusion_checkpoint_sizes(Path(record["run"]))
        if actual != record.get("checkpoint_sizes_bytes", {}):
            raise ValueError("fusion saved checkpoint size inventory changed")
        if any(type(size) is not int or size <= 0 for size in actual.values()):
            raise ValueError("fusion saved checkpoint size must be positive")
        sizes.extend(actual.values())
    # Current runner emits no smoke checkpoint. Never manufacture one or use
    # tiny synthetic smoke diagnostics as a checkpoint-size estimate.
    return max(sizes) if sizes else CHECKPOINT_BYTES


def validate_summary_sources(results: list[dict], root: Path, profile: Path,
                             timing: Path, metadata: dict) -> None:
    """Recheck collected bytes against admitted identities under controller/phase locks."""
    regular(profile)
    regular(timing)
    regular(INPUTS["protocol"])
    if (file_hash(timing) != metadata["timing_hash"]
            or controller_sources(new_motion="motion_package" in metadata)[
                "scripts/run_fusion_study.py"] != metadata["controller_hash"]
            or (controller_sources(new_motion=True) if "motion_package" in metadata
                else controller_sources()) != metadata["controller_sources"]):
        raise ValueError("fusion summary admitted controller/timing bytes changed")
    for row in results:
        run, identity = Path(row["run"]), row["identity"]
        if "motion_package" in metadata:
            if (runner.motion_run_policy(identity) != runner.NEW_RUN_POLICY
                    or identity["motion_package"] != metadata["motion_package"]):
                raise ValueError("fusion summary new motion graph/mode mismatch")
            validate_live(identity, timing, profile)
        elif "motion_package" in identity or identity.get("run_policy") == runner.NEW_RUN_POLICY:
            raise ValueError("fusion legacy summary cannot adopt new motion run")
        if (run.parent != root / "runs" or run.is_symlink() or not run.is_dir()
                or run.name != row["run_key"] or run.name != identity["run_key"]
                or identity["seed"] != row["seed"] or identity["arm"] != row["arm"]):
            raise ValueError("fusion summary run ownership mismatch")
        if (identity["timing_hash"] != file_hash(timing)
                or identity["initialization_profile_file_hash"] != file_hash(profile)
                or identity["initialization_profile_path"] != str(profile.resolve())
                or identity["bound_files"].get(str(INPUTS["protocol"].resolve()))
                != file_hash(INPUTS["protocol"])):
            raise ValueError("fusion summary admitted timing/profile/protocol bytes changed")
        sizes = runner.fusion_checkpoint_sizes(run)
        if sizes != row["checkpoint_sizes_bytes"]:
            raise ValueError("fusion summary checkpoint inventory changed")
        names = {"budget.json", "run-status.json", "selected-checkpoint.json",
                 "initialization.json", "gradient-smoke.json",
                 *[f"dev-{s}.json" for s in runner.DEV_STEPS],
                 *[f"step-{s}.ckpt" for s in runner.DEV_STEPS], *sizes}
        sources = row.get("source_files")
        if not isinstance(sources, dict) or set(sources) != {str(run / n) for n in names}:
            raise ValueError("fusion summary complete source inventory required")
        for path, sha in sources.items():
            regular(Path(path))
            if file_hash(Path(path)) != sha:
                raise ValueError("fusion summary collected artifact bytes changed")


def execute_phases(root: Path, profile: Path, timing: Path, metadata: dict, state_path: Path,
                   publish: Callable[[list[dict], list[dict]], None] | None = None):
    runs, results, accounting = {}, [], []
    with ExitStack() as stack:
        # Recover/stop BOTH seeds' owned orphan groups before launching seed0.
        # Keep both phase locks until the sequential six-run schedule finishes.
        motion_kwargs: dict = ({"motion_package": Path(metadata["motion_package"]["root"])}
                         if "motion_package" in metadata else {})
        credit = metadata.get("recovery", {}).get("inherited_seconds", 0)
        phases = [stack.enter_context(FusionPhase(
                      root / f"seed{seed}", seed,
                      **({**motion_kwargs, "inherited_seconds": credit} if seed == 0 and credit
                         else motion_kwargs)))
                  for seed in (0, 1)]
        for phase in phases:
            for arm, record in phase.state["runs"].items():
                run = Path(record["run"])
                expected = record["identity"]
                if (arm not in ARMS or expected.get("arm") != arm
                        or expected.get("seed") != phase.fusion_seed
                        or run.parent != root / "runs"):
                    raise ValueError("fusion inventory phase/run ownership mismatch")
                if "motion_package" in metadata:
                    if (runner.motion_run_policy(expected) != runner.NEW_RUN_POLICY
                            or expected["motion_package"] != metadata["motion_package"]):
                        raise ValueError("fusion inventory motion graph/policy mismatch")
                elif runner.motion_run_policy(expected) != runner.RUN_POLICY:
                    raise ValueError("fusion legacy inventory cannot adopt new motion run")
                phase.active_arm = arm
                validate_live(expected, timing, profile)
                phase.bind_run(arm, run, expected)
                complete = validate_completion(phase, run, expected)
                runs[f"{phase.fusion_seed}-{arm}"] = {
                    **record, "complete": complete,
                    "checkpoint_sizes_bytes": runner.fusion_checkpoint_sizes(run)}
        for phase in phases:
            results.extend(run_phase(phase, profile, timing, runs))
            accounting.append({"seed": phase.fusion_seed,
                               "spent_seconds": phase.state["spent_seconds"],
                               "stages": phase.state["stages"]})
            metadata.update(completed_seeds=list(range(phase.fusion_seed + 1)),
                            spent_seconds=sum(a["spent_seconds"] for a in accounting))
            atomic_json(state_path, metadata)
        if publish is not None:
            publish(results, accounting)  # Both seed locks still held through final publication.
    return results, accounting


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--motion-timing", type=Path, required=True)
    parser.add_argument("--motion-package", type=Path)
    parser.add_argument("--recovery-record", type=Path)
    parser.add_argument("--time-waiver-record", type=Path)
    parser.add_argument("--authorize-fusion-study", action="store_true")
    args = parser.parse_args(argv)
    if args.time_waiver_record is not None:
        os.environ["DESPAMO_FUSION_TIME_WAIVER"] = str(args.time_waiver_record)
        if not time_waived():
            raise ValueError("time waiver was not authorized")
    if not args.authorize_fusion_study:
        raise ValueError("fusion controller requires --authorize-fusion-study")
    regular(args.motion_timing)  # No controller/profile/GPU initialization without timing bytes.
    root = args.study_root or (NEW_DEFAULT_ROOT if args.motion_package is not None
                                else DEFAULT_ROOT)
    if time_waived():
        from scripts.fusion_time_waiver import STUDY
        from scripts.fusion_time_waiver import policy as time_policy

        waiver = time_policy()
        assert waiver is not None
        if root != STUDY or str(args.motion_package) != waiver["motion_package"]:
            raise ValueError("time waiver cannot apply to another controller/data study")
    package = None
    recovery = None
    policy = POLICY
    if args.motion_package is not None:
        frames = read_json(INPUTS["adapted-root"] / "complete/frame_rows.json")
        package = runner.validate_motion_package(args.motion_package, frames, verify_features=True)
        if (str(args.motion_timing) != package["timing_path"]
                or file_hash(args.motion_timing) != package["timing_hash"]):
            raise ValueError("new motion controller timing path/bytes mismatch")
        runner.validate_new_motion_production(package, frames)
        destination_args = argparse.Namespace(
            **{k.replace("-", "_"): p for k, p in INPUTS.items()},
            baseline_run=REFERENCE, motion_package=args.motion_package)
        runner._safe_destination(root, destination_args)
        runner.fusion_code_hashes(ROOT, new_motion=True)
        policy = NEW_POLICY
    if args.recovery_record is not None:
        from despamo.training.validation_recovery import validate_training_recovery

        if args.motion_package is None:
            raise ValueError("training recovery requires completed new motion package")
        recovery = validate_training_recovery(args.recovery_record, root, args.motion_package)
    if (any(p.is_symlink() for p in (root, *root.parents))
            or root.resolve().is_relative_to(ROOT)):
        raise ValueError("fusion study root must be external and regular")
    root.mkdir(parents=True, exist_ok=True)
    if (root / ".fusion-controller.lock").is_symlink():
        raise ValueError("fusion controller lock unsafe")
    lock = (root / ".fusion-controller.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "state.json"
        profile = root / "common-initialization.json"
        binding = {"study_policy": policy, "budget_seconds": 64800,
                   "per_seed_budget_seconds": PHASE_CAP, "per_run_budget_seconds": RUN_CAP,
                   "timing_path": str(args.motion_timing.resolve()),
                   "timing_hash": file_hash(args.motion_timing), "profile": str(profile),
                    "controller_hash": controller_sources(new_motion=package is not None)[
                        "scripts/run_fusion_study.py"]}
        binding["controller_sources"] = (controller_sources() if package is None
                                          else controller_sources(new_motion=True))
        if package is not None:
            binding["motion_package"] = runner.compact_motion_binding(package)
        if recovery is not None:
            binding["recovery"] = recovery
        if path.exists():
            regular(path)
            metadata = read_json(path)
            if any(metadata.get(k) != v for k, v in binding.items()):
                raise ValueError("fusion controller policy/budget/timing/source mismatch")
        else:
            metadata = {**binding, "status": "created"}
        metadata.update(**execution_fields())
        old_term = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
        try:
            metadata["status"] = "running"
            atomic_json(path, metadata)
            (root / "runs").mkdir(exist_ok=True)
            def publish(results, accounting):
                summary = aggregate_fusion_results(results)
                summary["study_policy"] = policy
                if package is not None:
                    summary["motion_package"] = binding["motion_package"]
                summary.update(budget_seconds=64800, phase_accounting=accounting,
                               spent_charged_seconds=sum(a["spent_seconds"] for a in accounting),
                               profile_sha256=file_hash(profile),
                               timing_sha256=file_hash(args.motion_timing),
                               controller_sha256=file_hash(Path(__file__)),
                                controller_sources=binding["controller_sources"],
                                protocol_sha256=file_hash(INPUTS["protocol"]))
                summary.update(**execution_fields())
                summary["content_hash"] = digest(summary)
                validate_summary_sources(results, root, profile, args.motion_timing, metadata)
                atomic_json(root / "summary.json", summary)
                metadata.update(status="complete", summary=str(root / "summary.json"),
                                summary_sha256=file_hash(root / "summary.json"))
                atomic_json(path, metadata)

            execute_phases(root, profile, args.motion_timing, metadata, path, publish)
            print(json.dumps({"status": "complete", "summary": metadata["summary"]}))
        except BaseException as error:
            metadata.update(status="incomplete", error=type(error).__name__,
                            error_message=str(error))
            atomic_json(path, metadata)
            raise
        finally:
            signal.signal(signal.SIGTERM, old_term)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
