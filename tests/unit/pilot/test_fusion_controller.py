"""Synthetic controller contracts, never historical timing or GPU evidence."""

import copy
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from scripts import run_projector_overnight as neutral


@pytest.mark.parametrize("kind", ["selection", "recovery"])
def test_production_identity_crosses_real_subprocess_without_large_argv(
    controller, tmp_path, monkeypatch, kind,
):
    identity, _, _ = identity_fixture(tmp_path)
    identity["bound_files"] = {
        f"/owned/receipts/train/{i:05d}-" + "x" * 140 + ".json": "a" * 64
        for i in range(7096)
    }
    assert len(json.dumps(identity).encode()) > 1_700_000
    run = tmp_path / "owned"
    run.mkdir()
    checkpoint = run / "step-5500.ckpt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(controller, "owned_status", lambda *args: {"status": "complete"})
    monkeypatch.setattr(controller, "require_production_telemetry", lambda *args: None)

    def query(command):
        # Real exec boundary, even before request transport exists (RED: E2BIG).
        subprocess.run(["/bin/true", *command], check=True, timeout=10)
        request = controller.read_fusion_query(Path(command[3]), command[4])
        assert request["expected_identity"] == identity
        child = subprocess.run([sys.executable, "-c",
            "import sys; from pathlib import Path; "
            "from scripts.run_fusion_study import read_fusion_query; "
            "print(read_fusion_query(Path(sys.argv[1]),sys.argv[2])['expected_identity_digest'])",
            *command[3:]], check=True, capture_output=True, text=True, timeout=30)
        assert child.stdout.strip() == digest(identity)
        return ({"status": "complete", "fusion_telemetry": {}} if kind == "selection"
                else {"step": 5500, "checkpoint": str(checkpoint)})

    phase = SimpleNamespace(root=tmp_path, cpu_query=query)
    if kind == "selection":
        controller.completion_selection(phase, run, identity)
    else:
        assert controller.recovery_checkpoint(phase, run, identity) == checkpoint


@pytest.fixture
def controller():
    assert importlib.util.find_spec("scripts.run_fusion_study"), "Task4 controller missing"
    return importlib.import_module("scripts.run_fusion_study")


def rows():
    data = [
        {"seed": seed, "arm": arm,
         "metrics": {k: float(seed + offset) for k in
                     ("bleu1", "bleu2", "bleu3", "bleu4", "rougeL_precision",
                      "rougeL_recall", "rougeL_f1")},
         "items": [{"clip_id": str(i), "reference": f"ref {i}", "prediction": "words"}
                   for i in range(582)],
         "hardware": {"uuid": "synthetic"}, "token_lengths": [42] * 582,
         "gpu_seconds": 100.0}
        for seed in (0, 1)
        for offset, arm in enumerate(("reference", "aligned_fixed", "aligned_learned"))
    ]
    for row in data:
        parameters = {"adapter": 4, "vt": 2, "flan": 3, "adapter_vt": 6, "total": 9}
        row.update(parameters=parameters, initialization={"trainable_parameters": parameters},
                   initialization_hash="a" * 64, fusion_telemetry=simulated_peaks(),
                   checkpoint_sizes_bytes={f"step-{s}.ckpt": 8192 for s in (5500, 10400, 16000)},
                   selected_step=5500, visual_token_lengths=[
                       {"clip_id": str(i), "valid_tokens": 42, "padded_tokens": 44}
                       for i in range(582)],
                   token_lengths_evidence="eval_diagnostic_forward_actual_output_mask")
        row["dev_trajectory"] = [
            {"checkpoint_step": s, "initialization_hash": row["initialization_hash"],
             "checkpoint_size_bytes": 8192, "visual_token_lengths": row["visual_token_lengths"],
             "visual_token_length_scope": "eval_diagnostic_forward_actual_output_mask",
             "cuda_memory": simulated_peaks()["dev"]} for s in (5500, 10400, 16000)]
    return data


def simulated_peaks():
    """Explicit CUDA-shaped metadata; measurements never run on actual CUDA."""
    return {"schema_version": 1, **{phase: {
        "device_type": "cuda", "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": allocated + 10, "sample_count": 3,
        "evidence_scope": "process_allocator_high_water_sampled_during_phase"}
        for phase, allocated in (("training", 456), ("dev", 789))}}


def test_summary_paired_contrasts_all_metrics(controller):
    result = controller.aggregate_fusion_results(rows())
    assert result["comparison_scope"] == "dev_only"
    assert result["signer07_scored"] is False
    for contrast in ("fixed-reference", "learned-fixed"):
        assert result["contrasts"][contrast]["bleu4"] == {
            "per_seed": [1.0, 1.0], "mean": 1.0, "sample_std": 0.0}
    assert result["contrasts"]["learned-reference"]["bleu4"]["mean"] == 2.0


@pytest.mark.parametrize("mutation", ["duplicate", "reference", "order", "nan", "tokens"])
def test_summary_rejects_unpaired_or_nonfinite_rows(controller, mutation):
    data = rows()
    if mutation == "duplicate":
        data[-1] = copy.deepcopy(data[0])
    elif mutation == "reference":
        data[-1]["items"][0]["reference"] = "changed"
    elif mutation == "order":
        data[-1]["items"].reverse()
    elif mutation == "nan":
        data[-1]["metrics"]["bleu1"] = float("nan")
    else:
        data[-1]["token_lengths"][0] += 1
    with pytest.raises(ValueError):
        controller.aggregate_fusion_results(data)


def test_commands_match_runner_parser_and_distinct_modes(controller, tmp_path):
    commands = [controller.stage_command("reference", 0, tmp_path / "profile.json",
                                         tmp_path / "timing.json", mode)
                for mode in ("initialize", "preflight", "smoke", "run")]
    assert len({neutral.command_signature(c) for c in commands}) == 4
    for command in commands:
        assert command[1].endswith("train_fusion_study.py")
        assert command[command.index("--gpu-cap-seconds") + 1] == "10800"
        assert command[command.index("--steps") + 1] == "16000"
        assert "--authorize-fusion-run" in command


def test_phase_restart_and_shared_run_cap(controller, tmp_path):
    command = [sys.executable, "-c", "pass"]
    with controller.FusionPhase(tmp_path, 0) as phase:
        phase.active_arm = "reference"
        phase.run_stage("reference-smoke", command, 10800)
        phase.state["stages"]["reference-smoke"]["elapsed_seconds"] = 10799.5
        phase.state["spent_seconds"] = 10799.5
        phase.save()
    with controller.FusionPhase(tmp_path, 0) as phase:
        assert phase.run_spent("reference") == 10799.5
        phase.active_arm = "reference"
        with pytest.raises(RuntimeError, match="budget"):
            phase.run_stage("reference-train", command, 10800)
        assert phase.state["budget_seconds"] == 32400


@pytest.mark.parametrize("conflict", ["policy", "seed", "cap"])
def test_phase_conflict_never_writes_incumbent(controller, tmp_path, conflict):
    with controller.FusionPhase(tmp_path, 0):
        pass
    path = tmp_path / "study-state.json"
    state = read_json(path)
    key, value = {"policy": ("study_policy", "other"), "seed": ("fusion_seed", 1),
                  "cap": ("budget_seconds", 36000)}[conflict]
    state[key] = value
    atomic_json(path, state)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        controller.FusionPhase(tmp_path, 0)
    assert path.read_bytes() == before


def test_phase_lock_failure_no_writes(controller, tmp_path):
    with controller.FusionPhase(tmp_path, 0):
        before = (tmp_path / "study-state.json").read_bytes()
        with pytest.raises(BlockingIOError):
            controller.FusionPhase(tmp_path, 0)
        assert (tmp_path / "study-state.json").read_bytes() == before


def test_recovery_uses_neutral_uncapped_boot_accounting(controller, tmp_path, monkeypatch):
    with controller.FusionPhase(tmp_path, 0) as phase:
        phase.state["stages"]["reference-train"] = {
            "status": "running", "elapsed_seconds": 100.0, "owned_arm": "reference",
            "command_hash": neutral.command_signature(["synthetic"]), "attempts": 1,
            "last_wall": 10.0, "boot_id": "old-boot", "reserved_seconds": 11000.0,
            "pid": None}
        phase.state["spent_seconds"] = 100.0
        phase.save()
    monkeypatch.setattr(neutral.time, "time", lambda: 20.0)
    with controller.FusionPhase(tmp_path, 0) as phase:
        assert phase.state["spent_seconds"] == 11100
        assert phase.run_spent("reference") == 11100
        assert phase.state["stages"]["reference-train"]["status"] == "interrupted"


def test_storage_initial_and_allocated_remaining_demand(controller, tmp_path):
    gib = 1024**3
    first = controller.storage_projection({}, 5.6 * gib)
    assert first["checkpoint_slots"] == 31
    assert first["required_free_bytes"] / gib == pytest.approx(223.6)
    owned = tmp_path / "run"
    owned.mkdir()
    ckpt = owned / "step-5500.ckpt"
    ckpt.write_bytes(b"x" * 8192)
    record = {"run": str(owned), "complete": False}
    second = controller.storage_projection({"0-reference": record}, 5.6 * gib)
    assert second["required_free_bytes"] == pytest.approx(
        first["required_free_bytes"] - ckpt.stat().st_blocks * 512)
    record["complete"] = True
    third = controller.storage_projection({"0-reference": record}, 5.6 * gib)
    assert third["checkpoint_slots"] == 26
    assert third["required_free_bytes"] / gib == pytest.approx(195.6)


@pytest.mark.parametrize("disk", ["linux", "windows"])
def test_independent_storage_failure(controller, tmp_path, disk, monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(controller.shutil, "disk_usage", lambda p:
                        SimpleNamespace(free=(49 if disk == "linux" else 300) * gib))
    monkeypatch.setattr(controller, "windows_free_bytes", lambda:
                        (49 if disk == "windows" else 300) * gib)
    with pytest.raises(ValueError, match=disk):
        controller.require_study_storage(tmp_path, {}, 5.6 * gib)


def identity_fixture(tmp_path):
    timing, profile = tmp_path / "timing.json", tmp_path / "profile.json"
    atomic_json(timing, {"synthetic": True})
    atomic_json(profile, {"synthetic": True})
    identity = {"seed": 0, "arm": "reference", "run_key": "owned",
                "gpu_cap_seconds": 10800, "run_policy": "signer-pilot-time-aligned-fusion-v1",
                "timing_hash": file_hash(timing), "recipe": {}, "code_hashes": {},
                "hardware": {"synthetic": True},
                "initialization_profile_file_hash": file_hash(profile),
                "initialization_profile_path": str(profile), "bound_files": {}}
    return identity, timing, profile


@pytest.mark.parametrize("mutation", ["raw", "mode", "nonce", "identity", "sources",
                                      "kind", "protocol", "root", "symlink", "phase"])
def test_owned_query_admission_rejects_tampering(controller, tmp_path, mutation):
    identity, _, _ = identity_fixture(tmp_path)
    phase = SimpleNamespace(root=tmp_path, fusion_seed=0, active_arm="reference",
                            motion_package=None)
    command = controller.fusion_query_command(phase, "selection", tmp_path / "owned", identity)
    path = Path(command[3])
    if mutation == "phase":
        phase.active_arm = "aligned_fixed"
    elif mutation == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        command[3] = str(link)
    elif mutation == "mode":
        path.chmod(0o600)
    else:
        request = read_json(path)
        if mutation == "nonce":
            request["nonce"] = "a" * 32
        elif mutation == "identity":
            request["expected_identity"]["seed"] = 1
        elif mutation == "sources":
            request["controller_sources"] = {}
        elif mutation == "kind":
            request["kind"] = "recovery"
        elif mutation == "protocol":
            request["protocol"] = "/foreign/protocol.json"
        elif mutation == "root":
            request["query_root"] = str(tmp_path.parent)
        else:
            request["padding"] = "changed raw bytes"
        atomic_json(path, request)
        path.chmod(0o400)
        if mutation != "raw":
            command[4] = file_hash(path)
    with pytest.raises(ValueError):
        controller.admit_fusion_query(command, phase)


@pytest.mark.parametrize("field", ["seed", "arm", "timing_hash", "checkpoint_hash"])
def test_wrong_partial_selection_rejected_before_launch(controller, tmp_path, field):
    identity, _, _ = identity_fixture(tmp_path)
    run = tmp_path / "owned"
    run.mkdir()
    atomic_json(run / "budget.json", {"identity": identity, "gpu_seconds": 10,
                                       "ceiling_seconds": 10800, "status": "closed"})
    atomic_json(run / "run-status.json", {"status": "partial", "run_key": "owned",
                                          "global_step": 16000})
    checkpoint = run / "step-5500.ckpt"
    checkpoint.write_bytes(b"synthetic")
    selected = {"seed": 0, "arm": "reference", "timing_hash": identity["timing_hash"],
                "run_identity_hash": digest(identity), "step": 5500,
                "checkpoint_hash": file_hash(checkpoint)}
    selected[field] = "wrong"
    atomic_json(run / "selected-checkpoint.json", selected)
    phase = SimpleNamespace(cpu_query=lambda c: pytest.fail("GPU/CPU child must not launch"))
    with pytest.raises(ValueError):
        controller.validate_completion(phase, run, identity)


@pytest.mark.parametrize("field", ["code_hashes", "hardware", "profile", "timing"])
def test_live_bindings_rejected_before_fit(controller, tmp_path, monkeypatch, field):
    identity, timing, profile = identity_fixture(tmp_path)
    monkeypatch.setattr(controller.runner, "fusion_code_hashes", lambda p: {})
    monkeypatch.setattr(controller.runner, "hardware_identity", lambda: {"synthetic": True})
    if field in ("code_hashes", "hardware"):
        identity[field] = {"changed": True}
    elif field == "profile":
        profile.write_bytes(b"changed")
    else:
        timing.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        controller.validate_live(identity, timing, profile)


def test_missing_timing_fails_before_controller_initialization(controller, tmp_path):
    root = tmp_path / "study"
    with pytest.raises(ValueError, match="timing"):
        controller.main(["--study-root", str(root), "--motion-timing", str(tmp_path / "missing"),
                         "--authorize-fusion-study"])
    assert not root.exists()


def test_controller_policy_failure_no_incumbent_write(controller, tmp_path):
    timing = tmp_path / "timing.json"
    atomic_json(timing, {})
    root = tmp_path / "study"
    root.mkdir()
    atomic_json(root / "state.json", {"study_policy": "incumbent"})
    before = (root / "state.json").read_bytes()
    with pytest.raises(ValueError, match="policy"):
        controller.main(["--study-root", str(root), "--motion-timing", str(timing),
                         "--authorize-fusion-study"])
    assert (root / "state.json").read_bytes() == before


@pytest.mark.parametrize("duplicate", [False, True])
def test_cpu_query_cannot_disguise_gpu_mode(controller, tmp_path, monkeypatch, duplicate):
    monkeypatch.setattr(neutral.Study, "cpu_query", lambda *a: pytest.fail("CPU gate bypass"))
    with controller.FusionPhase(tmp_path, 0) as phase:
        command = controller.stage_command("reference", 0, tmp_path / "profile.json",
                                           tmp_path / "timing.json",
                                           "preflight" if duplicate else "smoke")
        if duplicate:
            command.extend(("--mode", "run"))
        with pytest.raises(ValueError, match="CPU"):
            phase.cpu_query(command)
        assert not phase.state["stages"]


def test_prelaunch_completion_queries_reduce_shared_allowance(controller, tmp_path, monkeypatch):
    with controller.FusionPhase(tmp_path, 0) as phase:
        phase.state["stages"]["reference-smoke"] = {
            "status": "complete", "elapsed_seconds": 10790.0, "owned_arm": "reference"}
        phase.state["spent_seconds"] = 10790.0
        captured = []

        def stage(self, name, command, cap, completion_validator=None):
            captured.append(cap)

        def validate():
            phase.state["stages"]["reference-smoke"]["elapsed_seconds"] += 4
            phase.state["spent_seconds"] += 4
            return False

        monkeypatch.setattr(neutral.Study, "run_stage", stage)
        phase.run_stage("reference-train", [sys.executable, "-c", "pass"], 10800, validate)
        assert captured == [6]


def test_recovery_pid_mismatch_never_stops_unowned_process(controller, tmp_path, monkeypatch):
    with controller.FusionPhase(tmp_path, 0) as phase:
        boot = phase.boot_id
        phase.state["stages"]["reference-train"] = {
            "status": "running", "elapsed_seconds": 0.0, "owned_arm": "reference",
            "command_hash": neutral.command_signature(["synthetic"]), "attempts": 1,
            "last_wall": 10.0, "last_monotonic": 10.0, "boot_id": boot,
            "reserved_seconds": 30.0, "pid": 12345,
            "process_identity": {"start_ticks": "old"}, "watchdog_command": ["synthetic"]}
        phase.save()
    monkeypatch.setattr(neutral, "process_identity", lambda p: {"start_ticks": "new"})
    monkeypatch.setattr(neutral, "stop_group", lambda p: pytest.fail("unowned PID killed"))
    monkeypatch.setattr(neutral.time, "monotonic", lambda: 50.0)
    with pytest.raises(RuntimeError, match="identity"):
        controller.FusionPhase(tmp_path, 0)
    assert read_json(tmp_path / "study-state.json")["spent_seconds"] == 40.0


def test_controller_lock_failure_never_writes_incumbent(controller, tmp_path):
    import fcntl

    root = tmp_path / "study"
    root.mkdir()
    timing = tmp_path / "timing.json"
    atomic_json(timing, {})
    path = root / "state.json"
    atomic_json(path, {"study_policy": "incumbent"})
    before = path.read_bytes()
    with (root / ".fusion-controller.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            controller.main(["--study-root", str(root), "--motion-timing", str(timing),
                             "--authorize-fusion-study"])
    assert path.read_bytes() == before


def test_terminal_recovery_prefers_step_on_tie_and_run_mode(controller, completed):
    run, identity, profile, timing = completed
    atomic_json(run / "run-status.json", {"status": "partial", "run_key": run.name,
                                          "global_step": 16000})
    for name in ("resume-16000.ckpt", "resume-15500.ckpt"):
        (run / name).write_bytes(b"synthetic")

    def query(command):
        return {"step": 16000, "checkpoint": str(run / "step-16000.ckpt")}

    checkpoint = controller.recovery_checkpoint(SimpleNamespace(cpu_query=query), run, identity)
    assert checkpoint == run / "step-16000.ckpt"
    command = controller.stage_command("reference", 0, profile, timing, "run")
    command.extend(("--resume-checkpoint", str(checkpoint)))
    assert command[command.index("--mode") + 1] == "run"


def test_orchestration_smoke_init_accounting_and_six_run_order(controller, tmp_path, monkeypatch):
    profile, timing = tmp_path / "profile.json", tmp_path / "timing.json"
    atomic_json(timing, {})
    launches = []
    runs = {}
    monkeypatch.setattr(controller, "validate_live", lambda *a: None)
    monkeypatch.setattr(controller, "validate_completion", lambda phase, run, identity:
                        (run / "complete").exists())
    monkeypatch.setattr(controller, "require_study_storage", lambda *a: {"synthetic": True})
    monkeypatch.setattr(controller, "collect_result", lambda phase, arm, run, identity:
                        {"seed": phase.fusion_seed, "arm": arm, "checkpoint_sizes_bytes": {}})

    def query(self, command):
        arm = command[command.index("--arm") + 1]
        run = tmp_path / "runs" / f"{arm}-seed{self.fusion_seed}"
        identity = {"arm": arm, "seed": self.fusion_seed, "run_key": run.name,
                    "recipe": controller.runner.RECIPE, "gpu_cap_seconds": 10800}
        return {"status": "ready", "identity": identity, "output": str(run)}

    def stage(self, name, command, cap, completion_validator=None):
        mode = command[command.index("--mode") + 1]
        arm = command[command.index("--arm") + 1]
        launches.append((self.fusion_seed, arm, mode))
        self.state["stages"][name] = {"owned_arm": arm, "elapsed_seconds": 2,
                                      "status": "complete"}
        self.state["spent_seconds"] += 2
        if mode == "initialize":
            atomic_json(profile, {"synthetic": True})
        elif mode == "run":
            run = tmp_path / "runs" / f"{arm}-seed{self.fusion_seed}"
            run.mkdir(parents=True)
            (run / "complete").write_text("synthetic")

    monkeypatch.setattr(controller.FusionPhase, "cpu_query", query)
    monkeypatch.setattr(controller.FusionPhase, "run_stage", stage)
    for seed in (0, 1):
        with controller.FusionPhase(tmp_path / f"seed{seed}", seed) as phase:
            controller.run_phase(phase, profile, timing, runs)
            assert phase.state["spent_seconds"] == (14 if seed == 0 else 12)
    assert launches == [(0, "reference", "initialize")] + [
        (seed, arm, mode) for seed in (0, 1) for arm in controller.ARMS
        for mode in ("smoke", "run")]
    assert len(runs) == 6 and all(r["complete"] for r in runs.values())


@pytest.fixture
def completed(controller, tmp_path, monkeypatch):
    import torch

    runner = controller.runner
    identity, timing, profile = identity_fixture(tmp_path)
    identity.update(recipe=copy.deepcopy(runner.RECIPE), variant="E3_fusion_reference", steps=16000,
                    architecture=copy.deepcopy(runner.FUSION_CONFIGS["reference"]),
                    dev_checkpoints=list(runner.DEV_STEPS), protocol_hash="protocol",
                    split_hash="split", supervision_policy=runner.POLICY,
                    human_review_status="not_assessed",
                    initial_base_hash="b" * 64, common_plain_hash="c" * 64,
                    dev_frame_coverage={"clip_count": 582, "real_frame_count": 582 * 24,
                                        "supported_frame_count": 582 * 24},
                    config={"data": {"spatial_root": str(tmp_path / "features")}})
    identity.pop("run_key")
    identity["run_key"] = "fusion-reference-seed0-16000-" + digest(identity)
    run = tmp_path / identity["run_key"]
    run.mkdir()
    receipt = {"schema_version": 1, "kind": "fusion-initialization",
               "run_policy": runner.RUN_POLICY, "run_key": run.name,
               "run_identity_hash": digest(identity), "initial_base_hash": "b" * 64,
               "common_plain_hash": "c" * 64, "initial_variant_hash": "c" * 64,
               "trainable_parameters": {"adapter": 4, "vt": 2, "flan": 3,
                                        "adapter_vt": 6, "total": 9},
               "trainable_inventory": {
                   "visual_adapter.weight": {"shape": [2, 2], "numel": 4},
                   "vt_align.weight": {"shape": [2], "numel": 2},
                   "language_model.lora.weight": {"shape": [3], "numel": 3}}}
    receipt["content_hash"] = digest(receipt)
    atomic_json(run / "initialization.json", receipt)
    initialization_hash = file_hash(run / "initialization.json")
    telemetry = simulated_peaks()
    items = rows()[0]["items"]
    protocol = {"protocol_hash": "protocol", "split_hash": "split", "split": {
        "groups": {"dev": {"clip_ids": [i["clip_id"] for i in items]}}}}
    protocol_path = tmp_path / "protocol.json"
    atomic_json(protocol_path, protocol)
    monkeypatch.setitem(controller.INPUTS, "protocol", protocol_path)
    monkeypatch.setattr(runner, "_check_live", lambda identity: None)
    stats = {"clip_count": 582, "real_frame_count": 582 * 24,
             "supported_frame_count": 582 * 24, "gate_count": 0,
             **{k: None for k in ("gate_sum", "gate_sumsq", "gate_min", "gate_max",
                                  "gate_below_0_05", "gate_above_0_95", "correction_norm")}}
    for step in runner.DEV_STEPS:
        checkpoint = {"run_metadata": identity, "pilot_identity": identity, "global_step": step,
                      "gpu_seconds_cumulative": 50, "optimizer_states": [{}], "lr_schedulers": [{}],
                      "pilot_rng": {"synthetic": True}, "pilot_sampler": {"batches_yielded": step},
                      "initialization_hash": initialization_hash, "fusion_telemetry": telemetry}
        path = run / f"step-{step}.ckpt"
        torch.save(checkpoint, path)
        report = {"checkpoint_step": step, "checkpoint_hash": file_hash(path),
                  "split": "dev", "split_hash": "split", "decoding": runner.DECODER,
                  "items": items, "metrics": rows()[0]["metrics"], "fusion_stats": stats,
                  "initialization_hash": initialization_hash,
                  "checkpoint_size_bytes": path.stat().st_size,
                  "visual_token_lengths": [{"clip_id": i["clip_id"], "valid_tokens": 7,
                                             "padded_tokens": 9} for i in items],
                  "visual_token_length_scope": "eval_diagnostic_forward_actual_output_mask",
                  "cuda_memory": telemetry["dev"],
                  **runner.report_binding(identity)}
        report["content_hash"] = digest(report)
        atomic_json(run / f"dev-{step}.json", report)
    selected = read_json(run / "dev-5500.json")
    atomic_json(run / "selected-checkpoint.json", {
        "step": 5500, "checkpoint_hash": selected["checkpoint_hash"],
        "metrics": selected["metrics"], **runner.report_binding(identity),
        "initialization_hash": initialization_hash, "fusion_telemetry": telemetry,
        "fusion_telemetry_hash": digest(telemetry),
        "checkpoint_sizes_bytes": runner.fusion_checkpoint_sizes(run),
        "dev_report_hashes": {f"dev-{s}.json": file_hash(run / f"dev-{s}.json")
                              for s in runner.DEV_STEPS}})
    atomic_json(run / "budget.json", {"identity": identity, "gpu_seconds": 100,
                                       "ceiling_seconds": 10800, "status": "closed",
                                       "optimizer_step": 16000,
                                       "initialization_hash": initialization_hash,
                                       "fusion_telemetry": telemetry,
                                       "checkpoint_sizes_bytes":
                                       runner.fusion_checkpoint_sizes(run)})
    atomic_json(run / "run-status.json", {"status": "complete", "run_key": run.name,
                                          "global_step": 16000,
                                          "initialization_hash": initialization_hash,
                                          "fusion_telemetry_hash": digest(telemetry)})
    atomic_json(tmp_path / "features/complete/frame_rows.json", {"clips": {
        i["clip_id"]: {"frame_count": 24} for i in items}})
    atomic_json(run / "gradient-smoke.json", {
        label: {"cuda_peak_bytes": 123, "runtime_seconds": 2, "optimizer_updates": 0}
        for label in ("ordinary", "long")})
    return run, identity, profile, timing


def execute_cpu_query(command, monkeypatch):
    import contextlib
    import io
    import json

    monkeypatch.setattr(sys, "argv", ["-c", *command[3:]])
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(command[2], {"__name__": "__main__"})
    return json.loads(out.getvalue().splitlines()[-1])


def test_completion_real_runner_selection_and_all_dev_bytes(controller, completed, monkeypatch):
    run, identity, _, _ = completed
    phase = SimpleNamespace(cpu_query=lambda c: execute_cpu_query(c, monkeypatch))
    assert controller.validate_completion(phase, run, identity)
    report = run / "dev-10400.json"
    report.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="byte hashes"):
        controller.validate_completion(phase, run, identity)


def test_partial_matching_selection_real_terminal_recovery(controller, completed, monkeypatch):
    run, identity, _, _ = completed
    atomic_json(run / "run-status.json", {"status": "partial", "run_key": run.name,
                                          "global_step": 16000})
    phase = SimpleNamespace(cpu_query=lambda c: execute_cpu_query(c, monkeypatch))
    assert controller.validate_completion(phase, run, identity) is False
    assert controller.recovery_checkpoint(phase, run, identity) == run / "step-16000.ckpt"


def test_collect_binds_all_report_checkpoint_bytes_and_memory(controller, completed, monkeypatch):
    run, identity, _, _ = completed
    phase = SimpleNamespace(cpu_query=lambda c: execute_cpu_query(c, monkeypatch),
                            fusion_seed=0, run_spent=lambda arm: 123.0)
    result = controller.collect_result(phase, "reference", run, identity)
    assert len(result["source_files"]) == 11
    assert all(file_hash(Path(path)) == sha for path, sha in result["source_files"].items())
    assert result["token_lengths"] == [7] * 582  # Deliberately differs from structural 3.
    assert result["controller_charged_seconds"] == 123
    assert result["fusion_telemetry"] == simulated_peaks()
    assert result["parameters"] == read_json(run / "initialization.json")["trainable_parameters"]
    assert result["initialization_hash"] == file_hash(run / "initialization.json")
    assert result["token_lengths_evidence"] == "eval_diagnostic_forward_actual_output_mask"
    assert result["checkpoint_sizes_bytes"] == controller.runner.fusion_checkpoint_sizes(run)


def test_both_seed_recoveries_precede_any_schedule(controller, tmp_path, monkeypatch):
    events = []
    original_load = controller.FusionPhase._load

    def load(self, budget):
        events.append(("recovered", self.fusion_seed))
        original_load(self, budget)

    def run(phase, *args):
        events.append(("run", phase.fusion_seed))
        return []

    monkeypatch.setattr(controller.FusionPhase, "_load", load)
    monkeypatch.setattr(controller, "run_phase", run)

    def publish(results, accounting):
        assert results == [] and [a["seed"] for a in accounting] == [0, 1]
        for seed in (0, 1):
            with pytest.raises(BlockingIOError):
                controller.FusionPhase(tmp_path / f"seed{seed}", seed)
        events.append(("publish", None))

    controller.execute_phases(tmp_path, tmp_path / "profile", tmp_path / "timing",
                              {}, tmp_path / "state.json", publish)
    assert events == [("recovered", 0), ("recovered", 1), ("run", 0), ("run", 1),
                      ("publish", None)]


def test_adopted_runner_spend_cannot_reset_controller_cap(controller, completed, tmp_path):
    run, identity, _, _ = completed
    with controller.FusionPhase(tmp_path / "phase", 0) as phase:
        phase.bind_run("reference", run, identity)
        assert phase.run_spent("reference") == 100
        phase.bind_run("reference", run, identity)
        assert phase.state["spent_seconds"] == 100  # Not double counted.
    with controller.FusionPhase(tmp_path / "phase", 0) as phase:
        assert phase.run_spent("reference") == 100


@pytest.mark.parametrize("mode", ["initialize", "preflight", "smoke", "run"])
def test_command_consumed_by_actual_runner_parser_without_initialization(
        controller, tmp_path, monkeypatch, mode):
    seen = []

    def initialize(args):
        seen.append((args, "initialize"))
        return {"synthetic": True}

    def prepare(args):
        seen.append((args, "preflight"))
        return {"run_key": "synthetic"}, tmp_path / "synthetic", {}, {}, None

    def execute(args, *, smoke_only=False):
        seen.append((args, "smoke" if smoke_only else "run"))
        return {"synthetic": True}

    monkeypatch.setattr(controller.runner, "initialize_fusion", initialize)
    monkeypatch.setattr(controller.runner, "prepare_fusion", prepare)
    monkeypatch.setattr(controller.runner, "_execute", execute)
    command = controller.stage_command("reference", 1, tmp_path / "profile",
                                       tmp_path / "timing", mode)
    controller.runner.main(command[2:])
    args, actual_mode = seen[0]
    assert actual_mode == mode and args.seed == 1 and args.arm == "reference"
    assert args.gpu_cap_seconds == 10800 and args.steps == 16000
    assert args.motion_timing == tmp_path / "timing"
    assert args.output_base == tmp_path / "runs"
    assert args.resume_checkpoint is None


@pytest.mark.parametrize("drift", [None, "early-checkpoint", "early-receipt", "last-report",
                                  "resume", "missing-source", "wrong-hash", "foreign-source",
                                  "timing", "profile", "protocol"])
@pytest.mark.parametrize("row_index", range(6))
def test_summary_publication_byte_bindings_and_phase_accounting(
        controller, tmp_path, monkeypatch, drift, row_index):
    root, timing = tmp_path / "study", tmp_path / "timing.json"
    atomic_json(timing, {"synthetic": True})
    protocol = tmp_path / "protocol.json"
    atomic_json(protocol, {"synthetic": True})
    monkeypatch.setitem(controller.INPUTS, "protocol", protocol)

    def phases(root, profile, timing, metadata, state_path, publish=None):
        atomic_json(profile, {"synthetic": True})
        data = rows()
        names = ["budget.json", "run-status.json", "selected-checkpoint.json",
                 "initialization.json", "gradient-smoke.json",
                 *[f"dev-{s}.json" for s in controller.runner.DEV_STEPS],
                 *[f"step-{s}.ckpt" for s in controller.runner.DEV_STEPS], "resume-16000.ckpt"]
        for row in data:
            run = root / "runs" / f"{row['seed']}-{row['arm']}"
            run.mkdir()
            for name in names:
                (run / name).write_bytes(b"synthetic")
            row.update(run=str(run), run_key=run.name, identity={
                "seed": row["seed"], "arm": row["arm"], "run_key": run.name,
                "timing_hash": file_hash(timing),
                "initialization_profile_file_hash": file_hash(profile),
                "initialization_profile_path": str(profile.resolve()),
                "bound_files": {str(protocol): file_hash(protocol)}},
                source_files={str(run / n): file_hash(run / n) for n in names},
                checkpoint_sizes_bytes={**{f"step-{s}.ckpt": 9
                                          for s in controller.runner.DEV_STEPS},
                                        "resume-16000.ckpt": 9})
            for report in row["dev_trajectory"]:
                report["checkpoint_size_bytes"] = 9
        if drift in ("early-checkpoint", "early-receipt", "last-report", "resume"):
            name = {"early-checkpoint": "step-5500.ckpt",
                    "early-receipt": "gradient-smoke.json", "last-report": "dev-16000.json",
                    "resume": "resume-16000.ckpt"}[drift]
            (Path(data[row_index]["run"]) / name).write_bytes(b"changed!!")
        elif drift == "missing-source":
            data[row_index]["source_files"].pop(str(Path(data[row_index]["run"]) / "budget.json"))
        elif drift == "wrong-hash":
            data[row_index]["source_files"][
                str(Path(data[row_index]["run"]) / "step-5500.ckpt")] = "0" * 64
        elif drift == "foreign-source":
            data[row_index]["source_files"][str(protocol)] = file_hash(protocol)
        elif drift in ("timing", "profile", "protocol"):
            {"timing": timing, "profile": profile, "protocol": protocol}[drift].write_bytes(
                b"changed")
        accounting = [{"seed": seed, "spent_seconds": spent,
                       "stages": {"train": {"elapsed_seconds": spent}}}
                      for seed, spent in ((0, 701), (1, 702))]
        if publish is not None:
            publish(data, accounting)
        return data, accounting

    monkeypatch.setattr(controller, "execute_phases", phases)
    command = ["--study-root", str(root), "--motion-timing", str(timing),
               "--authorize-fusion-study"]
    if drift is not None:
        with pytest.raises(ValueError):
            controller.main(command)
        assert not (root / "summary.json").exists()
        assert read_json(root / "state.json")["status"] == "incomplete"
        return
    controller.main(command)
    summary, state = read_json(root / "summary.json"), read_json(root / "state.json")
    assert summary["content_hash"] == digest({k: v for k, v in summary.items()
                                               if k != "content_hash"})
    assert summary["spent_charged_seconds"] == 1403
    assert summary["profile_sha256"] == file_hash(root / "common-initialization.json")
    assert summary["timing_sha256"] == file_hash(timing)
    assert state["summary_sha256"] == file_hash(root / "summary.json")


def test_controller_helper_source_drift_rejected_without_state_write(
        controller, tmp_path, monkeypatch):
    with controller.FusionPhase(tmp_path, 0):
        pass
    before = (tmp_path / "study-state.json").read_bytes()
    monkeypatch.setattr(controller, "controller_sources", lambda: {"watchdog": "changed"})
    with pytest.raises(ValueError, match="source"):
        controller.FusionPhase(tmp_path, 0)
    assert (tmp_path / "study-state.json").read_bytes() == before


def test_owned_orphan_stopped_before_recovery_finishes(controller, tmp_path, monkeypatch):
    expected = {"start_ticks": "123", "command": ["synthetic"], "pgid": 12345,
                "session": 12345}
    with controller.FusionPhase(tmp_path, 0) as phase:
        phase.state["stages"]["reference-train"] = {
            "status": "running", "elapsed_seconds": 0.0, "owned_arm": "reference",
            "command_hash": neutral.command_signature(["synthetic"]), "attempts": 1,
            "last_wall": 10.0, "last_monotonic": 10.0, "boot_id": phase.boot_id,
            "reserved_seconds": 30.0, "pid": 12345, "command": ["synthetic"],
            "process_identity": expected, "watchdog_command": ["synthetic"]}
        phase.save()
    events = []
    monkeypatch.setattr(neutral, "process_identity", lambda pid: expected)
    monkeypatch.setattr(neutral, "stop_group", lambda pid: events.append(pid))
    monkeypatch.setattr(neutral.time, "monotonic", lambda: 50.0)
    with controller.FusionPhase(tmp_path, 0) as phase:
        assert events == [12345]
        assert phase.run_spent("reference") == 40  # Not clamped to 30 reserved.


def test_seed_budgets_independent_no_unused_transfer(controller, tmp_path):
    with controller.FusionPhase(tmp_path / "seed0", 0) as first:
        first.state["stages"]["reference-train"] = {
            "status": "complete", "owned_arm": "reference", "elapsed_seconds": 100}
        first.state["spent_seconds"] = 100
        first.save()
    with controller.FusionPhase(tmp_path / "seed1", 1) as second:
        assert second.state["budget_seconds"] == 32400
        assert second.state["spent_seconds"] == 0
        with pytest.raises(ValueError, match="cap"):
            second.run_stage("oversized", [sys.executable, "-c", "pass"], 10801)


def test_adopted_gpu_floor_does_not_erase_new_cpu_or_init_spend(controller, completed, tmp_path):
    run, identity, _, _ = completed
    with controller.FusionPhase(tmp_path / "phase", 0) as phase:
        phase.state["stages"] = {
            "preflight-1": {"owned_arm": "reference", "cpu_only": True,
                            "elapsed_seconds": 10, "status": "complete"},
            "common-initialization": {"owned_arm": "reference", "cpu_only": False,
                                      "elapsed_seconds": 2, "status": "complete"}}
        phase.state["spent_seconds"] = 12
        phase.bind_run("reference", run, identity)
        assert phase.run_spent("reference") == 112
        phase.bind_run("reference", run, identity)
        assert phase.state["spent_seconds"] == 112


def test_selection_query_returns_validated_actual_telemetry(controller, completed, monkeypatch):
    run, identity, _, _ = completed
    result = execute_cpu_query(controller.fusion_query_command(
        SimpleNamespace(root=run.parent), "selection", run, identity), monkeypatch)
    assert result["initialization"]["trainable_parameters"]["total"] == 9
    assert result["initialization_hash"] == file_hash(run / "initialization.json")
    assert result["fusion_telemetry"] == simulated_peaks()
    assert result["checkpoint_sizes_bytes"] == controller.runner.fusion_checkpoint_sizes(run)


@pytest.mark.parametrize("mutation", ["counts", "diagnostic_scope", "peak_scope", "cpu",
                                     "step_lengths", "selected_lengths", "sizes", "unobserved_dev"])
def test_summary_rejects_measured_field_mismatch_and_scopes(controller, mutation):
    data = rows()
    row = data[-1]
    if mutation == "counts":
        row["parameters"] = {**row["parameters"], "total": 10}
    elif mutation == "diagnostic_scope":
        row["dev_trajectory"][0]["visual_token_length_scope"] = "generation_forward_capture"
    elif mutation == "peak_scope":
        row["fusion_telemetry"]["training"]["evidence_scope"] = "exclusive_phase_peak"
    elif mutation == "cpu":
        row["fusion_telemetry"]["training"].update(
            device_type="cpu", peak_allocated_bytes=0, peak_reserved_bytes=0,
            evidence_scope="cpu_no_cuda_measurement")
    elif mutation == "step_lengths":
        report = row["dev_trajectory"][1]
        report["visual_token_lengths"] = copy.deepcopy(report["visual_token_lengths"])
        report["visual_token_lengths"][0]["valid_tokens"] = 41
    elif mutation == "selected_lengths":
        row["visual_token_lengths"] = copy.deepcopy(row["visual_token_lengths"])
        row["visual_token_lengths"][0]["padded_tokens"] += 1
    elif mutation == "sizes":
        row["dev_trajectory"][0]["checkpoint_size_bytes"] += 1
    else:
        row["dev_trajectory"][0]["cuda_memory"] = {
            "device_type": None, "peak_allocated_bytes": None, "peak_reserved_bytes": None,
            "sample_count": 0, "evidence_scope": "not_observed"}
    with pytest.raises(ValueError):
        controller.aggregate_fusion_results(data)


@pytest.mark.parametrize("mutation", ["missing_receipt", "receipt_bytes", "dev_lengths",
                                     "report_size", "inventory_size", "peak_maximum"])
def test_completion_rejects_tampered_measured_provenance(
        controller, completed, monkeypatch, mutation):
    run, identity, _, _ = completed
    if mutation == "missing_receipt":
        (run / "initialization.json").rename(run / "hidden-initialization.json")
    elif mutation == "receipt_bytes":
        path = run / "initialization.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation in ("dev_lengths", "report_size"):
        path = run / "dev-5500.json"
        report = read_json(path)
        if mutation == "dev_lengths":
            report["visual_token_lengths"].reverse()
        else:
            report["checkpoint_size_bytes"] += 1
        report["content_hash"] = digest({k: v for k, v in report.items() if k != "content_hash"})
        atomic_json(path, report)
        selected = read_json(run / "selected-checkpoint.json")
        selected["dev_report_hashes"][path.name] = file_hash(path)
        atomic_json(run / "selected-checkpoint.json", selected)
    else:
        for name in ("budget.json", "selected-checkpoint.json"):
            value = read_json(run / name)
            if mutation == "inventory_size":
                value["checkpoint_sizes_bytes"]["step-5500.ckpt"] += 1
            else:
                value["fusion_telemetry"]["dev"]["peak_allocated_bytes"] -= 1
                value["fusion_telemetry_hash"] = digest(value["fusion_telemetry"])
            atomic_json(run / name, value)
        if mutation == "peak_maximum":
            status = read_json(run / "run-status.json")
            status["fusion_telemetry_hash"] = read_json(
                run / "selected-checkpoint.json")["fusion_telemetry_hash"]
            atomic_json(run / "run-status.json", status)
    phase = SimpleNamespace(cpu_query=lambda c: execute_cpu_query(c, monkeypatch))
    with pytest.raises(ValueError):
        controller.validate_completion(phase, run, identity)


def test_saved_size_inventory_replaces_estimate_and_rejects_drift(controller, completed):
    run, identity, _, _ = completed
    inventory = controller.runner.fusion_checkpoint_sizes(run)
    assert controller.checkpoint_size({}) == 5.6 * 1024**3
    record = {"run": str(run), "identity": identity, "checkpoint_sizes_bytes": inventory}
    assert controller.checkpoint_size({"0-reference": record}) == max(inventory.values())
    inventory["step-5500.ckpt"] += 1
    with pytest.raises(ValueError, match="size"):
        controller.checkpoint_size({"0-reference": record})


def test_summary_memory_maxima_not_sums_and_parameter_deltas_are_actual(controller):
    data = rows()
    for row in data:
        if row["arm"] == "aligned_fixed":
            row["parameters"]["adapter"] += 17
            row["parameters"]["adapter_vt"] += 17
            row["parameters"]["total"] += 17
    summary = controller.aggregate_fusion_results(data)
    assert summary["production_cuda_high_water_maxima"] == simulated_peaks()
    assert summary["actual_total_parameter_deltas_vs_reference"]["aligned_fixed"] == [17, 17]
    assert summary["visual_token_length_scope"] == "eval_diagnostic_forward_actual_output_mask"


def test_actual_cpu_only_completion_cannot_claim_production_cuda(
        controller, completed, monkeypatch):
    import torch

    run, identity, _, _ = completed
    telemetry = {"schema_version": 1, **{phase: {
        "device_type": "cpu", "peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
        "sample_count": 1, "evidence_scope": "cpu_no_cuda_measurement"}
        for phase in ("training", "dev")}}
    for step in controller.runner.DEV_STEPS:
        path = run / f"step-{step}.ckpt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        saved["fusion_telemetry"] = telemetry
        torch.save(saved, path)
        report_path = run / f"dev-{step}.json"
        report = read_json(report_path)
        report.update(cuda_memory=telemetry["dev"], checkpoint_hash=file_hash(path),
                      checkpoint_size_bytes=path.stat().st_size)
        report["content_hash"] = digest({k: v for k, v in report.items() if k != "content_hash"})
        atomic_json(report_path, report)
    for name in ("budget.json", "selected-checkpoint.json"):
        value = read_json(run / name)
        value.update(fusion_telemetry=telemetry, fusion_telemetry_hash=digest(telemetry),
                     checkpoint_sizes_bytes=controller.runner.fusion_checkpoint_sizes(run))
        if name.startswith("selected"):
            value["checkpoint_hash"] = file_hash(run / "step-5500.ckpt")
            value["dev_report_hashes"] = {f"dev-{step}.json": file_hash(run / f"dev-{step}.json")
                                         for step in controller.runner.DEV_STEPS}
        atomic_json(run / name, value)
    status = read_json(run / "run-status.json")
    status["fusion_telemetry_hash"] = digest(telemetry)
    atomic_json(run / "run-status.json", status)
    protocol = read_json(controller.INPUTS["protocol"])
    ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
    # Producer permits correctly scoped CPU software fixtures; controller's real
    # study boundary must not relabel these as production CUDA evidence.
    assert controller.runner.validate_fusion_selection(run, protocol, ids, identity)
    phase = SimpleNamespace(cpu_query=lambda c: execute_cpu_query(c, monkeypatch))
    with pytest.raises(ValueError, match="CUDA"):
        controller.validate_completion(phase, run, identity)


@pytest.mark.parametrize("mutation", ["initialization_hash", "fusion_telemetry"])
def test_partial_checkpoint_telemetry_rejected_before_gpu_child(
        controller, completed, monkeypatch, mutation):
    import torch

    run, identity, _, _ = completed
    atomic_json(run / "run-status.json", {"status": "partial", "run_key": run.name,
                                          "global_step": 16000})
    path = run / "resume-16000.ckpt"
    checkpoint = torch.load(run / "step-16000.ckpt", map_location="cpu", weights_only=False)
    if mutation == "initialization_hash":
        checkpoint[mutation] = "0" * 64
    else:
        checkpoint[mutation]["training"]["evidence_scope"] = "exclusive_phase_peak"
    torch.save(checkpoint, path)
    # Step is preferred; make corrupted resume the highest durable candidate.
    (run / "step-16000.ckpt").rename(run / "hidden-step-16000.bin")
    phase = SimpleNamespace(cpu_query=lambda c: execute_cpu_query(c, monkeypatch))
    with pytest.raises(ValueError):
        controller.recovery_checkpoint(phase, run, identity)


@pytest.mark.parametrize("disk", ["linux", "windows"])
def test_measured_sparse_size_demand_keeps_allocated_credit_separate(
        controller, tmp_path, monkeypatch, disk):
    run = tmp_path / "owned"
    run.mkdir()
    with (run / "resume-500.ckpt").open("wb") as file:
        file.truncate(8 * 1024**3)
    runs = {"0-reference": {"run": str(run), "complete": False,
                             "checkpoint_sizes_bytes":
                             controller.runner.fusion_checkpoint_sizes(run)}}
    size = controller.checkpoint_size(runs)
    assert size == 8 * 1024**3
    projection = controller.storage_projection(runs, size)
    assert projection["allocated_owned_bytes"] == 0  # Sparse logical bytes are not allocated.
    assert projection["required_free_bytes"] == 298 * 1024**3
    monkeypatch.setattr(controller.shutil, "disk_usage", lambda p:
                        SimpleNamespace(free=(297 if disk == "linux" else 300) * 1024**3))
    monkeypatch.setattr(controller, "windows_free_bytes", lambda:
                        (297 if disk == "windows" else 300) * 1024**3)
    with pytest.raises(ValueError, match=disk):
        controller.require_study_storage(tmp_path, runs, size)


def test_collection_rejects_receipt_change_after_validated_selection(
        controller, completed, monkeypatch):
    run, identity, _, _ = completed

    def query(command):
        result = execute_cpu_query(command, monkeypatch)
        path = run / "initialization.json"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    phase = SimpleNamespace(cpu_query=query, fusion_seed=0, run_spent=lambda arm: 123)
    with pytest.raises(ValueError, match="artifact bytes changed"):
        controller.collect_result(phase, "reference", run, identity)


@pytest.fixture
def partial_without_selection(controller, completed, tmp_path):
    import torch

    run, identity, profile, timing = completed
    (tmp_path / "runs").mkdir()
    run = run.rename(tmp_path / "runs" / run.name)
    (run / "selected-checkpoint.json").rename(run / "hidden-selected.json")
    for step in (10400, 16000):
        for name in (f"dev-{step}.json", f"step-{step}.ckpt"):
            (run / name).rename(run / f"not-produced-{name}.bin")
    saved = torch.load(run / "step-5500.ckpt", map_location="cpu", weights_only=False)
    saved["global_step"] = 6000
    saved["pilot_sampler"]["batches_yielded"] = 6000
    torch.save(saved, run / "resume-6000.ckpt")
    budget = read_json(run / "budget.json")
    budget.update(optimizer_step=6000,
                  checkpoint_sizes_bytes=controller.runner.fusion_checkpoint_sizes(run))
    atomic_json(run / "budget.json", budget)
    atomic_json(run / "run-status.json", {"status": "partial", "global_step": 6000,
                                          "run_key": run.name})
    return run, identity, profile, timing


@pytest.mark.parametrize("mutation", ["identity", "content_hash", "order", "initialization",
                                     "masks", "size", "peak", "checkpoint_bytes",
                                     "checkpoint_identity", "orphan_report", "unknown_report"])
def test_selection_absent_corrupt_prior_dev_rejects_before_gpu_dispatch(
        controller, partial_without_selection, tmp_path, monkeypatch, mutation):
    import torch

    run, identity, profile, timing = partial_without_selection
    path = run / "dev-5500.json"
    report = read_json(path)
    if mutation == "identity":
        report["seed"] = 1
    elif mutation == "content_hash":
        report["content_hash"] = "0" * 64
    elif mutation == "order":
        report["items"].reverse()
    elif mutation == "initialization":
        report["initialization_hash"] = "0" * 64
    elif mutation == "masks":
        report["visual_token_lengths"].reverse()
    elif mutation == "size":
        report["checkpoint_size_bytes"] += 1
    elif mutation == "peak":
        report["cuda_memory"]["peak_allocated_bytes"] += 100
        report["cuda_memory"]["peak_reserved_bytes"] += 100
    elif mutation == "checkpoint_bytes":
        checkpoint = run / "step-5500.ckpt"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    elif mutation == "checkpoint_identity":
        checkpoint = run / "step-5500.ckpt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved["run_metadata"] = {**identity, "seed": 1}
        torch.save(saved, checkpoint)
        report.update(checkpoint_hash=file_hash(checkpoint),
                      checkpoint_size_bytes=checkpoint.stat().st_size)
    elif mutation == "orphan_report":
        (run / "step-5500.ckpt").rename(run / "hidden-step-5500.bin")
    else:
        path = run / "dev-12345.json"
    if mutation != "content_hash":
        report["content_hash"] = digest({k: v for k, v in report.items() if k != "content_hash"})
    atomic_json(path, report)
    monkeypatch.setattr(controller, "validate_live", lambda *args: None)
    monkeypatch.setattr(controller, "require_study_storage", lambda *args: {"synthetic": True})
    dispatched = []

    def stage(name, command, cap, completion_validator=None):
        if name != "common-initialization":
            dispatched.append(name)
            raise RuntimeError("GPU dispatch reached")

    def query(command):
        if command[1] == "-c":
            return execute_cpu_query(command, monkeypatch)
        return {"status": "ready", "identity": identity, "output": str(run)}

    with controller.FusionPhase(tmp_path / "seed0", 0) as phase:
        monkeypatch.setattr(phase, "cpu_query", query)
        monkeypatch.setattr(phase, "run_stage", stage)
        with pytest.raises(ValueError):
            controller.run_phase(phase, profile, timing, {})
    assert dispatched == []


def test_selection_absent_valid_partial_allows_missing_future_reports(
        controller, partial_without_selection, monkeypatch):
    run, identity, _, _ = partial_without_selection
    phase = SimpleNamespace(cpu_query=lambda c: execute_cpu_query(c, monkeypatch))
    assert not (run / "selected-checkpoint.json").exists()
    assert not (run / "dev-10400.json").exists() and not (run / "dev-16000.json").exists()
    assert controller.recovery_checkpoint(phase, run, identity) == run / "resume-6000.ckpt"
