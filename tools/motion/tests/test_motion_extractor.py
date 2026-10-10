"""Dedicated-environment tests: factual PNG consumption and durable execution."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from despamo.appearance.provenance import digest, file_hash
from tools.motion.records import MODEL, PROCESSOR, make_motion_identity


def measured_profile(identity):
    return {"schema_version": 3, "admission_policy": "measured-smoke-hard-cap-v1",
            "featurekey": digest(identity),
            "parity": [{"batch_size": n, "bit_equal": True} for n in (1, 8, 16)],
            "samples": [{"clip_id": "clip", "windows": 1}],
            "observation_sha256": {"clip": "a" * 64}}


@pytest.fixture
def api():
    try:
        return importlib.import_module("tools.motion.extractor")
    except ModuleNotFoundError:
        pass
    class API:
        def __getattr__(self, name):
            try:
                return getattr(importlib.import_module("tools.motion.extractor"), name)
            except ModuleNotFoundError:
                pytest.fail("Task2 extractor implementation missing")
    return API()


def pngs(root: Path, count: int) -> tuple[Path, ...]:
    root.mkdir(parents=True, exist_ok=True)
    paths = tuple(root / f"{i:04}.png" for i in range(count))
    for i, path in enumerate(paths):
        Image.new("RGB", (210, 260), (i % 256, i % 256, i % 256)).save(path)
    return paths


class RecordingEncoder(torch.nn.Module):
    def __init__(self, bad=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []
        self.bad = bad

    def forward(self, pixel_values, *, output_hidden_states, return_dict):
        assert output_hidden_states and return_dict
        assert not self.training and not torch.is_grad_enabled()
        assert not torch.is_autocast_enabled()
        assert pixel_values.dtype == torch.float32
        self.calls.append(pixel_values[:, :, 0, 0, 0].clone())
        b = len(pixel_values)
        hidden = torch.zeros(b, 2, 1024)
        hidden[:, 0, :] = pixel_values[:, 0, 0, 0, 0, None]
        hidden[:, 1, :] = 100
        if self.bad == "nan":
            hidden[0, 0, 0] = float("nan")
        if self.bad == "width":
            hidden = hidden[:, :, :32]
        if self.bad == "dtype":
            hidden = hidden.double()
        if self.bad == "rows":
            hidden = hidden[:0]
        return SimpleNamespace(hidden_states=(hidden,), last_hidden_state=hidden + 500)


def fixture_identity(api, tmp_path, paths):
    evidence = {}
    for name in ("frames", "annotation", "lock", "extractor", "helpers"):
        p = tmp_path / f"{name}.json"
        p.write_text("{}")
        evidence[name] = {"path": str(p), "sha256": file_hash(p)}
    row = {"frame_count": len(paths), "paths": [str(p) for p in paths], "feature_hash": "f" * 64,
           "source_indices": list(range(len(paths))),
           "source_hash": digest([(str(p), file_hash(p)) for p in paths])}
    frames = {"encoder_key": "synthetic-spatial", "clips": {"clip": row}}
    Path(evidence["frames"]["path"]).write_text(json.dumps(frames))
    evidence["frames"]["sha256"] = file_hash(Path(evidence["frames"]["path"]))
    identity = make_motion_identity(
        {"frame_root": str(paths[0].parent), "frame_rows": evidence["frames"],
         "annotation": evidence["annotation"]}, MODEL, PROCESSOR,
        {"lock": evidence["lock"]},
        {"extractor": evidence["extractor"], "helpers": evidence["helpers"]}, {})
    return identity, frames, tmp_path / f"motion-features-{digest(identity)}"


def test_actual_chunk_order_tail_and_first_hidden_token(api, tmp_path):
    paths = pngs(tmp_path / "frames", 153)  # 18 windows: chunks 16 + 2, frame152 tail
    model = RecordingEncoder()
    features, sources, windows = api.extract_motion_clip(paths, model, api.make_processor())
    assert [len(c) for c in model.calls] == [16, 2]
    observed = torch.cat(model.calls).numpy()
    wanted = np.array([[s + i for i in range(16)] for s in range(0, 137, 8)])
    np.testing.assert_allclose(observed, wanted / 255 * 2 - 1, atol=1e-7)
    np.testing.assert_array_equal(features[:, 0], observed[:, 0])
    assert features.shape == (18, 1024) and features.dtype == np.float32
    assert [w["input_source_indices"] for w in windows] == wanted.tolist()
    assert [w["row"] for w in windows] == list(range(18))
    assert len(sources) == 153 and sources[-1]["index"] == 152
    assert sources[-1]["sha256"] == hashlib.sha256(paths[-1].read_bytes()).hexdigest()


def test_short_repeat_only_inputs_and_single_byte_buffer(api, tmp_path, monkeypatch):
    paths = pngs(tmp_path / "frames", 10)
    original = Path.read_bytes
    reads = []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    model = RecordingEncoder()
    features, sources, windows = api.extract_motion_clip(paths, model, api.make_processor())
    assert reads == list(paths)
    assert len(sources) == 10 and features.shape == (1, 1024)
    assert windows == ({"row": 0, "input_source_indices": list(range(10)) + [9] * 6,
                        "real_start": 0, "real_end": 9, "centre": 7.5},)


@pytest.mark.parametrize("bad", ["nan", "width", "dtype", "rows"])
def test_invalid_observed_outputs_reject(api, tmp_path, bad):
    with pytest.raises(ValueError):
        api.extract_motion_clip(pngs(tmp_path / "frames", 1), RecordingEncoder(bad),
                                api.make_processor())


@pytest.mark.parametrize("batch", [0, 17, True])
def test_batch_limit(api, tmp_path, batch):
    with pytest.raises(ValueError):
        api.extract_motion_clip(pngs(tmp_path / "frames", 1), RecordingEncoder(),
                                api.make_processor(), batch_size=batch)


@pytest.mark.parametrize("batch", [1, 8, 16])
def test_real_processor_numpy_bit_parity(api, batch):
    videos = [[Image.new("RGB", shape, (13 + i, 41, 92)) for i in range(16)]
              for shape in [(210, 260), (9, 17), (260, 210), (224, 224)] * 4][:batch]
    processor = api.make_processor()
    tensor = api.processor_tensor(processor, videos)
    standard = processor(videos, return_tensors="pt")["pixel_values"]
    assert tensor.dtype == torch.float32 and torch.equal(tensor, standard)


def test_changed_same_filename_never_commits(api, tmp_path):
    paths = pngs(tmp_path / "frames", 10)
    identity, frames, root = fixture_identity(api, tmp_path, paths)
    Image.new("RGB", (210, 260), (233, 0, 0)).save(paths[0])
    with pytest.raises(ValueError, match="source"):
        api.generate_clip(root, "clip", frames["clips"]["clip"], identity,
                          RecordingEncoder(), api.make_processor())
    assert not (root / "receipts/train/clip.json").exists()
    assert not (root / "train/clip.npy").exists()


def test_owned_array_interruption_recompute_complete_reuse(api, tmp_path, monkeypatch):
    paths = pngs(tmp_path / "frames", 10)
    identity, frames, root = fixture_identity(api, tmp_path, paths)
    row = frames["clips"]["clip"]
    model = RecordingEncoder()
    original = api.atomic_json

    def crash(path, value):
        if path.name == "clip.json":
            raise OSError("after array before receipt")
        return original(path, value)

    monkeypatch.setattr(api, "atomic_json", crash)
    with pytest.raises(OSError):
        api.generate_clip(root, "clip", row, identity, model, api.make_processor())
    assert (root / "train/clip.npy").exists()
    monkeypatch.setattr(api, "atomic_json", original)
    api.generate_clip(root, "clip", row, identity, model, api.make_processor())
    assert len(model.calls) == 2
    api.generate_clip(root, "clip", row, identity, model, api.make_processor())
    assert len(model.calls) == 2
    (root / "train/clip.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        api.generate_clip(root, "clip", row, identity, model, api.make_processor())
    assert len(model.calls) == 2


def test_alias_and_foreign_debris_reject(api, tmp_path):
    paths = pngs(tmp_path / "frames", 1)
    identity, frames, root = fixture_identity(api, tmp_path, paths)
    root.mkdir()
    (root / "unexpected.tmp").write_text("retain")
    with pytest.raises(ValueError):
        api.check_owned_inventory(root, frames)
    assert (root / "unexpected.tmp").read_text() == "retain"
    link = tmp_path / "link"
    link.symlink_to(root)
    with pytest.raises(ValueError):
        with api.exclusive_writer(link):
            pass


def test_cap_resume_exception_uncapped_and_reboot_no_reset(api, tmp_path):
    now = {"mono": 100.0, "wall": 1000.0, "boot": "a"}
    path = tmp_path / "budget.json"
    clock = api.ExtractionClock(path, "key", resume=False,
                                clocks=lambda: (now["mono"], now["wall"], now["boot"]))
    clock.start("profile")
    now.update(mono=107.0, wall=1007.0)
    clock.finish("failed")
    assert json.loads(path.read_text())["spent_seconds"] == 7
    clock = api.ExtractionClock(path, "key", resume=True,
                                clocks=lambda: (now["mono"], now["wall"], now["boot"]))
    clock.start("run")
    now.update(mono=180.0, wall=1100.0, boot="b")
    # Unclosed running interval reconciles conservatively across reboot.
    recovered = api.ExtractionClock(path, "key", resume=True,
                                    clocks=lambda: (now["mono"], now["wall"], now["boot"]))
    assert recovered.spent == 100
    recovered.start("run")
    now.update(mono=18190.0, wall=19110.0)
    with pytest.raises(TimeoutError):
        recovered.check()
    recovered.finish("capped")
    assert recovered.spent == 18110  # actual uncapped elapsed, never min(cap, elapsed)
    raw = path.read_bytes()
    with pytest.raises(ValueError):
        api.ExtractionClock(path, "foreign", resume=True)
    assert path.read_bytes() == raw


def test_live_owner_lock_and_budget_unchanged(api, tmp_path):
    root = tmp_path / "owned"
    with api.exclusive_writer(root):
        path = root / "budget.json"
        clock = api.ExtractionClock(path, "key", resume=False)
        clock.start("profile")
        raw = path.read_bytes()
        with pytest.raises(RuntimeError):
            with api.exclusive_writer(root):
                pass
        with pytest.raises(RuntimeError):
            api.ExtractionClock(path, "key", resume=True)
        assert path.read_bytes() == raw
        clock.finish("profiled")


def test_complete_clock_read_only_adoption(api, tmp_path):
    path = tmp_path / "budget.json"
    clock = api.ExtractionClock(path, "key", resume=False)
    clock.start("run")
    clock.finish("complete")
    raw, stat = path.read_bytes(), path.stat()
    other = api.ExtractionClock(path, "key", resume=True)
    assert other.state["status"] == "complete"
    assert path.read_bytes() == raw and path.stat().st_mtime_ns == stat.st_mtime_ns


def test_parent_watchdog_stops_small_real_child(api, tmp_path):
    path = tmp_path / "budget.json"
    clock = api.ExtractionClock(path, "key", resume=False)
    clock.start("run")
    started = time.monotonic()
    result = api.supervise_command(
        [sys.executable, "-c", "import time; time.sleep(20)"], clock,
        deadline_seconds=0.25)
    assert result != 0 and time.monotonic() - started < 5
    assert clock.spent >= 0.25
    assert clock.state["child"] is None


def test_production_counts_not_bypassable(api):
    with pytest.raises(ValueError, match="production"):
        api.production_gate({"clips": {"small": {"frame_count": 10}}}, {})


def test_profile_quantiles_deterministic_including_long(api):
    frames = {"clips": {f"c{i:02}": {"frame_count": i + 1} for i in range(20)}}
    ids = api.profile_ids(frames)
    assert ids == api.profile_ids(frames)
    assert "c00" in ids and "c19" in ids


def test_partial_publication_resume_and_generation_immutable(api, tmp_path, monkeypatch):
    paths = pngs(tmp_path / "frames", 10)
    identity, frames, root = fixture_identity(api, tmp_path, paths)
    api.generate_clip(root, "clip", frames["clips"]["clip"], identity,
                      RecordingEncoder(), api.make_processor())
    api.finish_package(root, identity, frames)
    ledger = root / "generating-run.json"
    raw, ino = ledger.read_bytes(), ledger.stat().st_ino
    descriptor = api.finish_package(root, identity, frames)
    assert descriptor["identity_hash"] == digest(identity)
    assert ledger.read_bytes() == raw and ledger.stat().st_ino == ino
    assert descriptor["clip_ids"] == ["clip"]


def test_cross_platform_lock_checks_only_active_dependencies(api):
    locked = tomllib.loads((Path(__file__).parents[1] / "uv.lock").read_text())
    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in
                 importlib.metadata.distributions()}
    api.verify_distribution_lock(locked, installed)
    missing = dict(installed)
    missing.pop("tokenizers")
    with pytest.raises(ValueError):
        api.verify_distribution_lock(locked, missing)
    wrong = dict(installed, numpy="0.0.0")
    with pytest.raises(ValueError):
        api.verify_distribution_lock(locked, wrong)


def extraction_args(mode="profile", resume=False):
    return SimpleNamespace(mode=mode, resume=resume, gpu_cap_seconds=18000,
                           authorize_motion_extraction=True, model_snapshot=Path("/unused"))


def test_preflight_cpu_no_root_or_model_load(api, tmp_path, monkeypatch):
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 10))
    monkeypatch.setattr(api, "prepare", lambda args: (identity, frames, root))
    result = api.run_motion_extraction(extraction_args("preflight"))
    assert result["status"] == "ready" and result["package"] is None
    assert result["spent_seconds"] == 0 and not root.exists()


def test_profile_run_resume_reuses_clip_and_charges_once(api, tmp_path, monkeypatch):
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 10))
    monkeypatch.setattr(api, "prepare", lambda args: (identity, frames, root))
    install_fake_preparation(monkeypatch, api, tmp_path, identity, frames, root)
    model, processor = RecordingEncoder(), api.make_processor()

    def stage(command, clock, **kwargs):
        request = api.read_json(root / "stage-request.json")
        api.generate_clip(root, "clip", frames["clips"]["clip"], identity, model, processor)
        package = api.finish_package(root, identity, frames) if request["mode"] == "run" else None
        profile = None if package else measured_profile(identity)
        if profile:
            api.atomic_json(root / "profile.json", profile)
        api.atomic_json(root / "stage-result.json", {"package": package, "profile": profile,
                                                    "request_hash": digest(request)})
        return 0

    monkeypatch.setattr(api, "supervise_command", stage)
    result = api.run_motion_extraction(extraction_args())
    assert result["status"] == "profiled" and len(model.calls) == 1
    api.record_terminal_admission(result)
    spent = result["spent_seconds"]
    result = api.run_motion_extraction(extraction_args("run", True))
    assert result["status"] == "complete" and len(model.calls) == 1
    api.record_terminal_admission(result)
    assert result["spent_seconds"] > spent
    budget_path = Path(result["budget_path"])
    budget = budget_path.read_bytes()
    ledger = (root / "generating-run.json").read_bytes()
    result = api.run_motion_extraction(extraction_args("run", True))
    assert result["status"] == "complete" and len(model.calls) == 1
    assert budget_path.read_bytes() == budget
    assert (root / "generating-run.json").read_bytes() == ledger


def test_foreign_root_preserves_owner_status_budget(api, tmp_path, monkeypatch):
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare", lambda args: (identity, frames, root))
    install_fake_preparation(monkeypatch, api, tmp_path, identity, frames, root)
    root.mkdir()
    for name in ("owner.json", "status.json", "budget.json"):
        (root / name).write_text("{}")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = api.run_motion_extraction(extraction_args(resume=True))
    assert result["status"] == "failed"
    assert {p.name: p.read_bytes() for p in root.iterdir() if p.name != ".writer.lock"} == before


def test_failed_worker_preserves_incomplete_and_actual_spend(api, tmp_path, monkeypatch):
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare", lambda args: (identity, frames, root))
    install_fake_preparation(monkeypatch, api, tmp_path, identity, frames, root)

    def failure(command, clock, **kwargs):
        time.sleep(.02)
        raise OSError("load or inference interrupted")

    monkeypatch.setattr(api, "supervise_command", failure)
    result = api.run_motion_extraction(extraction_args())
    assert result["status"] == "failed" and result["spent_seconds"] >= .02
    assert not (root / "complete/completion.json").exists()


def test_interrupted_publication_recovers_missing_metadata_only(api, tmp_path, monkeypatch):
    package = importlib.import_module("tools.motion.package")
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    api.generate_clip(root, "clip", frames["clips"]["clip"], identity,
                      RecordingEncoder(), api.make_processor())
    original = package.atomic_json

    def fail_after_manifest(path, value):
        if path.name == "source-window-records.json":
            raise OSError("interrupted metadata")
        return original(path, value)

    monkeypatch.setattr(package, "atomic_json", fail_after_manifest)
    with pytest.raises(ValueError):
        api.finish_package(root, identity, frames)
    p = root / "complete/manifest.json"
    before = p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns
    monkeypatch.setattr(package, "atomic_json", original)
    descriptor = api.finish_package(root, identity, frames)
    assert descriptor["clip_ids"] == ["clip"]
    assert (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns) == before


def test_authorization_and_cap_arguments_fail_before_production_reads(api, tmp_path):
    args = extraction_args()
    args.authorize_motion_extraction = False
    with pytest.raises(ValueError, match="authorization"):
        api.run_motion_extraction(args)
    args = extraction_args()
    args.gpu_cap_seconds = 18001
    with pytest.raises(ValueError, match="cap"):
        api.run_motion_extraction(args)


def test_owned_initialization_crash_resumes_before_any_launch(api, tmp_path, monkeypatch):
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare", lambda args: (identity, frames, root))
    launch = install_fake_preparation(monkeypatch, api, tmp_path, identity, frames, root)
    bootstrap = tmp_path / "motion-extraction-bootstrap-v1"
    api.atomic_json(bootstrap / "owner.json", {
        "schema_version": 1, "policy": "motion-extraction-bootstrap-owner-v1",
        "bootstrapkey": digest(launch), "launch": launch})
    monkeypatch.setattr(api, "supervise_command", lambda *a, **kw: 1)
    result = api.run_motion_extraction(extraction_args(resume=True))
    assert result["status"] == "failed" and result["spent_seconds"] > 0


def test_run_requires_valid_profile_under_same_clock(api, tmp_path, monkeypatch):
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare", lambda args: (identity, frames, root))
    install_fake_preparation(monkeypatch, api, tmp_path, identity, frames, root)
    monkeypatch.setattr(api, "supervise_command", lambda *a, **kw: pytest.fail("no GPU launch"))
    result = api.run_motion_extraction(extraction_args("run"))
    assert result["status"] == "failed" and "profile" in result["error"]


def test_metrics_cover_consumed_bytes_and_actual_invocations(api, tmp_path):
    paths = pngs(tmp_path / "frames", 25)
    with api.measure_clip() as stats:
        api.extract_motion_clip(paths, RecordingEncoder(), api.make_processor())
    assert stats["source_bytes"] == sum(p.stat().st_size for p in paths)
    assert stats["source_frames"] == 25
    assert stats["inference_batches"] == 1 and stats["inference_windows"] == 2
    assert stats["input_frames"] == 32
    assert stats["hash_seconds"] > 0 and stats["decode_seconds"] > 0
    assert stats["inference_seconds"] > 0 and stats["read_seconds"] > 0


def test_global_gpu_lock_serializes_different_content_roots(api, tmp_path):
    lock = tmp_path / ".motion-extraction-gpu.lock"
    with api.file_lock(lock):
        with pytest.raises(RuntimeError):
            with api.file_lock(lock):
                pass


def test_profile_byte_drift_blocks_run_before_launch(api, tmp_path, monkeypatch):
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare", lambda args: (identity, frames, root))
    install_fake_preparation(monkeypatch, api, tmp_path, identity, frames, root)
    def stage(command, clock, **kwargs):
        request = api.read_json(root / "stage-request.json")
        profile = measured_profile(identity)
        api.atomic_json(root / "profile.json", profile)
        api.atomic_json(root / "stage-result.json", {"package": None, "profile": profile,
                                                    "request_hash": digest(request)})
        return 0
    monkeypatch.setattr(api, "supervise_command", stage)
    first = api.run_motion_extraction(extraction_args())
    assert first["status"] == "profiled"
    api.record_terminal_admission(first)
    (root / "profile.json").write_text("{}")
    monkeypatch.setattr(api, "supervise_command", lambda *a, **kw: pytest.fail("no launch"))
    result = api.run_motion_extraction(extraction_args("run", True))
    assert result["status"] == "failed" and "profile" in result["error"]


@pytest.mark.parametrize("resume", [False, True])
def test_precommit_deadline_failure_leaves_no_marker_or_handoff(api, tmp_path, resume):
    """Task2 must pass the read-only deadline guard to both producer routes."""
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    api.generate_clip(root, "clip", frames["clips"]["clip"], identity,
                      RecordingEncoder(), api.make_processor())
    if resume:
        (root / "complete").mkdir()
    request = {"boot_id": api.clocks()[2], "deadline_monotonic": time.monotonic() + 29}
    with pytest.raises(TimeoutError):
        api.finish_package(root, identity, frames, before_commit=lambda: api._deadline(request))
    assert not (root / "complete/completion.json").exists()
    assert not (root / "stage-result.json").exists()


def test_full_candidate_failure_before_marker(api, tmp_path, monkeypatch):
    package = importlib.import_module("tools.motion.package")
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    api.generate_clip(root, "clip", frames["clips"]["clip"], identity,
                      RecordingEncoder(), api.make_processor())
    def interruption(*a, **kw):
        raise InterruptedError("signal during full candidate before marker")
    monkeypatch.setattr(package, "_completion_candidate", interruption)
    with pytest.raises(ValueError, match="before marker"):
        api.finish_package(root, identity, frames)
    assert not (root / "complete/completion.json").exists()
    assert not (root / "stage-result.json").exists()


def test_clock_spend_cannot_fall_below_committed_stage(api, tmp_path):
    path = tmp_path / "budget.json"
    now = {"mono": 0.0, "wall": 1000.0}
    clock = api.ExtractionClock(path, "key", resume=False,
                                clocks=lambda: (now["mono"], now["wall"], "boot"))
    clock.start("profile")
    now.update(mono=10.0, wall=1010.0)
    clock.finish("profiled")
    record = json.loads(path.read_text())
    record["spent_seconds"] = 0
    api.atomic_json(path, record)
    raw = path.read_bytes()
    with pytest.raises(ValueError):
        api.ExtractionClock(path, "key", resume=True)
    assert path.read_bytes() == raw


def test_legacy_annotation_integer_records_and_duplicate_ids(api):
    row = {k: "value" for k in ("fileid", "signer", "gloss", "text", "en_text",
                               "es_text", "fr_text")}
    row.update(fileid="clip", signer="Signer03")
    assert api.annotation_records({"prefix": "legacy", 0: row}) == {"clip": row}
    with pytest.raises(ValueError):
        api.annotation_records({0: row, 1: row})
    with pytest.raises(ValueError):
        api.annotation_records({0: {**row, "text": ""}})


def test_watchdog_and_inherited_lock_survive_caller_sigkill(api, tmp_path):
    root = tmp_path / "orphan-test"
    code = '''import sys
from pathlib import Path
from tools.motion.extractor import ExtractionClock, exclusive_writer, supervise_command
root = Path(sys.argv[1])
with exclusive_writer(root) as fd:
    clock = ExtractionClock(root / 'budget.json', 'key', resume=False)
    clock.start('run')
    supervise_command([sys.executable, '-c', 'import time; time.sleep(20)'], clock,
                      lock_fd=fd, deadline_seconds=1.0)
'''
    parent = subprocess.Popen([sys.executable, "-c", code, str(root)],
                              env={**os.environ, "PYTHONPATH": "src:."})
    child = None
    try:
        until = time.monotonic() + 10
        while time.monotonic() < until:
            path = root / "budget.json"
            if path.exists():
                child = api.read_json(path)["child"]
                if child is not None:
                    break
            time.sleep(.01)
        assert child is not None
        parent.kill()
        parent.wait(timeout=2)
        with pytest.raises(RuntimeError):
            with api.exclusive_writer(root):
                pass
        until = time.monotonic() + 5
        while api.process_identity(child["pid"]) is not None and time.monotonic() < until:
            time.sleep(.02)
        with api.exclusive_writer(root):
            recovered = api.ExtractionClock(root / "budget.json", "key", resume=True)
            assert recovered.spent >= .9
            assert recovered.state["status"] == "interrupted"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child and api.process_identity(child["pid"]) == child:
            os.killpg(child["pid"], signal.SIGKILL)


def bootstrap_args(tmp_path, mode="profile", resume=False):
    return SimpleNamespace(mode=mode, resume=resume, gpu_cap_seconds=18000,
                           authorize_motion_extraction=True, output_base=tmp_path,
                           adapted_root=tmp_path / "adapted", annotation=tmp_path / "ann.npy",
                           environment_lock=tmp_path / "lock", model_snapshot=tmp_path / "snapshot")


def synthetic_launch(monkeypatch, api, tmp_path):
    launch = {"schema_version": 1, "policy": "motion-extraction-launch-v1",
              "paths": {k: str(getattr(bootstrap_args(tmp_path), k)) for k in
                        ("output_base", "adapted_root", "annotation", "environment_lock",
                         "model_snapshot")}, "authorization": api.AUTHORIZATION,
              "code": {"fixture": "pinned"}}
    monkeypatch.setattr(api, "bootstrap_launch_binding", lambda args: launch)
    return launch


def install_fake_preparation(monkeypatch, api, tmp_path, identity, frames, root):
    launch = synthetic_launch(monkeypatch, api, tmp_path)
    def preparation(args, bootstrap, launch, clock, **kwargs):
        if kwargs.get("adopt_complete"):
            api.atomic_json(bootstrap / "prepared.json", {
                "package": api.validate_motion_package(root, frames)})
        return identity, frames, root
    monkeypatch.setattr(api, "prepare_supervised", preparation)
    return launch


def test_prepare_failure_is_inside_owned_clock_before_any_hashing(api, tmp_path, monkeypatch):
    synthetic_launch(monkeypatch, api, tmp_path)
    def preparation(args, bootstrap, launch, clock, **kwargs):
        record = api.read_json(bootstrap / "budget.json")
        assert record["active"]["phase"] == "preparing"
        assert record["active"]["owner"]["start_ticks"]
        time.sleep(.03)
        raise OSError("source hashing failed")
    monkeypatch.setattr(api, "prepare_supervised", preparation)
    result = api.run_motion_extraction(bootstrap_args(tmp_path))
    assert result["status"] == "failed" and result["spent_seconds"] >= .03
    assert result["featurekey"] is None and result["package"] is None
    state = api.read_json(Path(result["budget_path"]))
    assert state["featurekey"] is None and state["active"] is None


def test_binding_keeps_same_clock_and_reboot_spend_without_new_root(api, tmp_path):
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    path = tmp_path / "bootstrap/budget.json"
    now = {"mono": 10.0, "wall": 1000.0, "boot": "a"}
    clock = api.ExtractionClock(path, None, bootstrapkey="launch", resume=False,
                                clocks=lambda: (now["mono"], now["wall"], now["boot"]))
    clock.start("profile", phase="preparing")
    now.update(mono=17.0, wall=1007.0)
    clock.bind_feature(identity, root)
    assert clock.spent == 7 and clock.path == path
    assert clock.state["featurekey"] == digest(identity)
    clock.finish("profiled")
    now.update(mono=20.0, wall=1010.0)
    clock = api.ExtractionClock(path, None, bootstrapkey="launch", resume=True,
                                clocks=lambda: (now["mono"], now["wall"], now["boot"]))
    clock.start("run", phase="preparing")
    now.update(mono=1.0, wall=1020.0, boot="b")
    clock = api.ExtractionClock(path, None, bootstrapkey="launch", resume=True,
                                clocks=lambda: (now["mono"], now["wall"], now["boot"]))
    assert clock.spent == 17
    before = path.read_bytes()
    with pytest.raises(ValueError):
        clock.bind_feature({**identity, "hardware": {"changed": True}}, root)
    assert path.read_bytes() == before


def test_bootstrap_launch_conflict_cannot_overwrite_clock_or_status(api, tmp_path, monkeypatch):
    launch = synthetic_launch(monkeypatch, api, tmp_path)
    def preparation(*args, **kwargs):
        raise OSError("fixture failed before feature binding")
    monkeypatch.setattr(api, "prepare_supervised", preparation)
    first = api.run_motion_extraction(bootstrap_args(tmp_path))
    bootstrap = Path(first["bootstrap_root"])
    before = {p.name: p.read_bytes() for p in bootstrap.iterdir() if p.is_file()}
    launch["code"] = {"fixture": "changed"}
    with pytest.raises(ValueError, match="launch"):
        api.run_motion_extraction(bootstrap_args(tmp_path, resume=True))
    assert {p.name: p.read_bytes() for p in bootstrap.iterdir() if p.is_file()} == before


def test_preparation_timeout_is_charged_uncapped(api, tmp_path, monkeypatch):
    synthetic_launch(monkeypatch, api, tmp_path)
    def preparation(args, bootstrap, launch, clock, **kwargs):
        rc = api.supervise_command([sys.executable, "-c", "import time; time.sleep(20)"],
                                   clock, deadline_seconds=.15, lock_fd=kwargs["lock_fd"])
        assert rc != 0
        raise TimeoutError("prepare deadline")
    monkeypatch.setattr(api, "prepare_supervised", preparation)
    result = api.run_motion_extraction(bootstrap_args(tmp_path))
    assert result["status"] == "capped" and result["spent_seconds"] >= .15
    assert result["package"] is None and result["featurekey"] is None


def test_successful_preparation_is_counted_once_not_retrocharged(api, tmp_path, monkeypatch):
    synthetic_launch(monkeypatch, api, tmp_path)
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    now = {"mono": 0.0, "wall": 1000.0}
    real_clock = api.ExtractionClock
    def clock_factory(*args, **kwargs):
        return real_clock(*args, **kwargs, clocks=lambda: (now["mono"], now["wall"], "boot"))
    monkeypatch.setattr(api, "ExtractionClock", clock_factory)
    def preparation(*args, **kwargs):
        now.update(mono=.05, wall=1000.05)
        return identity, frames, root
    monkeypatch.setattr(api, "prepare_supervised", preparation)
    monkeypatch.setattr(api, "supervise_command", lambda *a, **kw: 1)
    result = api.run_motion_extraction(bootstrap_args(tmp_path))
    state = api.read_json(Path(result["budget_path"]))
    assert state["stages"][-1]["preparation_seconds"] == .05
    assert state["spent_seconds"] == .05
    assert state["featurekey"] == digest(identity)
    assert not (root / "budget.json").exists()


def test_resume_hardware_drift_cannot_mint_new_feature_budget(api, tmp_path, monkeypatch):
    synthetic_launch(monkeypatch, api, tmp_path)
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare_supervised", lambda *a, **kw: (identity, frames, root))
    monkeypatch.setattr(api, "supervise_command", lambda *a, **kw: 1)
    first = api.run_motion_extraction(bootstrap_args(tmp_path))
    changed = {**identity, "hardware": {"changed": True}}
    new_root = tmp_path / f"motion-features-{digest(changed)}"
    monkeypatch.setattr(api, "prepare_supervised", lambda *a, **kw: (changed, frames, new_root))
    second = api.run_motion_extraction(bootstrap_args(tmp_path, resume=True))
    assert second["status"] == "failed" and second["package"] is None
    assert second["budget_path"] == first["budget_path"]
    assert second["spent_seconds"] >= first["spent_seconds"]
    assert not new_root.exists()


def preparation_shim(identity, frames, root, *, failure=False):
    return f'''from pathlib import Path
import time
import tools.motion.extractor as m
m.bootstrap_launch_binding = lambda args: m.read_json(Path(__import__('sys').argv[1]))['launch']
m.preparation_inputs = lambda args: {{'fixture': 'observed CPU inputs'}}
def fixture_prepare(args):
    time.sleep(.03)
    {"raise OSError('fixture hashing failed')" if failure else "pass"}
    return {identity!r}, {frames!r}, Path({str(root)!r})
m.prepare = fixture_prepare
'''


@pytest.mark.parametrize("failure", [False, True])
def test_real_supervised_preparation_cpu_result_or_failure(api, tmp_path, monkeypatch, failure):
    launch = synthetic_launch(monkeypatch, api, tmp_path)
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    original = api.supervise_command
    shim = preparation_shim(identity, frames, root, failure=failure)
    def supervised(command, clock, **kwargs):
        assert clock.state["active"]["phase"] == "preparing"
        command = list(command)
        command[2] = shim + command[2]
        return original(command, clock, **kwargs)
    monkeypatch.setattr(api, "supervise_command", supervised)
    bootstrap = tmp_path / "motion-extraction-bootstrap-v1"
    with api.exclusive_writer(bootstrap) as fd:
        clock = api.ExtractionClock(bootstrap / "budget.json", None,
                                    bootstrapkey=digest(launch), resume=False)
        clock.start("profile", phase="preparing")
        started = time.monotonic()
        if failure:
            with pytest.raises(RuntimeError, match="hashing failed"):
                api.prepare_supervised(bootstrap_args(tmp_path), bootstrap, launch,
                                       clock, lock_fd=fd)
            assert not (bootstrap / "prepared.json").exists()
            clock.finish("failed")
        else:
            actual, actual_frames, actual_root = api.prepare_supervised(
                bootstrap_args(tmp_path), bootstrap, launch, clock, lock_fd=fd)
            assert actual == identity and actual_root == root
            assert list(actual_frames["clips"]) == ["clip"]
            clock.bind_feature(actual, actual_root)
            clock.finish("profiled")
        elapsed = time.monotonic() - started
        assert elapsed - .05 <= clock.spent <= elapsed + .1
        assert clock.state["stages"][-1]["preparation_seconds"] >= .03
        assert clock.state["child"] is None


def test_prepare_crash_durable_charge_after_caller_sigkill(api, tmp_path):
    root = tmp_path / "motion-extraction-bootstrap-v1"
    code = '''import sys
from pathlib import Path
import tools.motion.extractor as m
base = Path(sys.argv[1])
launch = {'paths': {'output_base': str(base)}, 'code': {'fixture': 'pinned'}}
m.bootstrap_launch_binding = lambda args: launch
def preparation(args, bootstrap, launch, clock, **kw):
    m.supervise_command([sys.executable, '-c', 'import time; time.sleep(20)'], clock,
                        lock_fd=kw['lock_fd'], deadline_seconds=1.0)
    raise TimeoutError('prepare timeout')
m.prepare_supervised = preparation
args = type('Args', (), dict(mode='profile',resume=False,authorize_motion_extraction=True))()
m.run_motion_extraction(args)
'''
    parent = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)],
                              env={**os.environ, "PYTHONPATH": "src:."})
    child = None
    try:
        until = time.monotonic() + 10
        while time.monotonic() < until:
            if (root / "budget.json").exists():
                record = api.read_json(root / "budget.json")
                child = record["child"]
                if child is not None:
                    assert record["active"]["phase"] == "preparing"
                    break
            time.sleep(.01)
        assert child is not None
        parent.kill()
        parent.wait(timeout=2)
        with pytest.raises(RuntimeError):
            with api.exclusive_writer(root):
                pass
        until = time.monotonic() + 5
        while api.process_identity(child["pid"]) is not None and time.monotonic() < until:
            time.sleep(.02)
        with api.exclusive_writer(root):
            record = api.read_json(root / "budget.json")
            recovered = api.ExtractionClock(root / "budget.json", None,
                                            bootstrapkey=record["bootstrapkey"], resume=True)
            assert recovered.spent >= .9
            assert recovered.state["featurekey"] is None
            assert recovered.state["stages"][-1]["phase"] == "preparing"
            assert recovered.state["stages"][-1]["status"] == "interrupted"
            assert recovered.state["stages"][-1]["owner"]["start_ticks"]
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child and api.process_identity(child["pid"]) == child:
            os.killpg(child["pid"], signal.SIGKILL)


def test_atomic_failure_after_visible_marker_is_not_success_or_handoff(api, tmp_path, monkeypatch):
    package = importlib.import_module("tools.motion.package")
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    install_fake_preparation(monkeypatch, api, tmp_path, identity, frames, root)
    model, processor = RecordingEncoder(), api.make_processor()
    def stage(command, clock, **kwargs):
        request = api.read_json(root / "stage-request.json")
        api.generate_clip(root, "clip", frames["clips"]["clip"], identity, model, processor)
        if request["mode"] == "profile":
            profile = measured_profile(identity)
            api.atomic_json(root / "profile.json", profile)
            api.atomic_json(root / "stage-result.json", {
                "package": None, "profile": profile, "request_hash": digest(request)})
            return 0
        original = package.atomic_json
        def fail_after_visibility(path, value):
            original(path, value)
            if path.name == "completion.json":
                raise OSError("fsync failed after marker became visible")
        monkeypatch.setattr(package, "atomic_json", fail_after_visibility)
        with pytest.raises(ValueError, match="fsync"):
            api.finish_package(root, identity, frames, before_commit=lambda: api._deadline(request))
        return 1
    monkeypatch.setattr(api, "supervise_command", stage)
    first = api.run_motion_extraction(extraction_args())
    assert first["status"] == "profiled"
    api.record_terminal_admission(first)
    result = api.run_motion_extraction(extraction_args("run", True))
    assert (root / "complete/completion.json").is_file()
    assert result["status"] == "failed" and result["package"] is None
    assert api.read_json(Path(result["budget_path"]))["status"] == "failed"


def test_real_preparation_observation_drift_rejects_resume_before_prepare(
    api, tmp_path, monkeypatch,
):
    launch = synthetic_launch(monkeypatch, api, tmp_path)
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    args = bootstrap_args(tmp_path)
    args.annotation.write_bytes(b"first actual source bytes")
    original = api.supervise_command
    shim = preparation_shim(identity, frames, root)
    shim += "m.preparation_inputs = lambda args: {'source_hash': m.file_hash(args.annotation)}\n"
    def supervised(command, clock, **kwargs):
        command = list(command)
        command[2] = shim + command[2]
        return original(command, clock, **kwargs)
    monkeypatch.setattr(api, "supervise_command", supervised)
    bootstrap = tmp_path / "motion-extraction-bootstrap-v1"
    with api.exclusive_writer(bootstrap) as fd:
        clock = api.ExtractionClock(bootstrap / "budget.json", None,
                                    bootstrapkey=digest(launch), resume=False)
        clock.start("profile", phase="preparing")
        actual, _, actual_root = api.prepare_supervised(args, bootstrap, launch, clock, lock_fd=fd)
        clock.bind_feature(actual, actual_root)
        clock.finish("profiled")
        spent = clock.spent
        observation = (bootstrap / "input-observation.json").read_bytes()
        args.annotation.write_bytes(b"changed same source filename")
        clock = api.ExtractionClock(bootstrap / "budget.json", None,
                                    bootstrapkey=digest(launch), resume=True)
        clock.start("run", phase="preparing")
        with pytest.raises(RuntimeError, match="changed on resume"):
            api.prepare_supervised(args, bootstrap, launch, clock, lock_fd=fd)
        clock.finish("failed")
        assert clock.spent > spent
        assert (bootstrap / "input-observation.json").read_bytes() == observation
        assert not root.exists()


def test_bootstrap_lock_conflict_preserves_incumbent_before_preparation(api, tmp_path, monkeypatch):
    synthetic_launch(monkeypatch, api, tmp_path)
    def failure(*args, **kwargs):
        raise OSError("fixture stopped preparation")
    monkeypatch.setattr(api, "prepare_supervised", failure)
    first = api.run_motion_extraction(bootstrap_args(tmp_path))
    bootstrap = Path(first["bootstrap_root"])
    before = {p.name: p.read_bytes() for p in bootstrap.iterdir()}
    monkeypatch.setattr(api, "prepare_supervised", lambda *a, **kw: pytest.fail("no preparation"))
    with api.exclusive_writer(bootstrap), pytest.raises(RuntimeError):
        api.run_motion_extraction(bootstrap_args(tmp_path, resume=True))
    assert {p.name: p.read_bytes() for p in bootstrap.iterdir()} == before


def test_check_resamples_after_blocking_persistence(api, tmp_path, monkeypatch):
    now = {"mono": 0.0, "wall": 1000.0}
    clock = api.ExtractionClock(tmp_path / "budget.json", "key", resume=False,
                                clocks=lambda: (now["mono"], now["wall"], "boot"))
    clock.start("run")
    now.update(mono=17960.0, wall=18960.0)
    save = clock.save
    def slow_save():
        save()
        now["mono"] += 50
        now["wall"] += 50
    monkeypatch.setattr(clock, "save", slow_save)
    with pytest.raises(TimeoutError):
        clock.check()
    assert clock.spent == 18010


def test_final_close_cannot_complete_after_blocking_write_overrun(api, tmp_path, monkeypatch):
    now = {"mono": 0.0, "wall": 1000.0}
    clock = api.ExtractionClock(tmp_path / "budget.json", "key", resume=False,
                                clocks=lambda: (now["mono"], now["wall"], "boot"))
    clock.start("run")
    now.update(mono=17960.0, wall=18960.0)
    save = clock.save
    def slow_save():
        save()
        now["mono"] += 50
        now["wall"] += 50
    monkeypatch.setattr(clock, "save", slow_save)
    with pytest.raises(TimeoutError):
        clock.finish("complete")
    assert clock.spent >= 18010 and clock.state["status"] != "complete"


def profile_fixture(api, tmp_path):
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    receipt = api.generate_clip(root, "clip", frames["clips"]["clip"], identity,
                                RecordingEncoder(), api.make_processor())
    return identity, frames, root, receipt


def observed_sample(seconds=8.0):
    return {"clip_id": "clip", "seconds": seconds, "reused": False,
            "measurements": {"inference_seconds": seconds / 2,
                             "inference_windows": 1, "inference_batches": 1,
                             "source_frames": 1, "input_frames": 16}}


def test_profile_observation_survives_interrupt_and_binds_actual_receipt(api, tmp_path):
    identity, frames, root, receipt = profile_fixture(api, tmp_path)
    observation = api.record_profile_sample(root, identity, receipt, observed_sample())
    path = root / "profile-samples/clip.json"
    original = path.read_bytes()
    assert observation["receipt_sha256"] == file_hash(root / "receipts/train/clip.json")
    assert observation["array_sha256"] == receipt["array"]["sha256"]
    assert observation["source_hash"] == receipt["source_hash"]
    samples = api.valid_profile_samples(root, identity, frames)
    assert samples["clip"]["seconds"] == 8
    assert api.profile_rate(samples, None) == 10
    assert path.read_bytes() == original


def test_profile_retry_cannot_erase_prior_conservative_max(api, tmp_path):
    identity, frames, root, receipt = profile_fixture(api, tmp_path)
    api.record_profile_sample(root, identity, receipt, observed_sample())
    samples = api.valid_profile_samples(root, identity, frames)
    fresh = dict(samples["clip"], clip_id="fresh", seconds=1.0)
    merged = {**samples, "fresh": fresh}
    assert api.profile_rate(merged, {"seconds_per_window": 10.0}) == 10.0


def test_receipt_without_inference_observation_fails_closed_no_rewrite(api, tmp_path):
    identity, frames, root, receipt = profile_fixture(api, tmp_path)
    paths = [root / "train/clip.npy", root / "receipts/train/clip.json"]
    before = [(p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns) for p in paths]
    with pytest.raises(ValueError, match="observation"):
        api.valid_profile_samples(root, identity, frames, required=["clip"])
    assert before == [(p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns) for p in paths]


def test_profile_observation_commit_failure_never_infers_timing_from_reuse(
    api, tmp_path, monkeypatch,
):
    identity, frames, root, receipt = profile_fixture(api, tmp_path)
    original = api.atomic_json
    def interruption(path, value):
        if path.parent.name == "profile-samples":
            raise OSError("interrupt after feature/receipt, before observation")
        return original(path, value)
    monkeypatch.setattr(api, "atomic_json", interruption)
    with pytest.raises(OSError):
        api.record_profile_sample(root, identity, receipt, observed_sample())
    with pytest.raises(ValueError, match="observation"):
        api.valid_profile_samples(root, identity, frames, required=["clip"])


def test_directory_allowlist_constructed_once_each(api, tmp_path):
    class CountedClips(dict):
        iterations = 0
        def __iter__(self):
            self.iterations += 1
            return super().__iter__()
    root = tmp_path / "inventory"
    (root / "train").mkdir(parents=True)
    (root / "receipts/train").mkdir(parents=True)
    clips = CountedClips({f"c{i}": {} for i in range(4)})
    for clip in list(clips.keys()):
        (root / "train" / f"{clip}.npy").write_bytes(b"array fixture")
        (root / "receipts/train" / f"{clip}.json").write_bytes(b"{}")
    api.check_owned_inventory(root, {"clips": clips})
    assert clips.iterations == 2


def test_real_short_parent_closing_watchdog_prevents_success(api, tmp_path):
    root = tmp_path / "closing-watchdog"
    code = '''import sys,time
from pathlib import Path
from tools.motion.extractor import ExtractionClock,ParentDeadlineGuard,exclusive_writer
root=Path(sys.argv[1])
with exclusive_writer(root) as fd:
    clock=ExtractionClock(root/'budget.json','fixture-key',resume=False)
    clock.start('run')
    guard=ParentDeadlineGuard(clock,fd,timeout_seconds=1.0)
    try:
        guard.begin_closing(.15)
        time.sleep(2)
        clock.finish('complete')
        (root/'false-success').write_text('must not happen')
    except TimeoutError:
        clock.finish('capped')
    finally:
        guard.close()
'''
    result = subprocess.run([sys.executable, "-c", code, str(root)],
                            env={**os.environ, "PYTHONPATH": "src:."},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not (root / "false-success").exists()
    record = api.read_json(root / "budget.json")
    assert record["status"] == "capped" and record["spent_seconds"] >= .15
    assert list(root.glob("terminal-failure-*.json"))


def test_mixed_retry_worker_preserves_durable_slow_sample(api, tmp_path, monkeypatch):
    paths = pngs(tmp_path / "frames", 1)
    identity, frames, _ = fixture_identity(api, tmp_path, paths)
    frames["clips"]["fresh"] = dict(frames["clips"]["clip"])
    frame_path = Path(identity["inputs"]["frame_rows"]["path"])
    frame_path.write_text(json.dumps(frames))
    identity["inputs"]["frame_rows"]["sha256"] = file_hash(frame_path)
    identity["environment"]["model_files"] = []
    root = tmp_path / f"motion-features-{digest(identity)}"
    now = {"mono": 100.0}
    monkeypatch.setattr(api.time, "monotonic", lambda: now["mono"])
    class TimedEncoder(RecordingEncoder):
        def to(self, *args, **kwargs):
            return self
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            now["mono"] += 8 if len(self.calls) == 1 else 1
            return result
    model = TimedEncoder()
    monkeypatch.setattr(api, "load_encoder", lambda *a: (model, api.make_processor()))
    monkeypatch.setattr(api, "snapshot_files", lambda *a: [])
    monkeypatch.setattr(api, "hardware_identity", lambda: {})
    monkeypatch.setattr(api, "require_gpu_idle", lambda: None)
    monkeypatch.setattr(api, "require_storage", lambda *a: {})
    monkeypatch.setattr(api, "tensor_parity", lambda *a: [])
    for name in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(api.torch.cuda, name, lambda: None)
    monkeypatch.setattr(api.torch.cuda, "max_memory_allocated", lambda: 0)
    original = api.generate_clip
    def interrupt(root, clip, *args, **kwargs):
        if clip == "fresh":
            raise OSError("interrupted between durable samples")
        return original(root, clip, *args, **kwargs)
    monkeypatch.setattr(api, "generate_clip", interrupt)
    request = {"root": str(root), "identity": identity, "mode": "profile",
               "model_snapshot": "/fixture", "boot_id": api.clocks()[2],
               "deadline_monotonic": 1e9}
    with pytest.raises(OSError):
        api._worker(request)
    old_sample = (root / "profile-samples/clip.json").read_bytes()
    old_receipt = (root / "receipts/train/clip.json").read_bytes()
    monkeypatch.setattr(api, "generate_clip", original)
    result = api._worker(request)
    assert result["profile"]["admission_policy"] == "measured-smoke-hard-cap-v1"
    assert "projected_remaining_seconds" not in result["profile"]
    assert [s["seconds"] for s in result["profile"]["samples"]] == [8.0, 1.0]
    assert len(model.calls) == 2
    assert (root / "profile-samples/clip.json").read_bytes() == old_sample
    assert (root / "receipts/train/clip.json").read_bytes() == old_receipt


def test_late_check_persist_returns_capped_package_none(api, tmp_path, monkeypatch):
    synthetic_launch(monkeypatch, api, tmp_path)
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare_supervised", lambda *a, **kw: (identity, frames, root))
    real_clock = api.ExtractionClock
    now = {"mono": 0.0, "wall": 1000.0}
    def clock_factory(*args, **kwargs):
        return real_clock(*args, **kwargs, clocks=lambda: (now["mono"], now["wall"], "boot"))
    monkeypatch.setattr(api, "ExtractionClock", clock_factory)
    def candidate(*args):
        clock = args[-2]
        now.update(mono=17960.0, wall=18960.0)
        save = clock.save
        def slow_once():
            save()
            now["mono"] += 50
            now["wall"] += 50
            clock.save = save
        clock.save = slow_once
        clock.check()
        return "complete", {"would": "be unsafe handoff"}, None
    monkeypatch.setattr(api, "_run_prepared", candidate)
    result = api.run_motion_extraction(bootstrap_args(tmp_path))
    assert result["status"] == "capped" and result["package"] is None
    assert result["spent_seconds"] >= 18010


def test_success_requires_terminal_receipt_bound_to_closed_budget(api, tmp_path, monkeypatch):
    launch = synthetic_launch(monkeypatch, api, tmp_path)
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare_supervised", lambda *a, **kw: (identity, frames, root))
    monkeypatch.setattr(api, "_run_prepared", lambda *a: ("profiled", None, {}))
    result = api.run_motion_extraction(bootstrap_args(tmp_path))
    assert result["status"] == "profiled"
    assert api.read_json(Path(result["budget_path"]))["status"] == "terminal_pending"
    api.record_terminal_admission(result)
    clock = api.ExtractionClock(Path(result["budget_path"]), None,
                                bootstrapkey=digest(launch), resume=True)
    receipt = api.validate_terminal_close(clock)
    assert receipt["spent_upper_bound"] == result["spent_seconds"]
    assert receipt["budget_sha256"] == file_hash(clock.path)
    assert clock.state["spent_semantics"] == "observed_work_plus_conservative_closing_bound"
    assert receipt["observed_before_receipt_write"] < receipt["closing_deadline_monotonic"]


def test_slow_last_terminal_write_cannot_return_completed_package(api, tmp_path, monkeypatch):
    synthetic_launch(monkeypatch, api, tmp_path)
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare_supervised", lambda *a, **kw: (identity, frames, root))
    real_clock = api.ExtractionClock
    now = {"mono": 0.0, "wall": 1000.0}
    def clock_factory(*args, **kwargs):
        clock = real_clock(*args, **kwargs, clocks=lambda: (now["mono"], now["wall"], "boot"))
        original = clock.save
        def slow_terminal():
            original()
            if clock.state["status"] == "terminal_pending":
                now["mono"] += 50
                now["wall"] += 50
        clock.save = slow_terminal
        return clock
    monkeypatch.setattr(api, "ExtractionClock", clock_factory)
    def candidate(*args):
        now.update(mono=17960.0, wall=18960.0)
        return "complete", {"candidate": "must be discarded"}, None
    monkeypatch.setattr(api, "_run_prepared", candidate)
    result = api.run_motion_extraction(bootstrap_args(tmp_path))
    assert result["status"] == "capped" and result["package"] is None
    assert result["spent_seconds"] >= 18010
    assert api.read_json(Path(result["budget_path"]))["status"] == "capped"


def test_unsealed_parent_close_resume_reconciles_instead_of_adopting_complete(api, tmp_path):
    root = tmp_path / "unsealed-close"
    code = '''import sys,time
from pathlib import Path
from tools.motion.extractor import ExtractionClock,ParentDeadlineGuard,exclusive_writer
root=Path(sys.argv[1])
with exclusive_writer(root) as fd:
    clock=ExtractionClock(root/'budget.json','fixture-key',resume=False)
    clock.start('run')
    guard=ParentDeadlineGuard(clock,fd,timeout_seconds=1)
    try:
        guard.begin_closing(.2)
        clock.finish('complete')
        time.sleep(2)  # no seal/ack: persisted candidate is not bounded success
    except TimeoutError:
        pass
    finally:
        guard.close()
'''
    result = subprocess.run([sys.executable, "-c", code, str(root)],
                            env={**os.environ, "PYTHONPATH": "src:."},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    clock = api.ExtractionClock(root / "budget.json", "fixture-key", resume=True)
    assert clock.state["status"] == "interrupted" and clock.spent >= .2
    with pytest.raises(ValueError):
        api.validate_terminal_close(clock)


def test_pending_candidate_17965_then_18010_can_be_capped(api, tmp_path):
    now = {"mono": 0.0, "wall": 1000.0}
    clock = api.ExtractionClock(tmp_path / "budget.json", "key", resume=False,
                                clocks=lambda: (now["mono"], now["wall"], "boot"))
    clock.start("run")
    now.update(mono=17960.0, wall=18960.0)
    clock.parent_guard = SimpleNamespace(
        closing_upper_bound=17965.0,
        binding=lambda: {"receipt_path": str(tmp_path / "terminal.json"),
                         "failure_path": str(tmp_path / "failure.json"),
                         "candidate_path": str(tmp_path / "candidate.json"),
                         "spent_upper_bound": 17965.0, "nonce": "fixture"})
    clock.finish("complete")
    candidate = api.read_json(clock.path)
    assert candidate["status"] == "terminal_pending" and candidate["active"] is not None
    now.update(mono=18010.0, wall=19010.0)
    clock.tick()
    clock.finish("capped")
    assert clock.spent == 18010
    assert api.read_json(clock.path)["status"] == "capped"
    assert clock.state["stages"][-1]["end_spent"] == 18010


def concrete_seal_fixture(api, tmp_path, monkeypatch):
    guard = api.ParentDeadlineGuard.__new__(api.ParentDeadlineGuard)
    now = {"mono": 1.0}
    guard.clock = SimpleNamespace(path=tmp_path / "budget.json",
                                 state={"bootstrapkey": "launch", "featurekey": "key",
                                        "status": "terminal_pending",
                                        "terminal_candidate": {"status": "complete"}})
    guard.receipt = tmp_path / "terminal.json"
    guard.failure = tmp_path / "failure.json"
    guard.closing_deadline = 5.0
    guard.deadline = 18000.0
    guard.binding = lambda: {"candidate_path": str(tmp_path / "candidate.json")}
    guard.send = lambda value: None
    guard.ack_fd = 99999
    monkeypatch.setattr(api.os, "read", lambda *args: b"1")
    monkeypatch.setattr(api, "validate_terminal_close", lambda *args, **kwargs: {
        "candidate_sha256": "candidate", "budget_sha256": "budget"})
    monkeypatch.setattr(api, "file_hash", lambda *args: "fixture-hash")
    monkeypatch.setattr(api.time, "monotonic", lambda: now["mono"])
    return guard, now


@pytest.mark.parametrize("late", [6.0, 18001.0])
def test_actual_seal_rechecks_closing_and_global_deadline_after_wait(
    api, tmp_path, monkeypatch, late,
):
    guard, now = concrete_seal_fixture(api, tmp_path, monkeypatch)
    class DelayedExit:
        def wait(self, **kwargs):
            now["mono"] = late
            return 0
    guard.child = DelayedExit()
    with pytest.raises(TimeoutError):
        guard.seal(tmp_path / "status.json")


def test_final_cleanup_late_discards_package_and_charges_closing_anchor(
    api, tmp_path, monkeypatch,
):
    synthetic_launch(monkeypatch, api, tmp_path)
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare_supervised", lambda *a, **kw: (identity, frames, root))
    monkeypatch.setattr(api, "_run_prepared", lambda *a: ("complete", {"candidate": True}, None))
    real_monotonic = api.time.monotonic
    offset = {"seconds": 0.0}
    monkeypatch.setattr(api.time, "monotonic", lambda: real_monotonic() + offset["seconds"])
    original = api.ParentDeadlineGuard.close
    def delayed_cleanup(guard):
        original(guard)
        offset["seconds"] += 10
    monkeypatch.setattr(api.ParentDeadlineGuard, "close", delayed_cleanup)
    result = api.run_motion_extraction(bootstrap_args(tmp_path))
    assert result["status"] == "capped" and result["package"] is None
    assert result["spent_seconds"] >= 10
    assert api.read_json(Path(result["budget_path"]))["status"] == "capped"


@pytest.mark.parametrize("change", ["status", "upper", "anchor", "nan", "receipt"])
def test_return_admission_rejects_status_bound_anchor_and_receipt_tamper(
    api, tmp_path, monkeypatch, change,
):
    synthetic_launch(monkeypatch, api, tmp_path)
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare_supervised", lambda *a, **kw: (identity, frames, root))
    monkeypatch.setattr(api, "_run_prepared", lambda *a: ("profiled", None, {}))
    result = api.run_motion_extraction(bootstrap_args(tmp_path))
    assert result["status"] == "profiled"
    proof = result["terminal_admission"]
    path = Path(proof["terminal_close"]["admission_path"])
    if change == "status":
        proof["status"] = "complete"
    elif change == "upper":
        proof["spent_upper_bound"] += 1
    elif change == "anchor":
        proof["observed_spend_at_return_check"] = (
            proof["terminal_close"]["observed_start_spend"] - 1)
    elif change == "nan":
        proof["parent_checked_monotonic"] = float("nan")
    else:
        Path(proof["terminal_close"]["receipt_path"]).write_text("{}")
    with pytest.raises(ValueError):
        api.record_terminal_admission(result)
    assert not path.exists()
    assert api.read_json(Path(result["budget_path"]))["status"] == "terminal_pending"


def test_pending_resume_is_not_adoption_and_reboot_charges_anchor(api, tmp_path):
    # A committed caller admission, not a visible candidate, distinguishes immutable adoption.
    now = {"mono": 0.0, "wall": 1000.0}
    clock = api.ExtractionClock(tmp_path / "pending.json", "key", resume=False,
                                clocks=lambda: (now["mono"], now["wall"], "a"))
    clock.start("run")
    now.update(mono=7.0, wall=1007.0)
    clock.state.update(status="terminal_pending", terminal_close={
        "admission_path": str(tmp_path / "missing-admission.json")})
    clock.save()
    now.update(mono=1.0, wall=1017.0)
    recovered = api.ExtractionClock(tmp_path / "pending.json", "key", resume=True,
                                    clocks=lambda: (now["mono"], now["wall"], "b"))
    assert recovered.state["status"] == "interrupted" and recovered.spent == 17


def test_cli_records_admission_only_after_extractor_return(api, tmp_path, monkeypatch, capsys):
    cli = importlib.import_module("scripts.extract_motion_features")
    synthetic_launch(monkeypatch, api, tmp_path)
    identity, frames, root = fixture_identity(api, tmp_path, pngs(tmp_path / "frames", 1))
    monkeypatch.setattr(api, "prepare_supervised", lambda *a, **kw: (identity, frames, root))
    monkeypatch.setattr(api, "_run_prepared", lambda *a: ("profiled", None, {}))
    result = api.run_motion_extraction(bootstrap_args(tmp_path))
    admission_path = Path(result["terminal_admission"]["terminal_close"]["admission_path"])
    assert not admission_path.exists()
    monkeypatch.setattr(cli, "parse_args", lambda: bootstrap_args(tmp_path))
    monkeypatch.setattr(cli, "run_motion_extraction", lambda args: result)
    assert cli.main() == 0
    assert admission_path.is_file()
    assert json.loads(capsys.readouterr().out)["status"] == "profiled"


def terminal_sequence_fixture(api, tmp_path):
    """Real pending budget/candidate/seal with independently chosen coherent times."""
    clock = api.ExtractionClock(tmp_path / "budget.json", "key", resume=False,
                                clocks=lambda: (100.0, 1000.0, "boot"))
    clock.start("run")
    terminal = {
        "schema_version": 1, "policy": "motion-terminal-close-v1", "nonce": "sequence",
        "receipt_path": str(tmp_path / "terminal.json"),
        "failure_path": str(tmp_path / "failure.json"),
        "candidate_path": str(tmp_path / "candidate.json"),
        "admission_path": str(tmp_path / "admission.json"),
        "spent_upper_bound": 5.0, "closing_deadline_monotonic": 105.0,
        "parent_deadline_monotonic": 110.0, "boot_id": "boot", "clock_boot_id": "boot",
        "started_monotonic": 100.0, "started_wall": 1000.0, "observed_start_spend": 0.0,
    }
    clock.parent_guard = SimpleNamespace(closing_upper_bound=5.0, binding=lambda: terminal)
    clock.finish("complete")
    api.atomic_json(tmp_path / "status.json", {"status": "complete"})
    receipt = {
        **terminal, "candidate_sha256": file_hash(Path(terminal["candidate_path"])),
        "budget_sha256": file_hash(clock.path),
        "status_sha256": file_hash(tmp_path / "status.json"),
        "status": "complete", "clock_status": "terminal_pending",
        "featurekey": "key", "bootstrapkey": None, "observed_before_receipt_write": 101.0,
    }
    api.atomic_json(Path(terminal["receipt_path"]), receipt)
    proof = {
        "schema_version": 1, "policy": "motion-terminal-admission-v1",
        "terminal_close": terminal, "candidate_sha256": receipt["candidate_sha256"],
        "terminal_sha256": file_hash(Path(terminal["receipt_path"])),
        "budget_sha256": receipt["budget_sha256"], "bootstrapkey": None, "featurekey": "key",
        "status": "complete", "spent_upper_bound": 5.0,
        "observed_spend_at_return_check": 2.0, "parent_checked_monotonic": 102.0,
        "boot_id": "boot",
    }
    result = {"status": "complete", "bootstrap_root": str(tmp_path),
              "budget_path": str(clock.path), "terminal_admission": proof}
    return clock, receipt, result


@pytest.mark.parametrize("checked", [-1.0, 99.0, 100.0, 100.5, 105.0, 110.0,
                                     True, False, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("validator", ["projection", "caller"])
def test_admission_rejects_impossible_parent_check_sequence(
    api, tmp_path, checked, validator,
):
    clock, _, result = terminal_sequence_fixture(api, tmp_path)
    proof = result["terminal_admission"]
    proof["parent_checked_monotonic"] = checked
    before = clock.path.read_bytes()
    with pytest.raises(ValueError):
        if validator == "projection":
            api.admitted_terminal_snapshot(clock, admission=proof)
        else:
            api.record_terminal_admission(result)
    assert not Path(proof["terminal_close"]["admission_path"]).exists()
    assert clock.path.read_bytes() == before


@pytest.mark.parametrize("observation", [-1.0, 99.0, 105.0, True, False,
                                         float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("validator", ["close", "projection", "caller"])
def test_admission_rejects_invalid_receipt_observation(
    api, tmp_path, observation, validator,
):
    clock, receipt, result = terminal_sequence_fixture(api, tmp_path)
    proof = result["terminal_admission"]
    receipt["observed_before_receipt_write"] = observation
    receipt_path = Path(proof["terminal_close"]["receipt_path"])
    # Rebind the actual receipt bytes: rejection must be chronology, not a stale hash.
    receipt_path.write_text(json.dumps(receipt))
    proof["terminal_sha256"] = file_hash(receipt_path)
    before = clock.path.read_bytes()
    with pytest.raises(ValueError):
        if validator == "close":
            api.validate_terminal_close(clock, allow_pending=True)
        elif validator == "projection":
            api.admitted_terminal_snapshot(clock, admission=proof)
        else:
            api.record_terminal_admission(result)
    assert not Path(proof["terminal_close"]["admission_path"]).exists()
    assert clock.path.read_bytes() == before


@pytest.mark.parametrize("observation,checked", [(100.0, 100.0), (101.0, 101.0),
                                               (101.0, 102.0), (104.0, 104.999)])
def test_admission_accepts_coherent_sequence_and_projects_persisted_proof(
    api, tmp_path, observation, checked,
):
    clock, receipt, result = terminal_sequence_fixture(api, tmp_path)
    proof = result["terminal_admission"]
    receipt["observed_before_receipt_write"] = observation
    receipt_path = Path(proof["terminal_close"]["receipt_path"])
    api.atomic_json(receipt_path, receipt)
    proof["terminal_sha256"] = file_hash(receipt_path)
    proof["parent_checked_monotonic"] = checked
    proof["observed_spend_at_return_check"] = checked - 100.0
    before = clock.path.read_bytes()
    assert api.validate_terminal_close(clock, allow_pending=True) == receipt
    projected = api.admitted_terminal_snapshot(clock, admission=proof)
    assert projected["status"] == "complete" and projected["spent_seconds"] == 5.0
    api.record_terminal_admission(result)
    assert api.admitted_terminal_snapshot(clock) == projected
    assert clock.path.read_bytes() == before
