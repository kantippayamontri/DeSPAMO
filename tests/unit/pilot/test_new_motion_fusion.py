"""New-motion admission uses synthetic CPU bytes, never real models or profiles."""

import copy
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from omegaconf import OmegaConf

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.data.phoenix14t import Phoenix14T
from despamo.data.signer_split import PhoenixTrainView
from scripts import run_fusion_study as controller
from scripts import train_fusion_study as runner
from tools.motion.package import publish_motion_package
from tools.motion.records import (
    MODEL,
    PROCESSOR,
    make_clip_receipt,
    make_motion_identity,
    source_identity,
    window_rows,
)


def helper():
    assert importlib.util.find_spec("despamo.data.new_motion"), "new-motion admission missing"
    return importlib.import_module("despamo.data.new_motion")


@pytest.fixture
def package_views(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    rows = []
    for index in range(25):
        path = sources / f"{index:02}.png"
        path.write_bytes(bytes([index]))
        rows.append({"index": index, "path": path.name, "sha256": file_hash(path)})
    spatial_root = tmp_path / "spatial"
    (spatial_root / "train").mkdir(parents=True)
    np.save(spatial_root / "train/c.npy", np.full((25, 2048), 3, dtype=np.float32))
    frames = {"encoder_key": runner.FEATURE_KEY, "clips": {"c": {
        "frame_count": 25, "paths": [row["path"] for row in rows],
        "source_indices": list(range(25)),
        "feature_hash": file_hash(spatial_root / "train/c.npy"),
        "source_hash": source_identity(tuple(Path(r["path"]) for r in rows),
                                       tuple(r["sha256"] for r in rows)),
    }}}
    frame_path = spatial_root / "complete/frame_rows.json"
    atomic_json(frame_path, frames)
    annotation = tmp_path / "annotation.npy"
    record = {"fileid": "c", "signer": "Signer01", "gloss": "G", "text": "text",
              "en_text": "english", "es_text": "spanish", "fr_text": "french"}
    np.save(annotation, np.array({0: record}, dtype=object))
    evidence = {}
    for name in ("extractor", "helpers", "lock"):
        path = tmp_path / f"{name}.txt"
        path.write_text(name)
        evidence[name] = {"path": str(path), "sha256": file_hash(path)}
    identity = make_motion_identity(
        {"frame_root": str(sources), "frame_rows": {"path": str(frame_path),
         "sha256": file_hash(frame_path)}, "annotation": {"path": str(annotation),
         "sha256": file_hash(annotation)}}, dict(MODEL), dict(PROCESSOR),
        {"lock": evidence.pop("lock")}, evidence, {"device": "cpu-fixture"},
    )
    root = tmp_path / f"motion-features-{digest(identity)}"
    (root / "train").mkdir(parents=True)
    np.save(root / "train/c.npy", np.full((2, 1024), 7, dtype=np.float32))
    receipt = make_clip_receipt("c", frames["clips"]["c"], rows, window_rows(25),
                                root / "train/c.npy", identity)
    atomic_json(root / "receipts/train/c.json", receipt)
    atomic_json(root / "generating-run.json", {
        "schema_version": 1, "origin": "new_extraction_v1", "featurekey": digest(identity),
        "authorization": {"extraction_cap_seconds": 18000, "training_cap_seconds": 64800,
                          "cap_transfer": False},
        "code": identity["code"], "clips": {"c": file_hash(root / "receipts/train/c.json")},
    })
    package = publish_motion_package(root, identity, frames, annotation)
    old_root = tmp_path / "old-motion"
    (old_root / "train").mkdir(parents=True)
    np.save(old_root / "train/c.npy", np.full((2, 1024), 2, dtype=np.float32))
    spatial = FeatureManifest(1, "spatial", 2048,
                              (FeatureRecord("c", "train", "train/c.npy", 25, 2048, "float32"),))
    motion = FeatureManifest(1, "motion", 1024,
                             (FeatureRecord("c", "train", "train/c.npy", 2, 1024, "float32"),))
    parent = Phoenix14T(annotation, "train", spatial_root, old_root, spatial, motion)
    return {"train": PhoenixTrainView(parent, ("c",))}, package, root, frames


def test_old_lineage_new_actual_views_separate(package_views):
    views, package, root, _ = package_views
    parent = views["train"].parent
    old_records = copy.deepcopy(parent.records)
    actual = helper().load_new_motion_views(views, package, root)
    assert actual["train"].parent is not parent
    assert parent.records == old_records
    assert parent.motion_root != root
    assert views["train"][0].motion.eq(2).all()
    assert actual["train"][0].motion.eq(7).all()
    assert actual["train"][0].spatial.eq(3).all()
    assert actual["train"].motion_root == root
    assert actual["train"].spatial_manifest is parent.spatial_manifest
    assert actual["train"].records == views["train"].records


@pytest.mark.parametrize("mutation", ["source", "window", "array", "marker", "descriptor",
                                     "annotation-record", "spatial-array"])
def test_reject_foreign_source_package(package_views, mutation):
    views, package, root, _ = package_views
    if mutation == "source":
        Path(package["identity"]["inputs"]["frame_root"]).joinpath("00.png").write_bytes(b"drift")
    elif mutation == "window":
        path = root / "receipts/train/c.json"
        receipt = read_json(path)
        receipt["windows"][0]["input_source_indices"][0:2] = [1, 0]
        atomic_json(path, receipt)
    elif mutation == "array":
        np.save(root / "train/c.npy", np.zeros((2, 1024), dtype=np.float32))
    elif mutation == "marker":
        path = Path(package["completion"])
        value = read_json(path)
        value["identity_hash"] = "0" * 64
        atomic_json(path, value)
    elif mutation == "descriptor":
        package["manifest_hash"] = "0" * 64
    elif mutation == "annotation-record":
        views["train"].parent.records[0]["text"] = "foreign text"
    else:
        np.save(views["train"].spatial_root / "train/c.npy",
                np.zeros((25, 2048), dtype=np.float32))
    with pytest.raises(ValueError):
        helper().load_new_motion_views(views, package, root)


def test_compact_reference_revalidates_source_drift_with_unchanged_marker(package_views):
    _, package, root, _ = package_views
    api = helper()
    reference = api.compact_motion_binding(package)
    assert "bindings" not in reference and "identity" not in reference
    assert len(json.dumps(reference)) < 2400
    assert api.revalidate_motion_binding(reference) == package
    marker = Path(package["completion"]).read_bytes()
    Path(package["identity"]["inputs"]["frame_root"]).joinpath("24.png").write_bytes(b"drift")
    assert Path(package["completion"]).read_bytes() == marker
    with pytest.raises(ValueError):
        api.revalidate_motion_binding(reference)


def test_legacy_motion_mode_unchanged(tmp_path):
    command = controller.stage_command("reference", 0, tmp_path / "profile",
                                       tmp_path / "timing", "preflight")
    assert "--motion-package" not in command
    assert runner.report_binding({"variant": "v", "protocol_hash": "p", "timing_hash": "t",
                                  "arm": "reference", "seed": 0})["run_policy"] == (
                                      "signer-pilot-time-aligned-fusion-v1")


@pytest.mark.parametrize("mode", ["initialize", "preflight", "smoke", "run"])
def test_stage_commands_and_parser_propagate_optin(package_views, mode, monkeypatch, tmp_path):
    _, package, root, _ = package_views
    profile = tmp_path / "profile"
    command = controller.stage_command("reference", 0, profile, Path(package["timing_path"]),
                                       mode, motion_package=root)
    assert command[command.index("--motion-package") + 1] == str(root)
    def inspect_args(args, **kwargs):
        assert args.motion_package == root
        assert args.motion_root != root  # Frozen old-source CLI proof stays separate.
        return {"status": "fixture", "run_key": "fixture"}
    monkeypatch.setattr(runner, "initialize_fusion", inspect_args)
    monkeypatch.setattr(runner, "_execute", inspect_args)
    monkeypatch.setattr(runner, "prepare_fusion", lambda args: (
        inspect_args(args), tmp_path / "output", {}, {}, OmegaConf.create({})))
    runner.main(command[2:])


def test_new_motion_identity_cannot_adopt_legacy_run(package_views):
    _, package, root, _ = package_views
    reference = helper().compact_motion_binding(package)
    identity = {"run_policy": "signer-pilot-time-aligned-fusion-new-motion-v1",
                "motion_package": reference, "timing_hash": package["timing_hash"],
                "config": {"data": {"motion_root": str(root),
                                    "motion_manifest": package["manifest_path"]}},
                "bound_files": {package["completion"]: package["completion_hash"]}}
    assert runner.motion_run_policy(identity) == identity["run_policy"]
    for bad in ({**identity, "run_policy": runner.RUN_POLICY},
                {**identity, "motion_package": None},
                {**identity, "config": {"data": {"motion_root": "old"}}}):
        with pytest.raises(ValueError):
            runner.motion_run_policy(bad)


def test_destination_cannot_alias_package_sources(package_views):
    _, package, root, _ = package_views
    args = SimpleNamespace(motion_package=root)
    for destination in (root / "runs", root.parent,
                        Path(package["identity"]["inputs"]["frame_root"]) / "run"):
        with pytest.raises(ValueError, match="destination|input|source"):
            runner._safe_destination(destination, args)


def test_new_phase_cannot_adopt_old_phase_status(tmp_path):
    with controller.FusionPhase(tmp_path, 0):
        pass
    path = tmp_path / "study-state.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="policy"):
        controller.FusionPhase(tmp_path, 0, motion_package=tmp_path / "package")
    assert path.read_bytes() == before


def test_new_source_closure_requires_pipeline_before_profile(tmp_path):
    with pytest.raises(ValueError, match="NEEDS_CONTEXT"):
        runner.fusion_code_hashes(tmp_path, new_motion=True)


@pytest.fixture
def admitted_inputs(package_views, tmp_path, monkeypatch):
    """Only production size, GPU facts and archived baseline replaced with CPU fixtures."""
    views, package, root, frames = package_views
    original_config = OmegaConf.create({
        "seed": 0, "model": {"warm_up_steps": 2000, "use_in_context": False},
        "data": {"batch_size": 4, "num_workers": 0,
                 "motion_root": str(views["train"].motion_root),
                 "motion_manifest": str(tmp_path / "old-manifest.json")},
        "trainer": {"max_steps": 16000, "accumulate_grad_batches": 1, "precision": "bf16",
                    "gradient_clip_val": 1.0}, "optimizer": {"weight_decay": .01},
    })
    parent = views["train"].parent
    parent.motion_manifest.save(tmp_path / "old-manifest.json")
    args = SimpleNamespace(
        arm="reference", seed=0, steps=16000, gpu_cap_seconds=10800,
        motion_package=root, motion_root=parent.motion_root,
        motion_manifest=tmp_path / "old-manifest.json", motion_timing=Path(package["timing_path"]),
        annotation=Path(package["annotation_path"]), adapted_root=parent.spatial_root,
        baseline_run=tmp_path / "baseline", protocol=tmp_path / "protocol.json",
        text_manifest=tmp_path / "text.json", output_base=tmp_path / "runs",
        initialization_profile=tmp_path / "profile.json", resume_checkpoint=None,
    )
    protocol = {"protocol_hash": "fixture-protocol", "split_hash": "fixture-split",
                "decoder": runner.DECODER, "split": {"groups": {"train": {"clip_ids": ["c"]},
                "dev": {"clip_ids": ["c"]}, "test": {"clip_ids": []}}}}
    for path in (args.protocol, args.text_manifest, args.adapted_root / "complete/manifest.json",
                 args.adapted_root / "complete/identity.json", args.baseline_run / "budget.json",
                 args.baseline_run / "run-status.json"):
        atomic_json(path, {"fixture": True})
    atomic_json(args.baseline_run / "selected-checkpoint.json", {"step": 8000})
    (args.baseline_run / "step-8000.ckpt").write_bytes(b"fixture checkpoint, never loaded")
    atomic_json(args.adapted_root / "receipts/train/c.json", frames["clips"]["c"])
    def baseline(namespace):
        assert namespace.motion_root == parent.motion_root
        assert namespace.motion_manifest == tmp_path / "old-manifest.json"
        assert namespace.projector == "linear"
        return protocol, views, original_config, {"encoder_key": runner.FEATURE_KEY}, "old", {}
    monkeypatch.setattr(runner, "_prepare_projector_inputs", baseline)
    monkeypatch.setattr(runner, "validate_fusion_splits", lambda p, v: None, raising=False)
    monkeypatch.setattr(runner, "validate_new_motion_production", lambda p, f: None)
    monkeypatch.setattr(runner, "fusion_code_hashes", lambda p, **kwargs: {
        "src/despamo/appearance/schema.py": "s" * 64})
    monkeypatch.setattr(runner, "hardware_identity", lambda: {"device": "CPU fixture"})
    return args, original_config, views, package, frames


def test_prepare_retains_baseline_proof_but_changes_only_actual_motion(admitted_inputs):
    args, original, views, package, _ = admitted_inputs
    _, actual_views, config, binding = runner._prepare_inputs(args)
    assert config is not original
    expected = copy.deepcopy(OmegaConf.to_container(original))
    assert isinstance(expected, dict)
    expected["data"].update(motion_root=str(args.motion_package),
                            motion_manifest=package["manifest_path"])
    assert OmegaConf.to_container(config) == expected
    assert views["train"][0].motion.eq(2).all()
    assert actual_views["train"][0].motion.eq(7).all()
    assert binding["baseline_motion"]["manifest_path"] == str(args.motion_manifest)
    assert binding["baseline_motion"]["manifest_hash"] == file_hash(args.motion_manifest)
    assert binding["run_policy"] == "signer-pilot-time-aligned-fusion-new-motion-v1"
    assert binding["motion_package"]["completion_hash"] == package["completion_hash"]
    assert not any("sources/" in path for path in binding["bound_files"])
    assert len(json.dumps(binding)) < 13000


def test_timing_copy_with_identical_bytes_still_conflicts(admitted_inputs, tmp_path):
    args, _, _, package, _ = admitted_inputs
    alternate = tmp_path / "timing-copy.json"
    alternate.write_bytes(Path(package["timing_path"]).read_bytes())
    args.motion_timing = alternate
    with pytest.raises(ValueError, match="authoritative timing"):
        runner._prepare_inputs(args)


def test_small_fixture_never_qualifies_as_production(package_views):
    _, package, _, frames = package_views
    with pytest.raises(ValueError, match="production"):
        runner.validate_new_motion_production(package, frames)


def test_profile_rejects_mixed_new_and_legacy_seeds(package_views):
    _, package, root, _ = package_views
    new = {"run_policy": runner.NEW_RUN_POLICY,
           "motion_package": helper().compact_motion_binding(package),
           "config": {"data": {"motion_root": str(root),
                               "motion_manifest": package["manifest_path"]}},
           "timing_hash": package["timing_hash"],
           "bound_files": {package["completion"]: package["completion_hash"]}}
    bindings = {"0": {"run_policy": runner.RUN_POLICY, "seed": 0}, "1": {**new, "seed": 1}}
    entries = {s: {**b, "initial_base_hash": runner.E1_SHARED_HASH, "common_plain_hash": "a" * 64}
               for s, b in bindings.items()}
    profile = {"profile_policy": runner.INITIALIZATION_POLICY, "seeds": entries}
    profile["profile_hash"] = digest(profile)
    with pytest.raises(ValueError, match="policy|profile|mode"):
        runner._validate_profile(profile, bindings)


def new_identity(package):
    return {"run_policy": runner.NEW_RUN_POLICY,
            "motion_package": helper().compact_motion_binding(package),
            "config": {"data": {"motion_root": package["root"],
                                "motion_manifest": package["manifest_path"]}},
            "timing_hash": package["timing_hash"],
            "code_hashes": {"fixture-source": "a" * 64},
            "hardware": {"device": "cpu-fixture"},
            "bound_files": {package["completion"]: package["completion_hash"]}}


@pytest.mark.parametrize("kind", ["selection", "recovery"])
def test_new_motion_large_identity_real_request_boundary(
    package_views, tmp_path, monkeypatch, kind,
):
    _, package, _, _ = package_views
    expected = new_identity(package)
    run = tmp_path / "owned"
    run.mkdir()
    checkpoint = run / "step-5500.ckpt"
    checkpoint.write_bytes(b"fixture")
    expected.update(run_key=run.name, seed=0, arm="reference")
    expected["bound_files"].update({
        f"/synthetic/receipts/train/{i:05d}-" + "x" * 140 + ".json": "a" * 64
        for i in range(7096)})
    assert len(json.dumps(expected).encode()) > 1_700_000
    monkeypatch.setattr(controller, "owned_status", lambda *args: {"status": "complete"})
    monkeypatch.setattr(controller, "require_production_telemetry", lambda *args: None)
    code = """
import sys
from pathlib import Path
from scripts import train_fusion_study as runner
from scripts.run_fusion_study import read_fusion_query
# Synthetic production-count/live-hardware dependencies; actual package graph closes.
runner.validate_new_motion_production=lambda *args:None
runner._check_live=lambda *args:None
print(read_fusion_query(Path(sys.argv[1]),sys.argv[2])['expected_identity_digest'])
"""

    def query(command):
        child = subprocess.run([sys.executable, "-c", code, *command[3:]],
                               capture_output=True, text=True, timeout=30, check=True)
        assert child.stdout.strip() == digest(expected)
        return ({"status": "complete", "fusion_telemetry": {}} if kind == "selection"
                else {"step": 5500, "checkpoint": str(checkpoint)})

    phase = SimpleNamespace(root=tmp_path, cpu_query=query)
    if kind == "selection":
        controller.completion_selection(phase, run, expected)
    else:
        assert controller.recovery_checkpoint(phase, run, expected) == checkpoint


def test_readonly_new_profile_six_distinct_keys_and_policy(admitted_inputs, monkeypatch):
    args, original, _, _, _ = admitted_inputs
    _, _, config, binding = runner._prepare_inputs(args)
    bindings = runner._bindings(config, binding, 10800)
    entries = {s: {**b, "initial_base_hash": runner.E1_SHARED_HASH,
                   "common_plain_hash": "a" * 64} for s, b in bindings.items()}
    profile = {"profile_policy": "fusion-common-initialization-new-motion-v1", "seeds": entries}
    profile["profile_hash"] = digest(profile)
    atomic_json(args.initialization_profile, profile)
    monkeypatch.setattr(runner, "_safe_destination", lambda p, a: None)
    monkeypatch.setattr(runner, "build_seeded_projector_model",
                        lambda *a: pytest.fail("read-only preflight constructed model"))
    keys = []
    for seed in (0, 1):
        args.seed = seed
        for arm in runner.FUSION_ARMS:
            args.arm = arm
            identity, output, _, _, actual = runner.prepare_fusion(args)
            assert identity["run_policy"] == runner.NEW_RUN_POLICY
            assert output.name.startswith(f"fusion-new-motion-{arm}-seed{seed}-16000-")
            assert actual.data.motion_root != original.data.motion_root
            keys.append(identity["run_key"])
    assert len(set(keys)) == 6
    assert not args.output_base.exists()
    profile["profile_policy"] = runner.INITIALIZATION_POLICY
    profile["profile_hash"] = digest({k: v for k, v in profile.items() if k != "profile_hash"})
    atomic_json(args.initialization_profile, profile)
    with pytest.raises(ValueError, match="profile"):
        runner.prepare_fusion(args)


@pytest.mark.parametrize("boundary", ["adoption", "recovery", "completion", "ledger",
                                     "selection-query", "recovery-query"])
def test_boundaries_reclose_graph_before_checkpoint_or_output(package_views, tmp_path,
                                                              monkeypatch, boundary):
    _, package, _, _ = package_views
    expected = new_identity(package)
    monkeypatch.setattr(runner, "validate_new_motion_production", lambda p, f: None)
    monkeypatch.setattr(runner, "fusion_code_hashes", lambda p, **kw: expected["code_hashes"])
    monkeypatch.setattr(runner, "hardware_identity", lambda: expected["hardware"])
    run = tmp_path / "missing-run"
    expected.update(run_key=run.name, seed=0, arm="reference")
    assert controller.owned_status(run, expected) == {}
    # Unused tail frame still belongs to the graph; marker remains byte-identical.
    Path(package["identity"]["inputs"]["frame_root"]).joinpath("24.png").write_bytes(b"drift")
    code = ""
    if boundary.endswith("query"):
        # Execute generated queries, proving full expected binding survives file transport.
        run.mkdir()
        atomic_json(run / "budget.json", {"identity": expected})
        atomic_json(run / "run-status.json", {})
        atomic_json(run / "selected-checkpoint.json", {})
        protocol = tmp_path / "query-protocol.json"
        atomic_json(protocol, {"split": {"groups": {"dev": {
            "clip_ids": [f"d{i}" for i in range(582)]}}}})
        monkeypatch.setitem(controller.INPUTS, "protocol", protocol)
        command = controller.fusion_query_command(SimpleNamespace(root=tmp_path),
            "selection" if boundary == "selection-query" else "recovery", run, expected,
            None if boundary == "selection-query" else run / "step-5500.ckpt")
        monkeypatch.setattr("sys.argv", ["query", *command[3:]])
        code = (controller.selection_query_code() if boundary == "selection-query"
                else controller.recovery_query_code())
    def call() -> None:
        if boundary == "adoption":
            controller.owned_status(run, expected)
        elif boundary == "recovery":
            controller.recovery_checkpoint(None, run, expected)
        elif boundary == "completion":
            controller.completion_selection(None, run, expected)
        elif boundary == "ledger":
            args = SimpleNamespace(motion_package=Path(package["root"]), resume_checkpoint=None)
            with runner.owned_fusion_ledger(args, expected, run):
                pytest.fail("invalid graph opened owned ledger")
        else:
            exec(code, {})
    with pytest.raises(ValueError, match="source|evidence|receipt|inventory"):
        call()
    if not boundary.endswith("query"):
        assert not run.exists()


def test_new_default_controller_root_selected_before_any_output(package_views, tmp_path,
                                                                monkeypatch):
    _, package, root, frames = package_views
    destination = tmp_path / "new-default"
    monkeypatch.setattr(controller, "NEW_DEFAULT_ROOT", destination)
    monkeypatch.setitem(controller.INPUTS, "adapted-root",
                        Path(package["spatial_frame_rows_path"]).parents[1])
    monkeypatch.setattr(runner, "validate_new_motion_production", lambda p, f: None)
    seen = []
    monkeypatch.setattr(runner, "_safe_destination", lambda p, a: seen.append(p))
    def closure(project, *, new_motion=False):
        assert new_motion is True
        raise ValueError("NEEDS_CONTEXT: fixture Task4 pipeline missing")
    monkeypatch.setattr(runner, "fusion_code_hashes", closure)
    with pytest.raises(ValueError, match="NEEDS_CONTEXT"):
        controller.main(["--motion-package", str(root), "--motion-timing", package["timing_path"],
                         "--authorize-fusion-study"])
    assert seen == [destination]
    assert not destination.exists()
    assert frames["encoder_key"] == runner.FEATURE_KEY


def test_source_closure_new_mode_additions_and_drift(tmp_path):
    for name in (*runner.SOURCE_FILES, *runner.NEW_SOURCE_FILES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source fixture")
    old = runner.fusion_code_hashes(tmp_path)
    new = runner.fusion_code_hashes(tmp_path, new_motion=True)
    assert "tools/motion/extractor.py" not in old
    assert "tools/motion/extractor.py" in new
    assert "tools/motion/uv.lock" in new
    assert "scripts/run_motion_fusion_pipeline.py" in new
    (tmp_path / "tools/motion/records.py").write_text("source drift")
    assert runner.fusion_code_hashes(tmp_path, new_motion=True) != new


def test_reference_serialization_does_not_scale_with_827354_source_entries(package_views):
    _, package, _, _ = package_views
    reference = helper().compact_motion_binding(package)
    inflated = {**package, "bindings": dict.fromkeys(range(827354), "0" * 64),
                "clip_ids": list(range(7096))}
    # These intentionally false inventories are never admitted. Only reference size tested.
    assert json.dumps(helper().compact_motion_binding(inflated)) == json.dumps(reference)
    assert len(json.dumps(reference).encode()) < 2400


def test_summary_cannot_label_new_motion_as_legacy_or_mix_modes(package_views):
    _, package, _, _ = package_views
    rows = importlib.import_module("test_fusion_controller").rows()
    for row in rows:
        row["identity"] = {**new_identity(package), "arm": row["arm"], "seed": row["seed"]}
    summary = controller.aggregate_fusion_results(rows)
    assert summary["study_policy"] == "time-aligned-fusion-new-motion-controller-v1"
    assert summary["motion_package"] == helper().compact_motion_binding(package)
    rows[-1]["identity"] = {"run_policy": runner.RUN_POLICY}
    with pytest.raises(ValueError, match="mode|policy|motion"):
        controller.aggregate_fusion_results(rows)


def test_destination_rejects_actual_spatial_and_model_input_roots(package_views):
    _, package, root, _ = package_views
    identity_path = root / "complete/identity.json"
    identity = read_json(identity_path)
    # This guard also applies pre-admission; it must protect declared input roots.
    model_input = root.parent / "model-snapshot/weights.bin"
    model_input.parent.mkdir()
    model_input.write_bytes(b"synthetic weights")
    identity["inputs"]["annotation"]["path"] = str(root.parent / "annotations/annotation.npy")
    identity["environment"]["lock"]["path"] = str(root.parent / "locks/uv.lock")
    for name, item in identity["code"].items():
        item["path"] = str(root.parent / "code" / f"{name}.py")
    identity["environment"]["model_files"] = [{"path": str(model_input),
                                               "sha256": file_hash(model_input)}]
    atomic_json(identity_path, identity)
    args = SimpleNamespace(motion_package=root)
    for destination in (Path(package["spatial_frame_rows_path"]).parents[1] / "train/run",
                        model_input.parent / "runs"):
        with pytest.raises(ValueError, match="destination|input|source"):
            runner._safe_destination(destination, args)


@pytest.fixture
def production_facts(tmp_path):
    """Count/source gate only. Not a published package or proof of real generation."""
    clips, records = {}, {}
    for i in range(7096):
        clip = f"c{i:04}"
        count = 8 * (14 + (i < 984)) + 3 + (i < 3442)
        clips[clip] = {"frame_count": count}
        signer = f"Signer{[1, 2, 4, 5, 6, 8, 9][i % 7]:02}" if i < 5746 else (
            "Signer03" if i < 6328 else "Signer07")
        records[i] = {"fileid": clip, "signer": signer}
    annotation = tmp_path / "production-count-fixture.npy"
    np.save(annotation, np.array(records, dtype=object))
    names = [name for name in runner.NEW_SOURCE_FILES if name not in (
        "scripts/run_motion_fusion_pipeline.py", "src/despamo/data/new_motion.py",
        "tools/motion/uv.lock")]
    names.extend(("src/despamo/appearance/provenance.py", "src/despamo/appearance/schema.py",
                  "src/despamo/data/manifest.py", "src/despamo/data/motion_timing.py"))
    code = {"extractor" if i == 0 else "helpers" if i == 1 else f"closure_{i}": {
        "path": str(runner.PROJECT / name), "sha256": file_hash(runner.PROJECT / name)}
        for i, name in enumerate(names)}
    lock = runner.PROJECT / "tools/motion/uv.lock"
    model = tmp_path / "model.bin"
    model.write_bytes(b"synthetic model evidence")
    package = {"annotation_path": str(annotation), "clip_ids": sorted(clips),
               "identity": {"code": code, "environment": {
                   "lock": {"path": str(lock), "sha256": file_hash(lock)},
                   "model_files": [{"path": str(model), "sha256": file_hash(model)}]}}}
    return package, {"encoder_key": runner.FEATURE_KEY, "clips": clips}


@pytest.mark.parametrize("mutation", ["missing-model", "actor-label", "source-count",
                                     "window-count", "lock", "foreign-code",
                                     "missing-common-source"])
def test_production_cannot_waive_model_actor_or_count_proof(production_facts, mutation):
    package, frames = production_facts
    runner.validate_new_motion_production(package, frames)
    if mutation == "missing-model":
        package["identity"]["environment"].pop("model_files")
    elif mutation == "actor-label":
        code = package["identity"]["code"]
        code["extractor"], code["helpers"] = code["helpers"], code["extractor"]
    elif mutation == "source-count":
        frames["clips"]["c0000"]["frame_count"] += 1
    elif mutation == "window-count":
        frames["clips"]["c0000"]["frame_count"] -= 8
        frames["clips"]["c3442"]["frame_count"] += 4
        frames["clips"]["c3443"]["frame_count"] += 4
    elif mutation == "lock":
        package["identity"]["environment"]["lock"]["sha256"] = "0" * 64
    elif mutation == "foreign-code":
        package["identity"]["code"]["extractor"]["sha256"] = "0" * 64
    else:
        code = package["identity"]["code"]
        code.pop(next(k for k, v in code.items() if v["path"].endswith("/provenance.py")))
    with pytest.raises(ValueError, match="production|generator"):
        runner.validate_new_motion_production(package, frames)


def test_new_phase_bind_cannot_adopt_legacy_even_when_run_absent(package_views, tmp_path):
    _, package, root, _ = package_views
    # Admission tested separately; this isolates phase's policy guard without publishing state.
    phase = object.__new__(controller.FusionPhase)
    phase.fusion_seed = 0
    phase.motion_package = root
    phase.policy = controller.NEW_POLICY
    with pytest.raises(ValueError, match="motion|policy"):
        phase.bind_run("reference", tmp_path / "absent", {
            "arm": "reference", "seed": 0, "run_policy": runner.RUN_POLICY})
    foreign = new_identity(package)
    foreign.update(arm="reference", seed=0)
    phase.motion_package = tmp_path / "foreign-package"
    with pytest.raises(ValueError, match="motion|policy"):
        phase.bind_run("reference", tmp_path / "absent", foreign)


def test_closed_graph_still_rejects_training_actor_source_drift(package_views, monkeypatch):
    _, package, _, _ = package_views
    expected = new_identity(package)
    monkeypatch.setattr(runner, "validate_new_motion_production", lambda p, f: None)
    monkeypatch.setattr(runner, "fusion_code_hashes", lambda p, **kw: {"fixture-source": "b" * 64})
    monkeypatch.setattr(runner, "hardware_identity", lambda: expected["hardware"])
    with pytest.raises(ValueError, match="source/hardware"):
        runner.validate_motion_graph(expected)


def test_initialize_rechecks_changed_package_before_first_model(package_views, tmp_path,
                                                               monkeypatch):
    views, package, root, _ = package_views
    binding = new_identity(package)
    args = SimpleNamespace(arm="reference", seed=0, steps=16000, gpu_cap_seconds=10800,
                           motion_package=root, initialization_profile=tmp_path / "profile.json",
                           resume_checkpoint=None)
    config = OmegaConf.create(binding["config"])
    config.seed = 0
    monkeypatch.setattr(runner, "_prepare_inputs", lambda a: ({}, views, config, binding))
    monkeypatch.setattr(runner, "_safe_destination", lambda p, a: None)
    monkeypatch.setattr(runner, "build_seeded_projector_model",
                        lambda *a: pytest.fail("changed package reached model construction"))
    Path(package["identity"]["inputs"]["frame_root"]).joinpath("24.png").write_bytes(b"drift")
    with pytest.raises(ValueError, match="source|receipt|inventory"):
        runner.initialize_fusion(args)
    assert not args.initialization_profile.exists()


@pytest.mark.parametrize("seed", [0, 1])
def test_phase_propagates_package_to_every_launch_and_preflight(admitted_inputs, monkeypatch, seed):
    args, _, _, package, _ = admitted_inputs
    _, _, config, binding = runner._prepare_inputs(args)
    entries = {s: {**b, "initial_base_hash": runner.E1_SHARED_HASH, "common_plain_hash": "a" * 64}
               for s, b in runner._bindings(config, binding, 10800).items()}
    profile = {"profile_policy": runner.NEW_INITIALIZATION_POLICY, "seeds": entries}
    profile["profile_hash"] = digest(profile)
    atomic_json(args.initialization_profile, profile)
    monkeypatch.setattr(runner, "_safe_destination", lambda p, a: None)
    monkeypatch.setattr(controller, "validate_completion", lambda *a: False)
    monkeypatch.setattr(controller, "recovery_checkpoint", lambda *a: None)
    monkeypatch.setattr(controller, "require_study_storage", lambda *a: {"fixture": True})
    monkeypatch.setattr(controller, "collect_result", lambda *a: {"checkpoint_sizes_bytes": {}})
    launched = []
    phase = SimpleNamespace(
        fusion_seed=seed, motion_package=args.motion_package, root=args.output_base.parent,
        state={"runs": {}, "spent_seconds": 0}, save=lambda: None)
    def query(command):
        assert command[command.index("--motion-package") + 1] == package["root"]
        assert command[command.index("--mode") + 1] == "preflight"
        arm = command[command.index("--arm") + 1]
        args.arm, args.seed = arm, seed
        identity, output, *_ = runner.prepare_fusion(args)
        launched.append((arm, "preflight"))
        return {"status": "ready", "identity": identity, "output": str(output)}
    def stage(name, command, cap, **kwargs):
        assert cap == 10800
        assert command[command.index("--motion-package") + 1] == package["root"]
        assert command[command.index("--seed") + 1] == str(seed)
        launched.append((command[command.index("--arm") + 1],
                         command[command.index("--mode") + 1]))
    phase.cpu_query = query
    phase.run_stage = stage
    phase.bind_run = lambda arm, run, identity: controller.FusionPhase.bind_run(
        cast(controller.FusionPhase, phase), arm, run, identity)
    controller.run_phase(cast(controller.FusionPhase, phase), args.initialization_profile,
                         args.motion_timing, {})
    expected = ([("reference", "initialize")] if seed == 0 else []) + [
        (arm, mode) for arm in ("reference", "aligned_fixed", "aligned_learned")
        for mode in ("preflight", "smoke", "run")]
    assert launched == expected
    assert phase.state["status"] == "complete"


def test_final_summary_revalidates_full_graph_even_with_unchanged_marker(package_views, tmp_path,
                                                                      monkeypatch):
    _, package, _, _ = package_views
    profile, protocol = tmp_path / "profile.json", tmp_path / "protocol.json"
    atomic_json(profile, {"fixture": True})
    atomic_json(protocol, {"fixture": True})
    monkeypatch.setitem(controller.INPUTS, "protocol", protocol)
    sources = {"scripts/run_fusion_study.py": file_hash(Path(controller.__file__))}
    monkeypatch.setattr(controller, "controller_sources", lambda **kw: sources)
    monkeypatch.setattr(runner, "validate_new_motion_production", lambda p, f: None)
    expected = new_identity(package)
    monkeypatch.setattr(runner, "fusion_code_hashes", lambda p, **kw: expected["code_hashes"])
    monkeypatch.setattr(runner, "hardware_identity", lambda: expected["hardware"])
    run = tmp_path / "runs/owned-run"
    run.mkdir(parents=True)
    expected.update(run_key=run.name, seed=0, arm="reference",
                    initialization_profile_path=str(profile),
                    initialization_profile_file_hash=file_hash(profile))
    metadata = {"motion_package": expected["motion_package"], "timing_hash": package["timing_hash"],
                "controller_hash": file_hash(Path(controller.__file__)),
                "controller_sources": sources}
    row = {"run": str(run), "identity": expected, "run_key": run.name,
           "seed": 0, "arm": "reference"}
    marker = Path(package["completion"]).read_bytes()
    Path(package["identity"]["inputs"]["frame_root"]).joinpath("24.png").write_bytes(b"drift")
    with pytest.raises(ValueError, match="source|receipt|inventory"):
        controller.validate_summary_sources([row], tmp_path, profile, Path(package["timing_path"]),
                                             metadata)
    assert Path(package["completion"]).read_bytes() == marker


@pytest.mark.parametrize("mutation", ["config-type", "data-type", "files-type", "schema-bool",
                                     "hash-type", "policy-type"])
def test_new_mode_type_mismatches_fail_closed(package_views, mutation):
    _, package, _, _ = package_views
    identity = new_identity(package)
    if mutation == "config-type":
        identity["config"] = None
    elif mutation == "data-type":
        identity["config"]["data"] = None
    elif mutation == "files-type":
        identity["bound_files"] = None
    elif mutation == "schema-bool":
        identity["motion_package"]["schema_version"] = True
    elif mutation == "hash-type":
        identity["motion_package"]["identity_hash"] = False
    else:
        identity["run_policy"] = True
    with pytest.raises(ValueError):
        runner.motion_run_policy(identity)
