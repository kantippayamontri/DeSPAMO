"""Timing admission catches wrong support, chronology, identities and unsafe writes."""

import copy
import importlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from despamo.appearance.provenance import digest, file_hash

ROOT = Path(__file__).resolve().parents[3]


def inputs(count=25, indices=None):
    indices = list(range(count)) if indices is None else indices
    frames = {
        "encoder_key": "dino",
        "clips": {
            "clip": {
                "frame_count": count,
                "source_indices": indices,
                "paths": [f"/frames/images{i:04}.png" for i in indices],
                "source_hash": "a" * 64,
                "feature_hash": "b" * 64,
            }
        },
    }
    record = {
        "clip_id": "clip",
        "split": "train",
        "path": "train/clip.npy",
        "length": max(1, (count - 16) // 8 + 1),
        "width": 1024,
        "dtype": "float32",
    }
    manifest = {
        "schema_version": 1,
        "encoder": "MCG-NJU/videomae-large",
        "expected_dim": 1024,
        "records": [record],
    }
    provenance = {
        "schema_version": 1,
        "status": "reconstructed_source_convention",
        "physical_split": "train",
        "spatial_crop_mode": "full",
        "window_size": 16,
        "stride": 8,
        "tail_policy": "full_windows_only",
        "short_clip_policy": "repeat_last_to_16",
        "frame_order": "lexicographic",
        "motion_row_order": "ascending_window_start",
        "input_sha256": {"frame_rows": digest(frames), "motion_manifest": digest(manifest)},
        "file_sha256": {
            key: "c" * 64
            for key in ("frame_rows", "motion_manifest", "annotation", "extractor", "helpers")
        },
        "evidence": [
            {
                "path": "/evidence/receipt.json",
                "sha256": "d" * 64,
                "claims": [
                    "historical_window_parameters",
                    "motion_row_order",
                    "source_frame_order",
                ],
            }
        ],
        "clips": {
            "clip": {
                "frame_count": count,
                "source_hash": "a" * 64,
                "motion_record_sha256": digest(record),
            }
        },
    }
    return frames, manifest, provenance


def module():
    return importlib.import_module("despamo.data.motion_timing")


def test_window_centres_and_uncovered_tail():
    frames, manifest, provenance = inputs()
    timing = module().build_motion_timing(frames, manifest, provenance)
    clip = module().validate_motion_timing(timing, frames, manifest)["clip"]
    assert clip.windows == ((0, 15), (8, 23))
    assert [(start + end) / 2 for start, end in clip.windows] == [7.5, 15.5]
    assert all(not start <= 24 <= end for start, end in clip.windows)
    assert clip.source_indices == tuple(range(25))
    with pytest.raises(FrozenInstanceError):
        clip.frame_count = 3


@pytest.mark.parametrize("count,end", [(1, 0), (10, 9), (16, 15)])
def test_short_clip_real_support(count, end):
    frames, manifest, provenance = inputs(count)
    timing = module().build_motion_timing(frames, manifest, provenance)
    assert module().validate_motion_timing(timing, frames, manifest)["clip"].windows == ((0, end),)


@pytest.mark.parametrize(
    "mutation",
    [
        "count",
        "order",
        "duplicate_indices",
        "unknown",
        "status_only",
        "stride",
        "row_order",
        "source_hash",
        "record_hash",
        "input_hash",
        "duplicate_record",
        "missing_clip",
        "empty",
        "bool_count",
        "paths_order",
    ],
)
def test_reject_count_order_and_provenance_mismatch(mutation):
    frames, manifest, provenance = inputs()
    row = frames["clips"]["clip"]
    if mutation == "count":
        manifest["records"][0]["length"] = 3
    elif mutation == "order":
        row["source_indices"][0:2] = [1, 0]
    elif mutation == "duplicate_indices":
        row["source_indices"][1] = 0
    elif mutation == "unknown":
        provenance["status"] = "unknown"
    elif mutation == "status_only":
        provenance["evidence"] = []
    elif mutation == "stride":
        provenance["stride"] = 16
    elif mutation == "row_order":
        provenance["motion_row_order"] = "unknown"
    elif mutation == "source_hash":
        provenance["clips"]["clip"]["source_hash"] = "e" * 64
    elif mutation == "record_hash":
        provenance["clips"]["clip"]["motion_record_sha256"] = "e" * 64
    elif mutation == "input_hash":
        provenance["input_sha256"]["frame_rows"] = "e" * 64
    elif mutation == "duplicate_record":
        manifest["records"].append(copy.deepcopy(manifest["records"][0]))
    elif mutation == "missing_clip":
        provenance["clips"] = {}
    elif mutation == "empty":
        row["source_indices"] = []
        row["paths"] = []
    elif mutation == "bool_count":
        row["frame_count"] = True
    elif mutation == "paths_order":
        row["paths"].reverse()
    # Rebind changed inputs: structural failures must not hide behind stale hashes.
    if mutation != "input_hash":
        provenance["input_sha256"] = {
            "frame_rows": digest(frames),
            "motion_manifest": digest(manifest),
        }
    if mutation in {"count", "duplicate_record"}:
        provenance["clips"]["clip"]["motion_record_sha256"] = digest(manifest["records"][0])
    with pytest.raises(ValueError):
        module().build_motion_timing(frames, manifest, provenance)


def test_runtime_requires_full_source_rows():
    frames, manifest, provenance = inputs(indices=[0, 3, 9, 20, 24])
    timing = module().build_motion_timing(frames, manifest, provenance)
    assert timing["clips"]["clip"]["source_indices"] == [0, 3, 9, 20, 24]
    with pytest.raises(ValueError, match="full source rows"):
        module().validate_motion_timing(timing, frames, manifest)
    frames, manifest, provenance = inputs()
    provenance["spatial_crop_mode"] = "random"
    timing = module().build_motion_timing(frames, manifest, provenance)
    with pytest.raises(ValueError, match="crop"):
        module().validate_motion_timing(timing, frames, manifest)


@pytest.mark.parametrize("field", ["schema_version", "clips", "provenance", "input_sha256"])
def test_reject_altered_timing(field):
    frames, manifest, provenance = inputs()
    timing = module().build_motion_timing(frames, manifest, provenance)
    if field == "schema_version":
        timing[field] = True
    elif field == "clips":
        timing[field]["clip"]["windows"][1][1] = 24
    elif field == "provenance":
        timing[field]["file_sha256"]["extractor"] = "unknown"
    else:
        timing[field]["motion_manifest"] = "e" * 64
    with pytest.raises(ValueError):
        module().validate_motion_timing(timing, frames, manifest)


def cli_fixture(tmp_path):
    frames, manifest, provenance = inputs()
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "frame_rows",
            "motion_manifest",
            "annotation",
            "extractor",
            "helpers",
            "provenance",
        )
    }
    paths["frame_rows"].write_text(json.dumps(frames))
    paths["motion_manifest"].write_text(json.dumps(manifest))
    for name in ("annotation", "extractor", "helpers"):
        paths[name].write_text(f"synthetic {name}\n")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"synthetic_test_receipt": true}')
    provenance["evidence"][0].update(path=str(evidence), sha256=file_hash(evidence))
    provenance["file_sha256"] = {
        name: file_hash(path) for name, path in paths.items() if name != "provenance"
    }
    paths["provenance"].write_text(json.dumps(provenance))
    return paths, tmp_path / "motion-timing.json", evidence


def run_cli(paths, output):
    command = [sys.executable, str(ROOT / "scripts/prepare_motion_timing.py")]
    for name, path in paths.items():
        command += ["--" + name.replace("_", "-"), str(path)]
    return subprocess.run(
        command + ["--output", str(output)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src:."},
        text=True,
        capture_output=True,
    )


def test_cli_writes_verified_artifact(tmp_path):
    paths, output, _ = cli_fixture(tmp_path)
    result = run_cli(paths, output)
    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text())
    assert artifact["provenance_sha256"] == file_hash(paths["provenance"])
    assert artifact["clips"]["clip"]["windows"] == [[0, 15], [8, 23]]
    assert artifact["provenance_status"] == "reconstructed_source_convention"
    assert artifact["coverage"] == {
        "real_frame_count": 25,
        "covered_frame_count": 24,
        "uncovered_frame_count": 1,
        "fraction": 0.96,
    }
    assert not list(tmp_path.glob(".partial-*"))


@pytest.mark.parametrize(
    "kind",
    [
        "alias",
        "hardlink",
        "symlink",
        "parent_symlink",
        "changed_source",
        "changed_evidence",
        "duplicate_json",
    ],
)
def test_cli_rejects_unsafe_output_and_stale_evidence(tmp_path, kind):
    paths, output, evidence = cli_fixture(tmp_path)
    if kind == "alias":
        output = paths["annotation"]
    elif kind == "hardlink":
        output.hardlink_to(paths["annotation"])
    elif kind == "symlink":
        output.symlink_to(tmp_path / "absent")
    elif kind == "parent_symlink":
        link = tmp_path / "link"
        link.symlink_to(tmp_path, target_is_directory=True)
        output = link / "motion-timing.json"
    elif kind == "changed_source":
        paths["helpers"].write_text("changed bytes")
    elif kind == "changed_evidence":
        evidence.write_text("changed evidence")
    elif kind == "duplicate_json":
        paths["provenance"].write_text('{"status":"unknown","status":"original_receipt"}')
    before = {path: path.read_bytes() for path in paths.values()}
    result = run_cli(paths, output)
    assert result.returncode != 0
    assert "motion timing rejected:" in result.stderr
    assert {path: path.read_bytes() for path in paths.values()} == before
    if kind not in {"alias", "hardlink", "symlink"}:
        assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "status_type",
        "partial_evidence",
        "hash_type",
        "provenance_version",
        "index_type",
        "index_bounds",
        "missing_paths",
        "duplicate_paths",
        "wrong_width",
        "wrong_encoder",
        "foreign_id",
        "missing_input",
    ],
)
def test_reject_malformed_schema(mutation):
    frames, manifest, provenance = inputs()
    row = frames["clips"]["clip"]
    if mutation == "status_type":
        provenance["status"] = []
    elif mutation == "partial_evidence":
        provenance["evidence"][0]["claims"] = ["historical_window_parameters"]
    elif mutation == "hash_type":
        provenance["file_sha256"]["extractor"] = 42
    elif mutation == "provenance_version":
        provenance["schema_version"] = True
    elif mutation == "index_type":
        row["source_indices"][0] = False
    elif mutation == "index_bounds":
        row["source_indices"][-1] = 25
    elif mutation == "missing_paths":
        row.pop("paths")
    elif mutation == "duplicate_paths":
        row["paths"][1] = row["paths"][0]
    elif mutation == "wrong_width":
        manifest["expected_dim"] = 768
        manifest["records"][0]["width"] = 768
    elif mutation == "wrong_encoder":
        manifest["encoder"] = "other"
    elif mutation == "foreign_id":
        manifest["records"][0].update(clip_id="foreign", path="train/foreign.npy")
    elif mutation == "missing_input":
        provenance["file_sha256"].pop("annotation")
    provenance["input_sha256"] = {"frame_rows": digest(frames), "motion_manifest": digest(manifest)}
    with pytest.raises(ValueError):
        module().build_motion_timing(frames, manifest, provenance)


def test_original_receipt_and_short_coverage():
    frames, manifest, provenance = inputs(10)
    provenance["status"] = "original_receipt"
    timing = module().build_motion_timing(frames, manifest, provenance)
    assert timing["provenance_status"] == "original_receipt"
    assert timing["coverage"] == {
        "real_frame_count": 10,
        "covered_frame_count": 10,
        "uncovered_frame_count": 0,
        "fraction": 1.0,
    }
    assert module().validate_motion_timing(timing, frames, manifest)["clip"].frame_count == 10


@pytest.mark.parametrize("kind", ["existing_output", "evidence_alias", "cropped"])
def test_cli_preserves_existing_outputs_and_rejects_cropped_rows(tmp_path, kind):
    paths, output, evidence = cli_fixture(tmp_path)
    if kind == "existing_output":
        output.write_text("user work")
    elif kind == "evidence_alias":
        output = evidence
    else:
        provenance = json.loads(paths["provenance"].read_text())
        provenance["spatial_crop_mode"] = "random"
        paths["provenance"].write_text(json.dumps(provenance))
    before = output.read_bytes() if output.exists() else None
    result = run_cli(paths, output)
    assert result.returncode == 2
    assert "motion timing rejected:" in result.stderr
    assert (output.read_bytes() if output.exists() else None) == before
