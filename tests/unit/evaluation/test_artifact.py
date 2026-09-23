import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch

from despamo.evaluation.artifact import collect_runtime_metadata, write_result_artifact


def test_artifact_records_items_and_metadata(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"
    write_result_artifact(
        output,
        ["clip-2", "clip-1"],
        ["second", "first"],
        ["zwei", "eins"],
        {"bleu4": 1.5},
        {"generation": "deterministic"},
    )

    payload = json.loads(output.read_text())
    assert set(payload) == {"items", "metrics", "metadata"}
    assert payload["items"] == [
        {"clip_id": "clip-2", "prediction": "second", "reference": "zwei"},
        {"clip_id": "clip-1", "prediction": "first", "reference": "eins"},
    ]
    assert payload["metrics"] == {"bleu4": 1.5}
    assert payload["metadata"] == {"generation": "deterministic"}
    assert output.read_text().endswith("\n")


@pytest.mark.parametrize(
    ("clip_ids", "predictions", "references"),
    [
        ([], [], []),
        (["a"], [], ["ref"]),
        (["a"], ["pred"], []),
        ([], ["pred"], ["ref"]),
    ],
)
def test_artifact_rejects_empty_or_misaligned_items(
    tmp_path: Path, clip_ids: list[str], predictions: list[str], references: list[str]
) -> None:
    output = tmp_path / "result.json"

    with pytest.raises(ValueError, match="equal non-zero length"):
        write_result_artifact(output, clip_ids, predictions, references, {}, {})

    assert not output.exists()


def test_artifact_rejects_duplicate_clip_ids(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    with pytest.raises(ValueError, match="duplicate clip_id"):
        write_result_artifact(output, ["same", "same"], ["one", "two"], ["a", "b"], {}, {})

    assert not output.exists()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_artifact_rejects_nonfinite_metrics(tmp_path: Path, nonfinite: float) -> None:
    output = tmp_path / "result.json"

    with pytest.raises(ValueError, match="finite"):
        write_result_artifact(output, ["clip"], ["pred"], ["ref"], {"bleu4": nonfinite}, {})

    assert not output.exists()


def test_runtime_metadata_hashes_inputs_and_detects_untracked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".git" / "info" / "exclude").write_text("/uv.lock\n")
    lock = repo / "uv.lock"
    spatial = tmp_path / "spatial.json"
    motion = tmp_path / "motion.json"
    lock.write_bytes(b"lock\x00\xff")
    spatial.write_text("spatial")
    motion.write_text("motion")
    monkeypatch.chdir(repo)

    metadata = collect_runtime_metadata(7, lock, spatial, motion)

    assert metadata == {
        "seed": 7,
        "git_revision": "NO_COMMIT",
        "git_dirty": False,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "package_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "spatial_manifest_sha256": hashlib.sha256(spatial.read_bytes()).hexdigest(),
        "motion_manifest_sha256": hashlib.sha256(motion.read_bytes()).hexdigest(),
    }

    (repo / "new-file.txt").write_text("untracked")

    assert collect_runtime_metadata(7, lock, spatial, motion)["git_dirty"] is True


def test_runtime_metadata_uses_lockfile_repo_when_called_from_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    lock = repo / "uv.lock"
    lock.write_text("lock")
    spatial = tmp_path / "spatial.json"
    motion = tmp_path / "motion.json"
    spatial.write_text("spatial")
    motion.write_text("motion")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    metadata = collect_runtime_metadata(7, lock, spatial, motion)

    assert metadata["git_revision"] == "NO_COMMIT"
    assert metadata["git_dirty"] is True
