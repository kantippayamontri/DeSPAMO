import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from despamo.comparison import FLAN_SHA, comparison_code_sha256, summarize


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_run(root, source, seed, ids=("a", "b"), bleu4=10.0, rouge=0.3):
    protocol = {
        "enabled": True,
        "smoke": False,
        "clip_root": "clip",
        "clip_manifest": "clip.json",
        "dino_root": "dino",
        "dino_manifest": "dino/complete/manifest.json",
    }
    config = {
        "seed": seed,
        "comparison": protocol,
        "data": {
            "spatial_root": source,
            "spatial_manifest": (
                "clip.json" if source == "clip" else "dino/complete/manifest.json"
            ),
            "motion_manifest": "motion.json",
            "batch_size": 2,
        },
        "trainer": {
            "max_steps": 1000,
            "max_epochs": -1,
            "accumulate_grad_batches": 2,
            "precision": "bf16",
            "default_root_dir": str(root / source / f"seed-{seed}"),
        },
        "evaluation": {"generation": "deterministic", "beam_size": 5},
        "optimizer": {"learning_rate": 6e-4, "weight_decay": 0.01},
        "model": {
            "vt_pooling": "masked_mean",
            "spatial_crop_mode": "full",
            "revision": FLAN_SHA,
            "name": "google/flan-t5-xl",
            "prompt": "Translate the given sentence into {}.",
            "vt_weight": 1.0,
            "use_in_context": True,
            "num_in_context": 3,
            "warm_up_steps": 0,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "spatial_dim": 2048,
            "motion_dim": 1024,
        },
    }
    annotations = {split: _hash(f"{split}-annotations") for split in ("train", "dev", "test")}
    metadata = {
        "seed": seed,
        "generation": "deterministic",
        "baseline_accepted": False,
        "spatial_crop_mode": "full",
        "checkpoint_global_step": 1000,
        "comparison_source": source,
        "checkpoint_sha256": _hash(f"{source}-{seed}"),
        "spatial_manifest_sha256": _hash(source),
        "motion_manifest_sha256": _hash("motion"),
        "package_lock_sha256": _hash("lock"),
        "comparison_code_sha256": _hash("code"),
        "dino_frame_rows_sha256": _hash("dino-frame-rows"),
        "git_revision": "NO_COMMIT",
        "git_dirty": True,
        "annotation_sha256": dict(annotations),
        "checkpoint_metadata": {
            "config": config,
            "seed": seed,
            "spatial_manifest_sha256": _hash(source),
            "motion_manifest_sha256": _hash("motion"),
            "package_lock_sha256": _hash("lock"),
            "comparison_code_sha256": _hash("code"),
            "dino_frame_rows_sha256": _hash("dino-frame-rows"),
            "git_revision": "NO_COMMIT",
            "git_dirty": True,
            "annotation_sha256": dict(annotations),
            "resume_reproducibility": "fresh",
            "model_source": {
                "identifier": "google/flan-t5-xl",
                "tuning_type": "lora",
                "revision_status": "resolved",
                "revision": FLAN_SHA,
            },
        },
        "config": config,
    }
    path = root / source / f"seed-{seed}" / "test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "metrics": {"bleu4": bleu4, "rougeL_f1": rouge},
                "items": [
                    {"clip_id": clip_id, "reference": "same text", "prediction": "prediction"}
                    for clip_id in ids
                ],
            }
        )
    )
    return path


@pytest.fixture
def results(tmp_path, monkeypatch):
    import despamo.comparison as comparison

    monkeypatch.setattr(comparison, "EXPECTED", {"train": 1, "dev": 1, "test": 2})
    for seed in (0, 1, 2):
        write_run(tmp_path, "clip", seed, bleu4=10.0 + seed, rouge=0.3 + seed * 0.01)
        write_run(tmp_path, "dino", seed, bleu4=12.0 + seed, rouge=0.4 + seed * 0.01)
    return tmp_path


def change_run(path, update):
    payload = json.loads(path.read_text())
    update(payload)
    path.write_text(json.dumps(payload))


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("compare_encoders")


def test_pairwise_scores_use_three_matching_seeds_and_ordered_ids(results):
    report = summarize(results)
    assert report["paired_delta"]["bleu4"] == {"mean": 2.0, "std": 0.0}
    assert report["paired_delta"]["rougeL_f1"]["mean"] == pytest.approx(0.1)
    assert report["sources"]["clip"]["bleu4"] == {"mean": 11.0, "std": 1.0}
    assert report["annotation_sha256"]["train"] == _hash("train-annotations")
    assert len(report["runs"]) == 6
    assert all(run["test_items"] == 2 for run in report["runs"])
    json.dumps(report, allow_nan=False)
    (results / "dino/seed-1/test.json").unlink()
    with pytest.raises(FileNotFoundError):
        summarize(results)


def test_report_rejects_consistently_wrong_optimizer_protocol(results):
    for seed in (0, 1, 2):
        for source in ("clip", "dino"):
            path = results / source / f"seed-{seed}/test.json"

            def change(payload):
                for metadata in (payload["metadata"], payload["metadata"]["checkpoint_metadata"]):
                    metadata["config"]["optimizer"] = {
                        "learning_rate": 0.7,
                        "weight_decay": 0.01,
                    }

            change_run(path, change)
    with pytest.raises(ValueError, match="protocol"):
        summarize(results)


def test_report_rejects_dino_content_change_across_runs(results):
    path = results / "dino/seed-2/test.json"

    def change(payload):
        digest = _hash("changed feature contents")
        payload["metadata"]["dino_frame_rows_sha256"] = digest
        payload["metadata"]["checkpoint_metadata"]["dino_frame_rows_sha256"] = digest

    change_run(path, change)
    with pytest.raises(ValueError, match="DINO.*drift"):
        summarize(results)


@pytest.mark.parametrize("ids", [("b", "a"), ("a", "a"), ("a",)])
def test_rejects_reordered_duplicate_or_incomplete_ids(results, ids):
    write_run(results, "dino", 1, ids=ids)
    with pytest.raises(ValueError, match="(paired test IDs/references|unique full split IDs)"):
        summarize(results)


def test_rejects_unpaired_references(results):
    path = results / "dino/seed-0/test.json"
    change_run(path, lambda payload: payload["items"][0].update(reference="different"))
    with pytest.raises(ValueError, match="paired test IDs/references"):
        summarize(results)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("metadata", "checkpoint_global_step", 999, "1000 steps"),
        ("metadata", "baseline_accepted", True, "deterministic 1000 steps"),
        ("metadata", "generation", "upstream", "deterministic 1000 steps"),
        ("checkpoint_metadata", "resume_reproducibility", "optimizer_state_only", "provenance"),
        ("config", "seed", 2, "protocol mismatch"),
    ],
)
def test_rejects_unfinished_accepted_resumed_or_wrong_seed(results, section, key, value, message):
    path = results / "dino/seed-0/test.json"

    def change(payload):
        target = payload["metadata"] if section == "metadata" else payload["metadata"][section]
        target[key] = value

    change_run(path, change)
    with pytest.raises(ValueError, match=message):
        summarize(results)


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_rejects_text_only_annotation_drift_across_six_runs(results, split):
    path = results / "dino/seed-2/test.json"

    def change(payload):
        for metadata in (payload["metadata"], payload["metadata"]["checkpoint_metadata"]):
            metadata["annotation_sha256"][split] = _hash(f"same IDs, changed {split} text")

    change_run(path, change)
    with pytest.raises(ValueError, match=f"{split} annotation SHA256 drift"):
        summarize(results)


def test_rejects_annotation_mismatch_with_checkpoint(results):
    path = results / "dino/seed-2/test.json"
    change_run(
        path,
        lambda payload: payload["metadata"]["annotation_sha256"].update(train=_hash("changed")),
    )
    with pytest.raises(ValueError, match="checkpoint annotation SHA256 mismatch"):
        summarize(results)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("spatial_manifest_sha256", "manifest drift"),
        ("motion_manifest_sha256", "motion manifest drift"),
        ("package_lock_sha256", "baseline lock drift"),
    ],
)
def test_rejects_manifest_or_lock_drift(results, field, message):
    path = results / "dino/seed-2/test.json"

    def change(payload):
        payload["metadata"][field] = _hash("changed")
        payload["metadata"]["checkpoint_metadata"][field] = _hash("changed")

    change_run(path, change)
    with pytest.raises(ValueError, match=message):
        summarize(results)


@pytest.mark.parametrize("field", ["comparison_code_sha256", "git_revision", "git_dirty"])
def test_rejects_single_run_code_or_git_provenance_drift(results, field):
    path = results / "dino/seed-2/test.json"
    changed = {
        "comparison_code_sha256": _hash("changed"),
        "git_revision": "b" * 40,
        "git_dirty": False,
    }[field]

    def change(payload):
        payload["metadata"][field] = changed
        if field in ("comparison_code_sha256", "git_revision", "git_dirty"):
            payload["metadata"]["checkpoint_metadata"][field] = changed

    change_run(path, change)
    with pytest.raises(ValueError, match=f"{field} drift"):
        summarize(results)


def test_rejects_code_sha_not_matching_checkpoint(results):
    path = results / "dino/seed-2/test.json"
    change_run(path, lambda payload: payload["metadata"].update(comparison_code_sha256=_hash("x")))
    with pytest.raises(ValueError, match="checkpoint code SHA256 mismatch"):
        summarize(results)


@pytest.mark.parametrize("field", ["git_revision", "git_dirty"])
def test_rejects_git_provenance_not_matching_checkpoint(results, field):
    path = results / "dino/seed-2/test.json"
    change_run(
        path,
        lambda payload: payload["metadata"]["checkpoint_metadata"].update(
            {field: "b" * 40 if field == "git_revision" else False}
        ),
    )
    with pytest.raises(ValueError, match="checkpoint git provenance mismatch"):
        summarize(results)


def test_comparison_code_digest_uses_only_sorted_code_and_config_bytes(tmp_path, monkeypatch):
    allowed = {
        "src/despamo/a.py": b"a",
        "src/despamo/sub/b.py": b"b",
        "scripts/train.py": b"train",
        "configs/experiment/c.yaml": b"config",
        "pyproject.toml": b"project",
        "uv.lock": b"lock",
    }
    for name, contents in reversed(list(allowed.items())):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    excluded = tmp_path / "research" / "paper.pdf"
    excluded.parent.mkdir()
    excluded.write_bytes(b"not comparison code")
    original_read = Path.read_bytes

    def checked_read(path):
        assert path.relative_to(tmp_path).as_posix() in allowed
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", checked_read)
    initial = comparison_code_sha256(tmp_path)
    assert len(initial) == 64
    assert comparison_code_sha256(tmp_path) == initial
    (tmp_path / "src/despamo/a.py").write_bytes(b"changed uncommitted code")
    assert comparison_code_sha256(tmp_path) != initial


@pytest.mark.parametrize("field", ["root", "manifest"])
def test_rejects_both_sources_bound_to_same_spatial_input(results, field):
    for source in ("clip", "dino"):
        for seed in (0, 1, 2):
            path = results / source / f"seed-{seed}" / "test.json"

            def change(payload, source=source):
                for meta in (payload["metadata"], payload["metadata"]["checkpoint_metadata"]):
                    config = meta["config"]
                    config["comparison"][f"dino_{field}"] = config["comparison"][f"clip_{field}"]
                    if source == "dino":
                        config["data"][f"spatial_{field}"] = config["comparison"][f"clip_{field}"]

            change_run(path, change)
    with pytest.raises(ValueError, match="distinct spatial"):
        summarize(results)


def test_rejects_equal_spatial_manifest_bytes_despite_distinct_labels(results):
    for seed in (0, 1, 2):
        path = results / "dino" / f"seed-{seed}" / "test.json"

        def change(payload):
            for meta in (payload["metadata"], payload["metadata"]["checkpoint_metadata"]):
                meta["spatial_manifest_sha256"] = _hash("clip")

        change_run(path, change)
    with pytest.raises(ValueError, match="distinct spatial manifest SHA256"):
        summarize(results)


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("bleu4", -0.1),
        ("bleu4", 100.1),
        ("rougeL_f1", -0.1),
        ("rougeL_f1", 1.1),
        ("bleu4", True),
        ("rougeL_f1", float("nan")),
    ],
)
def test_rejects_invalid_metric_ranges(results, metric, value):
    path = results / "dino/seed-2/test.json"
    change_run(path, lambda payload: payload["metrics"].update({metric: value}))
    with pytest.raises(ValueError, match=f"invalid {metric}"):
        summarize(results)


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_sha256",
        "spatial_manifest_sha256",
        "motion_manifest_sha256",
        "package_lock_sha256",
    ],
)
def test_rejects_integer_digest_fields(results, field):
    path = results / "dino/seed-2/test.json"

    def change(payload):
        payload["metadata"][field] = int("1" * 64)
        if field != "checkpoint_sha256":
            payload["metadata"]["checkpoint_metadata"][field] = int("1" * 64)

    change_run(path, change)
    with pytest.raises(ValueError, match=f"invalid {field}"):
        summarize(results)


def test_rejects_flan_revision_drift(results):
    path = results / "dino/seed-2/test.json"
    change_run(
        path,
        lambda payload: payload["metadata"]["checkpoint_metadata"]["model_source"].update(
            revision="c" * 40
        ),
    )
    with pytest.raises(ValueError, match="Flan revision"):
        summarize(results)


@pytest.mark.parametrize("field", ["beam_size", "generation"])
def test_rejects_protocol_drift(results, field):
    path = results / "dino/seed-2/test.json"

    def change(payload):
        config = payload["metadata"]["config"]
        config["evaluation"][field] = 4 if field == "beam_size" else "upstream"

    change_run(path, change)
    with pytest.raises(ValueError, match="protocol mismatch"):
        summarize(results)


def test_rejects_swapped_source_or_run_root(results):
    path = results / "dino/seed-2/test.json"
    change_run(path, lambda payload: payload["metadata"].update(comparison_source="clip"))
    with pytest.raises(ValueError, match="deterministic 1000 steps"):
        summarize(results)
    write_run(results, "dino", 2)
    change_run(
        path,
        lambda payload: payload["metadata"]["config"]["trainer"].update(
            default_root_dir="wrong-directory"
        ),
    )
    with pytest.raises(ValueError, match="run root"):
        summarize(results)


def test_rejects_extra_run(results):
    write_run(results, "clip", 3)
    with pytest.raises(ValueError, match="exactly six"):
        summarize(results)


def test_reports_negative_paired_delta_without_improvement_threshold(results):
    for seed in (0, 1, 2):
        path = results / "dino" / f"seed-{seed}" / "test.json"
        change_run(path, lambda payload, score=8.0 + seed: payload["metrics"].update(bleu4=score))
    assert summarize(results)["paired_delta"]["bleu4"] == {"mean": -2.0, "std": 0.0}


def test_rejects_nonfinite_paired_delta_even_with_finite_run_scores(results):
    for source, value in (("clip", -1e308), ("dino", 1e308)):
        for seed in (0, 1, 2):
            path = results / source / f"seed-{seed}" / "test.json"
            change_run(path, lambda payload, score=value: payload["metrics"].update(bleu4=score))
    with pytest.raises(ValueError, match="invalid.*bleu4"):
        summarize(results)


def test_report_cli_writes_json_to_external_path_and_never_overwrites(results, monkeypatch, cli):
    output = results.parent / "reports" / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_encoders.py", "report", "--results-root", str(results), "--output", str(output)],
    )
    cli.main()
    assert json.loads(output.read_text())["paired_delta"]["bleu4"]["mean"] == 2.0
    with pytest.raises(ValueError, match="report exists"):
        cli.main()
    assert json.loads(output.read_text())["paired_delta"]["bleu4"]["mean"] == 2.0


def test_report_cli_rejects_individually_ignored_file_in_unignored_parent(
    results, tmp_path, monkeypatch, cli
):
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("configs/local.yaml\nartifacts/\n")
    configs = repo / "configs"
    configs.mkdir()
    monkeypatch.setattr(cli, "ROOT", repo)
    output = configs / "local.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_encoders.py", "report", "--results-root", str(results), "--output", str(output)],
    )
    with pytest.raises(ValueError, match="ignored by Git"):
        cli.main()
    assert list(configs.iterdir()) == []

    ignored_dir_output = repo / "artifacts" / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_encoders.py",
            "report",
            "--results-root",
            str(results),
            "--output",
            str(ignored_dir_output),
        ],
    )
    cli.main()
    assert json.loads(ignored_dir_output.read_text())["paired_delta"]["bleu4"]["mean"] == 2.0


def test_report_cli_error_does_not_create_output(results, monkeypatch, cli):
    output = results.parent / "reports" / "failure.json"
    (results / "dino/seed-2/test.json").unlink()
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_encoders.py", "report", "--results-root", str(results), "--output", str(output)],
    )
    with pytest.raises(FileNotFoundError):
        cli.main()
    assert not output.exists()


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_report_cli_failure_leaves_no_output_or_temp(results, monkeypatch, cli, failure):
    import os

    output = results / "reports" / "failed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_encoders.py", "report", "--results-root", str(results), "--output", str(output)],
    )
    if failure == "write":
        monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("write failed")))
    else:
        monkeypatch.setattr(
            os, "link", lambda *args: (_ for _ in ()).throw(OSError("publish failed"))
        )
    with pytest.raises(OSError, match="failed"):
        cli.main()
    assert not output.exists()
    assert list(output.parent.iterdir()) == []


def test_preflight_cli_checks_both_overlays_before_sources(monkeypatch, tmp_path, capsys, cli):
    for key in (
        "PHOENIX14T_ANNOTATION_ROOT",
        "DESPAMO_FEATURE_ROOT",
        "DESPAMO_HF_CACHE",
        "DINO_ROOT",
        "COMPARISON_RUN_DIR",
    ):
        monkeypatch.setenv(key, str(tmp_path / key))
    counts = {
        source: {split: 1 for split in ("train", "dev", "test")} for source in ("clip", "dino")
    }
    monkeypatch.setattr(cli, "preflight_sources", lambda config: counts)
    monkeypatch.setattr(sys, "argv", ["compare_encoders.py", "preflight"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == counts
