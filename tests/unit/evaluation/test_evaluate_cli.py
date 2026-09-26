import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from despamo.evaluation.artifact import collect_runtime_metadata


def _script():
    path = Path(__file__).resolve().parents[3] / "scripts/evaluate.py"
    spec = importlib.util.spec_from_file_location("despamo_evaluate_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAdapter(torch.nn.Module):
    def forward(self, spatial, spatial_mask, motion, motion_mask):
        assert spatial_mask.equal(motion_mask)
        assert spatial.equal(motion)
        return spatial, spatial_mask


class FakeLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def generate_text(self, visual, mask, prompts, mode, beam_size, max_length):
        assert not torch.is_grad_enabled()
        assert mask.all()
        self.calls.append((mode, beam_size, max_length))
        return [
            f"{int(item.item())} {prompt}"
            for item, prompt in zip(visual[:, 0, 0], prompts, strict=True)
        ]


class FakeModel(torch.nn.Module):
    def __init__(self, seed=7):
        super().__init__()
        self.visual_adapter = FakeAdapter()
        self.language_model = FakeLanguageModel()
        self.prompt_template = "Translate into {}."
        self.use_in_context = True
        self.num_in_context = 3
        self.rng = random.Random(seed)


def _fixture(
    tmp_path,
    monkeypatch,
    script,
    *,
    bad_clip=None,
    missing=False,
    bad_predictions=False,
    frame_count=1,
):
    config = OmegaConf.create(
        {
            "seed": 7,
            "evaluation": {"expected_test_items": 642, "beam_size": 5},
            "model": {
                "max_text_length": 64,
                "max_frame_len": 512,
                "spatial_crop_mode": "upstream_random",
            },
            "data": {
                "spatial_manifest": str(tmp_path / "spatial.json"),
                "motion_manifest": str(tmp_path / "motion.json"),
            },
        }
    )
    clip_ids = [f"clip-{i:03d}" for i in range(642)]
    records = [{"fileid": item} for item in clip_ids[:-1]]
    if not missing:
        records.append({"fileid": clip_ids[-1]})

    def batches():
        for offset, size in ((0, 320), (320, len(records) - 320)):
            ids = clip_ids[offset : offset + size]
            if bad_clip and offset == 320:
                ids[-1] = bad_clip
            values = torch.arange(offset, offset + size, dtype=torch.float32).view(-1, 1, 1)
            values = values.expand(-1, frame_count, -1)
            yield SimpleNamespace(
                clip_ids=tuple(ids),
                texts=tuple(f"TEXT {i}" for i in range(offset, offset + size)),
                en_texts=tuple(f"english {i}" for i in range(offset, offset + size)),
                fr_texts=tuple(f"francais {i}" for i in range(offset, offset + size)),
                es_texts=tuple(f"espanol {i}" for i in range(offset, offset + size)),
                spatial=values,
                spatial_mask=torch.ones(size, frame_count, dtype=torch.bool),
                motion=values,
                motion_mask=torch.ones(size, frame_count, dtype=torch.bool),
            )

    monkeypatch.setattr(
        script,
        "build_data",
        lambda config: SimpleNamespace(
            test_dataset=SimpleNamespace(
                records=records, spatial_crop_mode=config.model.spatial_crop_mode
            ),
            test_dataloader=batches,
        ),
    )
    models = []

    def build_model(config):
        model = FakeModel()
        if bad_predictions:
            model.language_model.generate_text = lambda *args: ["one prediction"]
        models.append(model)
        return model

    monkeypatch.setattr(script, "build_model", build_model)
    captured = []

    def metrics(predictions, references):
        captured.append((list(predictions), list(references)))
        return {"bleu1": 46.0, "bleu4": 25.08, "rougeL_f1": 0.4698}

    monkeypatch.setattr(script, "evaluate_translations", metrics)
    monkeypatch.setattr(
        script,
        "collect_runtime_metadata",
        lambda seed, lock, spatial, motion: {
            "seed": seed,
            "package_lock": str(lock),
            "spatial_manifest": str(spatial),
            "motion_manifest": str(motion),
        },
    )
    checkpoint = tmp_path / "small.pt"
    torch.save(
        {"state_dict": FakeModel().state_dict(), "metadata": {"source_sha256": "fixture"}},
        checkpoint,
    )
    return config, checkpoint, clip_ids, models, captured


def _mark_converted_released(checkpoint):
    torch.save(
        {
            "state_dict": FakeModel().state_dict(),
            "metadata": {
                "source_path": "spamo.ckpt",
                "source_sha256": "06a432cdd8e1da4ce0b0e4cff246b20ad7d6a60406f32dfdbdfd974f94d3eee6",
                "target_schema_sha256": "b" * 64,
                "source_tensor_count": 871,
                "target_tensor_count": 871,
            },
        },
        checkpoint,
    )


def _comparison_fixture(tmp_path, monkeypatch, script):
    import despamo.comparison as comparison
    from despamo.comparison import annotation_hashes, comparison_code_sha256
    from despamo.utils.hashing import sha256_file

    config, checkpoint, _, models, captured = _fixture(tmp_path, monkeypatch, script)
    ann = tmp_path / "annotations"
    ann.mkdir()
    for split in ("train", "dev", "test"):
        np.save(
            ann / f"{split}_info_ml.npy",
            {0: {"fileid": f"{split}-id", "num_frames": 5, "text": "original"}},
        )
    config.data.annotation_root = str(ann)
    config.data.batch_size = 2
    config.data.spatial_root = str(tmp_path / "clip")
    config.model.spatial_crop_mode = "full"
    config.model.name = "google/flan-t5-xl"
    config.model.revision = "7d6315df2c2fb742f0f5b556879d730926ca9001"
    config.model.vt_pooling = "masked_mean"
    config.model.prompt = "Translate the given sentence into {}."
    config.model.vt_weight = 1.0
    config.model.use_in_context = True
    config.model.num_in_context = 3
    config.model.warm_up_steps = 0
    config.model.lora_rank = 16
    config.model.lora_alpha = 32
    config.model.lora_dropout = 0.1
    config.model.spatial_dim = 2048
    config.model.motion_dim = 1024
    config.evaluation.generation = "deterministic"
    config.comparison = {
        "enabled": True,
        "smoke": False,
        "clip_root": config.data.spatial_root,
        "clip_manifest": config.data.spatial_manifest,
        "dino_root": str(tmp_path / "dino"),
        "dino_manifest": str(tmp_path / "dino/complete/manifest.json"),
    }
    config.trainer = {
        "max_steps": 1000,
        "max_epochs": -1,
        "accumulate_grad_batches": 2,
        "precision": "bf16",
    }
    config.optimizer = {"learning_rate": 6e-4, "weight_decay": 0.01}
    for modality in ("spatial", "motion"):
        Path(config.data[f"{modality}_manifest"]).write_text(modality)
    saved = {
        "seed": config.seed,
        "config": OmegaConf.to_container(config, resolve=True),
        "resume_reproducibility": "fresh",
        "spatial_manifest_sha256": sha256_file(Path(config.data.spatial_manifest)),
        "motion_manifest_sha256": sha256_file(Path(config.data.motion_manifest)),
        "annotation_sha256": annotation_hashes(config),
        "comparison_code_sha256": comparison_code_sha256(script.PROJECT_ROOT),
        "dino_frame_rows_sha256": "e" * 64,
        "model_source": {
            "identifier": "google/flan-t5-xl",
            "tuning_type": "lora",
            "revision_status": "resolved",
            "revision": config.model.revision,
        },
    }
    torch.save(
        {
            "pytorch-lightning_version": "1.9.5",
            "global_step": 1000,
            "state_dict": FakeModel().state_dict(),
            "run_metadata": saved,
            "optimizer_states": [
                {
                    "state": {
                        0: {
                            "step": torch.tensor(1000.0),
                            "exp_avg": torch.ones(1),
                            "exp_avg_sq": torch.ones(1),
                        }
                    },
                    "param_groups": [{"params": [0]}],
                }
            ],
            "lr_schedulers": [{"last_epoch": 1000}],
        },
        checkpoint,
    )
    monkeypatch.setattr(comparison, "validate_dino_content", lambda config: "e" * 64, raising=False)
    monkeypatch.setattr(script, "validate_dino_content", lambda config: "e" * 64, raising=False)
    return config, checkpoint, ann, saved, models, captured


@pytest.mark.parametrize("mutation", ["replace", "in_place"])
def test_evaluation_hashes_loaded_snapshot_when_source_changes_during_generation(
    tmp_path, monkeypatch, mutation
):
    script = _script()
    config, checkpoint, _, models, _ = _fixture(tmp_path, monkeypatch, script)
    original_bytes = checkpoint.read_bytes()
    output = tmp_path / "result.json"
    loaded_paths = []
    real_load = torch.load
    real_generate = FakeLanguageModel.generate_text
    changed = False

    def capture_load(path, *args, **kwargs):
        loaded_paths.append(Path(path))
        return real_load(path, *args, **kwargs)

    def mutate_source(self, *args):
        nonlocal changed
        if not changed:
            changed = True
            replacement = tmp_path / "replacement.pt"
            new_model = FakeModel()
            new_model.language_model.weight.data.fill_(7)
            torch.save(
                {"state_dict": new_model.state_dict(), "metadata": {"source_sha256": "other"}},
                replacement,
            )
            if mutation == "replace":
                replacement.replace(checkpoint)
            else:
                checkpoint.write_bytes(replacement.read_bytes())
        return real_generate(self, *args)

    monkeypatch.setattr(script.torch, "load", capture_load)
    monkeypatch.setattr(FakeLanguageModel, "generate_text", mutate_source)

    script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    payload = json.loads(output.read_text())
    assert payload["metadata"]["checkpoint"] == str(checkpoint)
    assert payload["metadata"]["checkpoint_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert payload["metadata"]["checkpoint_metadata"] == {"source_sha256": "fixture"}
    assert (
        payload["metadata"]["checkpoint_sha256"]
        != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert models[0].language_model.weight.item() == 1
    assert len(loaded_paths) == 1
    assert loaded_paths[0] != checkpoint
    assert loaded_paths[0].parent == output.parent
    assert not loaded_paths[0].exists()


def test_evaluation_removes_snapshot_after_inference_failure(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script, bad_predictions=True)
    output = tmp_path / "results" / "result.json"
    loaded_paths = []
    real_load = torch.load

    def capture_load(path, *args, **kwargs):
        loaded_paths.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(script.torch, "load", capture_load)

    with pytest.raises(ValueError, match="predictions must match"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert len(loaded_paths) == 1
    assert loaded_paths[0] != checkpoint
    assert not loaded_paths[0].exists()
    assert list(output.parent.iterdir()) == []
    assert checkpoint.is_file()
    assert not output.exists()


def test_cpu_fake_evaluates_all_clips_once_and_repeats_deterministically(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, clip_ids, models, captured = _fixture(tmp_path, monkeypatch, script)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    script.evaluate(config, checkpoint, "deterministic", first, device="cpu")
    script.evaluate(config, checkpoint, "deterministic", second, device="cpu")

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text())
    assert [item["clip_id"] for item in payload["items"]] == clip_ids
    assert len(set(item["clip_id"] for item in payload["items"])) == 642
    assert captured[0] == captured[1]
    assert captured[0][1] == [f"text {i}" for i in range(642)]
    assert [model.language_model.calls for model in models] == [[("deterministic", 5, 64)] * 2] * 2
    assert payload["metadata"]["generation"] == "deterministic"
    assert payload["metadata"]["spatial_crop_mode"] == "center"
    assert payload["metadata"]["checkpoint"] == str(checkpoint)
    assert (
        payload["metadata"]["checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert payload["metadata"]["checkpoint_metadata"] == {"source_sha256": "fixture"}
    effective_config = OmegaConf.to_container(config, resolve=True)
    effective_config["model"]["spatial_crop_mode"] = "center"
    assert payload["metadata"]["config"] == effective_config
    assert config.model.spatial_crop_mode == "upstream_random"
    assert payload["metadata"]["package_lock"] == str(
        Path(script.__file__).resolve().parents[1] / "uv.lock"
    )
    assert payload["metadata"]["spatial_manifest"] == config.data.spatial_manifest


def test_evaluation_accepts_training_checkpoint_provenance(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    torch.save(
        {
            "state_dict": FakeModel().state_dict(),
            "run_metadata": {"seed": 7, "model_source": {"revision_status": "unresolved"}},
        },
        checkpoint,
    )
    output = tmp_path / "training-result.json"

    script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert json.loads(output.read_text())["metadata"]["checkpoint_metadata"] == {
        "seed": 7,
        "model_source": {"revision_status": "unresolved"},
    }


@pytest.mark.parametrize("metadata_key", ["metadata", "run_metadata"])
def test_evaluation_rejects_secret_in_checkpoint_metadata_before_generation(
    tmp_path, monkeypatch, metadata_key
):
    script = _script()
    config, checkpoint, _, models, captured = _fixture(tmp_path, monkeypatch, script)
    torch.save(
        {
            "state_dict": FakeModel().state_dict(),
            metadata_key: {"source": [{"hf_token": "DUMMY_DO_NOT_WRITE"}]},
        },
        checkpoint,
    )
    output = tmp_path / "result.json"

    with pytest.raises(ValueError, match="hf_token") as error:
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert "DUMMY_DO_NOT_WRITE" not in str(error.value)
    assert not output.exists()
    assert not captured
    assert not models or not models[0].language_model.calls


def test_evaluation_outside_project_rejects_unignored_project_output_before_data(
    tmp_path, monkeypatch
):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    subprocess.run(["git", "init", "-q", str(elsewhere)], check=True)
    monkeypatch.chdir(elsewhere)
    output = Path(script.__file__).resolve().parents[1] / "training-provenance-eval-unignored.json"
    assert not output.exists()
    monkeypatch.setattr(script, "build_data", lambda config: pytest.fail("data loaded"))
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model loaded"))

    with pytest.raises(ValueError, match="output.*Git.*ignored"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert not output.exists()


def test_evaluation_cli_outside_project_rejects_unignored_project_output_before_cuda(
    tmp_path, monkeypatch
):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    config.model.spatial_dim = 2048
    config.model.motion_dim = 1024
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    subprocess.run(["git", "init", "-q", str(elsewhere)], check=True)
    monkeypatch.chdir(elsewhere)
    output = Path(script.__file__).resolve().parents[1] / "training-provenance-eval-unignored.json"
    assert not output.exists()
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: pytest.fail("CUDA checked"))
    monkeypatch.setattr(script, "evaluate", lambda *args, **kw: pytest.fail("evaluated"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--generation",
            "deterministic",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(ValueError, match="output.*Git.*ignored"):
        script.main()

    assert not output.exists()


@pytest.mark.parametrize("location", ["ignored", "external"])
def test_evaluation_outside_project_allows_ignored_or_external_output(
    tmp_path, monkeypatch, location
):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    subprocess.run(["git", "init", "-q", str(elsewhere)], check=True)
    monkeypatch.chdir(elsewhere)
    output = (
        Path(script.__file__).resolve().parents[1] / "artifacts" / "policy-test-not-written.json"
        if location == "ignored"
        else tmp_path / "external.json"
    )
    assert not output.exists()

    def allowed(config):
        raise RuntimeError("output policy passed")

    monkeypatch.setattr(script, "build_data", allowed)

    with pytest.raises(RuntimeError, match="output policy passed"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert not output.exists()


@pytest.mark.parametrize("key", ["token", "password", "client_secret", "hf_token", "apiKey"])
def test_evaluation_rejects_nested_secret_keys_before_data_or_model(tmp_path, monkeypatch, key):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    config.nested = [{"settings": {key: "DUMMY_DO_NOT_WRITE"}}]
    output = tmp_path / "result.json"
    monkeypatch.setattr(script, "build_data", lambda config: pytest.fail("data loaded"))
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model loaded"))

    with pytest.raises(ValueError, match=key) as error:
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert "DUMMY_DO_NOT_WRITE" not in str(error.value)
    assert not output.exists()


def test_evaluation_allows_tokenizer_config_key(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    config.model.tokenizer = "local"
    output = tmp_path / "result.json"

    script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert json.loads(output.read_text())["metadata"]["config"]["model"]["tokenizer"] == "local"


def test_evaluation_uses_project_lock_outside_project_cwd(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    for modality in ("spatial", "motion"):
        (tmp_path / f"{modality}.json").write_text(modality)
    monkeypatch.setattr(script, "collect_runtime_metadata", collect_runtime_metadata)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    subprocess.run(["git", "init", "-q", str(elsewhere)], check=True)
    (elsewhere / "uv.lock").write_text("different local lock")
    monkeypatch.chdir(elsewhere)
    output = tmp_path / "result.json"

    script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    project_lock = Path(script.__file__).resolve().parents[1] / "uv.lock"
    actual = json.loads(output.read_text())["metadata"]["package_lock_sha256"]
    assert actual == hashlib.sha256(project_lock.read_bytes()).hexdigest()
    assert actual != hashlib.sha256((elsewhere / "uv.lock").read_bytes()).hexdigest()


def test_evaluation_rejects_unignored_output_even_with_real_metadata(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".git" / "info" / "exclude").write_text("/uv.lock\n")
    (repo / "uv.lock").write_text("lock")
    monkeypatch.setattr(script, "PROJECT_ROOT", repo)
    monkeypatch.setattr(script, "collect_runtime_metadata", collect_runtime_metadata)
    monkeypatch.chdir(repo)
    output = repo / "results" / "first.json"

    with pytest.raises(ValueError, match="output.*Git.*ignored"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert not output.exists()


@pytest.mark.parametrize("relative_output", ["results/result.json", "result.json"])
def test_evaluation_rejects_nonignored_git_output_before_data_and_model(
    tmp_path, monkeypatch, relative_output
):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "uv.lock").write_text("lock")
    monkeypatch.setattr(script, "PROJECT_ROOT", repo)
    output = repo / relative_output
    monkeypatch.chdir(repo)
    original = checkpoint.read_bytes()
    monkeypatch.setattr(script, "build_data", lambda config: pytest.fail("data loaded"))
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model loaded"))
    monkeypatch.setattr(script.torch, "load", lambda *a, **kw: pytest.fail("checkpoint loaded"))
    monkeypatch.setattr(script, "sha256_file", lambda path: pytest.fail("checkpoint hashed"))

    with pytest.raises(ValueError, match="output.*Git.*ignored"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert checkpoint.read_bytes() == original
    assert not output.exists()


@pytest.mark.parametrize(
    "alias", ["symlink_repo", "symlink_external", "symlink_broken", "hardlink_repo"]
)
def test_evaluation_rejects_ignored_output_link_to_unrelated_file(tmp_path, monkeypatch, alias):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".git" / "info" / "exclude").write_text("/uv.lock\n/artifacts/\n")
    (repo / "uv.lock").write_text("lock")
    artifacts = repo / "artifacts"
    artifacts.mkdir()
    output = artifacts / "result.json"
    target = (
        tmp_path / "external.txt"
        if alias in {"symlink_external", "symlink_broken"}
        else repo / "unrelated.txt"
    )
    if alias != "symlink_broken":
        target.write_bytes(b"unrelated original contents")
    if alias in {"symlink_repo", "hardlink_repo"}:
        subprocess.run(["git", "add", "--", "unrelated.txt"], cwd=repo, check=True)
    try:
        if alias == "hardlink_repo":
            os.link(target, output)
        else:
            output.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"links unavailable: {exc}")
    original = target.read_bytes() if target.exists() else None
    checkpoint_bytes = checkpoint.read_bytes()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(script, "build_data", lambda config: pytest.fail("data loaded"))
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model loaded"))
    monkeypatch.setattr(script.torch, "load", lambda *a, **kw: pytest.fail("checkpoint loaded"))
    monkeypatch.setattr(script, "sha256_file", lambda path: pytest.fail("checkpoint hashed"))

    with pytest.raises(ValueError, match="output.*(symlink|hardlink)"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert checkpoint.read_bytes() == checkpoint_bytes
    assert (target.read_bytes() if target.exists() else None) == original
    if alias == "hardlink_repo":
        assert output.read_bytes() == original
    else:
        assert output.is_symlink()


def test_evaluation_resolves_ignored_symlink_parent_before_checking_git(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".git" / "info" / "exclude").write_text("/uv.lock\n/artifacts/\n")
    (repo / "uv.lock").write_text("lock")
    monkeypatch.setattr(script, "PROJECT_ROOT", repo)
    private = repo / "private"
    private.mkdir()
    sentinel = private / "original.txt"
    sentinel.write_bytes(b"original")
    artifacts = repo / "artifacts"
    artifacts.mkdir()
    try:
        (artifacts / "redirect").symlink_to(private, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    output = artifacts / "redirect" / "result.json"
    monkeypatch.chdir(repo)
    monkeypatch.setattr(script, "build_data", lambda config: pytest.fail("data loaded"))
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model loaded"))
    monkeypatch.setattr(script.torch, "load", lambda *a, **kw: pytest.fail("checkpoint loaded"))
    monkeypatch.setattr(script, "sha256_file", lambda path: pytest.fail("checkpoint hashed"))

    with pytest.raises(ValueError, match="output.*Git.*ignored"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert sentinel.read_bytes() == b"original"
    assert not (private / "result.json").exists()


def test_evaluation_allows_external_git_output_with_real_metadata(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".git" / "info" / "exclude").write_text("/uv.lock\n")
    (repo / "uv.lock").write_text("lock")
    (tmp_path / "spatial.json").write_text("spatial")
    (tmp_path / "motion.json").write_text("motion")
    monkeypatch.setattr(
        script,
        "collect_runtime_metadata",
        lambda seed, lock, spatial, motion: collect_runtime_metadata(
            seed, repo / "uv.lock", spatial, motion
        ),
    )
    monkeypatch.chdir(repo)
    output = tmp_path / "outside" / "result.json"

    script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert json.loads(output.read_text())["metadata"]["git_dirty"] is False


@pytest.mark.parametrize("other_change", ["clean", "staged_in_artifacts", "untracked_elsewhere"])
def test_ignored_artifacts_repeat_with_real_git_status(tmp_path, monkeypatch, other_change):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".git" / "info" / "exclude").write_text("/uv.lock\n/artifacts/\n")
    (repo / "uv.lock").write_text("lock")
    monkeypatch.setattr(script, "PROJECT_ROOT", repo)
    (tmp_path / "spatial.json").write_text("spatial")
    (tmp_path / "motion.json").write_text("motion")
    artifacts = repo / "artifacts"
    if other_change == "staged_in_artifacts":
        artifacts.mkdir()
        (artifacts / "unrelated.txt").write_text("unrelated")
        subprocess.run(["git", "add", "-f", "--", "artifacts/unrelated.txt"], cwd=repo, check=True)
    elif other_change == "untracked_elsewhere":
        (repo / "unrelated.txt").write_text("unrelated")
    monkeypatch.setattr(
        script,
        "collect_runtime_metadata",
        lambda seed, lock, spatial, motion: collect_runtime_metadata(
            seed, repo / "uv.lock", spatial, motion
        ),
    )
    monkeypatch.chdir(repo)
    first = artifacts / "first.json"
    second = artifacts / "second.json"

    script.evaluate(config, checkpoint, "deterministic", first, device="cpu")
    script.evaluate(config, checkpoint, "deterministic", second, device="cpu")

    assert first.read_bytes() == second.read_bytes()
    expected_dirty = other_change != "clean"
    assert json.loads(first.read_text())["metadata"]["git_dirty"] is expected_dirty


def test_cpu_fake_upstream_passes_generation_mode_and_writes_artifact(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, models, _ = _fixture(tmp_path, monkeypatch, script)
    output = tmp_path / "upstream.json"

    script.evaluate(config, checkpoint, "upstream", output, device="cpu")

    assert len(json.loads(output.read_text())["items"]) == 642
    assert json.loads(output.read_text())["metadata"]["spatial_crop_mode"] == "upstream_random"
    assert models[0].language_model.calls == [("upstream", 5, 64)] * 2


@pytest.mark.parametrize(
    ("generation", "expected_mode"),
    [("deterministic", "center"), ("upstream", "upstream_random")],
)
def test_evaluation_passes_effective_crop_to_factory_and_records_it(
    tmp_path, monkeypatch, generation, expected_mode
):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    calls = []
    original_build_data = script.build_data

    def build_data(run_config):
        calls.append((run_config.model.max_frame_len, run_config.model.spatial_crop_mode))
        return original_build_data(run_config)

    monkeypatch.setattr(script, "build_data", build_data)
    output = tmp_path / "result.json"

    script.evaluate(config, checkpoint, generation, output, device="cpu")

    assert calls == [(512, expected_mode)]
    assert config.model.spatial_crop_mode == "upstream_random"
    assert json.loads(output.read_text())["metadata"]["spatial_crop_mode"] == expected_mode


@pytest.mark.parametrize("generation", ["deterministic", "upstream"])
def test_evaluation_preserves_full_policy_for_long_sequences(tmp_path, monkeypatch, generation):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script, frame_count=520)
    config.model.spatial_crop_mode = "full"
    observed = []
    original_forward = FakeAdapter.forward

    def observe_forward(self, spatial, spatial_mask, motion, motion_mask):
        observed.append(spatial.shape[1])
        return original_forward(self, spatial, spatial_mask, motion, motion_mask)

    monkeypatch.setattr(FakeAdapter, "forward", observe_forward)
    output = tmp_path / "result.json"

    script.evaluate(config, checkpoint, generation, output, device="cpu")

    metadata = json.loads(output.read_text())["metadata"]
    assert observed == [520, 520]
    assert metadata["spatial_crop_mode"] == "full"
    assert metadata["config"]["model"]["spatial_crop_mode"] == "full"
    assert config.model.spatial_crop_mode == "full"


def test_evaluation_upstream_respects_explicit_center_policy(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    config.model.spatial_crop_mode = "center"
    output = tmp_path / "center.json"

    script.evaluate(config, checkpoint, "upstream", output, device="cpu")

    metadata = json.loads(output.read_text())["metadata"]
    assert metadata["spatial_crop_mode"] == "center"
    assert metadata["config"]["model"]["spatial_crop_mode"] == "center"


@pytest.mark.parametrize("missing", [True, False])
def test_cpu_fake_rejects_incomplete_or_wrong_split_before_loading(tmp_path, monkeypatch, missing):
    script = _script()
    config, checkpoint, _, models, captured = _fixture(
        tmp_path, monkeypatch, script, missing=missing, bad_clip=None if missing else "clip-000"
    )
    output = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="expected 642 test items" if missing else "test split"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert not output.exists()
    assert len(models) == (0 if missing else 1)
    assert not captured


def test_cpu_fake_rejects_partial_batch_predictions_without_success_artifact(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, captured = _fixture(
        tmp_path, monkeypatch, script, bad_predictions=True
    )
    output = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="predictions.*batch"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert not output.exists()
    assert not captured


@pytest.mark.parametrize(
    "state", [{}, {"language_model.weight": torch.tensor(1.0), "extra": torch.tensor(0)}]
)
def test_cpu_fake_requires_strict_checkpoint_state(tmp_path, monkeypatch, state):
    script = _script()
    config, checkpoint, _, _, captured = _fixture(tmp_path, monkeypatch, script)
    torch.save({"state_dict": state, "metadata": {}}, checkpoint)
    output = tmp_path / "invalid.json"

    with pytest.raises(RuntimeError, match="state_dict"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert not output.exists()
    assert not captured


def test_cpu_fake_rejects_upstream_metrics_before_writing_success_artifact(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    _mark_converted_released(checkpoint)
    monkeypatch.setattr(
        script,
        "RELEASED_CONVERTED_SHA256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        raising=False,
    )
    monkeypatch.setattr(
        script,
        "evaluate_translations",
        lambda predictions, references: {"bleu4": 0.0, "rougeL_f1": 0.4698},
    )
    output = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="BLEU-4 outside tolerance"):
        script.evaluate(config, checkpoint, "upstream", output, device="cpu", accept_baseline=True)

    assert not output.exists()


def test_upstream_research_score_writes_artifact_without_claiming_acceptance(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    config.model.spatial_crop_mode = "full"
    monkeypatch.setattr(
        script,
        "evaluate_translations",
        lambda predictions, references: {"bleu4": 18.0, "rougeL_f1": 0.3},
    )
    output = tmp_path / "research.json"

    metrics = script.evaluate(config, checkpoint, "upstream", output, device="cpu")

    payload = json.loads(output.read_text())
    assert metrics == payload["metrics"] == {"bleu4": 18.0, "rougeL_f1": 0.3}
    assert payload["metadata"]["spatial_crop_mode"] == "full"
    assert payload["metadata"]["baseline_accepted"] is False
    assert len(payload["items"]) == 642


def test_explicit_baseline_acceptance_records_only_valid_upstream_result(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    _mark_converted_released(checkpoint)
    monkeypatch.setattr(
        script,
        "RELEASED_CONVERTED_SHA256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        raising=False,
    )
    output = tmp_path / "accepted.json"

    metrics = script.evaluate(
        config, checkpoint, "upstream", output, device="cpu", accept_baseline=True
    )

    payload = json.loads(output.read_text())
    assert metrics == payload["metrics"]
    assert payload["metadata"]["baseline_accepted"] is True
    assert payload["metadata"]["spatial_crop_mode"] == "upstream_random"


def test_baseline_acceptance_rejects_deterministic_before_data_load(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    monkeypatch.setattr(script, "build_data", lambda config: pytest.fail("data loaded"))
    output = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="accept-baseline.*upstream"):
        script.evaluate(
            config, checkpoint, "deterministic", output, device="cpu", accept_baseline=True
        )

    assert not output.exists()


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hard_link"])
def test_evaluation_rejects_checkpoint_output_alias_before_preflight(tmp_path, monkeypatch, alias):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    output = tmp_path / "alias.pt"
    if alias == "same_path":
        output = checkpoint
    elif alias == "symlink":
        try:
            output.symlink_to(checkpoint)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
    else:
        try:
            os.link(checkpoint, output)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"hard links unavailable: {exc}")
    original = checkpoint.read_bytes()
    monkeypatch.setattr(script, "build_data", lambda config: pytest.fail("data preflight started"))
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model loaded"))
    monkeypatch.setattr(script.torch, "load", lambda *a, **kw: pytest.fail("checkpoint loaded"))
    monkeypatch.setattr(script, "sha256_file", lambda path: pytest.fail("checkpoint hashed"))

    with pytest.raises(ValueError, match="checkpoint and output must be different files"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert checkpoint.read_bytes() == original
    assert output.read_bytes() == original


@pytest.mark.parametrize("case", ["missing", "directory"])
def test_evaluation_preflights_checkpoint_file_before_model(tmp_path, monkeypatch, case):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    if case == "missing":
        checkpoint.unlink()
    else:
        checkpoint.unlink()
        checkpoint.mkdir()
    data_loader = script.build_data
    seen = []

    def check_data(config):
        seen.append("data preflight")
        return data_loader(config)

    monkeypatch.setattr(script, "build_data", check_data)
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("Flan download started"))
    output = tmp_path / "result.json"

    with pytest.raises(FileNotFoundError, match="checkpoint.*file"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")

    assert seen == ["data preflight"]
    assert not output.exists()


def test_cli_preserves_config_layer_order_and_checks_cuda(tmp_path, monkeypatch):
    script = _script()
    paths = []
    for index, content in enumerate(
        (
            "model:\n  spatial_dim: 2048\n  motion_dim: 1024\n  name: first\n",
            "model:\n  name: second\n",
            "seed: 7\nevaluation:\n  expected_test_items: 642\n",
        )
    ):
        path = tmp_path / f"{index}.yaml"
        path.write_text(content)
        paths.append(path)
    calls = []
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        script,
        "evaluate",
        lambda config, checkpoint, generation, output, **kwargs: (
            calls.append((config.model.name, checkpoint, generation, output)) or {"bleu4": 25.08}
        ),
    )
    output = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoint.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            *(arg for path in paths for arg in ("--config", str(path))),
            "--checkpoint",
            str(checkpoint),
            "--generation",
            "deterministic",
            "--output",
            str(output),
        ],
    )

    script.main()

    assert calls == [("second", checkpoint, "deterministic", output)]


def test_cli_passes_explicit_baseline_acceptance_to_evaluator(tmp_path, monkeypatch):
    script = _script()
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  spatial_dim: 2048\n  motion_dim: 1024\n")
    calls = []
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        script, "evaluate", lambda *args, **kwargs: calls.append(kwargs) or {"bleu4": 25.08}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            str(config),
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--generation",
            "upstream",
            "--accept-baseline",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )

    script.main()

    assert calls == [{"accept_baseline": True}]


def test_cli_rejects_deterministic_baseline_acceptance_before_cuda(tmp_path, monkeypatch, capsys):
    script = _script()
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  spatial_dim: 2048\n  motion_dim: 1024\n")
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: pytest.fail("CUDA checked"))
    monkeypatch.setattr(script, "evaluate", lambda *args, **kwargs: pytest.fail("evaluated"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            str(config),
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--generation",
            "deterministic",
            "--accept-baseline",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        script.main()
    assert "--accept-baseline requires --generation upstream" in capsys.readouterr().err


def test_cli_rejects_checkpoint_as_output_even_without_cuda(tmp_path, monkeypatch):
    script = _script()
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  spatial_dim: 2048\n  motion_dim: 1024\n")
    checkpoint = tmp_path / "small.pt"
    torch.save({"state_dict": {}}, checkpoint)
    original = checkpoint.read_bytes()
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(script, "build_data", lambda config: pytest.fail("data loaded"))
    monkeypatch.setattr(script.torch, "load", lambda *a, **kw: pytest.fail("checkpoint loaded"))
    monkeypatch.setattr(script, "sha256_file", lambda path: pytest.fail("checkpoint hashed"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--generation",
            "deterministic",
            "--output",
            str(checkpoint),
        ],
    )

    with pytest.raises(ValueError, match="checkpoint and output must be different files"):
        script.main()

    assert checkpoint.read_bytes() == original


def test_cli_without_cuda_fails_explicitly_before_weight_loading(tmp_path, monkeypatch):
    script = _script()
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  spatial_dim: 2048\n  motion_dim: 1024\n")
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("loaded model"))
    monkeypatch.setattr(script.torch, "load", lambda *a, **kw: pytest.fail("loaded checkpoint"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            str(config),
            "--checkpoint",
            str(tmp_path / "large.pt"),
            "--generation",
            "upstream",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )

    with pytest.raises(SystemExit, match="CUDA GPU required"):
        script.main()

    assert not (tmp_path / "result.json").exists()


def test_comparison_rejects_baseline_acceptance_even_with_upstream(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    config.comparison = {"enabled": True, "smoke": False}
    monkeypatch.setattr(script, "build_data", lambda config: pytest.fail("data built"))
    with pytest.raises(ValueError, match="released SpaMo"):
        script.evaluate(
            config,
            checkpoint,
            "upstream",
            tmp_path / "result.json",
            device="cpu",
            accept_baseline=True,
        )


def test_comparison_rejects_incomplete_checkpoint_before_model(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    config.comparison = {"enabled": True, "smoke": False}
    config.trainer = {"max_steps": 1000}
    config.evaluation.generation = "deterministic"
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model built"))
    with pytest.raises(ValueError, match="comparison.*step"):
        script.evaluate(config, checkpoint, "deterministic", tmp_path / "result.json", device="cpu")


def test_baseline_acceptance_rejects_other_converted_source(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    _mark_converted_released(checkpoint)
    content = torch.load(checkpoint, map_location="cpu", weights_only=True)
    content["metadata"]["source_sha256"] = "c" * 64
    torch.save(content, checkpoint)
    monkeypatch.setattr(
        script,
        "RELEASED_CONVERTED_SHA256",
        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        raising=False,
    )
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model built"))
    with pytest.raises(ValueError, match="converted released SpaMo"):
        script.evaluate(
            config,
            checkpoint,
            "upstream",
            tmp_path / "result.json",
            device="cpu",
            accept_baseline=True,
        )


def test_evaluation_cli_applies_seed_override_to_same_layers(tmp_path, monkeypatch):
    script = _script()
    common_path = tmp_path / "comparison.yaml"
    common_path.write_text(
        "seed: 0\nmodel:\n  spatial_dim: 2048\n  motion_dim: 1024\n"
        "trainer:\n  default_root_dir: ${oc.env:COMPARISON_RUN_DIR}\n"
    )
    source_path = tmp_path / "dino.yaml"
    source_path.write_text("data:\n  spatial_root: dino\n")
    monkeypatch.setenv("COMPARISON_RUN_DIR", str(tmp_path / "dino/seed-2"))
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: True)
    seen = []
    monkeypatch.setattr(
        script,
        "evaluate",
        lambda config, *args, **kwargs: (
            seen.append((config.seed, config.trainer.default_root_dir, config.data.spatial_root))
            or {"bleu4": 1.0}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            str(common_path),
            "--config",
            str(source_path),
            "--checkpoint",
            str(tmp_path / "model.ckpt"),
            "--generation",
            "deterministic",
            "--output",
            str(tmp_path / "test.json"),
            "seed=2",
        ],
    )
    script.main()
    assert seen == [(2, str(tmp_path / "dino/seed-2"), "dino")]


def test_comparison_text_only_annotation_drift_blocks_score_and_artifact(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, ann, saved, _, _ = _comparison_fixture(tmp_path, monkeypatch, script)
    accepted = tmp_path / "matched.json"
    script.evaluate(config, checkpoint, "deterministic", accepted, device="cpu")
    result_metadata = json.loads(accepted.read_text())["metadata"]
    assert result_metadata["comparison_source"] == "clip"
    assert result_metadata["checkpoint_global_step"] == 1000
    assert result_metadata["annotation_sha256"] == saved["annotation_sha256"]
    assert result_metadata["comparison_code_sha256"] == saved["comparison_code_sha256"]
    assert result_metadata["baseline_accepted"] is False
    raw = np.load(ann / "train_info_ml.npy", allow_pickle=True).item()
    raw[0]["text"] = "changed text without changing clip ID or num_frames"
    np.save(ann / "train_info_ml.npy", raw)
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model built"))
    monkeypatch.setattr(
        script, "evaluate_translations", lambda *args: pytest.fail("score calculated")
    )
    monkeypatch.setattr(
        script, "write_result_artifact", lambda *args: pytest.fail("result recorded")
    )
    output = tmp_path / "drifted.json"
    with pytest.raises(ValueError, match="annotation SHA256 mismatch"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")
    assert not output.exists()


def test_comparison_rejects_conflicting_metadata_before_model(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, saved, _, _ = _comparison_fixture(tmp_path, monkeypatch, script)
    content = torch.load(checkpoint, map_location="cpu", weights_only=True)
    content["metadata"] = {"annotation_sha256": saved["annotation_sha256"], "seed": 999}
    torch.save(content, checkpoint)
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model built"))
    output = tmp_path / "conflict.json"
    with pytest.raises(ValueError, match="comparison.*conflicting.*metadata"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")
    assert not output.exists()


def test_comparison_code_drift_rejects_before_model(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _, _ = _comparison_fixture(tmp_path, monkeypatch, script)
    content = torch.load(checkpoint, map_location="cpu", weights_only=True)
    content["run_metadata"]["comparison_code_sha256"] = "a" * 64
    torch.save(content, checkpoint)
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model built"))
    output = tmp_path / "code-drift.json"
    with pytest.raises(ValueError, match="comparison code SHA256 mismatch"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")
    assert not output.exists()


def test_comparison_rechecks_code_before_artifact_write(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _, _ = _comparison_fixture(tmp_path, monkeypatch, script)
    monkeypatch.setattr(script, "comparison_code_sha256", lambda root: "0" * 64)
    monkeypatch.setattr(
        script, "write_result_artifact", lambda *args: pytest.fail("artifact written")
    )
    output = tmp_path / "code-drift-after-model.json"
    with pytest.raises(ValueError, match="comparison code SHA256 mismatch before result write"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")
    assert not output.exists()


def test_comparison_state_dict_only_is_rejected_before_model(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _, _ = _comparison_fixture(tmp_path, monkeypatch, script)
    content = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for field in ("pytorch-lightning_version", "optimizer_states", "lr_schedulers"):
        del content[field]
    torch.save(content, checkpoint)
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model built"))
    with pytest.raises(ValueError, match="comparison.*Lightning"):
        script.evaluate(
            config, checkpoint, "deterministic", tmp_path / "invalid.json", device="cpu"
        )


def test_comparison_protocol_violation_blocks_model_even_with_matching_saved_config(
    tmp_path, monkeypatch
):
    script = _script()
    config, checkpoint, _, _, _, _ = _comparison_fixture(tmp_path, monkeypatch, script)
    config.model.vt_pooling = "legacy_mean"
    content = torch.load(checkpoint, map_location="cpu", weights_only=True)
    content["run_metadata"]["config"] = OmegaConf.to_container(config, resolve=True)
    torch.save(content, checkpoint)
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model built"))
    with pytest.raises(ValueError, match="comparison protocol mismatch"):
        script.evaluate(
            config, checkpoint, "deterministic", tmp_path / "invalid.json", device="cpu"
        )


def test_comparison_rechecks_annotation_bytes_after_inference(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, ann, _, _, _ = _comparison_fixture(tmp_path, monkeypatch, script)
    original_generate = FakeLanguageModel.generate_text
    changed = False

    def mutate_after_preflight(self, *args):
        nonlocal changed
        if not changed:
            changed = True
            path = ann / "test_info_ml.npy"
            raw = np.load(path, allow_pickle=True).item()
            raw[0]["text"] = "changed after checkpoint validation"
            np.save(path, raw)
        return original_generate(self, *args)

    monkeypatch.setattr(FakeLanguageModel, "generate_text", mutate_after_preflight)
    output = tmp_path / "drifted-after-inference.json"
    with pytest.raises(ValueError, match="annotation SHA256 mismatch before result write"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")
    assert changed and not output.exists()


def test_comparison_rechecks_dino_content_before_artifact_write(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _, _ = _comparison_fixture(tmp_path, monkeypatch, script)
    calls = []

    def content_hash(config):
        calls.append(config)
        return "f" * 64

    monkeypatch.setattr(script, "validate_dino_content", content_hash, raising=False)
    output = tmp_path / "content-drift.json"
    with pytest.raises(ValueError, match="DINO content SHA256 mismatch before result write"):
        script.evaluate(config, checkpoint, "deterministic", output, device="cpu")
    assert len(calls) == 1 and not output.exists()


def test_forged_released_source_metadata_cannot_claim_baseline_acceptance(tmp_path, monkeypatch):
    script = _script()
    config, checkpoint, _, _, _ = _fixture(tmp_path, monkeypatch, script)
    _mark_converted_released(checkpoint)
    monkeypatch.setattr(script, "build_model", lambda config: pytest.fail("model built"))
    output = tmp_path / "forged.json"
    with pytest.raises(ValueError, match="converted released SpaMo.*SHA256"):
        script.evaluate(config, checkpoint, "upstream", output, device="cpu", accept_baseline=True)
    assert not output.exists()
