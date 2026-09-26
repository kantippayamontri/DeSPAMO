import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.models.flan_t5 import FlanT5Backbone
from despamo.utils.hashing import sha256_file

FLAN_SHA = "7d6315df2c2fb742f0f5b556879d730926ca9001"


def _train_script():
    path = Path(__file__).resolve().parents[2] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("despamo_train_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_run(tmp_path, monkeypatch, *, root=None, revision=None, extra_config=None):
    train = _train_script()
    root = root or tmp_path / "artifacts"
    spatial = tmp_path / "spatial.json"
    motion = tmp_path / "motion.json"
    spatial.write_text("spatial")
    motion.write_text("motion")
    config_path = tmp_path / "config.yaml"
    config = {
        "seed": 7,
        "trainer": {"default_root_dir": str(root), "max_epochs": 1},
        "data": {"spatial_manifest": str(spatial), "motion_manifest": str(motion)},
        "model": {"spatial_dim": 2048, "motion_dim": 1024, "name": "google/flan-t5-xl"},
    }
    if extra_config:
        config.update(extra_config)
    OmegaConf.save(config, config_path)
    pretrained = SimpleNamespace(config=SimpleNamespace(_commit_hash=revision))
    model = SimpleNamespace(language_model=SimpleNamespace(model=pretrained))
    events = []

    class FakeTrainer:
        def __init__(self, **kwargs):
            events.append("trainer")

        def fit(self, fitted_model, *, datamodule, **kwargs):
            events.append(("fit", kwargs))
            assert fitted_model is model
            assert datamodule == "data"
            assert json.loads((root / "run_metadata.json").read_text()) == model.run_metadata

    def fake_data(config):
        events.append("data")
        return "data"

    def fake_model(config):
        events.append("model")
        assert (root / "run_metadata.json").is_file() or root.exists()
        return model

    monkeypatch.setattr(train.pl, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.pl, "Trainer", FakeTrainer)
    monkeypatch.setattr(train, "build_data", fake_data)
    monkeypatch.setattr(train, "build_model", fake_model)
    monkeypatch.setattr(train, "dino_frame_rows_hash", lambda config: "e" * 64, raising=False)
    return train, root, config_path, model, events


def _tiny_resume_checkpoint(metadata):
    return {
        "pytorch-lightning_version": "1.9.5",
        "global_step": 1,
        "state_dict": {"weight": torch.ones(1)},
        "optimizer_states": [
            {"state": {0: {"step": torch.tensor(1)}}, "param_groups": [{"params": [0], "lr": 0.1}]}
        ],
        "lr_schedulers": [{"last_epoch": 1}],
        "run_metadata": metadata,
    }


def test_model_build_failure_removes_only_provisional_manifest_and_fresh_retry_works(
    tmp_path, monkeypatch
):
    train, root, config_path, _, events = _fake_run(tmp_path, monkeypatch)
    original_build_model = train.build_model
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    def failed_model(config):
        assert (root / "run_metadata.json").is_file()
        raise RuntimeError("FLAN loader failed")

    monkeypatch.setattr(train, "build_model", failed_model)
    with pytest.raises(RuntimeError, match="FLAN loader failed"):
        train.main()

    assert not (root / "run_metadata.json").exists()
    monkeypatch.setattr(train, "build_model", original_build_model)
    train.main()
    assert (root / "run_metadata.json").is_file()
    assert events[-1] == ("fit", {})


def test_metadata_update_failure_removes_own_provisional_manifest(tmp_path, monkeypatch):
    train, root, config_path, _, _ = _fake_run(tmp_path, monkeypatch)
    original_write = train._write_run_metadata
    writes = 0

    def fail_update(*args):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("metadata update failed")
        return original_write(*args)

    monkeypatch.setattr(train, "_write_run_metadata", fail_update)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    with pytest.raises(OSError, match="metadata update failed"):
        train.main()

    assert not (root / "run_metadata.json").exists()


def test_build_failure_does_not_delete_concurrent_manifest_replacement(tmp_path, monkeypatch):
    train, root, config_path, _, _ = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    def replaced_then_failed(config):
        manifest = root / "run_metadata.json"
        replacement = root / "concurrent.json"
        replacement.write_bytes(manifest.read_bytes())
        replacement.replace(manifest)
        raise RuntimeError("FLAN loader failed")

    monkeypatch.setattr(train, "build_model", replaced_then_failed)
    with pytest.raises(RuntimeError, match="FLAN loader failed"):
        train.main()

    assert (root / "run_metadata.json").is_file()


def test_resume_build_failure_preserves_existing_manifest(tmp_path, monkeypatch):
    train, root, config_path, _, _ = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    manifest = root / "run_metadata.json"
    original = manifest.read_bytes()
    checkpoint = tmp_path / "tiny.ckpt"
    torch.save(_tiny_resume_checkpoint(json.loads(original)), checkpoint)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--resume", str(checkpoint), "--trusted-resume"],
    )

    def failed_model(config):
        raise RuntimeError("FLAN loader failed")

    monkeypatch.setattr(train, "build_model", failed_model)
    with pytest.raises(RuntimeError, match="FLAN loader failed"):
        train.main()

    assert manifest.read_bytes() == original
    assert set(root.iterdir()) == {manifest}


@pytest.mark.parametrize("key", ["hf_token", "client_secret", "apiKey", "clientPassword"])
def test_train_cli_rejects_nested_composite_secret_keys_before_model(tmp_path, monkeypatch, key):
    train, root, config_path, _, events = _fake_run(
        tmp_path, monkeypatch, extra_config={"nested": [{"deeper": {key: "DUMMY_DO_NOT_WRITE"}}]}
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    with pytest.raises(ValueError, match=key) as error:
        train.main()

    assert "DUMMY_DO_NOT_WRITE" not in str(error.value)
    assert not (root / "run_metadata.json").exists()
    assert "model" not in events


def test_train_cli_allows_non_secret_tokenizer_key(tmp_path, monkeypatch):
    train, root, config_path, _, _ = _fake_run(
        tmp_path, monkeypatch, extra_config={"nested": [{"tokenizer": "local"}]}
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    train.main()

    assert json.loads((root / "run_metadata.json").read_text())["config"]["nested"] == [
        {"tokenizer": "local"}
    ]


def test_resume_requires_explicit_trust_before_opening_even_missing_path(tmp_path, monkeypatch):
    train = _train_script()
    missing = tmp_path / "untrusted.ckpt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(tmp_path / "absent.yaml"), "--resume", str(missing)],
    )
    monkeypatch.setattr(train.torch, "load", lambda *a, **kw: pytest.fail("untrusted load"))

    with pytest.raises(ValueError, match="--trusted-resume"):
        train.main()


@pytest.mark.parametrize("in_root", [True, False])
@pytest.mark.parametrize("fit_fails", [True, False])
def test_resume_uses_single_snapshot_even_if_source_changes(
    tmp_path, monkeypatch, in_root, fit_fails
):
    train, root, config_path, model, _ = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    source = (root if in_root else tmp_path) / "source.ckpt"
    torch.save(
        _tiny_resume_checkpoint(json.loads((root / "run_metadata.json").read_text())), source
    )
    original = source.read_bytes()
    observed = []

    def fake_model(config):
        source.write_bytes(b"swapped after safe preflight")
        return model

    class FakeTrainer:
        def __init__(self, **kwargs):
            pass

        def fit(self, fitted_model, *, datamodule, ckpt_path):
            snapshot = Path(ckpt_path)
            observed.append(snapshot)
            assert snapshot != source
            assert snapshot.parent == root
            assert snapshot.read_bytes() == original
            assert fitted_model.run_metadata["resume_checkpoint_sha256"] == sha256_file(snapshot)
            if fit_fails:
                raise RuntimeError("fit failed")

    monkeypatch.setattr(train, "build_model", fake_model)
    monkeypatch.setattr(train.pl, "Trainer", FakeTrainer)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--resume", str(source), "--trusted-resume"],
    )

    if fit_fails:
        with pytest.raises(RuntimeError, match="fit failed"):
            train.main()
    else:
        train.main()

    assert len(observed) == 1
    assert not observed[0].exists()
    assert source.read_bytes() == b"swapped after safe preflight"


def test_resume_rejects_symlink_before_reading_checkpoint(tmp_path, monkeypatch):
    train, root, config_path, _, events = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    manifest = root / "run_metadata.json"
    before = manifest.read_bytes()
    target = tmp_path / "target.ckpt"
    torch.save(_tiny_resume_checkpoint(json.loads(before)), target)
    source = tmp_path / "linked.ckpt"
    source.symlink_to(target)
    events.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--resume", str(source), "--trusted-resume"],
    )

    with pytest.raises(ValueError, match="symlink"):
        train.main()

    assert manifest.read_bytes() == before
    assert set(root.iterdir()) == {manifest}
    assert "model" not in events


def test_safe_preflight_failure_keeps_previous_manifest_and_cleans_snapshot(tmp_path, monkeypatch):
    train, root, config_path, _, events = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    manifest = root / "run_metadata.json"
    before = manifest.read_bytes()
    source = tmp_path / "legacy.ckpt"
    torch.save({**_tiny_resume_checkpoint(json.loads(before)), "unsupported": np.arange(1)}, source)
    events.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--resume", str(source), "--trusted-resume"],
    )

    with pytest.raises(ValueError, match="weights_only=True"):
        train.main()

    assert manifest.read_bytes() == before
    assert set(root.iterdir()) == {manifest}
    assert "model" not in events


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_dict", {}),
        ("state_dict", {"weight": "not a tensor"}),
        ("optimizer_states", []),
        ("optimizer_states", [{"state": {}, "param_groups": []}]),
        ("lr_schedulers", []),
        ("pytorch-lightning_version", 1),
        ("global_step", -1),
        ("global_step", True),
    ],
)
def test_resume_rejects_malformed_lightning_state_before_changing_manifest(
    tmp_path, monkeypatch, field, value
):
    train, root, config_path, _, events = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    manifest = root / "run_metadata.json"
    before = manifest.read_bytes()
    source = tmp_path / "malformed.ckpt"
    torch.save({**_tiny_resume_checkpoint(json.loads(before)), field: value}, source)
    events.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--resume", str(source), "--trusted-resume"],
    )

    with pytest.raises(ValueError, match=field):
        train.main()

    assert manifest.read_bytes() == before
    assert set(root.iterdir()) == {manifest}
    assert "model" not in events


def test_tiny_lightning_trainer_checkpoint_round_trip_through_cli(tmp_path, monkeypatch):
    train = _train_script()
    root = tmp_path / "run"
    spatial = tmp_path / "spatial.json"
    motion = tmp_path / "motion.json"
    spatial.write_text("spatial")
    motion.write_text("motion")
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(
        {
            "seed": 7,
            "trainer": {
                "default_root_dir": str(root),
                "accelerator": "cpu",
                "devices": 1,
                "max_steps": 2,
                "max_epochs": 2,
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "limit_train_batches": 1,
            },
            "model": {"name": "tiny-local", "spatial_dim": 2048, "motion_dim": 1024},
            "data": {"spatial_manifest": str(spatial), "motion_manifest": str(motion)},
        },
        config_path,
    )

    class ToyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)
            self.loaded_steps = []
            self.start_optimizer = None
            self.start_scheduler = None

        def on_train_start(self):
            self.start_optimizer = deepcopy(self.trainer.optimizers[0].state_dict())
            self.start_scheduler = deepcopy(
                self.trainer.lr_scheduler_configs[0].scheduler.state_dict()
            )

        def training_step(self, batch, batch_idx):
            features, targets = batch
            return (self.layer(features) - targets).square().mean()

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        def on_save_checkpoint(self, checkpoint):
            checkpoint["run_metadata"] = self.run_metadata

        def on_load_checkpoint(self, checkpoint):
            train.validate_run_metadata(self.run_metadata, checkpoint["run_metadata"])
            self.loaded_steps.append(checkpoint["global_step"])

    class ToyData(pl.LightningDataModule):
        def train_dataloader(self):
            dataset = torch.utils.data.TensorDataset(torch.ones(2, 1), torch.zeros(2, 1))
            return torch.utils.data.DataLoader(dataset, batch_size=1)

    source = tmp_path / "lightning.ckpt"

    class SaveAndStopAfterFirstEpoch(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            if trainer.global_step == 1:
                trainer.save_checkpoint(source)
                trainer.should_stop = True

    trainers = []
    models = []
    real_trainer = pl.Trainer

    def build_trainer(**kwargs):
        if not trainers:
            kwargs["callbacks"] = [SaveAndStopAfterFirstEpoch()]
        trainer = real_trainer(**kwargs)
        trainers.append(trainer)
        return trainer

    def build_model(config):
        model = ToyModel()
        models.append(model)
        return model

    monkeypatch.setattr(train.pl, "Trainer", build_trainer)
    monkeypatch.setattr(train, "build_data", lambda config: ToyData())
    monkeypatch.setattr(train, "build_model", build_model)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    assert trainers[0].global_step == 1
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    assert checkpoint["global_step"] == 1
    assert checkpoint["run_metadata"]["resume_reproducibility"] == "fresh"
    assert checkpoint["optimizer_states"][0]["state"]
    assert checkpoint["lr_schedulers"][0]["last_epoch"] == 1
    previous = (root / "run_metadata.json").read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--resume", str(source), "--trusted-resume"],
    )

    train.main()

    assert trainers[1].global_step == 2
    assert models[1].loaded_steps == [1]
    assert models[1].run_metadata["resume_checkpoint_sha256"] == sha256_file(source)
    assert models[1].run_metadata["resume_reproducibility"] == "optimizer_state_only"
    resumed_checkpoint = tmp_path / "resumed.ckpt"
    trainers[1].save_checkpoint(resumed_checkpoint)
    resumed_metadata = torch.load(resumed_checkpoint, map_location="cpu", weights_only=True)[
        "run_metadata"
    ]
    assert resumed_metadata["resume_reproducibility"] == "optimizer_state_only"
    saved_optimizer = checkpoint["optimizer_states"][0]
    assert models[1].start_optimizer["param_groups"] == saved_optimizer["param_groups"]
    for param_id, moments in saved_optimizer["state"].items():
        for name, value in moments.items():
            torch.testing.assert_close(models[1].start_optimizer["state"][param_id][name], value)
    assert models[1].start_scheduler == checkpoint["lr_schedulers"][0]

    reference_config = tmp_path / "reference.yaml"
    reference = OmegaConf.load(config_path)
    reference.trainer.default_root_dir = str(tmp_path / "reference-run")
    OmegaConf.save(reference, reference_config)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(reference_config)])
    train.main()

    assert trainers[2].global_step == 2
    for name, parameter in models[1].state_dict().items():
        torch.testing.assert_close(parameter, models[2].state_dict()[name])
    resumed_optimizer = trainers[1].optimizers[0].state_dict()
    continuous_optimizer = trainers[2].optimizers[0].state_dict()
    assert resumed_optimizer["param_groups"] == continuous_optimizer["param_groups"]
    for param_id, moments in continuous_optimizer["state"].items():
        for name, value in moments.items():
            torch.testing.assert_close(resumed_optimizer["state"][param_id][name], value)
    resumed_scheduler = trainers[1].lr_scheduler_configs[0].scheduler
    continuous_scheduler = trainers[2].lr_scheduler_configs[0].scheduler
    assert resumed_scheduler.state_dict() == continuous_scheduler.state_dict()
    assert resumed_scheduler.last_epoch == 2
    assert trainers[1].optimizers[0].param_groups[0]["lr"] == pytest.approx(0.01 / 3)
    assert (root / "run_metadata.json").read_bytes() != previous
    assert set(root.iterdir()) == {root / "run_metadata.json"}


def test_train_cli_uses_project_lock_when_launched_elsewhere(tmp_path, monkeypatch):
    train, root, config_path, model, events = _fake_run(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    train.main()

    metadata = json.loads((root / "run_metadata.json").read_text())
    lock = Path(train.__file__).resolve().parents[1] / "uv.lock"
    assert metadata["package_lock_sha256"] == sha256_file(lock)
    assert model.run_metadata is not None
    assert events == ["trainer", "data", "model", ("fit", {})]


def test_train_cli_never_loads_checkpoint_without_resume(tmp_path, monkeypatch):
    train, _, config_path, _, _ = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    monkeypatch.setattr(
        train.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("loaded checkpoint without --resume"),
    )
    monkeypatch.setattr(train, "sha256_file", lambda path: pytest.fail("hashed checkpoint"))

    train.main()


def test_train_cli_resumes_only_explicit_checkpoint_and_records_its_hash(tmp_path, monkeypatch):
    train, root, config_path, model, events = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    previous = json.loads((root / "run_metadata.json").read_text())
    checkpoint_path = tmp_path / "tiny.ckpt"
    torch.save(_tiny_resume_checkpoint(previous), checkpoint_path)
    events.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--resume",
            str(checkpoint_path),
            "--trusted-resume",
        ],
    )

    train.main()

    assert events[:3] == ["trainer", "data", "model"]
    snapshot = Path(events[3][1]["ckpt_path"])
    assert snapshot != checkpoint_path and snapshot.parent == root
    assert not snapshot.exists()
    assert model.run_metadata["resume_checkpoint_sha256"] == sha256_file(checkpoint_path)
    assert json.loads((root / "run_metadata.json").read_text()) == model.run_metadata


def test_run_manifest_marks_fresh_and_subsequent_resumes_without_blocking(tmp_path, monkeypatch):
    train, root, config_path, model, _ = _fake_run(tmp_path, monkeypatch)
    manifest = root / "run_metadata.json"
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    fresh = json.loads(manifest.read_text())
    assert fresh["resume_reproducibility"] == "fresh"
    assert "resume_checkpoint_sha256" not in fresh

    previous = fresh
    for attempt in (1, 2):
        source = tmp_path / f"resume-{attempt}.ckpt"
        torch.save(_tiny_resume_checkpoint(previous), source)
        monkeypatch.setattr(
            sys,
            "argv",
            ["train.py", "--config", str(config_path), "--resume", str(source), "--trusted-resume"],
        )
        train.main()
        previous = json.loads(manifest.read_text())
        assert previous["resume_reproducibility"] == "optimizer_state_only"
        assert previous["resume_checkpoint_sha256"] == sha256_file(source)
        assert model.run_metadata == previous
    assert set(root.iterdir()) == {manifest}


def test_train_cli_requires_existing_resume_file_before_loading_model(tmp_path, monkeypatch):
    train, root, config_path, _, events = _fake_run(tmp_path, monkeypatch)
    missing = tmp_path / "missing.ckpt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--resume", str(missing), "--trusted-resume"],
    )

    with pytest.raises(FileNotFoundError, match="missing.ckpt"):
        train.main()

    assert "model" not in events
    assert not (root / "run_metadata.json").exists()


def test_train_cli_refuses_second_run_without_resume_and_keeps_manifest(tmp_path, monkeypatch):
    train, root, config_path, _, events = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    manifest = root / "run_metadata.json"
    before = manifest.read_bytes()
    events.clear()

    with pytest.raises(ValueError, match="resume"):
        train.main()

    assert manifest.read_bytes() == before
    assert "model" not in events


@pytest.mark.parametrize("checkpoint_kind", ["lightning_missing", "converted_missing", "mismatch"])
def test_train_cli_refuses_incompatible_resume_without_touching_manifest(
    tmp_path, monkeypatch, checkpoint_kind
):
    train, root, config_path, _, events = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    manifest = root / "run_metadata.json"
    before = manifest.read_bytes()
    checkpoint = _tiny_resume_checkpoint(json.loads(before))
    if checkpoint_kind == "mismatch":
        checkpoint["run_metadata"] = {**json.loads(before), "spatial_manifest_sha256": "bad"}
    if checkpoint_kind == "lightning_missing":
        del checkpoint["run_metadata"]
    if checkpoint_kind == "converted_missing":
        checkpoint = {"metadata": {"source_sha256": "converted"}}
    path = tmp_path / "tiny.ckpt"
    torch.save(checkpoint, path)
    events.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--resume", str(path), "--trusted-resume"],
    )

    with pytest.raises(ValueError, match="state_dict|run_metadata|spatial_manifest_sha256"):
        train.main()

    assert manifest.read_bytes() == before
    assert "model" not in events


def test_train_cli_rejects_resume_when_root_manifest_settings_changed(tmp_path, monkeypatch):
    train, root, config_path, _, events = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    manifest = root / "run_metadata.json"
    before = manifest.read_bytes()
    path = tmp_path / "tiny.ckpt"
    torch.save(_tiny_resume_checkpoint(json.loads(before)), path)
    (tmp_path / "spatial.json").write_text("changed")
    events.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--resume", str(path), "--trusted-resume"],
    )

    with pytest.raises(ValueError, match="spatial_manifest_sha256"):
        train.main()

    assert manifest.read_bytes() == before
    assert "model" not in events


def test_train_cli_rejects_changed_model_revision_on_resume_without_overwriting_manifest(
    tmp_path, monkeypatch
):
    train, root, config_path, model, events = _fake_run(tmp_path, monkeypatch, revision="a" * 40)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    manifest = root / "run_metadata.json"
    before = manifest.read_bytes()
    path = tmp_path / "tiny.ckpt"
    torch.save(_tiny_resume_checkpoint(json.loads(before)), path)
    model.language_model.model.config._commit_hash = "b" * 40
    events.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "--resume", str(path), "--trusted-resume"],
    )

    with pytest.raises(ValueError, match="model_source.revision"):
        train.main()

    assert manifest.read_bytes() == before
    assert not any(isinstance(event, tuple) and event[0] == "fit" for event in events)


@pytest.mark.parametrize("revision", ["a" * 40, "not-verified", None])
def test_train_cli_records_only_verified_model_revision(tmp_path, monkeypatch, revision):
    train, root, config_path, model, _ = _fake_run(tmp_path, monkeypatch, revision=revision)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    train.main()

    source = json.loads((root / "run_metadata.json").read_text())["model_source"]
    assert source["identifier"] == "google/flan-t5-xl"
    assert source["tuning_type"] == "lora"
    assert source["revision_status"] == ("resolved" if revision == "a" * 40 else "unresolved")
    assert source["revision"] == (revision if revision == "a" * 40 else None)
    assert model.run_metadata["model_source"] == source


def test_train_cli_records_peft_base_model_commit_when_exposed(tmp_path, monkeypatch):
    train, root, config_path, model, _ = _fake_run(tmp_path, monkeypatch)
    base = SimpleNamespace(config=SimpleNamespace(_commit_hash="b" * 40))
    model.language_model.model = SimpleNamespace(get_base_model=lambda: base)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    train.main()

    assert json.loads((root / "run_metadata.json").read_text())["model_source"] == {
        "identifier": "google/flan-t5-xl",
        "tuning_type": "lora",
        "revision": "b" * 40,
        "revision_status": "resolved",
    }


@pytest.mark.parametrize("key", ["credential", "secret", "password", "token", "api_key"])
def test_train_cli_does_not_serialize_secret_named_config_keys(tmp_path, monkeypatch, key):
    train, root, config_path, _, events = _fake_run(
        tmp_path, monkeypatch, extra_config={"nested": [{key: "DUMMY_DO_NOT_WRITE"}]}
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    with pytest.raises(ValueError, match=key) as error:
        train.main()

    assert "DUMMY_DO_NOT_WRITE" not in str(error.value)
    assert not (root / "run_metadata.json").exists()
    assert "model" not in events


def test_train_cli_rejects_unignored_root_inside_project_before_model_build(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    root = repo / "training-provenance-unignored"
    train, _, config_path, _, events = _fake_run(tmp_path, monkeypatch, root=root)
    monkeypatch.setattr(
        train, "_write_run_metadata", lambda *args: pytest.fail("wrote unignored output")
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    with pytest.raises(ValueError, match="Git.*ignored"):
        train.main()

    assert not root.exists()
    assert "model" not in events


@pytest.mark.parametrize("tuning_type", ["freeze", None])
def test_train_cli_writes_resolved_provenance_before_model_and_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tuning_type: str | None
) -> None:
    train = _train_script()
    spatial = tmp_path / "spatial.json"
    motion = tmp_path / "motion.json"
    spatial.write_text("spatial")
    motion.write_text("motion")
    root = tmp_path / "artifacts"
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "seed: 7\n"
        f"trainer:\n  max_epochs: 1\n  default_root_dir: {root}\n"
        "model:\n  spatial_dim: 2048\n  motion_dim: 1024\n"
        "  name: google/flan-t5-xl\n"
        + (f"  tuning_type: {tuning_type}\n" if tuning_type else "")
        + "data:\n"
        + f"  spatial_manifest: {spatial}\n  motion_manifest: {motion}\n"
        + "  batch_size: ${trainer.max_epochs}\n"
    )
    model = SimpleNamespace()
    events = []

    def fake_data(config):
        events.append("data")
        return "dataset"

    def fake_model(config):
        events.append("model")
        assert (root / "run_metadata.json").is_file()
        return model

    class FakeTrainer:
        def __init__(self, **kwargs):
            events.append("trainer")

        def fit(self, fitted_model, *, datamodule):
            events.append("fit")
            assert fitted_model is model
            assert datamodule == "dataset"
            metadata = json.loads((root / "run_metadata.json").read_text())
            assert fitted_model.run_metadata == metadata
            assert metadata["config"]["data"]["batch_size"] == 2
            assert metadata["config"]["trainer"]["max_epochs"] == 2
            assert metadata["seed"] == 7
            assert metadata["git_revision"]
            assert isinstance(metadata["git_dirty"], bool)
            assert metadata["torch_version"]
            assert "cuda_version" in metadata
            assert metadata["package_lock_sha256"] == sha256_file(Path("uv.lock"))
            assert metadata["spatial_manifest_sha256"] == sha256_file(spatial)
            assert metadata["motion_manifest_sha256"] == sha256_file(motion)
            assert metadata["model_source"] == {
                "identifier": "google/flan-t5-xl",
                "tuning_type": tuning_type or "lora",
                "revision": None,
                "revision_status": "unresolved",
            }

    monkeypatch.setattr(train.pl, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.pl, "Trainer", FakeTrainer)
    monkeypatch.setattr(train, "build_data", fake_data)
    monkeypatch.setattr(train, "build_model", fake_model)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(config_path), "trainer.max_epochs=2"],
    )

    train.main()

    assert events == ["trainer", "data", "model", "fit"]
    assert list(root.iterdir()) == [root / "run_metadata.json"]


def test_run_metadata_replace_failure_keeps_previous_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = _train_script()
    root = tmp_path / "artifacts"
    root.mkdir()
    manifest = root / "run_metadata.json"
    manifest.write_text('{"seed": 7}\n')

    def failed_replace(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(train.os, "replace", failed_replace)

    with pytest.raises(OSError, match="disk unavailable"):
        train._write_run_metadata(root, {"seed": 13})

    assert manifest.read_text() == '{"seed": 7}\n'
    assert list(root.iterdir()) == [manifest]


def test_train_cli_merges_explicit_layers_and_overrides_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = _train_script()
    layers = [
        "data:\n  batch_size: 2\n"
        f"  spatial_manifest: {tmp_path / 'spatial.json'}\n"
        f"  motion_manifest: {tmp_path / 'motion.json'}\n"
        "model:\n  spatial_dim: 2048\n  motion_dim: 1024\n",
        "model:\n  name: first\n",
        f"seed: 7\ntrainer:\n  max_epochs: 500\n  default_root_dir: {tmp_path / 'artifacts'}\n",
        "model:\n  name: smoke\ntrainer:\n  max_epochs: 1\n",
        "data:\n  batch_size: 3\nmodel:\n  name: local\n",
    ]
    paths = []
    for index, contents in enumerate(layers):
        path = tmp_path / f"layer-{index}.yaml"
        path.write_text(contents)
        paths.append(path)
    events = []
    model = SimpleNamespace()

    class FakeTrainer:
        def __init__(self, **kwargs):
            events.append(("trainer", kwargs))

        def fit(self, model, *, datamodule):
            events.append(("fit", model, datamodule))

    monkeypatch.setattr(
        train.pl, "seed_everything", lambda seed, workers: events.append(("seed", seed, workers))
    )
    monkeypatch.setattr(train.pl, "Trainer", FakeTrainer)

    def fake_data(config):
        events.append(("data", config.data.batch_size, config.model.name))
        return "dataset"

    def fake_model(config):
        events.append(("model", config.trainer.max_epochs))
        return model

    monkeypatch.setattr(train, "build_data", fake_data)
    monkeypatch.setattr(train, "build_model", fake_model)
    monkeypatch.setattr(train, "collect_runtime_metadata", lambda *args: {"seed": args[0]})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            *(arg for path in paths for arg in ("--config", str(path))),
            "trainer.max_epochs=2",
        ],
    )

    train.main()

    assert events == [
        ("seed", 7, True),
        ("trainer", {"max_epochs": 2, "default_root_dir": str(tmp_path / "artifacts")}),
        ("data", 3, "local"),
        ("model", 2),
        ("fit", model, "dataset"),
    ]


def test_train_cli_missing_manifest_fails_before_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = _train_script()
    path = tmp_path / "config.yaml"
    path.write_text(
        "seed: 0\ntrainer:\n  max_epochs: 1\nmodel:\n  spatial_dim: 2048\n  motion_dim: 1024\n"
        "data:\n  spatial_manifest: " + str(tmp_path / "missing.json") + "\n"
        "  motion_manifest: " + str(tmp_path / "also-missing.json") + "\n"
    )
    monkeypatch.setattr(train.pl, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train.pl, "Trainer", lambda **kwargs: SimpleNamespace(fit=lambda *args, **kwargs: None)
    )

    def unexpected_model(config):
        pytest.fail("weights loaded before checking data")

    monkeypatch.setattr(train, "build_model", unexpected_model)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(path)])

    with pytest.raises(FileNotFoundError, match="missing.json"):
        train.main()


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("manifest", "in manifest"),
        ("file", "missing spatial feature file"),
        ("dtype", "metadata mismatch"),
        ("length", "metadata mismatch"),
        ("header", "invalid spatial feature"),
    ],
)
def test_train_cli_preflights_second_clip_before_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, reason: str
) -> None:
    train = _train_script()
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    clips = {"train": ("train-1", "train-2"), "dev": ("dev-1",), "test": ("test-1",)}
    for split, filename in (
        ("train", "train_info_ml.npy"),
        ("dev", "dev_info_ml.npy"),
        ("test", "test_info_ml.npy"),
    ):
        np.save(
            annotations / filename,
            {
                index: {
                    "fileid": clip,
                    "signer": "Signer01",
                    "gloss": "WIND",
                    "text": "wind",
                    "en_text": "wind",
                    "es_text": "viento",
                    "fr_text": "vent",
                }
                for index, clip in enumerate(clips[split])
            },
        )
    data = {"annotation_root": str(annotations), "batch_size": 2, "num_workers": 0}
    for modality, width in (("spatial", 2048), ("motion", 1024)):
        root = tmp_path / modality
        records = []
        for split, split_clips in clips.items():
            (root / split).mkdir(parents=True)
            for clip in split_clips:
                bad_clip = modality == "spatial" and clip == "train-2"
                if bad_clip and case == "manifest":
                    continue
                records.append(
                    FeatureRecord(clip, split, f"{split}/{clip}.npy", 1, width, "float32")
                )
                path = root / split / f"{clip}.npy"
                if bad_clip and case == "file":
                    continue
                if bad_clip and case == "header":
                    path.write_bytes(b"not a numpy header")
                else:
                    length = 2 if bad_clip and case == "length" else 1
                    dtype = np.float64 if bad_clip and case == "dtype" else np.float32
                    np.save(path, np.zeros((length, width), dtype=dtype))
        manifest_path = tmp_path / f"{modality}.json"
        FeatureManifest(1, modality, width, tuple(records)).save(manifest_path)
        data[f"{modality}_root"] = str(root)
        data[f"{modality}_manifest"] = str(manifest_path)

    config_path = tmp_path / "config.yaml"
    OmegaConf.save(
        {
            "seed": 0,
            "trainer": {"max_epochs": 1},
            "model": {
                "spatial_dim": 2048,
                "motion_dim": 1024,
                "name": "google/flan-t5-xl",
                "cache_dir": "/unused",
                "max_text_length": 64,
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            },
            "data": data,
        },
        config_path,
    )
    monkeypatch.setattr(train.pl, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train.pl, "Trainer", lambda **kwargs: SimpleNamespace(fit=lambda *args, **kwargs: None)
    )

    def unexpected_loader(*args, **kwargs):
        pytest.fail("Flan weights loaded before checking second clip")

    monkeypatch.setattr(FlanT5Backbone, "from_pretrained", unexpected_loader)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    with pytest.raises((ValueError, FileNotFoundError)) as error:
        train.main()
    assert "spatial" in str(error.value)
    assert "train/train-2" in str(error.value)
    assert reason in str(error.value)
    if case == "dtype":
        assert "float64" in str(error.value)
    if case == "length":
        assert "(2, 2048)" in str(error.value)


def _fake_annotation_root(config_path, tmp_path):
    ann = tmp_path / "ann"
    ann.mkdir()
    for split in ("train", "dev", "test"):
        np.save(
            ann / f"{split}_info_ml.npy",
            {0: {"fileid": split, "num_frames": 5, "text": "original text"}},
        )
    config = OmegaConf.load(config_path)
    config.data.annotation_root = str(ann)
    OmegaConf.save(config, config_path)
    return ann


def test_comparison_preflight_stops_before_model_and_training(tmp_path, monkeypatch):
    root = tmp_path / "run"
    train, _, config_path, _, events = _fake_run(
        tmp_path,
        monkeypatch,
        root=root,
        extra_config={
            "comparison": {"enabled": True, "smoke": True},
            "trainer": {"default_root_dir": str(root), "max_steps": 1},
        },
    )
    monkeypatch.setattr(
        train,
        "preflight_run",
        lambda config, paths, overrides: (_ for _ in ()).throw(
            ValueError("DINO complete manifest missing")
        ),
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    with pytest.raises(ValueError, match="DINO complete manifest missing"):
        train.main()
    assert "model" not in events
    assert not (root / "run_metadata.json").exists()


def test_train_cli_rejects_dino_overridden_to_clip_before_model(tmp_path, monkeypatch):
    train = _train_script()
    configs = Path(__file__).resolve().parents[2] / "configs"
    feature_root = tmp_path / "features"
    monkeypatch.setenv("PHOENIX14T_ANNOTATION_ROOT", str(tmp_path / "ann"))
    monkeypatch.setenv("DESPAMO_FEATURE_ROOT", str(feature_root))
    monkeypatch.setenv("DESPAMO_HF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("DINO_ROOT", str(tmp_path / "dino-key"))
    monkeypatch.setenv("COMPARISON_RUN_DIR", str(tmp_path / "run"))
    layers = [
        configs / "data/phoenix14t.yaml",
        configs / "model/spamo_flan_t5_xl.yaml",
        configs / "experiment/phoenix14t_baseline.yaml",
        configs / "experiment/phoenix14t_encoder_comparison.yaml",
        configs / "experiment/phoenix14t_dinov3.yaml",
    ]
    spatial = feature_root / "manifests/phoenix14t_spatial.json"
    monkeypatch.setattr(
        "despamo.comparison.preflight_sources",
        lambda config: pytest.fail("feature I/O started before binding"),
    )
    monkeypatch.setattr(train, "build_data", lambda config: pytest.fail("data built"))
    monkeypatch.setattr(train, "build_model", lambda config: pytest.fail("model built"))
    monkeypatch.setattr(train.pl, "Trainer", lambda **kwargs: pytest.fail("trainer built"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            *(part for path in layers for part in ("--config", str(path))),
            f"data.spatial_root={feature_root / 'vit_feat_Phoenix14T'}",
            f"data.spatial_manifest={spatial}",
        ],
    )
    with pytest.raises(ValueError, match="dino spatial source binding"):
        train.main()
    assert not (tmp_path / "run/run_metadata.json").exists()


def test_comparison_saves_only_exact_step_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "run"
    train, _, config_path, model, _ = _fake_run(
        tmp_path,
        monkeypatch,
        root=root,
        revision=FLAN_SHA,
        extra_config={
            "comparison": {"enabled": True, "smoke": True},
            "trainer": {"default_root_dir": str(root), "max_steps": 1},
            "model": {
                "spatial_dim": 2048,
                "motion_dim": 1024,
                "name": "google/flan-t5-xl",
                "revision": FLAN_SHA,
            },
        },
    )
    ann = _fake_annotation_root(config_path, tmp_path)
    monkeypatch.setattr(train, "preflight_run", lambda config, paths, overrides: {})
    saved = []

    class FakeTrainer:
        global_step = 1

        def __init__(self, **kwargs):
            assert kwargs["max_steps"] == 1

        def fit(self, model, *, datamodule):
            pass

        def save_checkpoint(self, path):
            saved.append(Path(path))
            Path(path).write_bytes(b"fake full-state checkpoint")

    monkeypatch.setattr(train.pl, "Trainer", FakeTrainer)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    assert saved == [root / "checkpoints/final.ckpt"]
    assert saved[0].read_bytes() == b"fake full-state checkpoint"
    assert model.run_metadata["annotation_sha256"] == {
        split: sha256_file(ann / f"{split}_info_ml.npy") for split in ("train", "dev", "test")
    }
    from despamo.comparison import comparison_code_sha256

    assert model.run_metadata["comparison_code_sha256"] == comparison_code_sha256(
        train.PROJECT_ROOT
    )
    assert model.run_metadata["dino_frame_rows_sha256"] == "e" * 64


def test_comparison_rejects_wrong_resolved_flan_before_fit(tmp_path, monkeypatch):
    root = tmp_path / "run"
    train, _, config_path, _, events = _fake_run(
        tmp_path,
        monkeypatch,
        root=root,
        revision="c" * 40,
        extra_config={
            "comparison": {"enabled": True, "smoke": True},
            "trainer": {"default_root_dir": str(root), "max_steps": 1},
            "model": {
                "spatial_dim": 2048,
                "motion_dim": 1024,
                "name": "google/flan-t5-xl",
                "revision": FLAN_SHA,
            },
        },
    )
    _fake_annotation_root(config_path, tmp_path)
    monkeypatch.setattr(train, "preflight_run", lambda config, paths, overrides: {})
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    with pytest.raises(ValueError, match="comparison Flan revision mismatch"):
        train.main()
    assert not (root / "run_metadata.json").exists()
    assert not any(isinstance(event, tuple) and event[0] == "fit" for event in events)


def test_baseline_train_does_not_hash_annotations_or_add_metadata(tmp_path, monkeypatch):
    train, root, config_path, _, _ = _fake_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        train, "annotation_hashes", lambda config: pytest.fail("baseline hashed annotations")
    )
    monkeypatch.setattr(
        train, "comparison_code_sha256", lambda root: pytest.fail("baseline hashed code")
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    train.main()
    assert "annotation_sha256" not in json.loads((root / "run_metadata.json").read_text())
    assert "comparison_code_sha256" not in json.loads((root / "run_metadata.json").read_text())


def test_comparison_rejects_resume_before_preflight_and_model(tmp_path, monkeypatch):
    root = tmp_path / "run"
    train, _, config_path, _, events = _fake_run(
        tmp_path,
        monkeypatch,
        root=root,
        extra_config={"comparison": {"enabled": True, "smoke": True}},
    )
    monkeypatch.setattr(train, "preflight_run", lambda *args: pytest.fail("preflight started"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--resume",
            str(tmp_path / "missing.ckpt"),
            "--trusted-resume",
        ],
    )
    with pytest.raises(ValueError, match="--resume forbidden"):
        train.main()
    assert not events
    assert not (root / "run_metadata.json").exists()


def test_comparison_stopped_early_never_saves_final_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "run"
    train, _, config_path, _, _ = _fake_run(
        tmp_path,
        monkeypatch,
        root=root,
        revision=FLAN_SHA,
        extra_config={
            "comparison": {"enabled": True, "smoke": True},
            "trainer": {"default_root_dir": str(root), "max_steps": 1},
            "model": {
                "spatial_dim": 2048,
                "motion_dim": 1024,
                "name": "google/flan-t5-xl",
                "revision": FLAN_SHA,
            },
        },
    )
    _fake_annotation_root(config_path, tmp_path)
    monkeypatch.setattr(train, "preflight_run", lambda *args: {})

    class StoppedTrainer:
        global_step = 0

        def __init__(self, **kwargs):
            pass

        def fit(self, model, *, datamodule):
            pass

        def save_checkpoint(self, path):
            pytest.fail("saved incomplete checkpoint")

    monkeypatch.setattr(train.pl, "Trainer", StoppedTrainer)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    with pytest.raises(ValueError, match="exact optimizer-step budget"):
        train.main()
    assert not (root / "checkpoints/final.ckpt").exists()


def test_comparison_lightning_cpu_final_checkpoint_carries_full_state(tmp_path, monkeypatch):
    train = _train_script()
    root = tmp_path / "run"
    spatial, motion = tmp_path / "spatial.json", tmp_path / "motion.json"
    spatial.write_text("spatial")
    motion.write_text("motion")
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(
        {
            "seed": 0,
            "comparison": {"enabled": True, "smoke": True},
            "trainer": {
                "default_root_dir": str(root),
                "accelerator": "cpu",
                "devices": 1,
                "max_steps": 1,
                "max_epochs": -1,
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
            "data": {"spatial_manifest": str(spatial), "motion_manifest": str(motion)},
            "model": {
                "spatial_dim": 2048,
                "motion_dim": 1024,
                "name": "google/flan-t5-xl",
                "revision": FLAN_SHA,
            },
        },
        config_path,
    )
    ann = _fake_annotation_root(config_path, tmp_path)

    class ToyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def training_step(self, batch, batch_idx):
            return (self.weight * batch).square().mean()

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        def on_save_checkpoint(self, checkpoint):
            checkpoint["run_metadata"] = self.run_metadata

    class ToyData(pl.LightningDataModule):
        def train_dataloader(self):
            return torch.utils.data.DataLoader([torch.tensor([1.0])], batch_size=1)

    model = ToyModel()
    monkeypatch.setattr(train, "preflight_run", lambda *args: {})
    monkeypatch.setattr(train, "dino_frame_rows_hash", lambda config: "e" * 64)
    monkeypatch.setattr(train, "build_data", lambda config: ToyData())
    monkeypatch.setattr(train, "build_model", lambda config: model)
    monkeypatch.setattr(train, "_model_revision", lambda model: (FLAN_SHA, "resolved"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])

    train.main()

    checkpoint = torch.load(root / "checkpoints/final.ckpt", map_location="cpu", weights_only=True)
    assert checkpoint["global_step"] == 1
    assert checkpoint["optimizer_states"][0]["state"]
    assert checkpoint["lr_schedulers"]
    assert checkpoint["run_metadata"]["annotation_sha256"] == {
        split: sha256_file(ann / f"{split}_info_ml.npy") for split in ("train", "dev", "test")
    }
    from despamo.comparison import comparison_code_sha256

    assert checkpoint["run_metadata"]["comparison_code_sha256"] == comparison_code_sha256(
        train.PROJECT_ROOT
    )
    assert json.loads((root / "run_metadata.json").read_text()) == checkpoint["run_metadata"]
