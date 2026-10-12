import copy
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from despamo.appearance.provenance import atomic_json, digest, read_json
from despamo.losses.vt_align import VTAlignLoss
from despamo.models.residual_projector import enable_residual_projectors
from despamo.models.visual_adapter import SpaMoVisualAdapter
from despamo.training.baseline_module import SpaMoBaselineModule
from despamo.training.pilot_factory import make_variant, shared_tensor_hash
from despamo.training.pilot_replay import PilotBatchSampler
from scripts import train_projector_pilot as experiment
from scripts import train_signer_pilot
from tests.unit.pilot.test_projector_matched_16000 import _write_control


def _small_base(config):
    return SpaMoBaselineModule(
        SpaMoVisualAdapter(4, 3, 8, 6), torch.nn.Linear(6, 6), VTAlignLoss(),
        "Translate", False, 0, "mean", 1.0, 2000, 3e-4, 0.01, int(config.seed),
    )


@pytest.fixture
def small_builder(monkeypatch):
    # Replace only the cached 2.8B model loader; keep pilot wrapping and tensor hashes real.
    monkeypatch.setattr(experiment, "build_model", _small_base, raising=False)
    monkeypatch.setattr(train_signer_pilot, "build_model", _small_base)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.fixture
def verified_inputs(tmp_path, monkeypatch, small_builder):
    args = SimpleNamespace(
        mode="initialize", seed=1, projector="linear", steps=16000, gpu_cap_seconds=14400,
        output_base=tmp_path / "runs", adapted_root=tmp_path / "adapted",
        baseline_run=tmp_path / "baseline", protocol=tmp_path / "protocol.json",
        dataset=tmp_path / "dataset", text_manifest=tmp_path / "text.json",
        dino_root=tmp_path / "dino", motion_root=tmp_path / "motion",
        motion_manifest=tmp_path / "motion.json", annotation=tmp_path / "annotations/train.npy",
        initialization_profile=tmp_path / "runs/seed1-initialization.json",
        resume_checkpoint=None, matched_control_run=None, authorize_projector_run=True,
    )
    args.output_base.mkdir()
    adaptation = {"checkpoint_hash": "adaptation", "source_hash": "source"}
    atomic_json(args.adapted_root / "complete/identity.json", adaptation)
    protocol = {"protocol_hash": "protocol", "split_hash": "split", "split": {"groups": {
        name: {"clip_ids": [f"{name}-{i}" for i in range(8)]}
        for name in ("train", "dev", "test")
    }}}

    class View(list):
        @property
        def records(self):
            return [{"fileid": value} for value in self]

    views = {name: View(protocol["split"]["groups"][name]["clip_ids"])
             for name in ("train", "dev", "test")}
    config = {
        "seed": 0, "model": {"warm_up_steps": 2000, "use_in_context": False},
        "data": {"batch_size": 4, "num_workers": 0,
                 "spatial_root": str(args.adapted_root),
                 "spatial_manifest": str(args.adapted_root / "complete/manifest.json"),
                 "motion_root": str(args.motion_root),
                 "motion_manifest": str(args.motion_manifest),
                 "annotation_root": str(args.annotation.parent)},
        "trainer": {"max_steps": 8000, "accumulate_grad_batches": 1},
        "optimizer": {"learning_rate": 3e-4, "weight_decay": 0.01},
    }
    baseline = {"checkpoint_hash": "baseline", "identity": {
        "steps": 8000, "config": config, "adapted_encoder_key": digest(adaptation),
        "adaptation_checkpoint_hash": "adaptation",
    }}
    adapted = {"checkpoint_hash": "adaptation", "encoder_key": digest(adaptation),
               "manifest_hash": "manifest", "frame_rows_hash": "rows", "clips": 7096}
    code = {"scripts/train_projector_pilot.py": "c" * 64,
            "src/despamo/training/pilot_factory.py": "d" * 64}
    monkeypatch.setattr(experiment, "load_e3_views", lambda *a: (protocol, views))
    monkeypatch.setattr(experiment, "_validate_e3_selection", lambda *a: baseline)
    monkeypatch.setattr(experiment, "validate_adapted_manifest", lambda *a: adapted)
    monkeypatch.setattr(experiment, "projector_code_hashes", lambda *a: dict(code))
    return args, protocol, adapted, baseline, code


def test_seed1_repeats_independently_of_prior_rng_and_differs_from_seed0(small_builder):
    config = OmegaConf.create({"seed": 1})
    first = experiment.build_seeded_projector_model(config, 16000, 1)
    random.seed(77)
    np.random.seed(77)
    torch.manual_seed(77)
    repeated = experiment.build_seeded_projector_model(config, 16000, 1)
    seed0 = experiment.build_seeded_projector_model(OmegaConf.create({"seed": 0}), 16000, 0)
    assert shared_tensor_hash(first) == shared_tensor_hash(repeated)
    assert shared_tensor_hash(first) != shared_tensor_hash(seed0)
    assert first.rng.getstate() == random.Random(1).getstate()
    assert first.total_steps == 16000
    assert all(p.device.type == "cpu" for p in first.parameters())


def test_seed1_sets_python_numpy_cpu_and_cuda_rng(small_builder, monkeypatch):
    cuda_seeds = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", cuda_seeds.append)
    experiment.build_seeded_projector_model(OmegaConf.create({"seed": 1}), 16000, 1)
    assert random.random() == random.Random(1).random()
    assert np.random.rand() == np.random.RandomState(1).rand()
    assert cuda_seeds and set(cuda_seeds) == {1}
    actual = torch.rand(4)
    experiment.build_seeded_projector_model(OmegaConf.create({"seed": 1}), 16000, 1)
    torch.testing.assert_close(torch.rand(4), actual, rtol=0, atol=0)


def test_seed0_public_helper_keeps_frozen_path_and_optimizer(small_builder):
    config = OmegaConf.create({"seed": 0})
    old = train_signer_pilot._model(config, "E3_dino_lora", 768, 16000, {
        "supervision_policy": experiment.POLICY, "human_review_status": "not_assessed",
    })
    old_rng = torch.get_rng_state().clone()
    new = experiment.build_seeded_projector_model(config, 16000, 0)
    assert shared_tensor_hash(new) == shared_tensor_hash(old)
    assert new.rng.getstate() == old.rng.getstate()
    assert torch.equal(torch.get_rng_state(), old_rng)
    assert [g["lr"] for g in new.configure_optimizers().param_groups] == [3e-4, 5e-5]
    with pytest.raises(ValueError, match="variant/seed"):
        make_variant("E3_dino_lora", lambda: _small_base(config), lambda m: m, seed=1)


def test_linear_residual_pair_preserves_common_parameters_and_dropout_rng(small_builder):
    config = OmegaConf.create({"seed": 1})
    linear = experiment.build_seeded_projector_model(config, 16000, 1)
    expected_rng = torch.get_rng_state().clone()
    residual = experiment.build_seeded_projector_model(config, 16000, 1)
    assert shared_tensor_hash(linear) == shared_tensor_hash(residual)
    enable_residual_projectors(residual.visual_adapter, 12, seed=2)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in linear.named_parameters():
        key = name
        for projector in ("spatial_projector", "motion_projector"):
            key = key.replace(f"{projector}.", f"{projector}.base.")
        torch.testing.assert_close(dict(residual.named_parameters())[key], value, rtol=0, atol=0)
    x = torch.randn(2, 4)
    torch.testing.assert_close(residual.visual_adapter.spatial_projector(x),
                               linear.visual_adapter.spatial_projector(x), rtol=0, atol=0)
    assert residual.rng.getstate() == linear.rng.getstate()


def test_seed1_wrapper_must_preserve_shared_hash(small_builder, monkeypatch):
    wrap = experiment.PilotBaselineModule

    def corrupt(base, steps):
        result = wrap(base, steps)
        with torch.no_grad():
            result.visual_adapter.spatial_projector.weight.add_(1)
        return result

    monkeypatch.setattr(experiment, "PilotBaselineModule", corrupt)
    with pytest.raises(ValueError, match="shared tensors"):
        experiment.build_seeded_projector_model(OmegaConf.create({"seed": 1}), 16000, 1)


@pytest.mark.parametrize("seed", [-1, 2, True])
def test_model_rejects_unsupported_seeds(small_builder, seed):
    with pytest.raises(ValueError, match="seed"):
        experiment.build_seeded_projector_model(OmegaConf.create({"seed": seed}), 16000, seed)


def test_seed1_model_rejects_config_seed_mismatch(small_builder):
    with pytest.raises(ValueError, match="seed"):
        experiment.build_seeded_projector_model(OmegaConf.create({"seed": 0}), 16000, 1)


def test_seed1_data_replays_seed1_permutations_and_refuses_seed0_resume():
    data = experiment.build_seeded_projector_data(list(range(8)), 4, 1)
    expected = torch.randperm(8, generator=torch.Generator().manual_seed(1)).tolist()
    batches = iter(data.sampler)
    assert next(batches) == expected[:4]
    state = copy.deepcopy(data.state_dict())
    assert state["seed"] == 1 and state["batches_yielded"] == 1
    restored = experiment.build_seeded_projector_data(list(range(8)), 4, 1)
    restored.load_state_dict(state)
    assert list(restored.sampler) == list(batches)
    old = experiment.build_seeded_projector_data(list(range(8)), 4, 0)
    with pytest.raises(ValueError, match="identity/state"):
        old.load_state_dict(state)
    with pytest.raises(ValueError, match="sampler configuration"):
        PilotBatchSampler(8, 4, 1, 4)


def test_initialize_attests_cpu_model_config_code_and_verified_inputs(verified_inputs):
    args, protocol, adapted, baseline, code = verified_inputs
    result = experiment.initialize_projector(args)
    profile = read_json(args.initialization_profile)
    config = copy.deepcopy(baseline["identity"]["config"])
    config["seed"] = 1
    config["trainer"].update(max_steps=16000, default_root_dir=str(args.output_base))
    model = experiment.build_seeded_projector_model(OmegaConf.create(config), 16000, 1)
    assert result["status"] == "initialized"
    assert profile["seed"] == 1 and profile["steps"] == 16000
    assert profile["profile_policy"] == "projector-common-initialization-v1"
    assert profile["initial_base_hash"] == shared_tensor_hash(model)
    assert profile["initial_base_hash"] != experiment.E1_SHARED_HASH
    assert profile["config_hash"] == digest(config)
    assert profile["code_hashes"] == code
    assert profile["input_identity"]["protocol_hash"] == protocol["protocol_hash"]
    assert profile["input_identity"]["adapted_manifest_hash"] == adapted["manifest_hash"]
    assert profile["input_identity"]["baseline_checkpoint_hash"] == baseline["checkpoint_hash"]
    assert profile["profile_hash"] == digest({k: v for k, v in profile.items()
                                              if k != "profile_hash"})
    identity, _, _, _, resolved = experiment.prepare_projector(args)
    assert identity["seed"] == resolved.seed == 1
    assert identity["initial_base_hash"] == profile["initial_base_hash"]
    assert identity["initialization_profile_hash"] == profile["profile_hash"]


def test_initialize_reuses_matching_profile_without_loading_or_overwriting(verified_inputs,
                                                                         monkeypatch):
    args, *_ = verified_inputs
    experiment.initialize_projector(args)
    before = (args.initialization_profile.read_bytes(),
              args.initialization_profile.stat().st_mtime_ns)

    def forbidden(*a):
        pytest.fail("matching attestation should reuse CPU profile without model allocation")

    monkeypatch.setattr(experiment, "build_seeded_projector_model", forbidden)
    assert experiment.initialize_projector(args)["status"] == "reused"
    assert before == (args.initialization_profile.read_bytes(),
                      args.initialization_profile.stat().st_mtime_ns)


@pytest.mark.parametrize("drift", ["config", "code", "baseline", "features", "seed",
                                 "initial_hash", "profile_hash"])
def test_profile_mismatch_refuses_prepare_and_overwrite(verified_inputs, drift):
    args, _, adapted, baseline, code = verified_inputs
    experiment.initialize_projector(args)
    if drift == "config":
        baseline["identity"]["config"]["optimizer"]["weight_decay"] = 0.2
    elif drift == "code":
        code["scripts/train_projector_pilot.py"] = "e" * 64
    elif drift == "baseline":
        baseline["checkpoint_hash"] = "other-baseline"
    elif drift == "features":
        adapted["frame_rows_hash"] = "other-rows"
    else:
        profile = read_json(args.initialization_profile)
        if drift == "seed":
            profile["seed"] = 0
        elif drift == "initial_hash":
            profile["initial_base_hash"] = experiment.E1_SHARED_HASH
        else:
            profile["profile_hash"] = "f" * 64
        atomic_json(args.initialization_profile, profile)
    before = args.initialization_profile.read_bytes()
    with pytest.raises(ValueError, match="initialization profile"):
        experiment.prepare_projector(args)
    with pytest.raises(ValueError, match="initialization profile"):
        experiment.initialize_projector(args)
    assert args.initialization_profile.read_bytes() == before


def test_seed1_identity_requires_valid_bound_attestation(verified_inputs):
    args, protocol, adapted, baseline, code = verified_inputs
    experiment.initialize_projector(args)
    identity, _, _, _, config = experiment.prepare_projector(args)
    resolved = OmegaConf.to_container(config, resolve=True)
    profile = read_json(args.initialization_profile)
    with pytest.raises(ValueError, match="initialization profile"):
        experiment.build_projector_identity(protocol, adapted, resolved, "linear", 16000,
                                             14400, baseline["checkpoint_hash"], code, seed=1,
                                             initial_base_hash=identity["initial_base_hash"])
    with pytest.raises(ValueError, match="initialization profile"):
        experiment.build_projector_identity(protocol, adapted, resolved, "linear", 16000,
                                             14400, baseline["checkpoint_hash"], code, seed=1,
                                             initialization_profile=profile)
    with pytest.raises(ValueError, match="initialization profile"):
        experiment.build_projector_identity(protocol, adapted, resolved, "linear", 16000,
                                             14400, baseline["checkpoint_hash"], code, seed=1,
                                             initial_base_hash=None, initialization_profile=profile)
    linear = experiment.build_projector_identity(
        protocol, adapted, resolved, "linear", 16000, 14400, baseline["checkpoint_hash"], code,
        seed=1, initial_base_hash=profile["initial_base_hash"], initialization_profile=profile,
    )
    residual = experiment.build_projector_identity(
        protocol, adapted, resolved, "residual", 16000, 14400, baseline["checkpoint_hash"], code,
        seed=1, initial_base_hash=profile["initial_base_hash"], initialization_profile=profile,
    )
    assert linear["residual_seed"] is None and residual["residual_seed"] == 2
    assert linear["run_key"] != residual["run_key"]
    assert linear["initialization_profile_hash"] == residual["initialization_profile_hash"]
    resolved["seed"] = 0
    old = experiment.build_projector_identity(protocol, adapted, resolved, "linear", 16000,
                                              14400, baseline["checkpoint_hash"], code)
    assert old["seed"] == 0 and old["initial_base_hash"] == experiment.E1_SHARED_HASH
    assert "initialization_profile_hash" not in old
    assert old["run_key"] != linear["run_key"]
    assert old["run_key"] == "projector-linear-16000-" + digest({
        k: v for k, v in old.items() if k != "run_key"
    })


def test_prepare_seed1_requires_profile_before_any_model_build(verified_inputs, monkeypatch):
    args, *_ = verified_inputs
    args.mode = "preflight"

    def forbidden(*a):
        pytest.fail("preflight must validate profile without allocating model")

    monkeypatch.setattr(experiment, "build_seeded_projector_model", forbidden)
    with pytest.raises(ValueError, match="initialization profile"):
        experiment.prepare_projector(args)
    args.initialization_profile = None
    with pytest.raises(ValueError, match="initialization profile"):
        experiment.prepare_projector(args)


@pytest.mark.parametrize("change", [{"seed": 0}, {"projector": "residual"}, {"steps": 8000},
                                  {"initialization_profile": None}])
def test_initialize_only_allows_seed1_linear16000(verified_inputs, change):
    args, *_ = verified_inputs
    vars(args).update(change)
    with pytest.raises(ValueError, match="initialize"):
        experiment.initialize_projector(args)


def test_profile_symlink_refused_without_modifying_target(verified_inputs):
    args, *_ = verified_inputs
    target = args.output_base / "keep.json"
    target.write_text("{}")
    args.initialization_profile.symlink_to(target)
    with pytest.raises(ValueError, match="initialization profile"):
        experiment.initialize_projector(args)
    assert target.read_text() == "{}"


def test_seed0_prepare_accepts_legacy_namespace_without_profile(verified_inputs):
    args, *_ = verified_inputs
    del args.seed
    del args.initialization_profile
    args.mode = "preflight"
    identity, _, _, _, config = experiment.prepare_projector(args)
    assert identity["seed"] == config.seed == 0
    assert identity["initial_base_hash"] == experiment.E1_SHARED_HASH
    assert "initialization_profile_hash" not in identity


def test_matched_control_requires_same_profile_hash(tmp_path):
    root, candidate, _ = _write_control(tmp_path)
    budget = read_json(root / "budget.json")
    budget["identity"]["initialization_profile_hash"] = "a" * 64
    atomic_json(root / "budget.json", budget)
    candidate["initialization_profile_hash"] = "b" * 64
    with pytest.raises(ValueError, match="control"):
        experiment.validate_matched_linear_control(root, candidate, ("dev-id",))


def test_matched_seed1_control_accepts_attested_hash_instead_of_seed0_constant(tmp_path):
    root, candidate, _ = _write_control(tmp_path)
    budget = read_json(root / "budget.json")
    for identity in (budget["identity"], candidate):
        identity.update(seed=1, initial_base_hash="a" * 64, initialization_profile_hash="b" * 64)
        identity["config"]["seed"] = 1
    atomic_json(root / "budget.json", budget)
    assert experiment.validate_matched_linear_control(root, candidate, ("dev-id",))["run_key"]


def test_matched_seed1_control_requires_attestation_even_when_both_profiles_missing(tmp_path):
    root, candidate, _ = _write_control(tmp_path)
    budget = read_json(root / "budget.json")
    for identity in (budget["identity"], candidate):
        identity.update(seed=1, initial_base_hash="a" * 64)
        identity["config"]["seed"] = 1
    atomic_json(root / "budget.json", budget)
    with pytest.raises(ValueError, match="control"):
        experiment.validate_matched_linear_control(root, candidate, ("dev-id",))


def test_residual_prepare_binds_same_seed1_profile_and_completed_linear_control(verified_inputs,
                                                                             tmp_path):
    args, *_ = verified_inputs
    experiment.initialize_projector(args)
    linear, _, protocol, _, _ = experiment.prepare_projector(args)
    root, _, selected = _write_control(tmp_path)
    budget = read_json(root / "budget.json")
    budget["identity"] = linear
    new_root = root.with_name(linear["run_key"])
    root.rename(new_root)
    atomic_json(new_root / "budget.json", budget)
    selected["adaptation_checkpoint_hash"] = "adaptation"
    atomic_json(new_root / "selected-checkpoint.json", selected)
    # Fixture selection now covers the complete logical dev list from verified inputs.
    for step in (5500, 10400, 16000):
        path = new_root / f"dev-{step}.json"
        report = read_json(path)
        report["items"] = [{"clip_id": clip, "prediction": "text", "reference": "text"}
                           for clip in protocol["split"]["groups"]["dev"]["clip_ids"]]
        atomic_json(path, report)
    args.projector = "residual"
    args.matched_control_run = new_root
    residual, _, _, _, config = experiment.prepare_projector(args)
    assert residual["initial_base_hash"] == linear["initial_base_hash"]
    assert residual["initialization_profile_hash"] == linear["initialization_profile_hash"]
    assert residual["seed"] == config.seed == 1 and residual["residual_seed"] == 2
    assert residual["matched_linear_control"]["run_key"] == linear["run_key"]


def test_initialize_does_not_build_optimizer_or_trainer(verified_inputs, monkeypatch):
    args, *_ = verified_inputs

    def forbidden(*a, **k):
        pytest.fail("CPU initialization must not build training or optimizer state")

    monkeypatch.setattr(SpaMoBaselineModule, "configure_optimizers", forbidden)
    monkeypatch.setattr(experiment.PilotBaselineModule, "configure_optimizers", forbidden)
    monkeypatch.setattr(experiment.pl, "Trainer", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    assert experiment.initialize_projector(args)["status"] == "initialized"


def test_initialize_refuses_live_writer_lock_without_profile_mutation(verified_inputs):
    args, *_ = verified_inputs
    with experiment.writer(args.initialization_profile.parent):
        with pytest.raises(RuntimeError, match="writer"):
            experiment.initialize_projector(args)
    assert not args.initialization_profile.exists()


def test_initialize_refuses_code_change_during_cpu_construction(verified_inputs, monkeypatch):
    args, _, _, _, code = verified_inputs

    def changed_source(config):
        base = _small_base(config)
        code["scripts/train_projector_pilot.py"] = "e" * 64
        return base

    monkeypatch.setattr(experiment, "build_model", changed_source)
    with pytest.raises(ValueError, match="code changed"):
        experiment.initialize_projector(args)
    assert not args.initialization_profile.exists()


@pytest.mark.parametrize("projector", ["linear", "residual"])
@pytest.mark.parametrize("initial_drift", [False, True])
def test_runner_uses_attested_hash_training_data_seed_and_residual_seed(verified_inputs,
                                                                     monkeypatch, projector,
                                                                     initial_drift):
    args, *_ = verified_inputs
    experiment.initialize_projector(args)
    identity, output, protocol, views, config = experiment.prepare_projector(args)
    args.projector = projector
    identity.update(projector=projector, variant=f"E3_projector_{projector}",
                    residual_seed=2 if projector == "residual" else None)
    monkeypatch.setattr(experiment, "prepare_projector",
                        lambda a: (identity, output, protocol, views, config))
    monkeypatch.setattr(experiment.shutil, "disk_usage",
                        lambda p: SimpleNamespace(free=100 * 2**30))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (16 * 2**30, 16 * 2**30))
    monkeypatch.setattr(experiment, "collate_phoenix",
                        lambda values: SimpleNamespace(clip_ids=values))
    monkeypatch.setattr(experiment, "projector_gradient_smoke", lambda *a: {"status": "passed"})
    monkeypatch.setattr(SpaMoBaselineModule, "to", lambda self, *a: self)
    if initial_drift:
        build = experiment.build_seeded_projector_model

        def different_weights(config, steps, seed):
            model = build(config, steps, seed)
            with torch.no_grad():
                model.visual_adapter.spatial_projector.weight.add_(1)
            return model

        monkeypatch.setattr(experiment, "build_seeded_projector_model", different_weights)

    class ReachedFit(Exception):
        pass

    class Trainer:
        def __init__(self, **kwargs):
            pass

        def fit(self, model, datamodule, ckpt_path):
            assert model.rng.getstate() == random.Random(1).getstate()
            assert datamodule.sampler.seed == 1
            expected = torch.randperm(8, generator=torch.Generator().manual_seed(1)).tolist()
            assert next(iter(datamodule.sampler)) == expected[:4]
            assert ckpt_path is None
            if projector == "residual":
                reference = experiment.build_seeded_projector_model(config, 16000, 1)
                enable_residual_projectors(reference.visual_adapter, experiment.HIDDEN, seed=2)
                assert shared_tensor_hash(model) == shared_tensor_hash(reference)
            assert read_json(output / "initialization.json")["initial_base_hash"] == (
                identity["initial_base_hash"])
            raise ReachedFit

    monkeypatch.setattr(experiment.pl, "Trainer", Trainer)
    if initial_drift:
        with pytest.raises(ValueError, match="attested shared initialization"):
            experiment.run_projector(args)
        assert not (output / "initialization.json").exists()
        assert not (output / "gradient-smoke.json").exists()
    else:
        with pytest.raises(ReachedFit):
            experiment.run_projector(args)


def test_initialize_cli_accepts_new_seed_and_profile(verified_inputs, monkeypatch, capsys):
    args, *_ = verified_inputs
    argv = ["train_projector_pilot", "--mode", "initialize", "--seed", "1",
            "--projector", "linear", "--steps", "16000", "--gpu-cap-seconds", "14400",
            "--initialization-profile", str(args.initialization_profile)]
    for name in ("protocol", "dataset", "text_manifest", "dino_root", "motion_root",
                 "motion_manifest", "annotation", "adapted_root", "baseline_run", "output_base"):
        argv.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    monkeypatch.setattr(sys, "argv", argv)
    experiment.main()
    assert "initialized" in capsys.readouterr().out
    assert read_json(args.initialization_profile)["seed"] == 1
