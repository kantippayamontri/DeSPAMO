import copy
import json

import pytest

from despamo.appearance.provenance import file_hash
from scripts import train_projector_pilot as experiment


def _inputs():
    config = {
        "seed": 0, "model": {"warm_up_steps": 2000},
        "data": {"batch_size": 4, "num_workers": 0},
        "trainer": {"max_steps": 16000, "accumulate_grad_batches": 1},
        "optimizer": {"weight_decay": 0.01},
    }
    protocol = {"protocol_hash": "protocol", "split_hash": "split"}
    adapted = {"checkpoint_hash": "adapt", "encoder_key": "encoder",
               "manifest_hash": "manifest", "frame_rows_hash": "rows", "clips": 7096}
    return protocol, adapted, config


def _write_control(tmp_path):
    protocol, adapted, config = _inputs()
    identity = experiment.build_projector_identity(
        protocol, adapted, config, "linear", 16000, 14400, "baseline", {
            "scripts/train_projector_pilot.py": "original",
            "src/despamo/training/pilot_factory.py": "same-training-code",
        },
    )
    root = tmp_path / identity["run_key"]
    root.mkdir()
    (root / "budget.json").write_text(json.dumps({"status": "closed", "identity": identity}))
    (root / "run-status.json").write_text(json.dumps({"status": "complete", "global_step": 16000}))
    decoder = {"mode": "deterministic", "beam_size": 5, "max_length": 64, "in_context": False}
    reports = []
    for step, score in ((5500, 4.1), (10400, 5.4), (16000, 5.3)):
        checkpoint = root / f"step-{step}.ckpt"
        checkpoint.write_bytes(f"checkpoint-{step}".encode())
        report = {
            "split": "dev", "split_hash": "split", "checkpoint_step": step,
            "checkpoint_hash": file_hash(checkpoint), "metrics": {"bleu4": score},
            "model_variant": "E3_projector_linear", "decoding": decoder,
            "items": [{"clip_id": "dev-id", "prediction": "text", "reference": "text"}],
        }
        (root / f"dev-{step}.json").write_text(json.dumps(report))
        reports.append(report)
    selected = {
        "step": 10400, "checkpoint_hash": reports[1]["checkpoint_hash"],
        "metrics": reports[1]["metrics"], "model_variant": "E3_projector_linear",
        "protocol_hash": "protocol", "adaptation_checkpoint_hash": "adapt",
        "supervision_policy": "qwen-schema98-unreviewed-v1", "human_review_status": "not_assessed",
    }
    (root / "selected-checkpoint.json").write_text(json.dumps(selected))
    candidate = copy.deepcopy(identity)
    candidate.update(projector="residual", variant="E3_projector_residual",
                     residual_hidden_dim=1024, residual_seed=1)
    candidate["code_hashes"]["scripts/train_projector_pilot.py"] = "approved-extension"
    return root, candidate, selected


def test_residual_16000_has_matched_long_schedule_and_distinct_identity():
    protocol, adapted, config = _inputs()
    residual = experiment.build_projector_identity(
        protocol, adapted, config, "residual", 16000, 14400, "baseline", {"runner": "code"},
    )
    linear = experiment.build_projector_identity(
        protocol, adapted, config, "linear", 16000, 14400, "baseline", {"runner": "code"},
    )
    assert residual["run_key"] != linear["run_key"]
    for key in ("steps", "warmup_steps", "lr_warmup_steps", "dev_checkpoints",
                "config", "seed", "resume_retention", "gpu_cap_seconds", "adapted_encoder_key"):
        assert residual[key] == linear[key]
    assert residual["dev_checkpoints"] == [5500, 10400, 16000]
    assert residual["warmup_steps"] == 2000 and residual["lr_warmup_steps"] == 1600
    assert residual["residual_hidden_dim"] == 1024


def test_completed_linear_control_binds_selected_checkpoint_and_identity(tmp_path):
    root, candidate, selected = _write_control(tmp_path)
    result = experiment.validate_matched_linear_control(root, candidate, ("dev-id",))
    assert result["run_key"] == root.name
    assert result["checkpoint_hash"] == selected["checkpoint_hash"]
    assert result["selected_step"] == 10400
    assert result["metrics"]["bleu4"] == 5.4
    assert result["budget_hash"] == file_hash(root / "budget.json")


@pytest.mark.parametrize(
    "drift",
    ["incomplete", "features", "scheduler", "optimizer", "training_code", "decoder", "selected"],
)
def test_matched_control_rejects_unfinished_or_changed_recipe(tmp_path, drift):
    root, candidate, _ = _write_control(tmp_path)
    if drift in {"incomplete", "features", "scheduler", "optimizer", "training_code"}:
        path = root / "budget.json"
        value = json.loads(path.read_text())
        if drift == "incomplete":
            value["status"] = "running"
        elif drift == "features":
            value["identity"]["adapted_encoder_key"] = "other-features"
        elif drift == "scheduler":
            value["identity"]["lr_warmup_steps"] = 800
        elif drift == "optimizer":
            value["identity"]["config"]["optimizer"]["weight_decay"] = 0.1
        else:
            value["identity"]["code_hashes"]["src/despamo/training/pilot_factory.py"] = "changed"
    elif drift == "decoder":
        path = root / "dev-10400.json"
        value = json.loads(path.read_text())
        value["decoding"]["max_length"] = 32
    else:
        path = root / "selected-checkpoint.json"
        value = json.loads(path.read_text())
        value["step"] = 5500
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="control"):
        experiment.validate_matched_linear_control(root, candidate, ("dev-id",))
