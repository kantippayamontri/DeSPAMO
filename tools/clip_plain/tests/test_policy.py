import copy
import json
import time
from pathlib import Path

import pytest


def config():
    from tools.clip_plain import policy

    return policy.load_config(Path(policy.__file__).with_name("config.json"))


def test_plain_policy_contract():
    c = config()
    assert c["spatial_width"] == 1024 and c["scales"] == [224]
    assert c["tokens"] == "last_hidden_state_cls_no_postnorm"
    assert c["recipe"]["batch_size"] == 8 and c["seeds"] == [0, 1]
    assert not c["test_scoring"]
    assert c["evaluation"]["batch_size"] == 1 and c["evaluation"]["use_cache"]
    assert c["evaluation"]["beam_size"] == 5 and c["evaluation"]["max_length"] == 64
    assert (
        c["parent_manifest_sha256"]
        == "1e5280e35fa273f7fbc28d36802496d3ac1d60a2b62a80c5674339cd0dd587e4"
    )


@pytest.mark.parametrize(
    "field,value", [("spatial_width", 2048), ("scales", [224, 448]), ("test_scoring", True)]
)
def test_config_rejects_policy_changes(tmp_path, field, value):
    from tools.clip_plain.policy import load_config

    c = config()
    c[field] = value
    p = tmp_path / "config.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_config(p)


def approval_fixture(tmp_path, monkeypatch):
    from tools.clip_plain import policy as p

    c = config()
    c["study_root"] = str(tmp_path / "study")
    hold = tmp_path / "old-hold.json"
    hold.write_text('{"status":"stopped_by_user","automatic_recovery_disabled":true}')
    monkeypatch.setattr(p, "GPU_HOLD", hold)
    monkeypatch.setattr(p, "source_hashes", lambda: {"source": "abc"})
    driver = tmp_path / "driver.json"
    driver.write_text('{"reviewed":true}')
    record = dict(
        approved=True,
        token="attempt1",
        boot_id=p.boot_id(),
        expires_at=time.time() + 60,
        config_sha256=p.digest(c),
        source_hashes=p.source_hashes(),
        hold_sha256=p.sha256_file(hold),
        driver_review=str(driver),
        driver_review_sha256=p.sha256_file(driver),
        stages=["train", "dev"],
        seeds=[0],
        max_batches=2,
    )
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(record))
    return p, c, record, path


@pytest.mark.parametrize(
    "mutation", ["absent", "expired", "boot", "hold", "stage", "source", "consumed"]
)
def test_gpu_gate_rejects_before_any_device_call(tmp_path, monkeypatch, mutation):
    p, c, record, path = approval_fixture(tmp_path, monkeypatch)
    if mutation == "absent":
        path = None
    else:
        fields = dict(
            expired=("expires_at", 0),
            boot=("boot_id", "old"),
            hold=("hold_sha256", "x"),
            stage=("stages", ["other"]),
            source=("source_hashes", {}),
            consumed=("consumed", True),
        )
        key, value = fields[mutation]
        record[key] = value
        path.write_text(json.dumps(record))
    with pytest.raises((ValueError, RuntimeError)):
        p.require_gpu_scope(c, "dev", path)


def test_attempt_claim_is_one_shot_and_hold_unchanged(tmp_path, monkeypatch):
    p, c, _, path = approval_fixture(tmp_path, monkeypatch)
    before = p.GPU_HOLD.read_bytes()
    result = p.claim_attempt(c, path)
    assert result["owner"]["pid"] > 0
    with pytest.raises((RuntimeError, FileExistsError)):
        p.claim_attempt(c, path)
    assert p.GPU_HOLD.read_bytes() == before


def test_child_requires_exact_parent_owner(tmp_path, monkeypatch):
    p, c, _, path = approval_fixture(tmp_path, monkeypatch)
    attempt = p.claim_attempt(c, path)
    job = dict(parent_owner=attempt["owner"], attempt_token="attempt1", seed=0, max_batches=2)
    assert p.require_gpu_scope(c, "dev", path, job=job)["token"] == "attempt1"
    bad = copy.deepcopy(job)
    bad["parent_owner"]["start_ticks"] = "bad"
    with pytest.raises(RuntimeError):
        p.require_gpu_scope(c, "dev", path, job=bad)


def test_owner_alive_exact_identity():
    import os

    from tools.clip_plain.policy import owner_alive
    from tools.dinov3.official_runtime import process_owner

    current = process_owner(os.getpid())
    assert owner_alive(current)
    assert not owner_alive({**current, "boot_id": "old"})
    assert not owner_alive({**current, "start_ticks": "different"})


def test_exclusive_gpu_blocks_foreign_worker(monkeypatch):
    from tools.benchmark import feature_package
    from tools.clip_plain.policy import require_exclusive_gpu

    monkeypatch.setattr(feature_package, "other_gpu_pids", lambda: [999])
    with pytest.raises(RuntimeError, match="Another"):
        require_exclusive_gpu()
    monkeypatch.setattr(feature_package, "other_gpu_pids", lambda: [])
    require_exclusive_gpu()


def test_duplicate_seed_scope_cannot_fake_two_runs(tmp_path, monkeypatch):
    p, c, record, path = approval_fixture(tmp_path, monkeypatch)
    record["seeds"] = [0, 0]
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError):
        p.require_gpu_scope(c, "dev", path)
