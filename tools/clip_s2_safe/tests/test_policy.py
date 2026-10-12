import os
import time
from pathlib import Path

import pytest

from tools.clip_s2_safe.tests.helpers import original_fixture
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.storage import atomic_json


def test_policy_contract():
    from tools.clip_s2_safe import policy as p

    c = p.load_config(Path(p.__file__).with_name("config.json"))
    assert c["seed"] == 0 and c["spatial_width"] == 2048 and c["anchor_step"] == 1774
    assert c["recipe"]["batch_size"] == 8 and c["recipe"]["vt_warmup_steps"] == 4000
    e = p.evaluation_policy()
    assert (
        e["id"] == "clip-s2-cached-b1-beam5-v1"
        and e["use_cache"]
        and e["batch_size"] == 1
        and e["beam_size"] == 5
        and e["max_length"] == 64
    )


def test_original_identity_and_byte_pins(tmp_path):
    from tools.clip_s2_safe.policy import verify_original

    c, i, *_ = original_fixture(tmp_path)
    result = verify_original(c)
    assert result["identity"] == i
    Path(next(iter(i["code_hashes"]))).write_text("drift")
    with pytest.raises(ValueError):
        verify_original(c)


def authorization(tmp_path, monkeypatch):
    from tools.clip_s2_safe import policy as p

    c, i, *_ = original_fixture(tmp_path)
    monkeypatch.setattr(p, "source_hashes", lambda: {"runtime": "sha"})
    binding = dict(
        original_identity_hash=digest(i), sources=p.source_hashes(), config_hash=digest(c)
    )
    atomic_json(Path(c["runtime_root"]) / "migration.json", binding)
    review = tmp_path / "driver-review.json"
    atomic_json(review, dict(reviewed=True))
    record = dict(
        approved=True,
        consumed=False,
        token="attempt",
        mode="production",
        seed=0,
        stages=["supervisor", "train", "dev"],
        boot_id=p.boot_id(),
        expires_at=time.time() + 1000,
        binding_hash=digest(binding),
        source_hashes=p.source_hashes(),
        config_hash=digest(c),
        hold_sha256=sha256_file(Path(c["study_root"]) / "manual-stop.json"),
        driver_review=str(review),
        driver_review_sha256=sha256_file(review),
        max_train_updates=4,
        max_dev_clips=6,
    )
    path = tmp_path / "approval.json"
    atomic_json(path, record)
    return p, c, record, path


@pytest.mark.parametrize(
    "bad", ["absent", "expired", "consumed", "boot", "mode", "source", "hold", "seed", "cap"]
)
def test_gate_fails_without_device_access(tmp_path, monkeypatch, bad):
    p, c, r, path = authorization(tmp_path, monkeypatch)
    if bad == "absent":
        path = None
    else:
        key, value = dict(
            expired=("expires_at", 0),
            consumed=("consumed", True),
            boot=("boot_id", "old"),
            mode=("mode", "other"),
            source=("source_hashes", {}),
            hold=("hold_sha256", "bad"),
            seed=("seed", 1),
            cap=("max_train_updates", True),
        )[bad]
        r[key] = value
        atomic_json(path, r)
    with pytest.raises((RuntimeError, ValueError)):
        p.require_scope(c, "train", path)


def test_claim_once_and_original_stop_unchanged(tmp_path, monkeypatch):
    p, c, r, path = authorization(tmp_path, monkeypatch)
    hold = Path(c["study_root"]) / "manual-stop.json"
    before = hold.read_bytes()
    claim = p.claim_scope(c, path)
    assert claim["owner"]["pid"] == os.getpid()
    with pytest.raises(RuntimeError):
        p.claim_scope(c, path)
    assert hold.read_bytes() == before


def test_exact_parent_ticket_and_owner(tmp_path, monkeypatch):
    p, c, r, path = authorization(tmp_path, monkeypatch)
    claim = p.claim_scope(c, path)
    job = dict(
        stage="dev",
        mode="production",
        seed=0,
        token=r["token"],
        parent_owner=claim["owner"],
        binding_hash=r["binding_hash"],
        start_step=2,
        target_step=2,
        max_batches=None,
    )
    ticket = Path(c["runtime_root"]) / "attempts/attempt/jobs/dev2.json"
    atomic_json(ticket, job)
    job["ticket_path"] = str(ticket)
    atomic_json(ticket, job)
    assert p.require_scope(c, "dev", path, job=job)["approved"]
    job["parent_owner"]["start_ticks"] = "bad"
    with pytest.raises(RuntimeError):
        p.require_scope(c, "dev", path, job=job)


def test_owner_requires_same_boot_ticks_command():
    from tools.clip_s2_safe.policy import owner_alive
    from tools.dinov3.official_runtime import process_owner

    o = process_owner(os.getpid())
    assert owner_alive(o)
    assert not owner_alive({**o, "boot_id": "old"}) and not owner_alive(
        {**o, "start_ticks": "different"}
    )
