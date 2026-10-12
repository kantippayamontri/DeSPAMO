from pathlib import Path

import pytest

from tools.benchmark.compact_checkpoint import capture_state
from tools.clip_s2_safe.tests.test_state import binding_fixture
from tools.dinov3.identity import sha256_file


def snapshots(tmp_path, monkeypatch):
    s, c, b, model, optimizer, sampler = binding_fixture(tmp_path)
    monkeypatch.setattr(s, "require_archive_storage", lambda *args: {})
    anchor_before = Path(c["anchor"]).read_bytes()
    for step in range(3, 7):
        sampler.next_ids()
        payload = capture_state(
            model, optimizer, b["original_identity"], step, sampler.state_dict(), base_hash="frozen"
        )
        s.publish_checkpoint(
            c,
            payload,
            dict(last_update=step, ce_values=[1.0] * max(0, step - 4)),
            mode="production",
        )
    root = Path(c["runtime_root"])
    selected = root / "selected/epoch-4.pt"
    selected.parent.mkdir()
    selected.write_bytes((root / "checkpoints/production/candidate-step-4.pt").read_bytes())
    return s, c, b, anchor_before, selected


def test_retention_keeps_latest2_candidates_selection_and_original_anchor(tmp_path, monkeypatch):
    s, c, b, anchor_before, selected = snapshots(tmp_path, monkeypatch)
    root = Path(c["runtime_root"]) / "checkpoints/production"
    selected_before = selected.read_bytes()
    assert s.prune_owned_resumes(c, b, allowed=False) == []
    assert len(list(root.glob("resume-*.pt"))) == 4
    removed = s.prune_owned_resumes(c, b, allowed=True)
    assert removed == ["resume-step-3.pt", "resume-step-4.pt"]
    assert sorted(p.name for p in root.glob("resume-*.pt")) == [
        "resume-step-5.pt",
        "resume-step-6.pt",
    ]
    assert Path(c["anchor"]).read_bytes() == anchor_before
    assert selected.read_bytes() == selected_before
    for step in [4, 6]:
        candidate = root / f"candidate-step-{step}.pt"
        archive = Path(c["archive_root"]) / "candidates/production" / candidate.name
        assert sha256_file(archive) == sha256_file(candidate)
        for suffix in [".json", ".runtime.json", ".guard.json"]:
            assert (
                archive.with_suffix(suffix).read_bytes()
                == candidate.with_suffix(suffix).read_bytes()
            )


def test_archive_failure_prevents_any_resume_deletion(tmp_path, monkeypatch):
    s, c, b, _, _ = snapshots(tmp_path, monkeypatch)
    root = Path(c["runtime_root"]) / "checkpoints/production"
    before = {p.name: sha256_file(p) for p in root.glob("resume-*.pt")}

    def fail_archive(*args):
        raise OSError("archive publication failed")

    monkeypatch.setattr(s, "archive_candidate", fail_archive)
    with pytest.raises(OSError, match="archive publication"):
        s.prune_owned_resumes(c, b, allowed=True)
    assert {p.name: sha256_file(p) for p in root.glob("resume-*.pt")} == before
