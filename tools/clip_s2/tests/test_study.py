import pytest

from tools.clip_s2.study import predecessor_allowed, validate_selections


def test_two_reference_selections_only():
    rows = [dict(seed=s, arm="reference", selection_hash="hash") for s in (0, 1)]
    validate_selections(rows)
    with pytest.raises(ValueError):
        validate_selections(rows + [rows[0]])
    with pytest.raises(ValueError):
        validate_selections([rows[0], dict(rows[1], arm="aligned_fixed")])


def test_explicit_predecessor_switch_permitted_but_active_owner_blocks():
    assert predecessor_allowed("stopped_by_user", True, False)
    assert not predecessor_allowed("running", False, False)
    assert not predecessor_allowed("stopped_by_user", True, True)


def test_frozen_selection_hash_cannot_change(tmp_path):
    from tools.clip_s2.study import freeze_selections

    rows = [dict(seed=s, arm="reference", selection_hash="hash") for s in (0, 1)]
    target = tmp_path / "frozen.json"
    freeze_selections(target, rows)
    freeze_selections(target, rows[::-1])
    with pytest.raises(ValueError):
        freeze_selections(target, [rows[0], dict(rows[1], selection_hash="changed")])


def test_archive_is_dev_only_without_test_claim(tmp_path, monkeypatch):
    import json

    import tools.clip_s2.study as s

    monkeypatch.setattr(s, "require_archive_storage", lambda *a: None)
    run = tmp_path / "run"
    run.mkdir()
    (run / "complete.json").write_text(json.dumps({"status": "complete"}))
    (run / "selected-checkpoint.json").write_text("{}")
    (run / "resume.pt").write_bytes(b"checkpoint")
    s.archive_dev(run, tmp_path / "archive")
    assert (tmp_path / "archive/run/dev-final-payload/archive-manifest.json").is_file()
    assert not (run / "official-test").exists()
