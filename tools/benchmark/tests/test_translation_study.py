import importlib

import pytest


def module():
    return importlib.import_module("tools.benchmark.run_translation_study")


def test_profile_batch_selection_rejects_oom_or_bad_margin():
    good=dict(status="complete",generation_equal=True,gpu_free_bytes=4*1024**3,
              results={"long":dict(peak_reserved_bytes=20*1024**3)})
    assert module().choose_batch({8:good,4:good})==8
    assert module().choose_batch({8:{**good,"status":"oom"},4:good})==4
    assert module().choose_batch({8:{**good,"gpu_free_bytes":0},4:good})==4
    with pytest.raises(ValueError):
        module().choose_batch({8:dict(status="oom"),4:dict(status="oom")})


def test_test_stage_requires_all_six_frozen_selections():
    rows=[dict(seed=s,arm=a,selection_hash=f"{s}-{a}") for s in (0,1)
          for a in ("reference","aligned_fixed","aligned_learned")]
    module().validate_selection_inventory(rows)
    with pytest.raises(ValueError):
        module().validate_selection_inventory(rows[:-1])
    with pytest.raises(ValueError):
        module().validate_selection_inventory(rows[:-1]+[rows[0]])


def test_final_archive_retains_native_resumes_and_rejects_tampering(tmp_path,monkeypatch):
    from tools.dinov3.storage import atomic_json

    m=module()
    monkeypatch.setattr(m,"ARCHIVE",tmp_path/"archive")
    monkeypatch.setattr(m,"require_archive_storage",lambda *a,**k:{})
    run=tmp_path/"run"
    run.mkdir()
    atomic_json(run/"complete.json",dict(status="complete"))
    atomic_json(run/"official-test/report.json",dict(metrics={}))
    (run/"resume.pt").write_bytes(b"native state")
    path=m.archive_final_payload(run)
    assert (run/"resume.pt").is_file() and (path.parent/"resume.pt").read_bytes()==b"native state"
    (path.parent/"resume.pt").write_bytes(b"changed data")
    with pytest.raises(ValueError):
        m.archive_final_payload(run)
