import importlib

import pytest
import torch


def test_declared_training_schedule_is40epochs_and_dev_every2():
    m=importlib.import_module("tools.benchmark.translation_model")
    assert m.schedule(8)==dict(batch_size=8,steps_per_epoch=887,total_steps=35480,
        epochs=40,dev_epochs=list(range(2,41,2)),vt_warmup_steps=2000,lr_warmup_steps=3548)
    assert m.lr_factor(3548,35480)==pytest.approx(1.)
    assert m.lr_factor(35480,35480)==pytest.approx(0.)
    with pytest.raises(ValueError):
        m.schedule(24)


def test_plain_fusion_models_have_identical_common_initialization():
    m=importlib.import_module("tools.benchmark.translation_model")
    from despamo.models.visual_adapter import SpaMoVisualAdapter

    torch.manual_seed(0)
    base=SpaMoVisualAdapter(2048,1024,768,2048)
    expected=m.plain_adapter(base,0)
    before={n:p.detach().clone() for n,p in expected.named_parameters()}
    from despamo.models.aligned_fusion import enable_aligned_fusion
    for arm in ["aligned_fixed","aligned_learned"]:
        combined=enable_aligned_fusion(expected,arm,0)
        assert all(torch.equal(p,before[n]) for n,p in combined.named_parameters() if n in before)


def test_dev_selection_ties_use_earlier_epoch():
    m=importlib.import_module("tools.benchmark.official_translation")
    reports=[dict(epoch=4,metrics=dict(bleu4=6.)),dict(epoch=2,metrics=dict(bleu4=6.)),
             dict(epoch=6,metrics=dict(bleu4=5.))]
    assert m.select_report(reports)["epoch"]==2
    with pytest.raises(ValueError):
        m.select_report([dict(epoch=2,metrics=dict(bleu4=float('nan')))])


def test_existing_dev_report_recovers_missing_selection_pointer(tmp_path):
    from tools.dinov3.identity import digest, sha256_file
    from tools.dinov3.storage import atomic_json

    m=importlib.import_module("tools.benchmark.official_translation")
    identity={"run":"owned"}
    archived=tmp_path/"archive/model-epoch-2.pt"
    archived.parent.mkdir()
    archived.write_bytes(b"compact candidate")
    receipt=dict(sha256=sha256_file(archived),identity=identity,bytes=archived.stat().st_size,step=20)
    atomic_json(archived.with_suffix('.json'),receipt)
    root=tmp_path/"run"
    root.mkdir()
    report=dict(epoch=2,metrics=dict(bleu4=5.),checkpoint_hash=sha256_file(archived),
                checkpoint_path=str(archived),run_identity_hash=digest(identity))
    atomic_json(root/"dev-epoch-2.json",report)
    best=m.reconcile_selection(root,identity)
    assert best["epoch"]==2 and (root/"selected-epoch-2.pt").read_bytes()==b"compact candidate"
    assert (root/"selected-checkpoint.json").is_file()


def test_interrupted_archive_receipt_recovers_from_owned_native_candidate(tmp_path,monkeypatch):
    m=importlib.import_module("tools.benchmark.official_translation")
    monkeypatch.setattr(m,"ARCHIVE",tmp_path/"archive")
    monkeypatch.setattr(m,"require_archive_storage",lambda *a,**k:{})
    root=tmp_path/"run"
    root.mkdir()
    state=dict(schema_version=1,identity={"run":"owned"},step=20,
               weights={"mutable":torch.ones(1)},frozen_fingerprint="same",prompt_rng=None)
    publish=m._archive_json
    def fail_once(path,value):
        raise OSError("receipt write interrupted")
    monkeypatch.setattr(m,"_archive_json",fail_once)
    with pytest.raises(OSError):
        m.candidate_to_archive(root,state,2)
    monkeypatch.setattr(m,"_archive_json",publish)
    dest=m.candidate_to_archive(root,state,2)
    assert dest.is_file() and dest.with_suffix('.json').is_file()
