import importlib
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tools.dinov3.e3_model import DINOFactorHeads, LoRALinear
from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.dinov3.official_schedule import AdaptSchedule


def module():
    return importlib.import_module("tools.dinov3.official_adapt")


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = LoRALinear(torch.nn.Linear(3, 4))

    def forward(self, pixel_values):
        value = self.layer(pixel_values.mean(dim=(-1, -2)))
        return SimpleNamespace(last_hidden_state=value[:, None, :])


def setup():
    torch.manual_seed(0)
    model, heads = Tiny(), DINOFactorHeads(visual_width=8, text_width=8)
    batch = dict(images224=torch.randn(2, 5, 3, 4, 4), images448=torch.randn(2, 5, 3, 4, 4),
                 teacher=torch.randn(2, 5, 8, requires_grad=True), vectors=torch.randn(2, 19, 8),
                 masks=torch.ones(2, 19, dtype=torch.bool),
                 labels=tuple(tuple(f"{i}-{j}" for j in range(19)) for i in range(2)))
    schedule = AdaptSchedule(train_clips=4, batch_size=2, epochs=4, calibration_epochs=1,
                             lora_warmup_epochs=1, grl_ramp_epochs=1, checkpoint_epochs=1)
    opt = module().make_optimizer(model, heads)
    return model, heads, batch, schedule, opt


def test_calibration_does_not_decay_or_update_lora_then_joint_has_all_terms():
    m, h, b, s, opt = setup()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    cal = module().train_official_update(m, h, opt, b, 1, s)
    assert cal["grl"] == 0 and cal["lora_lr"] == 0
    assert all(torch.equal(p, before[n]) for n, p in m.named_parameters())
    assert all(p.grad is None for p in m.parameters())
    losses = module().compute_official_losses(m, h, b, 4, s)
    expected = .05*sum(losses[n] for n in ("biometric", "clothing", "hair", "background"))
    expected += .1*sum(losses[n] for n in ("left_hand", "right_hand", "mouth"))
    expected += .1*(losses["reference_loss"]+losses["relational_pattern"]
                    +losses["relational_spread"])
    assert torch.allclose(losses["combined_loss"], expected)
    result = module().train_official_update(m, h, opt, b, 4, s)
    assert result["lora_grad_norm"] > 0 and b["teacher"].grad is None
    assert not torch.equal(before["layer.up.weight"], m.layer.up.weight)
    assert torch.equal(before["layer.base.weight"], m.layer.base.weight)


def test_nonfinite_loss_cannot_update_parameters():
    m, h, b, s, opt = setup()
    b["teacher"] = torch.full_like(b["teacher"], float("nan"))
    before = m.layer.up.weight.detach().clone()
    with pytest.raises((ValueError, FloatingPointError)):
        module().train_official_update(m, h, opt, b, 3, s)
    assert torch.equal(before, m.layer.up.weight)


def test_exact_kernel_rng_optimizer_sampler_resume():
    m, h, b, s, opt = setup()
    sampler = OfficialBatchSampler(("a", "b", "c", "d"), batch_size=2)
    for k in (1, 2, 3, 4):
        sampler.next_ids()
        module().train_official_update(m, h, opt, b, k, s)
    snapshot = deepcopy(dict(model=m.state_dict(), heads=h.state_dict(), opt=opt.state_dict(),
                             rng=module().capture_rng(), sampler=sampler.state_dict()))
    next_batch = sampler.next_ids()
    expected = module().train_official_update(m, h, opt, b, 5, s)
    expected_weights = {n: p.clone() for n, p in m.named_parameters()}
    restored, rh, rb, rs, ropt = setup()
    restored.load_state_dict(snapshot["model"])
    rh.load_state_dict(snapshot["heads"])
    ropt.load_state_dict(snapshot["opt"])
    replay = OfficialBatchSampler(("a", "b", "c", "d"), batch_size=2)
    replay.load_state_dict(snapshot["sampler"])
    module().restore_rng(snapshot["rng"])
    assert replay.next_ids() == next_batch
    actual = module().train_official_update(restored, rh, ropt, rb, 5, rs)
    assert actual == expected
    assert all(torch.equal(p, expected_weights[n]) for n, p in restored.named_parameters())


def test_config_rejects_changed_recipe_and_accepts_batch24():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    cfg = json.loads((root/"configs/adaptation/phoenix14t_dino_merged40.json").read_text())
    module().validate_config(cfg)
    for key, value in (("batch_size", 16), ("epochs", 39), ("calibration_epochs", 4)):
        changed = deepcopy(cfg)
        changed["schedule"][key] = value
        with pytest.raises(ValueError):
            module().validate_config(changed)
    cfg["losses"]["relational_weight"] = 0.
    with pytest.raises(ValueError):
        module().validate_config(cfg)


def test_ledger_preserves_spend_and_rejects_live_owner(tmp_path):
    identity = dict(test="owned")
    ledger = module().OfficialLedger(tmp_path, identity, initial_seconds=5.)
    ledger.charge(7, "test")
    ledger.close("interrupted")
    prior = ledger.last
    restored = module().OfficialLedger(tmp_path, identity)
    assert restored.last >= prior and restored.completed == 7
    with pytest.raises(RuntimeError):
        module().OfficialLedger(tmp_path, identity)
    restored.close("interrupted")


def test_storage_checks_both_backing_and_linux_reserve(tmp_path, monkeypatch):
    runtime = importlib.import_module("tools.dinov3.official_runtime")
    monkeypatch.setattr(runtime, "windows_free", lambda drive: 51*1024**3)
    monkeypatch.setattr(runtime.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=100*1024**3))
    with pytest.raises(ValueError):
        module().require_official_storage(tmp_path, 2*1024**3)


def test_nonfinite_gradient_rejected_before_optimizer_step():
    m,h,b,s,opt=setup()
    base=m.layer.base.weight.detach().clone()
    m.layer.up.weight.register_hook(lambda grad: torch.full_like(grad,float("nan")))
    with pytest.raises(FloatingPointError,match="gradient"):
        module().train_official_update(m,h,opt,b,4,s)
    assert torch.equal(base,m.layer.base.weight) and not opt.state


def test_config_requires_explicit_duration_null_not_missing():
    import json
    from pathlib import Path

    cfg=json.loads((Path(__file__).resolve().parents[3]/
                   "configs/adaptation/phoenix14t_dino_merged40.json").read_text())
    del cfg["duration_limit_seconds"]
    with pytest.raises(ValueError):
        module().validate_config(cfg)


def test_closed_ledger_refuses_mismatched_reused_owner(tmp_path,monkeypatch):
    runtime=importlib.import_module("tools.dinov3.official_runtime")
    ledger=module().OfficialLedger(tmp_path,{"identity":"x"})
    ledger.close("interrupted")
    monkeypatch.setattr(runtime,"process_owner",lambda pid:{"pid":pid,"boot_id":"different"})
    with pytest.raises(RuntimeError):
        module().OfficialLedger(tmp_path,{"identity":"x"})


def test_failed_smoke_charges_are_inherited_without_erasure(tmp_path):
    from tools.dinov3.identity import digest
    from tools.dinov3.storage import atomic_json

    cfg={"working_root":str(tmp_path),"identity":"same"}
    root=tmp_path/"smoke-attempts"/digest(cfg)
    atomic_json(root/"a.json",dict(config_hash=digest(cfg),charged_seconds=3.,status="failed"))
    atomic_json(root/"b.json",dict(config_hash=digest(cfg),charged_seconds=4.,status="complete"))
    assert module().smoke_setup_charge(cfg)==7.


def test_factor_projection_fp32_inside_bf16_training():
    m,h,b,s,opt=setup()
    dtypes=[]
    h.heads["left_hand"].register_forward_hook(lambda mod,args,out:dtypes.append(out.dtype))
    module().train_official_update(m,h,opt,b,4,s)
    assert dtypes==[torch.float32]


def test_resume_charges_new_setup_without_double_counting_old_smoke(tmp_path):
    ledger=module().OfficialLedger(tmp_path,{"identity":"x"},initial_seconds=10.)
    ledger.close("interrupted")
    prior=ledger.last
    restored=module().OfficialLedger(tmp_path,{"identity":"x"},initial_seconds=10.,setup_seconds=3.)
    assert prior+3.<=restored.last<prior+3.1
    restored.close("interrupted")


def test_durable_checkpoint_replays_kernel_weights_and_rng(tmp_path):
    from tools.dinov3.official_checkpoints import publish_milestone, validate_official_resume

    m,h,b,s,opt=setup()
    sampler=OfficialBatchSampler(("a","b","c","d"),batch_size=2)
    for k in range(1,5):
        sampler.next_ids()
        module().train_official_update(m,h,opt,b,k,s)
    run=tmp_path/"official-dino-adapt-test"
    run.mkdir()
    identity=dict(policy="official-phoenix14t-merged-dino-adaptation-v1",
                  run_key=run.name,run_root=str(run))
    state=module()._state(m,h,opt,sampler,s,identity,10.)
    publish_milestone(run,state,2)
    expected_batch=sampler.next_ids()
    expected=module().train_official_update(m,h,opt,b,5,s)
    expected_weights={n:p.clone() for n,p in m.named_parameters()}
    restored=validate_official_resume(run/"resume-epoch-2.pt",identity,
                                     dict(charged_seconds=11.,completed_updates=4))
    rm,rh,rb,rs,ropt=setup()
    with torch.no_grad():
        for name,p in rm.named_parameters():
            if name in restored["model"]:
                p.copy_(restored["model"][name])
    rh.load_state_dict(restored["heads"])
    ropt.load_state_dict(restored["optimizer"])
    replay=OfficialBatchSampler(("a","b","c","d"),batch_size=2)
    replay.load_state_dict(restored["sampler"])
    module().restore_rng(restored["rng"])
    assert replay.next_ids()==expected_batch
    actual=module().train_official_update(rm,rh,ropt,rb,5,rs)
    assert actual==expected
    assert all(torch.equal(p,expected_weights[n]) for n,p in rm.named_parameters())
