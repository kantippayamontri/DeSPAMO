import importlib
import random

import numpy as np
import pytest
import torch


def module():
    return importlib.import_module("tools.benchmark.compact_checkpoint")


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base=torch.nn.Linear(4,4).requires_grad_(False)
        self.head=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4),
                                      torch.nn.Dropout(.2),torch.nn.Linear(4,2))

    def forward(self,x):
        return self.head(self.base(x))


def build():
    torch.manual_seed(0)
    m=Tiny()
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
    return m,opt


def test_compact_excludes_frozen_parameters_but_retains_mutable_buffers(tmp_path):
    m,opt=build()
    x=torch.arange(16,dtype=torch.float32).reshape(4,4)/16
    m(x).square().mean().backward()
    opt.step()
    frozen=module().frozen_fingerprint(m)
    state=module().capture_state(m,opt,dict(identity="run"),1,{"cursor":1})
    assert "base.weight" not in state["weights"]
    assert "head.1.running_mean" in state["weights"]
    p=tmp_path/"snapshot.pt"
    module().save_snapshot(p,state)
    fresh,fopt=build()
    module().load_snapshot(p,fresh,fopt,dict(identity="run"))
    assert module().frozen_fingerprint(fresh)==frozen
    m.eval()
    fresh.eval()
    assert torch.equal(m(x),fresh(x))


def test_compact_optimizer_dropout_and_rng_replay_identical(tmp_path):
    m,opt=build()
    random.seed(1)
    np.random.seed(1)
    x=torch.randn(4,4)
    m(x).square().mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    state=module().capture_state(m,opt,{"run":"same"},1,{"cursor":1})
    p=tmp_path/"snapshot.pt"
    module().save_snapshot(p,state)
    expected=m(x)
    expected.sum().backward()
    opt.step()
    expected_weights={n:v.detach().clone() for n,v in m.state_dict().items()}
    fresh,fopt=build()
    restored=module().load_snapshot(p,fresh,fopt,{"run":"same"})
    assert restored["sampler"]["cursor"]==1
    actual=fresh(x)
    actual.sum().backward()
    fopt.step()
    assert torch.equal(expected,actual)
    assert all(torch.equal(v,expected_weights[n]) for n,v in fresh.state_dict().items())


def test_wrong_base_missing_buffer_and_extra_tensor_rejected(tmp_path):
    m,opt=build()
    state=module().capture_state(m,opt,{"run":"same"},0,{"cursor":0})
    p=tmp_path/"snapshot.pt"
    module().save_snapshot(p,state)
    fresh,fopt=build()
    with torch.no_grad():
        fresh.base.weight.add_(1)
    with pytest.raises(ValueError):
        module().load_snapshot(p,fresh,fopt,{"run":"same"})
    for bad in ["missing","extra"]:
        m,opt=build()
        state=module().capture_state(m,opt,{"run":"same"},0,{"cursor":0})
        if bad=="missing":
            state["weights"].pop("head.1.running_mean")
        else:
            state["weights"]["unknown"]=torch.ones(1)
        path=tmp_path/f"{bad}.pt"
        module().save_snapshot(path,state)
        with pytest.raises(ValueError):
            module().load_snapshot(path,m,opt,{"run":"same"})
