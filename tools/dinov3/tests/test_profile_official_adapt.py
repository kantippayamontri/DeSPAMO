import importlib
from types import SimpleNamespace

import pytest
import torch

from tools.dinov3.e3_model import DINOFactorHeads, LoRALinear


def module():
    return importlib.import_module("tools.dinov3.profile_official_adapt")


def test_joint_profile_updates_lora_heads_and_initializes_every_optimizer_state():
    torch.manual_seed(0)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = LoRALinear(torch.nn.Linear(3, 4))

        def forward(self, pixel_values):
            x = self.layer(pixel_values.mean(dim=(-1, -2)))
            return SimpleNamespace(last_hidden_state=x[:, None, :])

    model, heads = Model(), DINOFactorHeads(visual_width=8, text_width=8)
    lora = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([dict(params=lora, lr=1e-5),
                            dict(params=heads.parameters(), lr=3e-5)], weight_decay=.05)
    batch = dict(images224=torch.randn(2, 5, 3, 4, 4), images448=torch.randn(2, 5, 3, 4, 4),
                 teacher=torch.randn(2, 5, 8), vectors=torch.randn(2, 19, 8),
                 masks=torch.ones(2, 19, dtype=torch.bool),
                 labels=tuple(tuple(f"clip-{i}-slot-{j}" for j in range(19)) for i in range(2)))
    base = model.layer.base.weight.detach().clone()
    up = model.layer.up.weight.detach().clone()
    assert module().optimizer_state_report(opt)["initialized_parameters"] == 0
    for _ in range(2):
        result = module().joint_update(model, heads, opt, batch)
        assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    state = module().optimizer_state_report(opt)
    assert state["initialized_parameters"] == state["expected_parameters"] == 16
    assert state["tensor_bytes"] > 0
    assert not torch.equal(up, model.layer.up.weight)
    assert torch.equal(base, model.layer.base.weight)
    assert model.layer.base.weight.grad is None


def test_profile_batches_nested_and_signer_diverse():
    admission = dict(train_ids=tuple(str(i) for i in range(40)),
                     clips={str(i): dict(signer=f"Signer{i % 9+1:02}") for i in range(40)})
    sixteen = module().profile_ids(admission, 16)
    thirtytwo = module().profile_ids(admission, 32)
    assert sixteen == thirtytwo[:16]
    assert len(set(thirtytwo)) == 32
    assert len({admission["clips"][i]["signer"] for i in sixteen}) == 9
    with pytest.raises(ValueError):
        module().profile_ids(admission, 64)


def test_invalid_joint_gradient_is_rejected_before_optimizer_step():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, pixel_values):
            return SimpleNamespace(last_hidden_state=self.weight * torch.full(
                (pixel_values.shape[0], 1, 4), float("nan")))

    model, heads = Model(), DINOFactorHeads(visual_width=8, text_width=8)
    opt = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()))
    batch = dict(images224=torch.ones(2, 5, 3, 4, 4), images448=torch.ones(2, 5, 3, 4, 4),
                 teacher=torch.ones(2, 5, 8), vectors=torch.ones(2, 19, 8),
                 masks=torch.ones(2, 19, dtype=torch.bool),
                 labels=tuple(tuple(f"{i}-{j}" for j in range(19)) for i in range(2)))
    with pytest.raises((ValueError, FloatingPointError)):
        module().joint_update(model, heads, opt, batch)
    assert not opt.state


def test_profile_identity_round_trips_native_atomic_json(tmp_path):
    import json

    from tools.dinov3.storage import atomic_json

    identity = module().profile_identity(
        16, {"manifest_sha256": "x"}, tuple(str(i) for i in range(16)))
    path = tmp_path / "identity.json"
    atomic_json(path, identity)
    assert json.loads(path.read_text()) == identity
    assert identity["clip_ids"] == [str(i) for i in range(16)]


def test_batch24_is_nested_and_uses_five_measured_updates():
    admission = dict(train_ids=tuple(str(i) for i in range(40)),
                     clips={str(i): dict(signer=f"Signer{i % 9+1:02}") for i in range(40)})
    ids = module().profile_ids(admission, 24)
    assert len(ids) == 24 and ids == module().profile_ids(admission, 32)[:24]
    identity = module().profile_identity(24, {"manifest_sha256": "x"}, ids)
    assert identity["warmup_updates"] == 2 and identity["measured_updates"] == 5
