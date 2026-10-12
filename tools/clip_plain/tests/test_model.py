import torch
from torch import nn


class Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(8, 8)
        from types import SimpleNamespace

        self.model.config = SimpleNamespace(use_cache=True)
        self.model.gradient_checkpointing_enable = lambda: setattr(
            self.model, "checkpointing", True
        )
        self.model.enable_input_require_grads = lambda: None


def test_model_spatial_input_1024_and_language_output_2048(monkeypatch):
    from tools.clip_plain import model as m

    monkeypatch.setattr(m.FlanT5Backbone, "from_pretrained", lambda *a, **k: Language())
    a, _ = m.build_model(0, "unused")
    b, _ = m.build_model(0, "unused")
    assert a.visual_adapter.spatial_projector[0].weight.shape == (1024, 1024)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])
    v, mask = a.visual_adapter.eval()(
        torch.ones(2, 20, 1024),
        torch.ones(2, 20, dtype=torch.bool),
        torch.ones(2, 1, 1024),
        torch.ones(2, 1, dtype=torch.bool),
    )
    assert v.shape[-1] == 2048 and mask.any(1).all()
    assert a.language_model.model.config.use_cache is False
    assert a.language_model.model.checkpointing and a.warm_up_steps == 4000
    assert a.vt_pooling == "legacy_mean" and a.language_model.max_text_length == 1024
