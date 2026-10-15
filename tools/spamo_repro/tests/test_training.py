import pytest
import torch


def test_accumulated_VT_only_update_does_not_call_translation_loss():
    from tools.spamo_repro.train import accumulated_update

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    model = Model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    calls = []

    def losses(m, batch, joint):
        calls.append(joint)
        return m.weight.square(), None

    result = accumulated_update(model, opt, [1, 2], 1, 17760, losses)
    assert calls == [False, False]
    assert result["optimizer_update"] == 1
    assert len(opt.state) == 1 and next(iter(opt.state.values()))["step"] == 1


def test_accumulation_tail_retains_declared_divisor_two():
    from tools.spamo_repro.train import accumulated_update

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    def losses(m, batch, joint):
        return m.weight.sum(), None

    result = accumulated_update(model, opt, [1], 1, 17760, losses)
    assert result["grad_norm"] == pytest.approx(0.5)
