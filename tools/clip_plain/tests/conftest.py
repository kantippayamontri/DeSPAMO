import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU suite must not touch CUDA")

    for name in ["init", "synchronize", "empty_cache", "get_rng_state_all", "set_rng_state_all"]:
        monkeypatch.setattr(torch.cuda, name, forbidden)
