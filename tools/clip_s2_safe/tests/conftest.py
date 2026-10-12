import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU implementation tests must not call CUDA")

    for name in [
        "init",
        "synchronize",
        "empty_cache",
        "get_rng_state_all",
        "set_rng_state_all",
        "device_count",
    ]:
        monkeypatch.setattr(torch.cuda, name, forbidden)
