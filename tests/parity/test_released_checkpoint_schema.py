import os
from pathlib import Path

import pytest
import torch

from despamo.checkpoints import convert_spamo_state_dict


@pytest.mark.integration
def test_released_checkpoint_maps_all_871_tensors() -> None:
    checkpoint = os.environ.get("DESPAMO_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("DESPAMO_CHECKPOINT is not configured")
    source = torch.load(Path(checkpoint), map_location="meta", weights_only=True)["state_dict"]
    converted = convert_spamo_state_dict(source)
    assert len(source) == 871
    assert len(converted) == 871
