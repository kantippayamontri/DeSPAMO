import os

import pytest
import torch

from despamo.models.visual_adapter import SpaMoVisualAdapter


@pytest.mark.integration
def test_temporal_weights_and_outputs_match_spamo(monkeypatch: pytest.MonkeyPatch) -> None:
    source = os.environ.get("SPAMO_PROJECT_PATH")
    if source is None:
        pytest.skip("SPAMO_PROJECT_PATH is not configured")
    monkeypatch.syspath_prepend(source)
    from spamo.tconv import TemporalConv as SourceTemporalConv

    torch.manual_seed(7)
    target = SpaMoVisualAdapter(2048, 1024, 768, 2048).eval()
    source_temporal = SourceTemporalConv(768, 768).eval()
    source_temporal.load_state_dict(target.temporal_encoder.state_dict(), strict=True)
    features = torch.randn(2, 768, 24)
    lengths = torch.tensor([20, 24])

    target_output, target_lengths = target.temporal_encoder(features, lengths)
    source_result = source_temporal(features, lengths)

    torch.testing.assert_close(target_output.transpose(0, 1), source_result["visual_feat"])
    assert target_lengths.tolist() == source_result["feat_len"].to(torch.int).tolist()
