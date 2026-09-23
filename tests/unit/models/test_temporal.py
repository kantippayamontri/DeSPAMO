import pytest
import torch

from despamo.models.temporal import TemporalConv


def test_temporal_conv_updates_lengths() -> None:
    encoder = TemporalConv(768, 768)
    features = torch.randn(2, 768, 24)

    output, lengths = encoder(features, torch.tensor([20, 24]))

    assert output.shape == (2, 3, 768)
    assert lengths.tolist() == [2, 3]


def test_temporal_conv_accepts_minimum_fused_length() -> None:
    encoder = TemporalConv(4, 4)

    output, lengths = encoder(torch.randn(1, 4, 16), torch.tensor([16]))

    assert output.shape == (1, 1, 4)
    assert lengths.tolist() == [1]


@pytest.mark.parametrize("length", [0, 15])
def test_temporal_conv_rejects_short_fused_length(length: int) -> None:
    encoder = TemporalConv(4, 4)

    with pytest.raises(ValueError, match="at least 16 tokens"):
        encoder(torch.randn(1, 4, 16), torch.tensor([length]))


def test_temporal_conv_rejects_length_exceeding_available_frames() -> None:
    encoder = TemporalConv(4, 4)

    with pytest.raises(ValueError, match="exceed input time"):
        encoder(torch.randn(1, 4, 20), torch.tensor([21]))


def test_temporal_conv_preserves_spamo_parameter_names() -> None:
    encoder = TemporalConv(4, 4)

    assert set(encoder.state_dict()) == {
        "temporal_conv.0.weight",
        "temporal_conv.0.bias",
        "temporal_conv.1.weight",
        "temporal_conv.1.bias",
        "temporal_conv.1.running_mean",
        "temporal_conv.1.running_var",
        "temporal_conv.1.num_batches_tracked",
        "temporal_conv.4.weight",
        "temporal_conv.4.bias",
        "temporal_conv.5.weight",
        "temporal_conv.5.bias",
        "temporal_conv.5.running_mean",
        "temporal_conv.5.running_var",
        "temporal_conv.5.num_batches_tracked",
    }
