import json
from pathlib import Path

import torch

from tools.spamo_repro.model import build_visual_adapter, configure_training


def test_transformer_checkpointing_keeps_frozen_base_and_supports_visual_input_gradients():
    from transformers import T5Config, T5ForConditionalGeneration

    transformer = T5ForConditionalGeneration(
        T5Config(
            vocab_size=32,
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            d_kv=8,
            dropout_rate=0.0,
            decoder_start_token_id=0,
        )
    )
    transformer.requires_grad_(False)
    configure_training(transformer)
    assert transformer.is_gradient_checkpointing
    assert transformer.config.use_cache is False
    assert all(not p.requires_grad for p in transformer.parameters())
    visual = torch.randn(2, 3, 16, requires_grad=True)
    loss = transformer(inputs_embeds=visual, labels=torch.tensor([[1, 2], [3, 4]])).loss
    loss.backward()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    assert all(p.grad is None for p in transformer.parameters())


def test_original_linear_projections_tcn_and_connector_shapes():
    config = json.loads((Path(__file__).parents[1] / "config.json").read_text())
    adapter = build_visual_adapter(config)
    assert isinstance(adapter.spatial_projector, torch.nn.Linear)
    assert (adapter.spatial_projector.in_features, adapter.spatial_projector.out_features) == (
        2048,
        768,
    )
    assert (adapter.motion_projector.in_features, adapter.motion_projector.out_features) == (
        1024,
        768,
    )
    temporal = adapter.temporal_encoder.temporal_conv
    assert [type(layer) for layer in temporal] == [
        torch.nn.Conv1d,
        torch.nn.BatchNorm1d,
        torch.nn.ReLU,
        torch.nn.MaxPool1d,
        torch.nn.Conv1d,
        torch.nn.BatchNorm1d,
        torch.nn.ReLU,
        torch.nn.MaxPool1d,
    ]
    assert temporal[0].kernel_size == (5,) and temporal[4].kernel_size == (5,)
    connector = adapter.multimodal_projector
    assert isinstance(connector[1], torch.nn.GELU)
    assert connector[0].weight.shape == (2048, 768)
    assert connector[2].weight.shape == (2048, 2048)
    adapter.eval()
    with torch.no_grad():
        output, mask = adapter(
            torch.randn(2, 20, 2048),
            torch.ones(2, 20, dtype=torch.bool),
            torch.randn(2, 16, 1024),
            torch.ones(2, 16, dtype=torch.bool),
        )
    assert output.shape == (2, 6, 2048) and mask.shape == (2, 6)
    assert torch.isfinite(output).all() and mask.all()
