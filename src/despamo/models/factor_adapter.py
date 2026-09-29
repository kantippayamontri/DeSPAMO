"""Expose baseline-compatible projected spatial tokens for factor heads."""

import torch
from torch.nn.utils.rnn import pad_sequence

from despamo.models.visual_adapter import SpaMoVisualAdapter, lengths_to_mask


class FactorVisualAdapter(SpaMoVisualAdapter):
    def forward_with_spatial(self, spatial, spatial_mask, motion, motion_mask):
        projected = self.spatial_projector(spatial)
        projected_motion = self.motion_projector(motion)
        fused = [
            torch.cat((projected[i, spatial_mask[i]], projected_motion[i, motion_mask[i]]), dim=0)
            for i in range(projected.shape[0])
        ]
        lengths = torch.tensor([item.shape[0] for item in fused], device=projected.device)
        encoded, output_lengths = self.temporal_encoder(
            pad_sequence(fused, batch_first=True).transpose(1, 2), lengths
        )
        tokens = self.multimodal_projector(encoded)
        mask = lengths_to_mask(output_lengths, tokens.shape[1], tokens.device)
        return tokens, mask, projected

    def forward(self, spatial, spatial_mask, motion, motion_mask):
        tokens, mask, _ = self.forward_with_spatial(spatial, spatial_mask, motion, motion_mask)
        return tokens, mask
