import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from despamo.models.temporal import TemporalConv


def lengths_to_mask(lengths: torch.Tensor, max_length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(max_length, device=device)[None, :] < lengths[:, None]


class SpaMoVisualAdapter(nn.Module):
    def __init__(
        self, spatial_dim: int, motion_dim: int, adapter_dim: int, language_dim: int
    ) -> None:
        super().__init__()
        self.spatial_projector = nn.Linear(spatial_dim, adapter_dim)
        self.motion_projector = nn.Linear(motion_dim, adapter_dim)
        self.temporal_encoder = TemporalConv(adapter_dim, adapter_dim)
        self.multimodal_projector = nn.Sequential(
            nn.Linear(adapter_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        spatial_mask: torch.Tensor,
        motion: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = self.spatial_projector(spatial)
        motion = self.motion_projector(motion)
        fused = [
            torch.cat((spatial[i, spatial_mask[i]], motion[i, motion_mask[i]]), dim=0)
            for i in range(spatial.shape[0])
        ]
        lengths = torch.tensor([item.shape[0] for item in fused], device=spatial.device)
        padded = pad_sequence(fused, batch_first=True)
        encoded, output_lengths = self.temporal_encoder(padded.transpose(1, 2), lengths)
        tokens = self.multimodal_projector(encoded)
        mask = lengths_to_mask(output_lengths, tokens.shape[1], tokens.device)
        return tokens, mask
