import torch
from torch import nn


class TemporalConv(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=5),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths - 4
        lengths = torch.div(lengths, 2, rounding_mode="floor")
        lengths = lengths - 4
        return torch.div(lengths, 2, rounding_mode="floor")

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.any(lengths < 16):
            raise ValueError(f"all fused sequences must contain at least 16 tokens: {lengths}")
        if torch.any(lengths > features.shape[-1]):
            raise ValueError(
                f"fused lengths cannot exceed input time {features.shape[-1]}: {lengths}"
            )
        output = self.temporal_conv(features).transpose(1, 2)
        return output, self.output_lengths(lengths)
