from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        dropout: float = 0.0,
        final_activation: bool = False,
    ):
        super().__init__()
        dims: List[int] = [input_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for left, right in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(left, right))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        if final_activation:
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierTimeFeatures(nn.Module):
    """Small fixed Fourier map for scalar time and dose values."""

    def __init__(self, bands: int = 6):
        super().__init__()
        frequencies = torch.tensor([2.0**idx for idx in range(bands)], dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)

    @property
    def output_dim(self) -> int:
        return int(self.frequencies.numel() * 4)

    def forward(self, time: torch.Tensor, dose: torch.Tensor) -> torch.Tensor:
        scalars = torch.stack([time, dose], dim=-1)
        angles = scalars.unsqueeze(-1) * self.frequencies.view(1, 1, -1) * math.pi
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return features.flatten(start_dim=1)


def population_stats(cells: torch.Tensor) -> torch.Tensor:
    mean = cells.mean(dim=1)
    std = cells.std(dim=1, unbiased=False)
    q25 = torch.quantile(cells, 0.25, dim=1)
    q75 = torch.quantile(cells, 0.75, dim=1)
    return torch.cat([mean, std, q25, q75], dim=-1)

