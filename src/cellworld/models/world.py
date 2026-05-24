from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from cellworld.models.layers import FourierTimeFeatures, MLP, population_stats


@dataclass(frozen=True)
class ModelConfig:
    genes: int
    cell_types: int
    latent_dim: int = 48
    hidden_dim: int = 96
    message_steps: int = 3
    dropout: float = 0.05


class LearnedGeneGraph(nn.Module):
    """Signed gene interaction map with penalties exposed for training."""

    def __init__(self, genes: int):
        super().__init__()
        raw = torch.empty(genes, genes)
        nn.init.normal_(raw, mean=0.0, std=0.025)
        self.raw_edges = nn.Parameter(raw)
        self.register_buffer("eye", torch.eye(genes))

    def adjacency(self) -> torch.Tensor:
        signed = torch.tanh(self.raw_edges)
        return signed * (1.0 - self.eye)

    def l1_penalty(self) -> torch.Tensor:
        return self.adjacency().abs().mean()

    def acyclicity_penalty(self) -> torch.Tensor:
        # NOTEARS-style smooth acyclicity surrogate on squared edge weights.
        matrix = self.adjacency().pow(2)
        return torch.trace(torch.matrix_exp(matrix)) - matrix.shape[0]

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        return cells @ self.adjacency().T


class CounterfactualCellWorld(nn.Module):
    """Distribution-level counterfactual model for single-cell perturbations."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gene_graph = LearnedGeneGraph(config.genes)
        self.cell_type_embedding = nn.Embedding(config.cell_types, config.hidden_dim)
        self.time_features = FourierTimeFeatures(bands=6)

        context_dim = (
            config.genes * 4
            + config.genes
            + config.hidden_dim
            + self.time_features.output_dim
        )
        self.context_encoder = MLP(
            context_dim,
            [config.hidden_dim, config.hidden_dim],
            config.hidden_dim,
            dropout=config.dropout,
            final_activation=True,
        )
        self.cell_encoder = MLP(
            config.genes * 2 + config.hidden_dim,
            [config.hidden_dim, config.hidden_dim],
            config.latent_dim,
            dropout=config.dropout,
            final_activation=True,
        )
        self.transition = nn.ModuleList(
            [
                MLP(
                    config.latent_dim + config.hidden_dim,
                    [config.hidden_dim],
                    config.latent_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.message_steps)
            ]
        )
        self.decoder = MLP(
            config.latent_dim + config.hidden_dim,
            [config.hidden_dim, config.hidden_dim],
            config.genes * 2,
            dropout=config.dropout,
        )

    def forward(
        self,
        source: torch.Tensor,
        perturbation: torch.Tensor,
        cell_type: torch.Tensor,
        time: torch.Tensor,
        dose: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        context = self._encode_context(source, perturbation, cell_type, time, dose)
        graph_signal = self.gene_graph(source)
        per_cell_context = context.unsqueeze(1).expand(-1, source.shape[1], -1)
        latent = self.cell_encoder(torch.cat([source, graph_signal, per_cell_context], dim=-1))

        for block in self.transition:
            update = block(torch.cat([latent, per_cell_context], dim=-1))
            latent = latent + 0.35 * torch.tanh(update)

        decoded = self.decoder(torch.cat([latent, per_cell_context], dim=-1))
        mean, raw_logvar = decoded.chunk(2, dim=-1)
        logvar = torch.clamp(raw_logvar, min=-6.0, max=2.5)
        return {
            "mean": mean,
            "logvar": logvar,
            "latent": latent,
            "adjacency": self.gene_graph.adjacency(),
        }

    def sample_counterfactual(
        self,
        source: torch.Tensor,
        perturbation: torch.Tensor,
        cell_type: torch.Tensor,
        time: torch.Tensor,
        dose: torch.Tensor,
        samples: int = 1,
    ) -> torch.Tensor:
        prediction = self.forward(source, perturbation, cell_type, time, dose)
        mean = prediction["mean"]
        std = torch.exp(0.5 * prediction["logvar"])
        draws = []
        for _ in range(samples):
            draws.append(mean + std * torch.randn_like(std))
        return torch.stack(draws, dim=1)

    def regularization(self) -> Dict[str, torch.Tensor]:
        return {
            "graph_l1": self.gene_graph.l1_penalty(),
            "acyclicity": self.gene_graph.acyclicity_penalty(),
        }

    def _encode_context(
        self,
        source: torch.Tensor,
        perturbation: torch.Tensor,
        cell_type: torch.Tensor,
        time: torch.Tensor,
        dose: torch.Tensor,
    ) -> torch.Tensor:
        stats = population_stats(source)
        type_features = self.cell_type_embedding(cell_type)
        time_features = self.time_features(time, dose)
        return self.context_encoder(torch.cat([stats, perturbation, type_features, time_features], dim=-1))


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + (target - mean).pow(2) / torch.exp(logvar)).mean()


def mmd_rbf(source: torch.Tensor, target: torch.Tensor, bandwidth: float = 1.5) -> torch.Tensor:
    """Population-level MMD averaged across a batch."""

    source_dist = torch.cdist(source, source).pow(2)
    target_dist = torch.cdist(target, target).pow(2)
    cross_dist = torch.cdist(source, target).pow(2)
    k_xx = torch.exp(-source_dist / (2.0 * bandwidth**2)).mean(dim=(1, 2))
    k_yy = torch.exp(-target_dist / (2.0 * bandwidth**2)).mean(dim=(1, 2))
    k_xy = torch.exp(-cross_dist / (2.0 * bandwidth**2)).mean(dim=(1, 2))
    return (k_xx + k_yy - 2.0 * k_xy).mean()


def training_loss(
    prediction: Dict[str, torch.Tensor],
    target: torch.Tensor,
    mmd_weight: float,
    graph_l1: torch.Tensor,
    graph_l1_weight: float,
    acyclicity: torch.Tensor,
    acyclicity_weight: float,
) -> Dict[str, torch.Tensor]:
    nll = gaussian_nll(prediction["mean"], prediction["logvar"], target)
    mmd = mmd_rbf(prediction["mean"], target)
    total = nll + mmd_weight * mmd + graph_l1_weight * graph_l1 + acyclicity_weight * acyclicity
    return {
        "loss": total,
        "nll": nll.detach(),
        "mmd": mmd.detach(),
        "graph_l1": graph_l1.detach(),
        "acyclicity": acyclicity.detach(),
    }


def deterministic_baseline(source: torch.Tensor, perturbation: torch.Tensor) -> torch.Tensor:
    return torch.tanh(source + perturbation.unsqueeze(1))


def mean_shift_baseline(source: torch.Tensor, perturbation: torch.Tensor) -> torch.Tensor:
    source_mean = source.mean(dim=1, keepdim=True)
    centered = source - source_mean
    shifted_mean = torch.tanh(source_mean + perturbation.unsqueeze(1))
    return shifted_mean + centered

