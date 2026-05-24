from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from cellworld.models.world import mmd_rbf


def evaluate_population_predictions(
    prediction: torch.Tensor,
    target: torch.Tensor,
    logvar: torch.Tensor,
) -> Dict[str, float]:
    with torch.no_grad():
        mse = torch.mean((prediction - target).pow(2)).item()
        mae = torch.mean(torch.abs(prediction - target)).item()
        mmd = mmd_rbf(prediction, target).item()
        energy = _energy_distance(prediction, target).item()
        correlation = _mean_gene_correlation(prediction, target)
        coverage = _gaussian_coverage(prediction, logvar, target, z_value=1.64)
    return {
        "mse": mse,
        "mae": mae,
        "mmd": mmd,
        "energy_distance": energy,
        "mean_gene_correlation": correlation,
        "coverage_90": coverage,
    }


def _energy_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cross = torch.cdist(prediction, target).mean(dim=(1, 2))
    pred = torch.cdist(prediction, prediction).mean(dim=(1, 2))
    tgt = torch.cdist(target, target).mean(dim=(1, 2))
    return (2.0 * cross - pred - tgt).mean()


def _mean_gene_correlation(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = prediction.detach().cpu().numpy().reshape(-1, prediction.shape[-1])
    target_np = target.detach().cpu().numpy().reshape(-1, target.shape[-1])
    values = []
    for gene in range(pred_np.shape[1]):
        left = pred_np[:, gene]
        right = target_np[:, gene]
        if np.std(left) < 1e-8 or np.std(right) < 1e-8:
            continue
        values.append(float(np.corrcoef(left, right)[0, 1]))
    return float(np.mean(values)) if values else 0.0


def _gaussian_coverage(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    z_value: float,
) -> float:
    std = torch.exp(0.5 * logvar)
    lower = mean - z_value * std
    upper = mean + z_value * std
    return torch.logical_and(target >= lower, target <= upper).float().mean().item()

