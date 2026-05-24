from __future__ import annotations

import torch

from cellworld.models import CounterfactualCellWorld, ModelConfig
from cellworld.models.world import training_loss


def test_model_forward_shapes_and_loss() -> None:
    model = CounterfactualCellWorld(ModelConfig(genes=12, cell_types=3, latent_dim=16, hidden_dim=24))
    source = torch.randn(2, 8, 12)
    target = torch.randn(2, 8, 12)
    perturbation = torch.zeros(2, 12)
    perturbation[:, 3] = -1.0
    cell_type = torch.tensor([0, 2])
    time = torch.ones(2)
    dose = torch.ones(2)

    prediction = model(source, perturbation, cell_type, time, dose)
    regularization = model.regularization()
    losses = training_loss(
        prediction,
        target,
        mmd_weight=0.2,
        graph_l1=regularization["graph_l1"],
        graph_l1_weight=0.001,
        acyclicity=regularization["acyclicity"],
        acyclicity_weight=0.001,
    )

    assert prediction["mean"].shape == target.shape
    assert prediction["logvar"].shape == target.shape
    assert prediction["adjacency"].shape == (12, 12)
    assert torch.isfinite(losses["loss"])

