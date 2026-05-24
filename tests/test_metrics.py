from __future__ import annotations

import torch

from cellworld.eval import evaluate_population_predictions


def test_metrics_reward_matching_population() -> None:
    target = torch.randn(2, 10, 6)
    close = target + 0.01 * torch.randn_like(target)
    far = target + 2.0
    logvar = torch.zeros_like(target)

    close_metrics = evaluate_population_predictions(close, target, logvar)
    far_metrics = evaluate_population_predictions(far, target, logvar)

    assert close_metrics["mse"] < far_metrics["mse"]
    assert close_metrics["mmd"] < far_metrics["mmd"]

