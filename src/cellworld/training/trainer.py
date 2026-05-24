from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cellworld.data import (
    CellPopulationDataset,
    SyntheticWorldConfig,
    build_synthetic_splits,
    collate_conditions,
)
from cellworld.data.synthetic import SyntheticSplitConfig
from cellworld.data.types import summarize_conditions
from cellworld.eval import evaluate_population_predictions
from cellworld.models import CounterfactualCellWorld, ModelConfig
from cellworld.models.world import deterministic_baseline, mean_shift_baseline, training_loss
from cellworld.utils.config import ensure_dir


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 12
    learning_rate: float = 1.5e-3
    weight_decay: float = 1e-4
    mmd_weight: float = 0.25
    graph_l1_weight: float = 5e-4
    acyclicity_weight: float = 1e-4
    device: str = "auto"


def train_synthetic(config: Dict[str, Any], output_root: str = "runs") -> Path:
    seed = int(config.get("seed", 7))
    _seed_everything(seed)

    world_config = SyntheticWorldConfig(**config["world"])
    split_config = SyntheticSplitConfig(**config["dataset"])
    training_config = TrainingConfig(**config["training"])
    model_config = ModelConfig(
        genes=world_config.genes,
        cell_types=world_config.cell_types,
        **config["model"],
    )
    run_name = str(config.get("output", {}).get("run_name", "synthetic_small"))
    run_dir = ensure_dir(str(Path(output_root) / run_name))

    splits, ground_truth = build_synthetic_splits(world_config, split_config, seed)
    datasets = {name: CellPopulationDataset(items) for name, items in splits.items()}
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=training_config.batch_size,
            shuffle=True,
            collate_fn=collate_conditions,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=training_config.batch_size,
            shuffle=False,
            collate_fn=collate_conditions,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=training_config.batch_size,
            shuffle=False,
            collate_fn=collate_conditions,
        ),
    }

    device = _resolve_device(training_config.device)
    model = CounterfactualCellWorld(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_path = run_dir / "best_model.pt"

    for epoch in tqdm(range(1, training_config.epochs + 1), desc="training"):
        train_metrics = _run_epoch(model, loaders["train"], optimizer, training_config, device)
        val_metrics = _evaluate_loader(model, loaders["val"], device)
        row = {"epoch": float(epoch)}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        if val_metrics["mmd"] < best_val:
            best_val = val_metrics["mmd"]
            torch.save({"model": model.state_dict(), "config": config}, best_path)

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])

    test_metrics = _evaluate_loader(model, loaders["test"], device)
    baseline_metrics = _baseline_metrics(loaders["test"], device)
    artifacts = _save_artifacts(
        run_dir=run_dir,
        model=model,
        ground_truth=ground_truth.adjacency,
        history=history,
        config=config,
        split_summary={
            split: summarize_conditions(items)
            for split, items in splits.items()
        },
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        loader=loaders["test"],
        device=device,
    )
    return artifacts


def _run_epoch(
    model: CounterfactualCellWorld,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    batches = 0
    for batch in loader:
        source, target, perturbation, cell_type, time, dose = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(source, perturbation, cell_type, time, dose)
        regularization = model.regularization()
        losses = training_loss(
            prediction=prediction,
            target=target,
            mmd_weight=config.mmd_weight,
            graph_l1=regularization["graph_l1"],
            graph_l1_weight=config.graph_l1_weight,
            acyclicity=regularization["acyclicity"],
            acyclicity_weight=config.acyclicity_weight,
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def _evaluate_loader(
    model: CounterfactualCellWorld,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            source, target, perturbation, cell_type, time, dose = _move_batch(batch, device)
            prediction = model(source, perturbation, cell_type, time, dose)
            rows.append(
                evaluate_population_predictions(
                    prediction=prediction["mean"],
                    target=target,
                    logvar=prediction["logvar"],
                )
            )
    return _average_rows(rows)


def _baseline_metrics(loader: DataLoader, device: torch.device) -> Dict[str, Dict[str, float]]:
    rows = {"direct_shift": [], "mean_shift": []}
    with torch.no_grad():
        for batch in loader:
            source, target, perturbation, _, _, _ = _move_batch(batch, device)
            direct = deterministic_baseline(source, perturbation)
            shifted = mean_shift_baseline(source, perturbation)
            zeros = torch.zeros_like(target)
            rows["direct_shift"].append(evaluate_population_predictions(direct, target, zeros))
            rows["mean_shift"].append(evaluate_population_predictions(shifted, target, zeros))
    return {name: _average_rows(values) for name, values in rows.items()}


def _save_artifacts(
    run_dir: Path,
    model: CounterfactualCellWorld,
    ground_truth: np.ndarray,
    history: List[Dict[str, float]],
    config: Dict[str, Any],
    split_summary: Dict[str, Dict[str, Any]],
    test_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Dict[str, float]],
    loader: DataLoader,
    device: torch.device,
) -> Path:
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    np.save(run_dir / "ground_truth_graph.npy", ground_truth)
    np.save(run_dir / "learned_graph.npy", model.gene_graph.adjacency().detach().cpu().numpy())

    predictions = _collect_predictions(model, loader, device)
    np.savez_compressed(run_dir / "test_predictions.npz", **predictions)

    report = {
        "config": config,
        "split_summary": split_summary,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return run_dir


def _collect_predictions(
    model: CounterfactualCellWorld,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    source_rows = []
    target_rows = []
    prediction_rows = []
    perturbation_rows = []
    with torch.no_grad():
        for batch in loader:
            source, target, perturbation, cell_type, time, dose = _move_batch(batch, device)
            prediction = model(source, perturbation, cell_type, time, dose)
            source_rows.append(source.detach().cpu().numpy())
            target_rows.append(target.detach().cpu().numpy())
            prediction_rows.append(prediction["mean"].detach().cpu().numpy())
            perturbation_rows.append(perturbation.detach().cpu().numpy())
    return {
        "source": np.concatenate(source_rows, axis=0),
        "target": np.concatenate(target_rows, axis=0),
        "prediction": np.concatenate(prediction_rows, axis=0),
        "perturbation": np.concatenate(perturbation_rows, axis=0),
    }


def _move_batch(
    batch: Dict[str, object],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["source"].to(device),  # type: ignore[union-attr]
        batch["target"].to(device),  # type: ignore[union-attr]
        batch["perturbation"].to(device),  # type: ignore[union-attr]
        batch["cell_type"].to(device),  # type: ignore[union-attr]
        batch["time"].to(device),  # type: ignore[union-attr]
        batch["dose"].to(device),  # type: ignore[union-attr]
    )


def _average_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
