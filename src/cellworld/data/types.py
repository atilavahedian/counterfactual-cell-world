from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CellCondition:
    """A paired source and target cell population under one intervention."""

    name: str
    source: np.ndarray
    target: np.ndarray
    perturbation: np.ndarray
    cell_type: int
    time: float
    dose: float
    genes: Tuple[int, ...]
    split: str

    @property
    def gene_count(self) -> int:
        return int(self.source.shape[-1])

    @property
    def cell_count(self) -> int:
        return int(self.source.shape[0])


class CellPopulationDataset(Dataset):
    """Torch dataset over paired cell-population conditions."""

    def __init__(self, conditions: Sequence[CellCondition]):
        if not conditions:
            raise ValueError("CellPopulationDataset needs at least one condition.")
        gene_counts = {condition.gene_count for condition in conditions}
        cell_counts = {condition.cell_count for condition in conditions}
        if len(gene_counts) != 1:
            raise ValueError("All conditions must share the same gene count.")
        if len(cell_counts) != 1:
            raise ValueError("All conditions must share the same cell count.")
        self.conditions = list(conditions)
        self.gene_count = next(iter(gene_counts))
        self.cell_count = next(iter(cell_counts))

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, index: int) -> Dict[str, object]:
        condition = self.conditions[index]
        return {
            "name": condition.name,
            "source": torch.tensor(condition.source, dtype=torch.float32),
            "target": torch.tensor(condition.target, dtype=torch.float32),
            "perturbation": torch.tensor(condition.perturbation, dtype=torch.float32),
            "cell_type": torch.tensor(condition.cell_type, dtype=torch.long),
            "time": torch.tensor(condition.time, dtype=torch.float32),
            "dose": torch.tensor(condition.dose, dtype=torch.float32),
            "genes": condition.genes,
            "split": condition.split,
        }


def collate_conditions(items: Iterable[Dict[str, object]]) -> Dict[str, object]:
    batch = list(items)
    tensor_keys = ["source", "target", "perturbation", "cell_type", "time", "dose"]
    collated: Dict[str, object] = {}
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch])  # type: ignore[list-item]
    collated["name"] = [item["name"] for item in batch]
    collated["genes"] = [item["genes"] for item in batch]
    collated["split"] = [item["split"] for item in batch]
    return collated


def summarize_conditions(conditions: Sequence[CellCondition]) -> Dict[str, object]:
    """Return a compact split summary for logs and model cards."""

    split_counts: Dict[str, int] = {}
    combo_orders: Dict[int, int] = {}
    for condition in conditions:
        split_counts[condition.split] = split_counts.get(condition.split, 0) + 1
        order = len(condition.genes)
        combo_orders[order] = combo_orders.get(order, 0) + 1
    return {
        "conditions": len(conditions),
        "splits": split_counts,
        "combo_orders": combo_orders,
        "genes": conditions[0].gene_count if conditions else 0,
        "cells_per_condition": conditions[0].cell_count if conditions else 0,
    }

