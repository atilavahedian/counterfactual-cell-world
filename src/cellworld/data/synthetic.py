from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

from cellworld.data.types import CellCondition


@dataclass(frozen=True)
class SyntheticWorldConfig:
    genes: int = 32
    programs: int = 5
    cell_types: int = 3
    perturbation_genes: int = 10
    cells_per_condition: int = 64
    noise_scale: float = 0.06
    nonlinear_scale: float = 0.75


@dataclass(frozen=True)
class SyntheticSplitConfig:
    train_conditions: int = 72
    val_conditions: int = 18
    test_conditions: int = 18
    max_combo_order: int = 2
    heldout_combo_fraction: float = 0.35


@dataclass(frozen=True)
class SyntheticGroundTruth:
    adjacency: np.ndarray
    program_loadings: np.ndarray
    cell_type_offsets: np.ndarray
    perturbable_genes: np.ndarray


def build_synthetic_splits(
    world_config: SyntheticWorldConfig,
    split_config: SyntheticSplitConfig,
    seed: int,
) -> Tuple[Dict[str, List[CellCondition]], SyntheticGroundTruth]:
    rng = np.random.default_rng(seed)
    ground_truth = _build_ground_truth(world_config, rng)

    candidate_gene_sets = _candidate_gene_sets(
        ground_truth.perturbable_genes,
        max_combo_order=split_config.max_combo_order,
    )
    rng.shuffle(candidate_gene_sets)

    heldout_count = max(1, int(len(candidate_gene_sets) * split_config.heldout_combo_fraction))
    heldout_gene_sets = set(candidate_gene_sets[:heldout_count])
    train_gene_sets = candidate_gene_sets[heldout_count:]

    splits = {
        "train": _sample_conditions(
            train_gene_sets,
            split="train",
            count=split_config.train_conditions,
            world_config=world_config,
            ground_truth=ground_truth,
            rng=rng,
        ),
        "val": _sample_conditions(
            train_gene_sets,
            split="val",
            count=split_config.val_conditions,
            world_config=world_config,
            ground_truth=ground_truth,
            rng=rng,
        ),
        "test": _sample_conditions(
            tuple(heldout_gene_sets),
            split="test",
            count=split_config.test_conditions,
            world_config=world_config,
            ground_truth=ground_truth,
            rng=rng,
        ),
    }
    return splits, ground_truth


def _build_ground_truth(config: SyntheticWorldConfig, rng: np.random.Generator) -> SyntheticGroundTruth:
    loadings = rng.normal(0.0, 0.55, size=(config.genes, config.programs))
    mask = rng.uniform(size=loadings.shape) < 0.42
    loadings = loadings * mask

    program_coupling = rng.normal(0.0, 0.35, size=(config.programs, config.programs))
    program_coupling = program_coupling * (rng.uniform(size=program_coupling.shape) < 0.55)
    adjacency = loadings @ program_coupling @ loadings.T
    adjacency += rng.normal(0.0, 0.04, size=adjacency.shape)
    adjacency *= rng.uniform(size=adjacency.shape) < 0.22
    np.fill_diagonal(adjacency, 0.0)

    spectral_radius = max(float(np.max(np.abs(np.linalg.eigvals(adjacency)))), 1e-6)
    adjacency = adjacency / spectral_radius * 0.82

    cell_type_offsets = rng.normal(0.0, 0.35, size=(config.cell_types, config.genes))
    perturbable = np.argsort(np.abs(loadings).sum(axis=1))[::-1][: config.perturbation_genes]

    return SyntheticGroundTruth(
        adjacency=adjacency.astype(np.float32),
        program_loadings=loadings.astype(np.float32),
        cell_type_offsets=cell_type_offsets.astype(np.float32),
        perturbable_genes=perturbable.astype(np.int64),
    )


def _candidate_gene_sets(genes: Sequence[int], max_combo_order: int) -> List[Tuple[int, ...]]:
    candidates: List[Tuple[int, ...]] = []
    for order in range(1, max_combo_order + 1):
        for gene_set in combinations([int(gene) for gene in genes], order):
            candidates.append(tuple(sorted(gene_set)))
    return candidates


def _sample_conditions(
    gene_sets: Sequence[Tuple[int, ...]],
    split: str,
    count: int,
    world_config: SyntheticWorldConfig,
    ground_truth: SyntheticGroundTruth,
    rng: np.random.Generator,
) -> List[CellCondition]:
    if not gene_sets:
        raise ValueError("Need at least one intervention gene set.")
    conditions: List[CellCondition] = []
    for index in range(count):
        genes = tuple(gene_sets[int(rng.integers(0, len(gene_sets)))])
        cell_type = int(rng.integers(0, world_config.cell_types))
        dose = float(rng.uniform(0.55, 1.45))
        time = float(rng.uniform(0.5, 2.0))
        source = _sample_source_population(world_config, ground_truth, cell_type, rng)
        perturbation = np.zeros(world_config.genes, dtype=np.float32)
        direction = -1.0 if rng.uniform() < 0.78 else 1.0
        for gene in genes:
            perturbation[gene] = direction * dose
        target = _evolve_population(
            source,
            perturbation,
            cell_type,
            time,
            world_config,
            ground_truth,
            rng,
        )
        gene_label = "-".join(str(gene) for gene in genes)
        conditions.append(
            CellCondition(
                name=f"{split}_{index:03d}_ct{cell_type}_g{gene_label}",
                source=source.astype(np.float32),
                target=target.astype(np.float32),
                perturbation=perturbation.astype(np.float32),
                cell_type=cell_type,
                time=time,
                dose=dose,
                genes=genes,
                split=split,
            )
        )
    return conditions


def _sample_source_population(
    config: SyntheticWorldConfig,
    ground_truth: SyntheticGroundTruth,
    cell_type: int,
    rng: np.random.Generator,
) -> np.ndarray:
    program_state = rng.normal(0.0, 1.0, size=(config.cells_per_condition, config.programs))
    expression = program_state @ ground_truth.program_loadings.T
    expression += ground_truth.cell_type_offsets[cell_type]
    expression += rng.normal(0.0, config.noise_scale, size=expression.shape)
    return np.tanh(expression).astype(np.float32)


def _evolve_population(
    source: np.ndarray,
    perturbation: np.ndarray,
    cell_type: int,
    time: float,
    config: SyntheticWorldConfig,
    ground_truth: SyntheticGroundTruth,
    rng: np.random.Generator,
) -> np.ndarray:
    state = source.copy()
    steps = max(3, int(round(8 * time)))
    dt = time / float(steps)
    direct = perturbation.reshape(1, -1)
    propagated = direct @ ground_truth.adjacency.T
    type_drive = ground_truth.cell_type_offsets[cell_type].reshape(1, -1) * 0.12

    for _ in range(steps):
        regulatory = state @ ground_truth.adjacency.T
        nonlinear = np.tanh(regulatory + type_drive + 0.25 * direct + 0.95 * propagated)
        drift = -0.32 * state + config.nonlinear_scale * nonlinear
        state = state + dt * drift

    state += rng.normal(0.0, config.noise_scale * np.sqrt(max(time, 0.1)), size=state.shape)
    return np.tanh(state).astype(np.float32)
