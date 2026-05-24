from __future__ import annotations

from cellworld.data.synthetic import SyntheticSplitConfig, SyntheticWorldConfig, build_synthetic_splits


def test_synthetic_world_has_heldout_combos() -> None:
    splits, ground_truth = build_synthetic_splits(
        SyntheticWorldConfig(genes=16, programs=4, cell_types=2, perturbation_genes=7, cells_per_condition=12),
        SyntheticSplitConfig(train_conditions=10, val_conditions=4, test_conditions=4),
        seed=11,
    )

    train_genes = {condition.genes for condition in splits["train"]}
    test_genes = {condition.genes for condition in splits["test"]}

    assert splits["train"][0].source.shape == (12, 16)
    assert ground_truth.adjacency.shape == (16, 16)
    assert train_genes.isdisjoint(test_genes)

