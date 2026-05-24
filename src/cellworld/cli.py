from __future__ import annotations

import argparse

from cellworld.training import train_synthetic
from cellworld.utils.config import load_yaml


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train Counterfactual Cell World.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--output-root", default="runs", help="Directory for run artifacts.")
    args = parser.parse_args()

    run_dir = train_synthetic(load_yaml(args.config), output_root=args.output_root)
    print(f"wrote run artifacts to {run_dir}")

