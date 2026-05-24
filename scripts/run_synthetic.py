from __future__ import annotations

import argparse

from cellworld.training import train_synthetic
from cellworld.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic_small.yaml")
    parser.add_argument("--output-root", default="runs")
    args = parser.parse_args()
    run_dir = train_synthetic(load_yaml(args.config), output_root=args.output_root)
    print(f"wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()

