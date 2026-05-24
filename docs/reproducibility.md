# Reproducibility

The checked run was produced from `configs/synthetic_small.yaml`.

```bash
uv sync --extra dev
uv run python scripts/run_synthetic.py --config configs/synthetic_small.yaml
uv run python scripts/make_figures.py --run-dir runs/synthetic_small
uv run pytest
```

The committed result files live in `results/synthetic_small`. The scratch run directory is ignored so repeated experiments do not pollute Git history.

## Held-Out Split

The synthetic split hides intervention combinations from training:

- train: 72 conditions
- validation: 18 conditions
- test: 18 conditions
- genes: 32
- cells per condition: 64

## Checked Result

On the held-out split:

| method | MSE | MAE | MMD | energy distance | mean gene correlation |
|---|---:|---:|---:|---:|---:|
| model | 0.0386 | 0.1422 | 0.1021 | 0.3241 | 0.8714 |
| direct shift | 0.1090 | 0.2356 | 0.1711 | 0.5967 | 0.7218 |
| mean shift | 0.1428 | 0.2692 | 0.1905 | 0.6980 | 0.7271 |

The numbers are not a claim about real cells. They show that the implementation can learn reusable perturbation structure in a controlled world where the test interventions are held out.

