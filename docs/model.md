# Model Note

The model predicts a target cell population, not a single averaged expression vector.

For each condition we observe:

- source cells `X`
- perturbation vector `p`
- cell-type id `c`
- dose `d`
- time `t`
- target cells `Y`

The model learns:

1. a signed gene interaction matrix `A`
2. a context vector from population statistics, intervention, cell type, dose, and time
3. a latent transition over source cells
4. a Gaussian decoder for target expression

The learned graph is not treated as biological truth. It is a modeling object with sparsity and acyclicity pressure. The graph is useful only if it improves held-out perturbation prediction and shows non-random agreement with the simulator graph.

## Objective

The training objective combines pointwise likelihood and population matching:

```text
L = NLL(Y | mu, sigma)
  + lambda_mmd MMD(mu, Y)
  + lambda_l1 |A|
  + lambda_dag h(A)
```

The likelihood term checks cell-level reconstruction. The MMD term checks whether the predicted cell population has the right distribution. The graph penalties keep the learned interaction map from becoming an unstructured dense shortcut.

## Why The Synthetic Split Matters

The test split hides intervention combinations. If the model only memorizes observed perturbation labels, test MMD should stay weak. If it learns reusable intervention structure, it should move target distributions closer than direct-shift and mean-shift baselines.

## Known Limits

This version does not claim wet-lab validity. The simulator is a controlled stress test for the algorithm. The next version should add real Perturb-seq splits, dose/time consistency checks, and a stronger uncertainty benchmark.

