# Dataset Notes

The first benchmark in this repo is synthetic on purpose. For counterfactual perturbation modeling, a synthetic world gives us something real single-cell datasets usually do not give us: known regulatory edges, known intervention targets, and a clean held-out split for unseen perturbation combinations.

The path toward real data is separate:

- Use public Perturb-seq or CRISPR screen datasets when the task is perturbation response.
- Use CELLxGENE metadata to discover single-cell atlases, but do not mix atlas data into perturbation training unless the experimental design is compatible.
- Treat batch, donor, tissue, assay, and cell-type labels as possible confounders, not harmless metadata.

Live checks made while setting up this repo:

- CELLxGENE Discover returned 377 public collections from its curation API on May 24, 2026.
- PubMed returned current 2026 single-cell perturbation and generative-cell-model papers, including work on multi-omics perturbation prediction and large single-cell foundation models.

The next serious step is not a bigger model. It is a real-data benchmark with a frozen train/validation/test split that hides combinatorial interventions and checks whether the model preserves distribution shape, not only mean expression.

