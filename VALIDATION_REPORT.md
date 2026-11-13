# Validation report

## Checks completed

- All four notebooks pass `nbformat` validation.
- All code cells in all four notebooks pass Python syntax parsing.
- All four notebooks have clean execution metadata: no stale cell outputs or execution counters are embedded.
- Five unit tests pass for the shared implementation:
  - analytical gradient versus finite differences;
  - analytical Hessian versus finite differences;
  - model fitting and prediction with the implemented optimizers;
  - exclusion of the intercept from L2 regularization;
  - deterministic SGD-style shuffling with a fixed random state.
- Notebook 01 and Notebook 02 execute from fresh kernels in the available environment.
- Notebook 03 completes in a synthetic-data workflow test.
- Notebook 04's complete execution flow, including the real-data analysis and reduced-size quadratic test workflow, completes in a synthetic-data QA run.

## Not verified in this environment

The Banknote Authentication and Ionosphere datasets are fetched from OpenML. Network access to OpenML was unavailable during this cleanup, so Notebook 03 and the full Notebook 04 were not executed against the actual OpenML datasets here. Their data-loading path is implemented in `src/datasets.py`; the actual-data runs should be performed from a machine with OpenML access.

The synthetic-data QA is a code-path test only. It is not a substitute for the project's real-data results.

## Important reproducibility change

Hyperparameter tuning now resets the shuffle sequence for each learning-rate candidate by passing a fixed `random_state`. This makes candidates more directly comparable. Because this can change the selected learning rates, rerun Notebook 04 from the first cell before using its generated result tables in the paper.

## Statistical-reporting correction

The quadratic summary now reports updates-to-convergence only among runs that actually met the convergence criterion. Non-converged runs are reported separately, alongside total updates used and final loss. This avoids treating the 5,000-update cap as if it were a successful convergence time.
