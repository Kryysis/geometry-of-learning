# Validation Report

**Project:** The Geometry of Learning\
**Scope:** Code, notebook, experimental-output, and reproducibility
checks

## 1. Validation summary

The repository has been updated to include the revised real-dataset
experimental protocol and cross-dataset replication analysis.

The following checks and results are recorded for this project:

-   The four project notebooks were previously checked for valid
    notebook structure and Python syntax.
-   Five unit tests were previously reported as passing for the shared
    implementation.
-   The revised Notebook 04 was executed successfully by the project
    author against the three real datasets.
-   The generated real-dataset experiment tables and cross-dataset
    summary files were inspected for consistency.
-   The fixed test-holdout audit and the expected experiment coverage
    were checked.

This report distinguishes checks performed during project development
from the author's execution of the latest full experiment. It does not
claim that every notebook or experiment was independently rerun as part
of this repository update.

## 2. Shared implementation tests

Five unit tests were previously reported as passing:

1.  Analytical gradient compared with finite differences.
2.  Analytical Hessian compared with finite differences.
3.  Model fitting and prediction using the implemented optimizers.
4.  Exclusion of the intercept from L2 regularization.
5.  Deterministic SGD-style shuffling with a fixed random state.

These tests cover important mathematical and implementation properties
of the shared logistic-regression and optimization code.

## 3. Real-dataset experiment protocol

The final repeated real-dataset study uses:

-   **Datasets:** Breast Cancer Wisconsin Diagnostic, Banknote
    Authentication, and Ionosphere.
-   **Feature conditions:** original and standardized.
-   **Optimizers:** Batch Gradient Descent, Stochastic Gradient Descent
    (SGD), Momentum, and Adam.
-   **Evaluation seeds:** ten development-split seeds.
-   **Total:** 3 datasets × 2 feature conditions × 4 optimizers × 10
    seeds = **240 model runs**.

For each dataset, one stratified 20% test holdout is set aside using
random seed 2026 before hyperparameter tuning. The remaining 80% is the
development pool. A 75/25 train/validation split within that pool gives
approximately 60% training, 20% validation, and 20% test observations
overall.

Hyperparameter selection and early stopping use development-pool data
only. The fixed test holdout is evaluated after the best validation
parameters are restored. Across the ten evaluation seeds, the
train/validation split changes while the same test observations are
reused for that dataset. Feature filtering and standardization are
fitted on the training partition only.

The inspected fixed-holdout audit records the following test-set sizes:

  Dataset                                Fixed test observations
  ------------------------------------ -------------------------
  Breast Cancer Wisconsin Diagnostic                         114
  Banknote Authentication                                    275
  Ionosphere                                                  71

## 4. Cross-dataset replication

Cross-dataset validation is implemented as replication of optimizer and
feature-scaling patterns across datasets with different feature spaces.
It is **not** a train-on-one-dataset/test-on-another transfer-learning
experiment.

The generated analysis includes comparisons of:

-   Standardization effects on accuracy and log loss.
-   Optimizer rankings across datasets.
-   Local Hessian condition numbers before and after standardization.
-   Optimizer consistency and rank summaries.
-   Per-seed scaling effects and fixed-test-holdout audit information.

The three datasets provide a useful comparison across distinct
classification problems, but they are not sufficient to establish
universal generalization claims.

## 5. Output inspection

The updated experiment outputs were inspected and the following checks
were reported:

-   The repeated-experiment results contain 240 model-run records.
-   The experiment coverage spans all three datasets, both feature
    conditions, all four optimizers, and ten seeds.
-   No duplicate experiment keys or missing numerical values were found
    in the inspected result tables.
-   Per-seed scaling summaries were consistent with the corresponding
    records in the repeated-experiment results.
-   Cross-dataset summary tables and figures were generated.

Relevant generated files are stored under `results/tables/` and
`results/figures/`. The notebooks are the source of these outputs;
rerunning the relevant notebook cells may overwrite generated files.

## 6. Statistical interpretation and limitations

The ten evaluation seeds do **not** represent ten independent test
cohorts. They reuse the same fixed test holdout within each dataset
while varying the development split and training process.

Consequently:

-   Comparisons across seeds should be interpreted conditionally on the
    fixed test holdout.
-   Statistical tests across these repeated fits are exploratory and
    should not be presented as population-level inference from ten
    independent test samples.
-   Cross-dataset consistency counts and optimizer rank summaries are
    descriptive.
-   Results from three datasets do not establish that an optimizer or
    preprocessing choice is universally best.

Hessian condition numbers are local diagnostics evaluated at the
specified parameter point; they should not be interpreted as a complete
description of the entire nonlinear loss landscape. Hyperparameter
conclusions are also limited by the candidate learning-rate grids and
the chosen training budgets.

## 7. Reproducibility notes

-   The Banknote Authentication and Ionosphere datasets are retrieved
    from OpenML; internet access may be required on the first run.
-   Breast Cancer Wisconsin Diagnostic is available through
    scikit-learn.
-   Use the dependencies listed in `requirements.txt`.
-   Run the notebooks in numerical order from fresh kernels when
    reproducing the complete workflow.
-   Execute Notebook 04 to regenerate the final repeated-experiment
    outputs and cross-dataset summaries.

## 8. Validation scope

This report records the checks and results available during the project
update. It is not a claim of independent replication by a third party,
nor does it certify the results on every machine or software
environment. For a new environment, rerun the unit tests and notebooks
and retain the resulting logs and outputs as the local validation
record.
