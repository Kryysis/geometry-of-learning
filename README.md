# From Curvature to Convergence: Feature Scaling and First-Order Optimization in Logistic Regression

This repository contains an experimental study of how feature scaling and local objective curvature interact with first-order optimization in binary logistic regression. The implementation uses NumPy and compares Batch Gradient Descent, Stochastic Gradient Descent (SGD), normalized exponential-moving-average Momentum, and Adam. A separate diagonal-quadratic experiment studies convergence as a prescribed condition number increases.

## Project structure

```text
geometry-of-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── datasets.py
│   ├── logistic_regression.py
│   └── optimizers.py
├── notebooks/
│   ├── 01_Mathematical_Foundations.ipynb
│   ├── 02_Optimizer_Comparison.ipynb
│   ├── 03_Real_Dataset_Experiments.ipynb
│   └── 04_Final_Experiments.ipynb
├── data/
├── results/
│   ├── figures/
│   └── tables/
├── paper/
└── tests/
    └── test_core.py
```

## Environment setup

The notebooks record Python 3.14.8 in their kernel metadata. The repository's `requirements.txt` specifies minimum dependency versions rather than a fully locked environment; exact package versions used to generate the archived experimental outputs are not recorded. For a fresh environment, use Python 3.10 or newer and install the declared dependencies:

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To record the environment used for a future reproduction, save:

```bash
python --version
python -m pip freeze > environment-freeze.txt
```

Run the core implementation tests:

```bash
python -m unittest discover -s tests -v
```

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Run the notebooks in numerical order, preferably from a fresh kernel for each notebook. Each notebook locates the repository root from the working directory and imports the shared source modules. Restart the kernel after modifying a source module before rerunning a notebook from the top.

## Notebook guide

| Notebook | Role |
|---|---|
| `01_Mathematical_Foundations.ipynb` | Defines the logistic-regression objective, gradient, and Hessian; verifies the analytical derivatives numerically. |
| `02_Optimizer_Comparison.ipynb` | Demonstrates the four optimizers on a reproducible synthetic binary-classification problem. This is illustrative, not the final repeated benchmark. |
| `03_Real_Dataset_Experiments.ipynb` | Profiles the three datasets and performs an initial local Hessian-conditioning comparison using its own stratified 80/20 split (seed 42). Its condition numbers are a separate preliminary diagnostic and are not expected to equal the final-protocol values in Notebook 04. |
| `04_Final_Experiments.ipynb` | Runs the final repeated experiments on three real datasets, exploratory statistical comparisons, descriptive cross-dataset replication summaries, and controlled quadratic conditioning experiments. |

## Datasets

The study uses:

- **Breast Cancer Wisconsin Diagnostic** (loaded through scikit-learn)
- **Banknote Authentication** (OpenML, version 1)
- **Ionosphere** (OpenML, version 1)

The two OpenML datasets require an internet connection for their first download. Scikit-learn caches successfully downloaded OpenML datasets locally, so a later run can usually use the cache. The project does not place external dataset copies in the repository.

## Experimental design summary

### Final real-dataset experiments

The final repeated study evaluates three datasets, two feature conditions (`original` and `standardized`), four optimizers, and ten development-split seeds, resulting in **240 final evaluation fits**.

Each dataset receives one fixed, stratified 20% test holdout (seed 2026) before hyperparameter tuning. The remaining 80% forms the development pool. A 75/25 train/validation split within that development pool gives approximately 60% training, 20% validation, and 20% test observations overall.

Hyperparameter tuning and early stopping use development-pool data only. The fixed test holdout is evaluated after restoring the best validation parameters. Each final seed changes the train/validation split but reuses the same test observations for that dataset. Constant-feature removal and standardization are fitted using the training partition only.

### Local Hessian conditioning

The final-paper condition numbers are calculated at the zero parameter vector on the **tuning training partition from the development pool**. They are therefore the values produced by Notebook 04's final experimental protocol. Notebook 03 uses a different, preliminary stratified 80/20 split with seed 42; its condition numbers (approximately 2.18×10¹² and 1.04×10⁵ for Breast Cancer, 132.0 and 12.7 for Banknote, and 396.4 and 153.4 for Ionosphere, for original and standardized features respectively) are not directly interchangeable with Notebook 04's final-protocol values. Both are local diagnostics at zero initialization, not global properties of the loss surface.

### Cross-dataset replication analysis

The same final experimental protocol is applied independently to Breast Cancer Wisconsin Diagnostic, Banknote Authentication, and Ionosphere. This is replication of optimizer and feature-scaling patterns across distinct datasets, not train-on-one-dataset/test-on-another transfer.

Optimizer ranks are computed within each dataset/feature-condition combination and summarized across these groups. Rank summaries, optimizer orderings, and consistency counts are **descriptive only**; they do not establish a universally best optimizer or support universal generalization from three datasets.

### Exploratory statistical comparisons

Notebook 04 computes Friedman omnibus tests for each dataset × feature-condition × metric group (12 tests total), followed by six paired Wilcoxon signed-rank comparisons within each group (72 pairwise tests total). Holm adjustments are reported both within each six-comparison group and globally across all 72 pairwise comparisons; Friedman p-values are also adjusted globally across the 12 omnibus tests.

These analyses are **exploratory and conditional on the fixed holdout**, not confirmatory population-level inference. The ten seeds share the same test observations and use overlapping development partitions, so they are not ten independent test cohorts. In addition, the Friedman chi-square approximation is tentative with only ten blocks and four related optimizer samples. Interpret descriptive effects and their consistency as primary; do not use thresholded p-values to claim a generally superior optimizer.

### Controlled quadratic experiment

The objective is

$$
f(\\theta) = \\frac{1}{2}\\theta^{T}
\\begin{bmatrix}
1 & 0 \\\
0 & \\kappa
\\end{bmatrix}\\theta,
\\quad
\\kappa \\in \\{1, 10, 100, 1000, 10000\\}.
$$

Learning rates are tuned independently for each optimizer and condition number. Evaluation uses ten seeds per optimizer/condition-number pair, producing **200 runs**. Convergence requires the loss to remain at or below `1e-8` for ten consecutive updates, with a maximum of 5,000 updates. A stable run that does not meet the threshold is reported as non-converged, not unstable.

The Momentum implementation uses an exponential moving average of gradients:

$$
v_t = \\beta v_{t-1} + (1-\\beta)g_t,
\\qquad
\\theta_{t+1} = \\theta_t - \\eta v_t.
$$

This is not the unnormalized heavy-ball convention. The quadratic SGD experiment uses unbiased stochastic coordinate sampling, not observation mini-batches. Equal update counts do not imply equal arithmetic cost, and the diagonal Hessian is aligned with the coordinate axes.

## Outputs

- Generated CSV tables are saved under `results/tables/`; generated plots are saved under `results/figures/`.
- Outputs include repeated optimizer experiments, selected hyperparameters, Hessian-conditioning diagnostics, exploratory statistical comparisons, and descriptive cross-dataset replication summaries.
- The cross-dataset analysis includes descriptive standardization effects, optimizer ranks, and Hessian conditioning across the three datasets.
- Run the notebooks to regenerate outputs using the current experimental code. Existing output files may be overwritten by the corresponding cells.

## Interpretation cautions

- Hessian condition numbers are local diagnostics evaluated at the zero parameter vector; the final-paper values use Notebook 04's tuning training partition, not Notebook 03's separate preliminary split.
- Notebook 02 is illustrative and is not a controlled equal-compute comparison.
- Learning rates are selected from finite candidate grids. A boundary selection means the best observed candidate may lie outside the tested range.
- In **94 of the 240 saved final fits**, the best validation checkpoint occurred at the 300-epoch cap. This count is from the archived `final_results` produced in Notebook 04; it indicates that the training budget may have constrained some fits.
- The ten evaluation seeds reuse one fixed test holdout per dataset and have overlapping development partitions. Standard deviations describe fitted-model variation conditional on the holdout, not variation across independent test cohorts.
- Friedman and Wilcoxon/Holm outputs are exploratory diagnostics under this dependent repeated-split design; corrected p-values should not be interpreted as confirmatory evidence.
- Optimizer ranks and rank summaries are descriptive, not inferential rankings or claims of a universal best method.
- Equal optimizer-update budgets in the quadratic experiment do not imply equal arithmetic cost.
- The real-dataset study is not cross-dataset transfer learning or train-on-one/test-on-another evaluation.
