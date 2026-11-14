# The Geometry of Learning

**Investigating the Effects of Loss Landscape Conditioning on Optimization Algorithms in Machine Learning**

This repository contains a reproducible experimental study of how objective-function curvature and feature scaling interact with first-order optimizers. The implementation uses logistic regression written with NumPy and compares Batch Gradient Descent, Stochastic Gradient Descent (SGD), Momentum, and Adam.

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

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
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
| `02_Optimizer_Comparison.ipynb` | Demonstrates the four optimizers on a reproducible synthetic binary-classification problem. This is illustrative, not the final statistical benchmark. |
| `03_Real_Dataset_Experiments.ipynb` | Profiles the three datasets and compares local Hessian conditioning before and after standardization. |
| `04_Final_Experiments.ipynb` | Conducts repeated experiments on three real datasets, cross-dataset replication analysis, and controlled quadratic conditioning experiments. Includes ablation studies, statistical tests, and generated figures. |

## Datasets

The study uses:

- **Breast Cancer Wisconsin Diagnostic** (available through scikit-learn's dataset loader)
- **Banknote Authentication** (OpenML, version 1)
- **Ionosphere** (OpenML, version 1)

The two OpenML datasets require an internet connection for their first download. Scikit-learn caches successfully downloaded OpenML datasets locally, so a later run can usually use the cache. The project does not place external dataset copies in the repository.

## Experimental Design Summary

### Real-Dataset Experiments

The final repeated study evaluates three datasets, two feature conditions (`original` and `standardized`), four optimizers, and ten development-split seeds, resulting in **240 final evaluation fits**.

Each dataset receives one fixed, stratified 20% test holdout (seed 2026) before hyperparameter tuning. The remaining 80% forms the development pool. A 75/25 train/validation split within that development pool gives approximately 60% training, 20% validation, and 20% test observations overall.

Hyperparameter tuning and early stopping use development-pool data only. The fixed test holdout is evaluated after restoring the best validation parameters. Each final seed changes the train/validation split but reuses the same test observations for that dataset. Constant-feature removal and standardization are fitted using the training partition only.

### Cross-Dataset Replication Validation

The same experimental protocol is applied independently to Breast Cancer Wisconsin Diagnostic, Banknote Authentication, and Ionosphere.

Cross-dataset validation is treated as replication of optimizer and feature-scaling patterns across datasets with different feature spaces, not as train-on-one-dataset/test-on-another transfer. The analysis compares standardization effects, optimizer rankings, and local Hessian condition numbers across the three datasets.

Consistency counts and rank summaries are descriptive. Three datasets do not support claims of universal generalization.

### Controlled Quadratic Experiment

The objective is

$$
f(\theta) = \frac{1}{2}\theta^{T}
\begin{bmatrix}
1 & 0 \\
0 & \kappa
\end{bmatrix}\theta,
\quad
\kappa \in \{1, 10, 100, 1000, 10000\}.
$$

Learning rates are tuned independently for each optimizer and condition number. Evaluation uses ten seeds per optimizer/condition-number pair, producing **200 runs**.

Convergence requires the loss to remain at or below `1e-8` for ten consecutive updates, with a maximum of 5,000 updates. A stable run that does not meet the threshold is reported as non-converged, not unstable.

The Momentum implementation uses an exponential moving average of gradients:

$$
v_t = \beta v_{t-1} + (1-\beta)g_t,
\qquad
\theta_{t+1} = \theta_t - \eta v_t.
$$

This is not the unnormalized heavy-ball convention. The quadratic SGD experiment uses unbiased stochastic coordinate sampling, not observation mini-batches.

## Outputs

- Generated CSV tables are saved under `results/tables/`; generated plots are saved under `results/figures/`.
- The outputs include the repeated optimizer experiments, hyperparameter selections, Hessian conditioning diagnostics, statistical comparisons, and cross-dataset replication summaries.
- The cross-dataset analysis includes comparisons of standardization effects, optimizer rankings, and Hessian conditioning across the three datasets.
- The notebooks should be executed to regenerate the results using the current experimental code. Existing output files may be overwritten when the corresponding cells are executed.

## Interpretation cautions

- Hessian condition numbers in Notebook 03 are local diagnostics evaluated at the zero parameter vector.
- Notebook 02 is an illustrative optimizer demonstration and is not a controlled equal-compute comparison.
- Learning rates are tuned within finite candidate grids. A selected boundary rate means the best observed candidate may lie outside the tested range.
- The ten evaluation seeds reuse the same fixed test holdout within each dataset. They represent repeated training and development-split variation, not independent test cohorts. Statistical comparisons across seeds should therefore be interpreted conditionally on the fixed holdout and should not be treated as population-level inference from ten independent test samples.
- Equal optimizer-update budgets in the quadratic experiment do not imply equal arithmetic cost.
- Ten evaluation seeds provide repeated evidence but still yield limited statistical power. Discuss descriptive effect sizes alongside corrected p-values.
- The real-dataset study evaluates the optimizer patterns independently across datasets; it is not a cross-dataset train-on-one/test-on-another transfer experiment.
