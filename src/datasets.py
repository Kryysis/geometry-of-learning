"""Dataset loading utilities for the real-data experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer


def _validate_binary_dataset(name: str, X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)

    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError(f"{name}: feature and target dimensions do not match.")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name}: features contain missing or non-finite values.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{name}: expected exactly two classes encoded as 0 and 1.")
    return X, y


def load_binary_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load Breast Cancer, Banknote Authentication, and Ionosphere.

    The Breast Cancer data are bundled with scikit-learn. The Banknote and
    Ionosphere datasets are retrieved from OpenML and cached by scikit-learn
    after the first successful download. ``pandas.factorize`` preserves the
    first-seen class ordering used in the original experiments.
    """
    breast = load_breast_cancer()
    datasets = {
        "Breast Cancer": _validate_binary_dataset(
            "Breast Cancer", breast.data, breast.target
        )
    }

    try:
        banknote = fetch_openml(
            name="banknote-authentication", version=1, as_frame=False
        )
        ionosphere = fetch_openml(
            name="ionosphere", version=1, as_frame=False
        )
    except Exception as exc:
        raise RuntimeError(
            "The Banknote Authentication and Ionosphere datasets could not be "
            "retrieved from OpenML. Check the internet connection; after the "
            "first successful download, scikit-learn normally reuses its local cache."
        ) from exc

    banknote_y = pd.factorize(np.asarray(banknote.target).reshape(-1))[0]
    ionosphere_y = pd.factorize(np.asarray(ionosphere.target).reshape(-1))[0]

    datasets["Banknote"] = _validate_binary_dataset(
        "Banknote", banknote.data, banknote_y
    )
    datasets["Ionosphere"] = _validate_binary_dataset(
        "Ionosphere", ionosphere.data, ionosphere_y
    )
    return datasets
