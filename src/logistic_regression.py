"""Numerically stable logistic-regression primitives and model implementation.

The objective is mean binary cross-entropy plus an optional L2 penalty:

    J(theta) = mean_i BCE(y_i, sigmoid(x_i @ theta))
               + (l2 / 2) * ||theta[1:]||_2^2.

The first parameter is treated as the intercept and is not regularized.
Feature matrices passed to the loss, gradient, and Hessian functions must
already contain an intercept column when an intercept is desired.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


def _validate_l2(l2: float) -> float:
    l2 = float(l2)
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be a finite, non-negative number.")
    return l2


def _validate_feature_matrix(X: Any) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain at least one row and one column.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    return X


def _validate_targets(y: Any, n_samples: int) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.shape[0] != n_samples:
        raise ValueError("X and y contain different numbers of samples.")
    if not np.all(np.isfinite(y)) or not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y must contain only finite binary labels (0 and 1).")
    return y


def _validate_parameters(theta: Any, n_parameters: int) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.shape[0] != n_parameters:
        raise ValueError("theta length must match the number of columns in X.")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must contain only finite values.")
    return theta


def sigmoid(z: Any) -> np.ndarray:
    """Apply the logistic sigmoid with clipping for numerical stability."""
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def add_intercept(X: Any) -> np.ndarray:
    """Prepend a column of ones to a two-dimensional feature matrix."""
    X = _validate_feature_matrix(X)
    return np.column_stack((np.ones(X.shape[0], dtype=float), X))


def binary_cross_entropy(X: Any, y: Any, theta: Any, l2: float = 0.0) -> float:
    """Return mean binary cross-entropy plus an L2 penalty on non-bias weights.

    Uses ``logaddexp`` rather than taking logarithms of predicted
    probabilities, which is more stable for large-magnitude logits.
    """
    X = _validate_feature_matrix(X)
    y = _validate_targets(y, X.shape[0])
    theta = _validate_parameters(theta, X.shape[1])
    l2 = _validate_l2(l2)

    logits = X @ theta
    data_loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
    regularization = 0.5 * l2 * np.sum(theta[1:] ** 2)
    return float(data_loss + regularization)


def compute_gradient(X: Any, y: Any, theta: Any, l2: float = 0.0) -> np.ndarray:
    """Return the analytical gradient of the regularized objective."""
    X = _validate_feature_matrix(X)
    y = _validate_targets(y, X.shape[0])
    theta = _validate_parameters(theta, X.shape[1])
    l2 = _validate_l2(l2)

    probabilities = sigmoid(X @ theta)
    gradient = (X.T @ (probabilities - y)) / X.shape[0]
    gradient[1:] += l2 * theta[1:]  # Do not regularize the intercept.
    return gradient


def compute_hessian(X: Any, theta: Any, l2: float = 0.0) -> np.ndarray:
    """Return the Hessian of binary cross-entropy plus the L2 penalty.

    The Hessian is ``X.T @ diag(p * (1-p)) @ X / m`` plus the L2
    diagonal penalty on all parameters except the first (intercept) term.
    The diagonal matrix is applied by row scaling to avoid materializing it.
    """
    X = _validate_feature_matrix(X)
    theta = _validate_parameters(theta, X.shape[1])
    l2 = _validate_l2(l2)

    probabilities = sigmoid(X @ theta)
    curvature = probabilities * (1.0 - probabilities)
    weighted_X = X * curvature[:, np.newaxis]
    hessian = (X.T @ weighted_X) / X.shape[0]

    regularization_matrix = np.eye(X.shape[1], dtype=float)
    regularization_matrix[0, 0] = 0.0
    hessian += l2 * regularization_matrix
    return hessian


class LogisticRegression:
    """Binary logistic regression trained with a pluggable optimizer."""

    def __init__(self, l2: float = 0.0) -> None:
        self.l2 = _validate_l2(l2)
        self.theta: np.ndarray | None = None
        self.loss_history: list[float] = []
        self.n_features_in_: int | None = None

    def fit(
        self,
        X: Any,
        y: Any,
        optimizer: Any,
        epochs: int = 100,
        batch_size: int | None = None,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> "LogisticRegression":
        """Fit the model using mini-batch updates.

        ``batch_size=None`` means full-batch gradient descent. A batch size of
        one gives stochastic gradient descent. Supplying ``random_state`` makes
        the per-epoch shuffling reproducible and is recommended for experiments.
        """
        X = _validate_feature_matrix(X)
        y = _validate_targets(y, X.shape[0])

        if not isinstance(epochs, Integral) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if batch_size is None:
            batch_size = X.shape[0]
        if not isinstance(batch_size, Integral) or not 1 <= batch_size <= X.shape[0]:
            raise ValueError("batch_size must be an integer between 1 and n_samples.")
        if not hasattr(optimizer, "reset") or not hasattr(optimizer, "step"):
            raise TypeError("optimizer must provide reset() and step() methods.")

        X_bias = add_intercept(X)
        n_samples, n_parameters = X_bias.shape
        self.n_features_in_ = X.shape[1]
        self.theta = np.zeros(n_parameters, dtype=float)
        self.loss_history = []
        optimizer.reset(n_parameters)

        rng = np.random.default_rng(random_state) if random_state is not None else None

        for _ in range(int(epochs)):
            if shuffle:
                indices = (
                    rng.permutation(n_samples)
                    if rng is not None
                    else np.random.permutation(n_samples)
                )
            else:
                indices = np.arange(n_samples)

            X_epoch = X_bias[indices]
            y_epoch = y[indices]

            for start in range(0, n_samples, int(batch_size)):
                end = min(start + int(batch_size), n_samples)
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]

                gradient = compute_gradient(
                    X_batch, y_batch, self.theta, l2=self.l2
                )
                updated_theta = np.asarray(
                    optimizer.step(self.theta, gradient), dtype=float
                ).reshape(-1)

                if updated_theta.shape != self.theta.shape:
                    raise ValueError("optimizer.step() returned an invalid parameter shape.")
                self.theta = updated_theta

            epoch_loss = binary_cross_entropy(
                X_bias, y, self.theta, l2=self.l2
            )
            self.loss_history.append(epoch_loss)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return predicted probabilities for class 1."""
        if self.theta is None or self.n_features_in_ is None:
            raise ValueError("The model must be fitted before prediction.")
        X = _validate_feature_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data.")
        return sigmoid(add_intercept(X) @ self.theta)

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using the supplied probability threshold."""
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        return (self.predict_proba(X) >= threshold).astype(int)
