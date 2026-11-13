"""First-order optimization algorithms used in the project."""

from __future__ import annotations

import numpy as np


def _validate_learning_rate(learning_rate: float) -> float:
    learning_rate = float(learning_rate)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive number.")
    return learning_rate


def _validate_vector(theta, gradient):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    gradient = np.asarray(gradient, dtype=float).reshape(-1)
    if theta.shape != gradient.shape:
        raise ValueError("theta and gradient must have the same shape.")
    return theta, gradient


class GradientDescent:
    """Gradient descent; batch size is determined by the model's data loader."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = _validate_learning_rate(learning_rate)

    def reset(self, n_parameters: int) -> None:
        """Reset optimizer state (there is no state for plain GD)."""
        if n_parameters < 1:
            raise ValueError("n_parameters must be positive.")

    def step(self, theta, gradient) -> np.ndarray:
        theta, gradient = _validate_vector(theta, gradient)
        return theta - self.learning_rate * gradient


class Momentum:
    """Gradient descent using an exponential moving average of gradients.

    This project uses ``v_t = beta*v_(t-1) + (1-beta)*g_t`` and
    ``theta_(t+1) = theta_t - learning_rate*v_t``. This is a normalized
    exponential-moving-average convention; it is not the unnormalized
    heavy-ball update often presented in introductory optimization texts.
    """

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9) -> None:
        self.learning_rate = _validate_learning_rate(learning_rate)
        self.beta = float(beta)
        if not np.isfinite(self.beta) or not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must be in the interval [0, 1).")
        self.velocity: np.ndarray | None = None

    def reset(self, n_parameters: int) -> None:
        if n_parameters < 1:
            raise ValueError("n_parameters must be positive.")
        self.velocity = np.zeros(n_parameters, dtype=float)

    def step(self, theta, gradient) -> np.ndarray:
        theta, gradient = _validate_vector(theta, gradient)
        if self.velocity is None or self.velocity.shape != theta.shape:
            self.velocity = np.zeros_like(theta)
        self.velocity = (
            self.beta * self.velocity + (1.0 - self.beta) * gradient
        )
        return theta - self.learning_rate * self.velocity


class Adam:
    """Adaptive Moment Estimation with bias-corrected first and second moments."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.learning_rate = _validate_learning_rate(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

        if not np.isfinite(self.beta1) or not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must be in the interval [0, 1).")
        if not np.isfinite(self.beta2) or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta2 must be in the interval [0, 1).")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be a finite positive number.")

        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t = 0

    def reset(self, n_parameters: int) -> None:
        if n_parameters < 1:
            raise ValueError("n_parameters must be positive.")
        self.m = np.zeros(n_parameters, dtype=float)
        self.v = np.zeros(n_parameters, dtype=float)
        self.t = 0

    def step(self, theta, gradient) -> np.ndarray:
        theta, gradient = _validate_vector(theta, gradient)
        if self.m is None or self.v is None or self.m.shape != theta.shape:
            self.reset(len(theta))

        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient ** 2)

        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        return theta - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
