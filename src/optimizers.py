import numpy as np


class GradientDescent:
    """Standard gradient descent."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def reset(self, n_parameters):
        pass

    def step(self, theta, gradient):
        return theta - self.learning_rate * gradient


class Momentum:
    """Gradient descent with momentum."""

    def __init__(self, learning_rate=0.01, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = None

    def reset(self, n_parameters):
        self.velocity = np.zeros(n_parameters)

    def step(self, theta, gradient):
        self.velocity = (
            self.beta * self.velocity
            + (1 - self.beta) * gradient
        )

        return theta - self.learning_rate * self.velocity


class Adam:
    """Adaptive Moment Estimation optimizer."""

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def reset(self, n_parameters):
        self.m = np.zeros(n_parameters)
        self.v = np.zeros(n_parameters)
        self.t = 0

    def step(self, theta, gradient):
        self.t += 1

        self.m = (
            self.beta1 * self.m
            + (1 - self.beta1) * gradient
        )

        self.v = (
            self.beta2 * self.v
            + (1 - self.beta2) * (gradient ** 2)
        )

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return theta - (
            self.learning_rate
            * m_hat
            / (np.sqrt(v_hat) + self.epsilon)
        )