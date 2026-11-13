
import numpy as np


def sigmoid(z):
    """
    Convert real-valued inputs into probabilities between 0 and 1.
    """
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500, 500)

    return 1.0 / (1.0 + np.exp(-z))


def add_intercept(X):
    """
    Add a column of ones to represent the bias parameter.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    return np.column_stack((np.ones(X.shape[0]), X))


def binary_cross_entropy(X, y, theta, l2=0.0):
    """
    Calculate the average binary cross-entropy loss.

    L2 regularization is applied to weights, not the bias.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    m = X.shape[0]

    z = X @ theta

    # Numerically stable binary cross-entropy
    data_loss = np.mean(np.logaddexp(0.0, z) - y * z)

    # Exclude bias (theta[0]) from regularization
    regularization = (l2 / (2 * m)) * np.sum(theta[1:] ** 2)

    return data_loss + regularization


def compute_gradient(X, y, theta, l2=0.0):
    """
    Calculate the gradient of binary cross-entropy.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    m = X.shape[0]

    probabilities = sigmoid(X @ theta)

    gradient = (X.T @ (probabilities - y)) / m

    # L2 regularization gradient (excluding bias)
    gradient[1:] += (l2 / m) * theta[1:]

    return gradient


def compute_hessian(X, theta, l2=0.0):
    """
    Calculate the Hessian matrix of binary cross-entropy.

    The Hessian describes the local curvature of the loss.
    """
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    m = X.shape[0]

    probabilities = sigmoid(X @ theta)

    curvature = probabilities * (1.0 - probabilities)

    # X.T @ diag(curvature) @ X, without explicitly
    # constructing the diagonal matrix.
    weighted_X = X * curvature[:, np.newaxis]

    hessian = (X.T @ weighted_X) / m

    # L2 regularization Hessian (excluding bias)
    regularization_matrix = np.eye(X.shape[1])
    regularization_matrix[0, 0] = 0.0

    hessian += (l2 / m) * regularization_matrix

    return hessian


class LogisticRegression:
    """
    Logistic regression with a manually implemented
    gradient descent optimizer.
    """

    def __init__(self, learning_rate=0.01, epochs=1000, l2=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2

        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("Invalid feature or target dimensions.")

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Targets must contain only 0 and 1.")

        X_bias = add_intercept(X)

        self.theta = np.zeros(X_bias.shape[1])
        self.loss_history = []

        for _ in range(self.epochs):
            gradient = compute_gradient(
                X_bias, y, self.theta, self.l2
            )

            self.theta -= self.learning_rate * gradient

            loss = binary_cross_entropy(
                X_bias, y, self.theta, self.l2
            )

            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        if self.theta is None:
            raise ValueError("Model must be fitted first.")

        X_bias = add_intercept(X)

        return sigmoid(X_bias @ self.theta)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)