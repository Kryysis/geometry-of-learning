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
    regularization = (l2 / 2) * np.sum(theta[1:] ** 2)
    
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
    gradient[1:] += l2 * theta[1:]

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

    hessian += l2 * regularization_matrix

    return hessian


class LogisticRegression:
    """
    Logistic regression with pluggable optimization algorithms.
    """

    def __init__(self, l2=0.0):
        self.l2 = l2
        self.theta = None
        self.loss_history = []

    def fit(
        self,
        X,
        y,
        optimizer,
        epochs=100,
        batch_size=None,
        shuffle=True
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("Invalid feature or target dimensions.")

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Targets must contain only 0 and 1.")

        X_bias = add_intercept(X)

        m = X_bias.shape[0]
        n_parameters = X_bias.shape[1]

        if batch_size is None:
            batch_size = m

        if not isinstance(batch_size, int) or not 1 <= batch_size <= m:
            raise ValueError("Invalid batch_size.")

        self.theta = np.zeros(n_parameters)
        self.loss_history = []

        optimizer.reset(n_parameters)

        for epoch in range(epochs):

            # Shuffle training observations
            if shuffle:
                indices = np.random.permutation(m)
            else:
                indices = np.arange(m)

            X_epoch = X_bias[indices]
            y_epoch = y[indices]

            # Process mini-batches
            for start in range(0, m, batch_size):

                end = min(start + batch_size, m)

                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]

                gradient = compute_gradient(
                    X_batch,
                    y_batch,
                    self.theta,
                    self.l2
                )

                self.theta = optimizer.step(
                    self.theta,
                    gradient
                )

            # Record full-dataset loss after each epoch
            epoch_loss = binary_cross_entropy(
                X_bias,
                y,
                self.theta,
                self.l2
            )

            self.loss_history.append(epoch_loss)

        return self

    def predict_proba(self, X):
        if self.theta is None:
            raise ValueError("Model must be fitted first.")

        X_bias = add_intercept(X)

        return sigmoid(X_bias @ self.theta)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)