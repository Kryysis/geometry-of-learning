import unittest
import numpy as np

from src import logistic_regression, optimizers


class LogisticRegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.X = rng.normal(size=(80, 3))
        probabilities = logistic_regression.sigmoid(
            0.2 + self.X @ np.array([0.7, -1.1, 0.4])
        )
        self.y = (rng.random(80) < probabilities).astype(int)
        self.X_bias = logistic_regression.add_intercept(self.X)
        self.theta = np.array([0.1, -0.2, 0.3, 0.05])

    def test_gradient_matches_finite_difference(self):
        def objective(theta):
            return logistic_regression.binary_cross_entropy(
                self.X_bias, self.y, theta, l2=0.03
            )

        epsilon = 1e-6
        numeric = np.zeros_like(self.theta)
        for j in range(self.theta.size):
            plus, minus = self.theta.copy(), self.theta.copy()
            plus[j] += epsilon
            minus[j] -= epsilon
            numeric[j] = (objective(plus) - objective(minus)) / (2 * epsilon)

        analytic = logistic_regression.compute_gradient(
            self.X_bias, self.y, self.theta, l2=0.03
        )
        np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=1e-5)

    def test_hessian_matches_finite_difference_of_gradient(self):
        epsilon = 1e-5
        numeric = np.zeros((self.theta.size, self.theta.size))
        for j in range(self.theta.size):
            plus, minus = self.theta.copy(), self.theta.copy()
            plus[j] += epsilon
            minus[j] -= epsilon
            grad_plus = logistic_regression.compute_gradient(
                self.X_bias, self.y, plus, l2=0.03
            )
            grad_minus = logistic_regression.compute_gradient(
                self.X_bias, self.y, minus, l2=0.03
            )
            numeric[:, j] = (grad_plus - grad_minus) / (2 * epsilon)

        analytic = logistic_regression.compute_hessian(
            self.X_bias, self.theta, l2=0.03
        )
        np.testing.assert_allclose(analytic, numeric, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(analytic, analytic.T, atol=1e-12)

    def test_model_fit_with_all_optimizers(self):
        optimizers_to_test = [
            optimizers.GradientDescent(0.1),
            optimizers.Momentum(0.1, beta=0.9),
            optimizers.Adam(0.01),
        ]
        for optimizer in optimizers_to_test:
            model = logistic_regression.LogisticRegression(l2=0.01)
            model.fit(
                self.X,
                self.y,
                optimizer=optimizer,
                epochs=20,
                batch_size=len(self.y),
                random_state=17,
            )
            self.assertEqual(len(model.loss_history), 20)
            self.assertEqual(model.predict(self.X).shape, self.y.shape)
            self.assertTrue(np.all(np.isfinite(model.predict_proba(self.X))))

    def test_sgd_batching_and_random_state_are_reproducible(self):
        models = []
        for _ in range(2):
            model = logistic_regression.LogisticRegression(l2=0.01)
            model.fit(
                self.X,
                self.y,
                optimizer=optimizers.GradientDescent(0.01),
                epochs=8,
                batch_size=1,
                random_state=29,
            )
            models.append(model)
        np.testing.assert_allclose(models[0].theta, models[1].theta)
        np.testing.assert_allclose(models[0].loss_history, models[1].loss_history)

    def test_regularization_excludes_intercept(self):
        gradient_no_l2 = logistic_regression.compute_gradient(
            self.X_bias, self.y, self.theta, l2=0.0
        )
        gradient_l2 = logistic_regression.compute_gradient(
            self.X_bias, self.y, self.theta, l2=0.5
        )
        self.assertAlmostEqual(gradient_no_l2[0], gradient_l2[0], places=12)
        self.assertFalse(np.allclose(gradient_no_l2[1:], gradient_l2[1:]))


if __name__ == "__main__":
    unittest.main()
