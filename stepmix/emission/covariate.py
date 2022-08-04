import numpy as np
from scipy.special import softmax

from stepmix.emission.emission import Emission
from stepmix.utils import check_in, print_parameters


class Covariate(Emission):
    """Covariate model with simple gradient update."""

    def __init__(
        self, tol=1e-4, max_iter=1, lr=1e-3, intercept=True, method="gradient", **kwargs
    ):
        super().__init__(**kwargs)
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr
        self.intercept = intercept
        self.method = method

    def check_parameters(self):
        super().check_parameters()
        check_in(["gradient", "newton-raphson"], method=self.method)

    # the full matrix (with column of 1s if self.intercept=True) is assumed to be given here for rapidity concerns
    def _forward(self, X_full):
        return softmax(X_full @ self.parameters["beta"], axis=1)

    def initialize(self, X, resp, random_state=None):
        n, D = X.shape
        D += self.intercept
        _, K = resp.shape

        self.check_parameters()
        random_state = self.check_random_state(random_state)

        # Parameter initialization
        # if self.intercept: beta[0,:]=intercept and beta[1:,:] = coefficients
        # Note: initial coefficients must be close to 0 for NR to be relatively stable
        self.parameters["beta"] = random_state.normal(0, 1e-3, size=(D, K))

    def get_full_matrix(self, X):
        n, _ = X.shape
        if self.intercept:
            return np.concatenate((np.ones((n, 1)), X), axis=1)
        else:
            return X

    # m-step using Newton-Raphson instead or gradient descent
    # Adapted from code by Thalles Silva: https://towardsdatascience.com/logistic-regression-the-good-parts-55efa68e11df
    def m_step(self, X, resp):
        X_full = self.get_full_matrix(X)
        n, D = X_full.shape
        _, K = resp.shape
        beta_shape = self.parameters["beta"].shape

        for _ in range(self.max_iter):
            logits = self._forward(X_full)

            if self.method == "newton-raphson":
                HT = np.zeros((D, K, D, K))
                # calculate the hesssian
                for i in range(K):
                    for j in range(K):
                        r = np.multiply(logits[:, i], ((i == j) - logits[:, j]))
                        HT[:, i, :, j] = np.dot(np.multiply(X_full.T, r), X_full)
                H = np.reshape(HT, (D * K, D * K))

            # gradient of the cross-entropy
            G = np.dot(X_full.T, (logits - resp))

            # stopping criterion: iterations stop when all the components of the gradient are under tol
            if np.max(np.abs(G)) < self.tol:
                break

            if self.method == "newton-raphson":
                # Newton's update
                self.parameters["beta"] = self.parameters["beta"].reshape(-1) - np.dot(
                    np.linalg.pinv(H), G.reshape(-1)
                )
                self.parameters["beta"] = np.reshape(
                    self.parameters["beta"], beta_shape
                )
            elif self.method == "gradient":
                # follow the gradient with GD
                self.parameters["beta"] = self.parameters["beta"] - self.lr * G / n

    def log_likelihood(self, X):
        X_full = self.get_full_matrix(X)
        n, D = X_full.shape
        prob = np.clip(self._forward(X_full), 1e-15, 1 - 1e-15)
        return np.log(prob)

    def predict(self, X):
        X_full = self.get_full_matrix(X)
        n, D = X_full.shape
        prob = self._forward(X_full)
        return prob.argmax(axis=1)

    def sample(self, class_no, n_samples):
        raise NotImplementedError

    def print_parameters(self, indent=1):
        print_parameters(
            self.parameters["beta"].T,
            "Covariate",
            np_precision=2,
            indent=indent,
            intercept=True,
        )

    @property
    def n_parameters(self):
        return self.parameters["beta"].shape[0] * self.parameters["beta"].shape[1]
