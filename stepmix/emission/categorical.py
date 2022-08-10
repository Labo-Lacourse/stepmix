"""Categorical emission models."""
import numpy as np

from stepmix.emission.emission import Emission
from stepmix.utils import print_parameters


class Bernoulli(Emission):
    """Bernoulli (binary) emission model."""

    def m_step(self, X, resp):
        pis = X.T @ resp
        pis /= resp.sum(axis=0, keepdims=True)
        pis = np.clip(pis, 1e-15, 1 - 1e-15)  # avoid probabilities 0 or 1
        self.parameters["pis"] = pis

    def log_likelihood(self, X):
        # compute log emission probabilities
        pis = np.clip(
            self.parameters["pis"], 1e-15, 1 - 1e-15
        )  # avoid probabilities 0 or 1
        log_eps = X @ np.log(pis) + (1 - X) @ np.log(1 - pis)
        return log_eps

    def sample(self, class_no, n_samples):
        feature_weights = self.parameters["pis"][:, class_no].reshape((1, -1))
        K = feature_weights.shape[1]  # number of features
        X = (self.random_state.uniform(size=(n_samples, K)) < feature_weights).astype(
            int
        )
        return X

    def print_parameters(self, indent=1):
        print_parameters(
            self.parameters["pis"].T, "Bernoulli", indent=indent, np_precision=4
        )

    @property
    def n_parameters(self):
        return self.parameters["pis"].shape[0] * self.parameters["pis"].shape[1]


class BernoulliNan(Bernoulli):
    """Bernoulli (binary) emission model supporting missing values (Full Information Maximum Likelihood)."""

    def m_step(self, X, resp):
        is_observed = ~np.isnan(X)

        # Replace all nans with 0
        X = np.nan_to_num(X, nan=0)
        pis = X.T @ resp

        # Compute normalization factor over observed values for each feature
        for i in range(pis.shape[0]):
            resp_i = resp[is_observed[:, i]]
            pis[i] /= resp_i.sum(axis=0)

        self.parameters["pis"] = pis

    def log_likelihood(self, X):
        is_observed = ~np.isnan(X)

        # Replace all nans with 0
        X = np.nan_to_num(X, nan=0)

        # compute log emission probabilities
        pis = np.clip(
            self.parameters["pis"], 1e-15, 1 - 1e-15
        )  # avoid probabilities 0 or 1
        log_eps = X @ np.log(pis) + ((1 - X) * is_observed) @ np.log(1 - pis)

        return log_eps


class Multinoulli(Emission):
    """Multinoulli (categorical) emission model

    Uses one-hot encoded features. Expected data formatting:
    X[n,k*L+l]=1 if l is the observed outcome for the kth attribute of data point n,
    where n is the number of observations, K=n_features, L=n_outcomes for each multinoulli

    Parameters:
    pis[k*L+l,c]=P[ X[n,k*L+l]=1 | n belongs to class c]
    """

    def __init__(self, n_components=2, random_state=None, n_outcomes=2):
        super().__init__(n_components=n_components, random_state=random_state)
        self.n_outcomes = n_outcomes

    def get_n_features(self):
        n_features_x_n_outcomes = self.parameters["pis"].shape[0]
        n_features = int(n_features_x_n_outcomes / self.n_outcomes)
        return n_features

    def m_step(self, X, resp):
        pis = X.T @ resp
        pis /= resp.sum(axis=0, keepdims=True)
        pis = np.clip(pis, 1e-15, 1 - 1e-15)  # avoid probabilities 0 or 1
        self.parameters["pis"] = pis

    def log_likelihood(self, X):
        # compute log emission probabilities
        pis = np.clip(self.parameters["pis"], 1e-15, 1 - 1e-15)
        log_eps = X @ np.log(pis)
        return log_eps

    def sample(self, class_no, n_samples):
        pis = self.parameters["pis"]
        n_features = self.get_n_features()
        feature_weights = pis[:, class_no].reshape(n_features, self.n_outcomes)
        X = np.array(
            [
                self.random_state.multinomial(1, feature_weights[k], size=n_samples)
                for k in range(n_features)
            ]
        )
        X = np.reshape(np.swapaxes(X, 0, 1), (n_samples, n_features * self.n_outcomes))
        return X

    def print_parameters(self, indent=1):
        print_parameters(
            self.parameters["pis"].T,
            "Multinoulli",
            n_outcomes=self.n_outcomes,
            indent=indent,
            np_precision=4,
        )

    @property
    def n_parameters(self):
        n_params = self.parameters["pis"].shape[0] * self.parameters["pis"].shape[1]
        return n_params


class MultinoulliNan(Multinoulli):
    """Multinoulli (categorical) emission model supporting missing values (Full Information Maximum Likelihood)."""

    def m_step(self, X, resp):
        is_observed = ~np.isnan(X)

        # Replace all nans with 0
        X = np.nan_to_num(X, nan=0)
        pis = X.T @ resp

        # Compute normalization factor over observed values for each feature
        for i in range(pis.shape[0]):
            resp_i = resp[is_observed[:, i]]
            pis[i] /= resp_i.sum(axis=0)

        self.parameters["pis"] = pis

    def log_likelihood(self, X):
        is_observed = ~np.isnan(X)

        # Replace all nans with 0
        X = np.nan_to_num(X, nan=0)

        # compute log emission probabilities
        pis = np.clip(self.parameters["pis"], 1e-15, 1 - 1e-15)
        log_eps = X @ np.log(pis)
        return log_eps
