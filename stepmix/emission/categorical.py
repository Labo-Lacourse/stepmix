"""Categorical emission models."""
import numpy as np

from stepmix.emission.emission import Emission
from stepmix.utils import print_parameters, max_one_hot


class Bernoulli(Emission):
    """Bernoulli (binary) emission model."""

    def m_step(self, X, resp):
        pis = X.T @ resp
        pis /= resp.sum(axis=0, keepdims=True)
        pis = np.clip(pis, 1e-15, 1 - 1e-15)  # avoid probabilities 0 or 1
        self.parameters["pis"] = pis.T

    def log_likelihood(self, X):
        # compute log emission probabilities
        pis = np.clip(
            self.parameters["pis"].T, 1e-15, 1 - 1e-15
        )  # avoid probabilities 0 or 1
        log_eps = X @ np.log(pis) + (1 - X) @ np.log(1 - pis)
        return log_eps

    def sample(self, class_no, n_samples):
        feature_weights = self.parameters["pis"][class_no, :].reshape((1, -1))
        K = feature_weights.shape[1]  # number of features
        X = (self.random_state.uniform(size=(n_samples, K)) < feature_weights).astype(
            int
        )
        return X

    def print_parameters(self, indent=1):
        print_parameters(
            self.parameters["pis"], "Bernoulli", indent=indent, np_precision=4
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

        self.parameters["pis"] = pis.T

    def log_likelihood(self, X):
        is_observed = ~np.isnan(X)

        # Replace all nans with 0
        X = np.nan_to_num(X, nan=0)

        # compute log emission probabilities
        pis = np.clip(
            self.parameters["pis"].T, 1e-15, 1 - 1e-15
        )  # avoid probabilities 0 or 1
        log_eps = X @ np.log(pis) + ((1 - X) * is_observed) @ np.log(1 - pis)

        return log_eps


class Multinoulli(Emission):
    """Multinoulli (categorical) emission model

    Uses one-hot encoded features. Expected data formatting:
    X[n,k*L+l]=1 if l is the observed outcome for the kth attribute of data point n,
    where n is the number of observations, K=n_features, L=max_n_outcomes for each multinoulli where max_n_outcomes
    represents the maximum number of outcomes for a given feature.

    If integer_codes is set to True, the model will expect integer-encoded categories and will one-hot
    encode the data itself. In this case, max_n_outcomes and total_outcomes are inferred by the model.

    Parameters
    ----------
    n_components : int, default=2
        The number of latent classes.
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters. Pass an int for reproducible output across multiple function calls.
    integer_codes : bool, default=True
        Input X should be integer-encoded zero-indexed categories.
    max_n_outcomes : int, default=None
        Maximum number of outcomes for a single categorical feature.
        Each column in the input will have max_n_outcomes associated columns in the one-hot encoding.
        If None and integer_codes=True, will be inferred from the data.
    total_outcomes : int, default=None
        Total outcomes over all features. E.g., if we provide a categorical variable with two outcomes and another
        with 4 outcomes, total_outcomes = 6.
        If None and integer_codes=True, will be inferred from the data.

    Attributes
    ----------
    pis[k*L+l,c]=P[ X[n,k*L+l]=1 | n belongs to class c]
    """

    def __init__(
        self,
        n_components=2,
        random_state=None,
        integer_codes=True,
        max_n_outcomes=None,
        total_outcomes=None,
    ):
        super().__init__(n_components=n_components, random_state=random_state)
        self.integer_codes = integer_codes
        self.parameters["max_n_outcomes"] = max_n_outcomes
        self.parameters["total_outcomes"] = total_outcomes

        if max_n_outcomes is None and not integer_codes:
            raise ValueError(
                "max_n_outcomes can only be set to None with integer_codes=True."
            )
        if total_outcomes is None and not integer_codes:
            raise ValueError(
                "total_outcomes can only be set to None with integer_codes=True."
            )

    def get_n_features(self):
        n_features_x_max_n_outcomes = self.parameters["pis"].shape[1]
        n_features = int(
            n_features_x_max_n_outcomes / self.parameters["max_n_outcomes"]
        )
        return n_features

    def encode_features(self, X):
        if self.integer_codes:
            # Self.n_outcomes will only be updated if it is still None
            (
                X,
                self.parameters["max_n_outcomes"],
                self.parameters["total_outcomes"],
            ) = max_one_hot(
                X,
                max_n_outcomes=self.parameters["max_n_outcomes"],
                total_outcomes=self.parameters["total_outcomes"],
            )

        return X

    def m_step(self, X, resp):
        X = self.encode_features(X)
        pis = X.T @ resp
        pis /= resp.sum(axis=0, keepdims=True)
        pis = np.clip(pis, 1e-15, 1 - 1e-15)  # avoid probabilities 0 or 1
        self.parameters["pis"] = pis.T

    def log_likelihood(self, X):
        X = self.encode_features(X)
        # compute log emission probabilities
        pis = np.clip(self.parameters["pis"].T, 1e-15, 1 - 1e-15)
        log_eps = X @ np.log(pis)
        return log_eps

    def sample(self, class_no, n_samples):
        pis = self.parameters["pis"].T
        n_features = self.get_n_features()
        feature_weights = pis[:, class_no].reshape(
            n_features, self.parameters["max_n_outcomes"]
        )
        X = np.array(
            [
                self.random_state.multinomial(1, feature_weights[k], size=n_samples)
                for k in range(n_features)
            ]
        )
        X = np.reshape(
            np.swapaxes(X, 0, 1),
            (n_samples, n_features * self.parameters["max_n_outcomes"]),
        )
        return X

    def print_parameters(self, indent=1):
        print_parameters(
            self.parameters["pis"],
            "Multinoulli",
            n_outcomes=self.parameters["max_n_outcomes"],
            indent=indent,
            np_precision=4,
        )

    @property
    def n_parameters(self):
        n_classes = self.parameters["pis"].shape[0]

        # Only n_outcomes - 1 free parameters per feature since probabilities sum to 1
        n_free_parameters_per_class = (
            self.parameters["total_outcomes"] - self.get_n_features()
        )
        return n_classes * n_free_parameters_per_class


class MultinoulliNan(Multinoulli):
    """Multinoulli (categorical) emission model supporting missing values (Full Information Maximum Likelihood)."""

    def m_step(self, X, resp):
        X = self.encode_features(X)
        is_observed = ~np.isnan(X)

        # Replace all nans with 0
        X = np.nan_to_num(X, nan=0)
        pis = X.T @ resp

        # Compute normalization factor over observed values for each feature
        for i in range(pis.shape[0]):
            resp_i = resp[is_observed[:, i]]
            pis[i] /= resp_i.sum(axis=0)

        self.parameters["pis"] = pis.T

    def log_likelihood(self, X):
        X = self.encode_features(X)
        is_observed = ~np.isnan(X)

        # Replace all nans with 0
        X = np.nan_to_num(X, nan=0)

        # compute log emission probabilities
        pis = np.clip(self.parameters["pis"].T, 1e-15, 1 - 1e-15)
        log_eps = X @ np.log(pis)
        return log_eps
