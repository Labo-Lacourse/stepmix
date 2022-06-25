import numpy as np

from lca.emission.emission import Emission


class Bernoulli(Emission):
    """Bernoulli (binary) emission model."""

    def m_step(self, X, resp):
        pis = X.T @ resp
        pis /= resp.sum(axis=0, keepdims=True)
        self.parameters['pis'] = pis

    def log_likelihood(self, X):
        # compute log emission probabilities
        pis = np.clip(self.parameters['pis'], 1e-15, 1 - 1e-15)  # avoid probabilities 0 or 1
        log_eps = X @ np.log(pis) + (1 - X) @ np.log(1 - pis)
        return log_eps

    def sample(self, class_no, n_samples):
        feature_weights = self.parameters['pis'][:, class_no].reshape((1, -1))
        K = feature_weights.shape[1]  # number of features
        X = (self.random_state.uniform(size=(n_samples, K)) < feature_weights).astype(int)
        return X


class BernoulliNan(Bernoulli):
    """Bernoulli (binary) emission model supporting missing values (Full Information Maximum Likelihood)."""

    def m_step(self, X, resp):
        is_observed = ~np.isnan(X)

        # Replace all nans with 0
        X = np.nan_to_num(X)
        pis = X.T @ resp

        # Compute normalization factor over observed values for each feature
        for i in range(pis.shape[0]):
            resp_i = resp[is_observed[:, i]]
            pis[i] /= resp_i.sum(axis=0)

        self.parameters['pis'] = pis

    def log_likelihood(self, X):
        is_observed = ~np.isnan(X)

        # Replace all nans with 0
        X = np.nan_to_num(X)

        # compute log emission probabilities
        pis = np.clip(self.parameters['pis'], 1e-15, 1 - 1e-15)  # avoid probabilities 0 or 1
        log_eps = X @ np.log(pis) + ((1 - X) * is_observed) @ np.log(1 - pis)

        return log_eps


class Multinoulli(Emission):
    """Multinoulli (categorical) emission model."""

    def m_step(self, X, resp):
        pis = np.swapaxes((X.T @ resp).T, 0, 1)
        pis /= np.swapaxes((X.T @ resp).T.sum(axis=2, keepdims=True), 0, 1)
        pis = np.clip(pis, 1e-15, 1 - 1e-15)  # avoid probabilities 0 or 1
        self.parameters['pis'] = pis

    def log_likelihood(self, X):
        # compute log emission probabilities
        pis = np.clip(self.parameters['pis'], 1e-15, 1 - 1e-15)  # avoid probabilities 0 or 1
        n, K, L = X.shape  # n individuals, K features, L possible outcomes for each multinoulli
        K, C, L = pis.shape  # C latent classes
        log_eps = np.reshape(X, (n, K * L)) @ np.reshape(np.swapaxes(np.log(pis), 0, 1), (C, K * L)).T
        return log_eps

    def sample(self, class_no, n_samples):
        feature_weights = self.parameters['pis'][:, class_no, :]
        X = np.array([self.random_state.multinomial(1, feature_weights[k], size=n_samples) for k in range(K)])
        return X
