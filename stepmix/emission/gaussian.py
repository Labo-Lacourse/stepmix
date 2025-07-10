"""Gaussian emission models."""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
    _estimate_gaussian_parameters,
    _compute_precision_cholesky,
)

from stepmix.emission.emission import Emission
from stepmix.utils import check_in, check_nonneg, cov_np_to_df


class GaussianUnit(Emission):
    """Gaussian emission model with fixed unit variance.

    sklearn.mixture.GaussianMixture does not have an implementation for fixed unit variance, so we provide one.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_str = "gaussian_unit"

    def m_step(self, X, resp):
        self.parameters["means"] = (resp[..., np.newaxis] * X[:, np.newaxis, :]).sum(
            axis=0
        ) / resp.sum(axis=0, keepdims=True).T

    def log_likelihood(self, X):
        n, D = X.shape
        log_eps = np.zeros((n, self.n_components))
        for c in range(self.n_components):
            log_eps[:, c] = multivariate_normal.logpdf(
                x=X, mean=self.parameters["means"][c], cov=1
            )
        return log_eps

    def sample(self, class_no, n_samples):
        D = self.parameters["means"].shape[1]
        X = self.random_state.normal(
            loc=self.parameters["means"][class_no],
            scale=np.ones(D),
            size=(n_samples, D),
        )
        return X

    @property
    def n_parameters(self):
        return self.parameters["means"].shape[0] * self.parameters["means"].shape[1]


class Gaussian(Emission):
    """Gaussian emission model with various covariance options.

    This class spoofs the scikit-learn Gaussian Mixture class by reusing the same attributes and calls its methods.
    """

    def __init__(
        self,
        n_components=2,
        covariance_type="spherical",
        init_params="random",
        reg_covar=1e-6,
        random_state=None,
    ):
        super().__init__(n_components=n_components, random_state=random_state)
        self.covariance_type = covariance_type
        self.init_params = init_params
        self.reg_covar = reg_covar

        # Actual parameters
        # We spoof the sklearn GaussianMixture class
        # This is not needed for typical children of the Emission class. We do this only to be compatible
        # with the sklearn GaussianMixture machinery.
        # Normally you should keep all parameters in a dict under self.parameters
        self.parameters = None
        self.means_ = None
        self.covariances_ = None
        self.precisions_cholesky_ = None

        # This is for compatibility with Gaussian mixture methods. Could be implemented in the future.
        self.means_init = None
        self.weights_init = None
        self.precisions_init = None

    def check_parameters(self):
        super().check_parameters()
        check_in(
            ["spherical", "tied", "diag", "full"], covariance_type=self.covariance_type
        )
        check_in(["random"], init_params=self.init_params)
        check_nonneg(reg_covar=self.reg_covar)

    def initialize(self, X, resp, random_state=None):
        self.check_parameters()

        # Required to get the initial means, covariances and precisions right
        # Already performs the M-step
        GaussianMixture._initialize(self, X, resp)

    def m_step(self, X, resp):
        """M step.

        Adapted from the gaussian mixture class to accept responsibilities instead of log responsibilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
            Posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

        # Save attributes to dict for compatible api with other stepmix models
        self.parameters = self.get_parameters()

    def log_likelihood(self, X):
        return GaussianMixture._estimate_log_prob(self, X)

    def sample(self, class_no, n_samples):
        if self.covariance_type == "full":
            X = self.random_state.multivariate_normal(
                self.means_[class_no], self.covariances_[class_no], n_samples
            )
        elif self.covariance_type == "tied":
            X = self.random_state.multivariate_normal(
                self.means_[class_no], self.covariances_, n_samples
            )
        else:
            n_features = self.means_.shape[1]
            X = self.means_[class_no] + self.random_state.standard_normal(
                size=(n_samples, n_features)
            ) * np.sqrt(self.covariances_[class_no])
        return X

    def get_parameters(self):
        return dict(
            means=self.means_.copy(),
            covariances=self.covariances_.copy(),
            precisions_cholesky=self.precisions_cholesky_.copy(),
        )

    def set_parameters(self, params):
        # We spoof the sklearn GaussianMixture class
        # This is not needed for typical children of the Emission class. We do this only to be compatible
        # with the sklearn GaussianMixture machinery.
        # This will update self.means_, self.covariances_ and self.precisions_cholesky_
        if "precisions_cholesky" not in params:
            params["precisions_cholesky"] = _compute_precision_cholesky(
                params["covariances"], self.covariance_type
            )
        GaussianMixture._set_parameters(
            self,
            (
                None,
                params["means"],
                params["covariances"],
                params["precisions_cholesky"],
            ),
        )

        # Save attributes to dict for compatible api with other stepmix models
        self.parameters = self.get_parameters()

    @property
    def n_parameters(self):
        n = GaussianMixture._n_parameters(self)

        # Remove class weights from the count
        # We add them back in the main StepMix class
        n -= self.n_components - 1

        return n

    def permute_classes(self, perm, axis=0):
        # Latent classes are on first axis
        self.means_ = self.means_[perm]

        # Tied has a single covariance (and precisions) shared among all classes
        # No need to permute
        if self.covariance_type != "tied":
            self.covariances_ = self.covariances_[perm]
            self.precisions_cholesky_ = self.precisions_cholesky_[perm]


class GaussianFull(Gaussian):
    def __init__(self, **kwargs):
        # Make sure no other covariance_type is specified
        kwargs.pop("covariance_type", None)
        super().__init__(covariance_type="full", **kwargs)
        self.model_str = "gaussian_full"

    def print_parameters(
        self,
        indent=1,
        feature_names=None,
        index=["class_no", "param"],
        columns=["model_name", "variable"],
    ):
        """Flipping class_no and variable is nicer for full covariances."""
        super().print_parameters(
            indent=indent, feature_names=feature_names, index=index, columns=columns
        )

    def get_parameters_df(self, feature_names=None):
        """Return self.parameters into a long dataframe.

        Call self._to_df or implement custom method."""
        if feature_names is None:
            n_features = self.parameters["means"].shape[1]
            feature_names = self.get_default_feature_names(n_features)

        means = self._to_df(
            param_dict=self.parameters, keys=["means"], feature_names=feature_names
        )
        cov = cov_np_to_df(
            self.parameters["covariances"], feature_names, self.model_str
        )

        return pd.concat([means, cov])


class GaussianSpherical(Gaussian):
    def __init__(self, **kwargs):
        # Make sure no other covariance_type is specified
        kwargs.pop("covariance_type", None)
        super().__init__(covariance_type="spherical", **kwargs)
        self.model_str = "gaussian_spherical"

    def get_parameters_df(self, feature_names=None):
        params = self.get_parameters()
        n_features = params["means"].shape[1]
        params["covariances"] = np.tile(params["covariances"], (n_features, 1)).T
        return self._to_df(
            param_dict=params,
            keys=["means", "covariances"],
            feature_names=feature_names,
        )


class GaussianDiag(Gaussian):
    def __init__(self, **kwargs):
        # Make sure no other covariance_type is specified
        kwargs.pop("covariance_type", None)
        super().__init__(covariance_type="diag", **kwargs)
        self.model_str = "gaussian_diag"

    def get_parameters_df(self, feature_names=None):
        params = self.get_parameters()
        return self._to_df(
            param_dict=params,
            keys=["means", "covariances"],
            feature_names=feature_names,
        )


class GaussianTied(Gaussian):
    def __init__(self, **kwargs):
        # Make sure no other covariance_type is specified
        kwargs.pop("covariance_type", None)
        super().__init__(covariance_type="tied", **kwargs)
        self.model_str = "gaussian_tied"

    def print_parameters(
        self,
        indent=1,
        feature_names=None,
        index=["class_no", "param"],
        columns=["model_name", "variable"],
    ):
        """Flipping class_no and variable is nicer for full covariances."""
        super().print_parameters(
            indent=indent, feature_names=feature_names, index=index, columns=columns
        )

    def get_parameters_df(self, feature_names=None):
        """Return self.parameters into a long dataframe.

        Call self._to_df or implement custom method."""
        if feature_names is None:
            n_features = self.parameters["means"].shape[1]
            feature_names = self.get_default_feature_names(n_features)

        means = self._to_df(
            param_dict=self.parameters, keys=["means"], feature_names=feature_names
        )
        cov = np.tile(self.parameters["covariances"], (self.n_components, 1, 1))
        cov = cov_np_to_df(cov, feature_names, self.model_str)

        return pd.concat([means, cov])


class GaussianNan(Emission):
    """Gaussian emission model supporting missing values (Full Information Maximum
    Likelihood)

    This class assumes a diagonal covariance structure. The covariances are therefore represented as a
    (n_components, n_features) array."""

    def __init__(self, debug_likelihood=False, **kwargs):
        super().__init__(**kwargs)
        self.debug_likelihood = debug_likelihood

    def m_step(self, X, resp):
        is_observed = ~np.isnan(X)

        # Replace all nans with 0
        resp_sums = list()

        # Compute normalization factor over observed values for each feature
        for i in range(X.shape[1]):
            resp_i = resp[is_observed[:, i]]
            resp_sums.append(resp_i.sum(axis=0))

        self.parameters["means"] = self._compute_means(X, resp, resp_sums)
        self.parameters["covariances"] = self._compute_cov(X, resp, resp_sums)

    def _compute_means(self, X, resp, resp_sums):
        X = np.nan_to_num(X)
        means = (resp[..., np.newaxis] * X[:, np.newaxis, :]).sum(axis=0)

        # Normalize
        for i in range(means.shape[1]):
            means[:, i] /= resp_sums[i]

        return means

    def _compute_cov(self, X, resp, resp_sums):
        raise NotImplementedError("No covariance estimator is implemented.")

    def log_likelihood(self, X):
        if not self.debug_likelihood:
            return self._log_likelihood(X)
        else:
            # Compute the log likelihood naively. Useful for debugging, but slow
            return self._debug_ll(X)

    def _log_likelihood(self, X):
        # To be tested, but this should work for any diagonal covariance
        is_observed = ~np.isnan(X)
        n, D = X.shape
        log_eps = np.zeros((n, self.n_components))

        for c in range(self.n_components):
            diff = X - self.parameters["means"][c].reshape(1, -1)

            # Zero out the nans
            diff = np.nan_to_num(diff)

            # First compute the likelihood from the term (x-\mu)^T \Sigma^-1 (x-\mu)
            precision_c = 1 / self.parameters["covariances"][c]
            ll_diff = ((diff**2) * precision_c.reshape(1, -1)).sum(axis=1)

            # Then compute the likelihood from the term log det(2*\pi*\Sigma)
            pi_cov_c = 2 * np.pi * np.tile(self.parameters["covariances"][c], (n, 1))

            # Replace nan dimensions with 1 since this won't affect the product of the diagonal (determinant)
            pi_cov_c = np.where(is_observed, pi_cov_c, 1)

            log_dets = np.log(pi_cov_c).sum(axis=1)

            log_eps[:, c] = -0.5 * (log_dets + ll_diff)

        return log_eps

    def _debug_ll(self, X):
        # Naive loopy way to compute the log likelihood
        # Useful for debugging, otherwise self._log_likelihood should be preferred
        is_observed = ~np.isnan(X)

        n, D = X.shape
        log_eps = np.zeros((n, self.n_components))
        for i in range(n):
            # Only select observed dimensions
            mask = is_observed[i]
            if mask.any():
                for c in range(self.n_components):
                    x_i = X[i, mask]
                    mean_c = self.parameters["means"][c, mask]
                    cov_c = self.parameters["covariances"][c, mask]

                    log_eps[i, c] = multivariate_normal.logpdf(
                        x=x_i, mean=mean_c, cov=cov_c
                    )
            else:
                # Undefined log likelihood
                # We use 0, as it won't affect the overall likelihood when summing over independent models
                log_eps[i, :] = 0
        return log_eps

    def sample(self, class_no, n_samples):
        D = self.parameters["means"].shape[1]
        X = self.random_state.normal(
            loc=self.parameters["means"][class_no],
            scale=self.parameters["covariances"][class_no],
            size=(n_samples, D),
        )
        return X


class GaussianUnitNan(GaussianNan):
    """Gaussian emission model with unit covariance supporting missing values (Full Information Maximum
    Likelihood)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_str = "gaussian_unit_nan"

    def _compute_cov(self, X, resp, resp_sums):
        """No estimate. Simply return diagonal covariance 1 for all features."""
        return np.ones_like(self.parameters["means"])

    @property
    def n_parameters(self):
        return self.parameters["means"].shape[0] * self.parameters["means"].shape[1]


class GaussianSphericalNan(GaussianNan):
    """Gaussian emission model with spherical covariance supporting missing values (Full Information Maximum
    Likelihood)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_str = "gaussian_spherical_nan"

    def _compute_cov(self, X, resp, resp_sums):
        """One covariance parameter per component."""
        covs = list()
        is_observed = ~np.isnan(X)
        is_observed_row = is_observed.sum(axis=1)

        for c in range(self.n_components):
            diff = X - self.parameters["means"][c].reshape(1, -1)
            diff = np.nan_to_num(diff)  # Zero out the nans
            cov_c = resp[:, c][..., np.newaxis] * (diff**2)
            z = resp[:, c] @ is_observed_row
            covs.append(cov_c.sum() / z)

        covs = np.array(covs)

        # Keep a diagonal format for compatibility with parent class
        result = np.ones_like(self.parameters["means"]) * covs.reshape((-1, 1))

        return result

    @property
    def n_parameters(self):
        mean_params = (
            self.parameters["means"].shape[0] * self.parameters["means"].shape[1]
        )

        # Covariance are n latent class x n features, but we only have one degree of freedom per class
        cov_params = self.parameters["covariances"].shape[0]

        return mean_params + cov_params


class GaussianDiagNan(GaussianNan):
    """Gaussian emission model with diagonal covariance supporting missing values (Full Information Maximum
    Likelihood)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_str = "gaussian_diag_nan"

    def _compute_cov(self, X, resp, resp_sums):
        """One covariance parameter per column."""
        covs = list()
        for c in range(self.n_components):
            diff = X - self.parameters["means"][c].reshape(1, -1)
            diff = np.nan_to_num(diff)  # Zero out the nans
            cov_c = resp[:, c][..., np.newaxis] * (diff**2)
            covs.append(cov_c.sum(axis=0))

        covs = np.vstack(covs)

        # Normalize
        for i in range(covs.shape[1]):
            covs[:, i] /= resp_sums[i]

        return covs

    @property
    def n_parameters(self):
        mean_params = (
            self.parameters["means"].shape[0] * self.parameters["means"].shape[1]
        )
        cov_params = (
            self.parameters["covariances"].shape[0]
            * self.parameters["covariances"].shape[1]
        )

        return mean_params + cov_params
