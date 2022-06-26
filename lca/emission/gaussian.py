import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters, _compute_precision_cholesky

from lca.emission.emission import Emission
from lca.utils import check_in, check_nonneg


class GaussianUnit(Emission):
    """Gaussian emission model with fixed unit variance.

    sklearn.mixture.GaussianMixture does not have an implementation for fixed unit variance, so we provide one.
    """

    def m_step(self, X, resp):
        self.parameters['means'] = (resp[..., np.newaxis] * X[:, np.newaxis, :]).sum(axis=0) / resp.sum(axis=0,
                                                                                                        keepdims=True).T

    def log_likelihood(self, X):
        n, D = X.shape
        log_eps = np.zeros((n, self.n_components))
        for c in range(self.n_components):
            log_eps[:, c] = multivariate_normal.logpdf(x=X, mean=self.parameters['means'][c], cov=1)
        return log_eps

    def sample(self, class_no, n_samples):
        D = self.parameters['means'].shape[1]
        X = self.random_state.normal(loc=self.parameters['means'][class_no], scale=np.ones(D), size=(n_samples, D))
        return X


class GaussianUnitNan(GaussianUnit):
    """Gaussian emission model with fixed unit variance supporting missing values (Full Information Maximum Likelihood)
    """

    def m_step(self, X, resp):
        is_observed = ~np.isnan(X)

        # Replace all nans with 0
        X = np.nan_to_num(X)
        means = (resp[..., np.newaxis] * X[:, np.newaxis, :]).sum(axis=0)

        # Compute normalization factor over observed values for each feature
        for i in range(means.shape[1]):
            resp_i = resp[is_observed[:, i]]
            means[:, i] /= resp_i.sum(axis=0)

        self.parameters['means'] = means

    def log_likelihood(self, X):
        n, D = X.shape
        log_eps = np.zeros((n, self.n_components))
        for c in range(self.n_components):
            mean_c = self.parameters['means'][c]

            # Impute nans with the mean to effectively ignore them in the log likelihood
            # TODO: Review this. Particularly in the case of the determinant, should it be
            # TODO: reduced if there are missing values in the sample? E.g., if we have a 3D multivariate gaussian
            # TODO: should we use a 2D log pdf for observations where we are missing one dimension?
            X_c = X.copy()
            for k in range(X_c.shape[1]):
                X_c[:, k] = np.nan_to_num(X[:, k], nan=mean_c[k])

            log_eps[:, c] = multivariate_normal.logpdf(x=X_c, mean=mean_c, cov=1)
        return log_eps


class Gaussian(Emission):
    """Gaussian emission model with various covariance options.

    This class spoofs the scikit-learn Gaussian Mixture class by reusing the same attributes and calls its methods."""

    def __init__(self, n_components=2, covariance_type="spherical", init_params="random", reg_covar=1e-6,
                 random_state=None):
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
        check_in(["spherical", "tied", "diag", "full"], covariance_type=self.covariance_type)
        check_in(["random"], init_params=self.init_params)
        check_nonneg(reg_covar=self.reg_covar)

    def initialize(self, X, resp, random_state=None):
        self.check_parameters()
        # Currently unused, for future random initializations
        random_state = self.check_random_state(random_state)

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

    def log_likelihood(self, X):
        return GaussianMixture._estimate_log_prob(self, X)

    def sample(self, class_no, n_samples):
        if self.covariance_type == "full":
            X = self.random_state.multivariate_normal(self.means_[class_no], self.covariances_[class_no], n_samples)
        elif self.covariance_type == "tied":
            X = self.random_state.multivariate_normal(self.means_[class_no], self.covariances_, n_samples)
        else:
            n_features = self.means_.shape[1]
            X = self.means_[class_no] + self.random_state.standard_normal(size=(n_samples, n_features)) * np.sqrt(
                self.covariances_[class_no])
        return X

    def get_parameters(self):
        return dict(means=self.means_.copy(), covariances=self.covariances_.copy(),
                    precisions_cholesky=self.precisions_cholesky_.copy())

    def set_parameters(self, params):
        # We spoof the sklearn GaussianMixture class
        # This is not needed for typical children of the Emission class. We do this only to be compatible
        # with the sklearn GaussianMixture machinery.
        # This will update self.means_, self.covariances_ and self.precisions_cholesky_
        if 'precisions_cholesky' not in params:
            params['precisions_cholesky'] = _compute_precision_cholesky(params['covariances'], self.covariance_type)
        GaussianMixture._set_parameters(self,
                                        (None, params['means'], params['covariances'], params['precisions_cholesky']))


class GaussianFull(Gaussian):
    def __init__(self, **kwargs):
        # Make sure no other covariance_type is specified
        kwargs.pop('covariance_type', None)
        super().__init__(covariance_type='full', **kwargs)


class GaussianSpherical(Gaussian):
    def __init__(self, **kwargs):
        # Make sure no other covariance_type is specified
        kwargs.pop('covariance_type', None)
        super().__init__(covariance_type='spherical', **kwargs)


class GaussianDiag(Gaussian):
    def __init__(self, **kwargs):
        # Make sure no other covariance_type is specified
        kwargs.pop('covariance_type', None)
        super().__init__(covariance_type='diag', **kwargs)


class GaussianTied(Gaussian):
    def __init__(self, **kwargs):
        # Make sure no other covariance_type is specified
        kwargs.pop('covariance_type', None)
        super().__init__(covariance_type='tied', **kwargs)
