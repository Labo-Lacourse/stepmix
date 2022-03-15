"""Emission models.

Encapsulate the M-step and log-likelihood computations of different conditional emission models."""
from abc import ABC, abstractmethod
import copy

from sklearn.utils.validation import check_random_state
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import numpy as np

from .utils import check_in, check_int, check_positive, check_nonneg


class Emission(ABC):
    def __init__(self, n_components, random_state):
        self.n_components = n_components
        self.random_state = random_state

        # Dict including all parameters for estimation
        self.parameters = dict()

    def check_parameters(self):
        check_int(n_components=self.n_components)
        check_positive(n_components=self.n_components)

    def initialize(self, X, log_resp, random_state=None):
        self.check_parameters()
        # Currently unused, for future random initializations
        random_state = self.check_random_state(random_state)

        # Measurement and structural models are initialized by running their M-step on the initial log responsibilities
        # obtained via kmeans or sampled uniformly (See LCA._initialize_parameters)
        self.m_step(X, log_resp)

    def check_random_state(self, random_state=None):
        if random_state is None:
            # If no random state is provided, use mine
            random_state = check_random_state(self.random_state)
        else:
            # Use the provided random_state
            random_state = check_random_state(random_state)
        return random_state

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, params):
        self.parameters = params

    @abstractmethod
    def m_step(self, X, log_resp):
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(self, X):
        raise NotImplementedError


class Bernoulli(Emission):
    def __init__(self, clip_eps=1e-15, n_components=2, random_state=None):
        super().__init__(n_components=n_components, random_state=random_state)
        self.clip_eps = clip_eps

    def check_parameters(self):
        super().check_parameters()
        check_nonneg(clip_eps=self.clip_eps)

    def m_step(self, X, log_resp):
        resp = np.exp(log_resp)
        pis = X.T @ resp
        pis /= resp.sum(axis=0, keepdims=True)
        pis = np.clip(pis, self.clip_eps, 1 - self.clip_eps)  # avoid probabilities 0 or 1
        self.parameters['pis'] = pis

    def log_likelihood(self, X):
        # compute log emission probabilities
        pis = np.clip(self.parameters['pis'], self.clip_eps, 1 - self.clip_eps)  # avoid probabilities 0 or 1
        log_eps = X @ np.log(pis) + (1 - X) @ np.log(1 - pis)

        return log_eps


class GaussianUnit(Emission):
    """sklearn.mixture.GaussianMixture does not have an implementation for fixed unit variance, so we provide one."""

    def __init__(self, n_components=2, random_state=None):
        super().__init__(n_components=n_components, random_state=random_state)

    def m_step(self, X, log_resp):
        resp = np.exp(log_resp)
        self.parameters['means'] = (resp[..., np.newaxis] * X[:, np.newaxis, :]).sum(axis=0) / resp.sum(axis=0, keepdims=True).T

    def log_likelihood(self, X):
        n, D = X.shape
        log_eps = np.zeros((n, self.n_components))
        for c in range(self.n_components):
            log_eps[:, c] = multivariate_normal.logpdf(x=X, mean=self.parameters['means'][c], cov=1)
        return log_eps


class Gaussian(Emission):
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

    def initialize(self, X, log_resp, random_state=None):
        self.check_parameters()
        # Currently unused, for future random initializations
        random_state = self.check_random_state(random_state)

        # Required to get the initial means, covariances and precisions right
        # Already performs the M-step
        GaussianMixture._initialize(self, X, np.exp(log_resp))

    def m_step(self, X, log_resp):
        # This will update self.means_, self.covariances_ and self.precisions_cholesky_
        GaussianMixture._m_step(self, X, log_resp)

    def log_likelihood(self, X):
        return GaussianMixture._estimate_log_prob(self, X)

    def get_parameters(self):
        return dict(means=self.means_.copy(), covariances=self.covariances_.copy(), precisions_cholesky=self.precisions_cholesky_.copy())

    def set_parameters(self, params):
        # We spoof the sklearn GaussianMixture class
        # This is not needed for typical children of the Emission class. We do this only to be compatible
        # with the sklearn GaussianMixture machinery.
        # This will update self.means_, self.covariances_ and self.precisions_cholesky_
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


EMISSION_DICT = {
    'gaussian_unit': GaussianUnit,
    'gaussian_full': GaussianFull,
    'gaussian_spherical': GaussianSpherical,
    'gaussian_diag': GaussianDiag,
    'gaussian_tied': GaussianTied,
    'bernoulli': Bernoulli,
}
