"""Emission models.

Encapsulate the M-step and log-likelihood computations of different conditional emission models."""
from abc import ABC, abstractmethod
import copy

import numpy as np
from sklearn.utils.validation import check_random_state
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky, _estimate_gaussian_parameters
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
from scipy.special import softmax

from .utils import check_in, check_int, check_positive, check_nonneg, modal


class Emission(ABC):
    """Abstract class for Emission models.

    Emission models can be used by the LCA class for both the structural and the measurement model. Emission instances
    encapsulate maximum likelihood computations for a given model.

    All model parameters should be values of the self.parameters dict attribute. See the Bernoulli and GaussianUnit
    implementations for reference.

    To add an emission model, you must :
        - Inherit from Emission.
        - Implement the m_step and log_likelihood methods.
        - Add a corresponding string in the EMISSION_DICT at the end of emission.py.
        - Update the LCA docstring for the measurement and structural arguments!

    Parameters
    ----------
    n_components : int, default=2
        The number of latent classes.
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the parameters.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    self.parameters : dict
        Dictionary with all model parameters.

    """

    def __init__(self, n_components, random_state):
        self.n_components = n_components
        self.random_state = random_state

        # Dict including all parameters for estimation
        self.parameters = dict()

    def check_parameters(self):
        """Validate class attributes."""
        check_int(n_components=self.n_components)
        check_positive(n_components=self.n_components)

    def check_random_state(self, random_state=None):
        """Use a provided random state, otherwise use self.random_state.

        Parameters
        ----------
        random_state : int, RandomState instance or None, default=None
            Controls the random seed given to the method chosen to initialize the parameters.
            Pass an int for reproducible output across multiple function calls.
        """
        if random_state is None:
            # If no random state is provided, use mine
            random_state = check_random_state(self.random_state)
        else:
            # Use the provided random_state
            random_state = check_random_state(random_state)
        return random_state

    def initialize(self, X, resp, random_state=None):
        """Initialize parameters.

        Simply performs the m-step on the current responsibilities to initialize parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data for this emission model.
        resp : ndarray of shape (n_samples, n_components)
            Responsibilities, i.e., posterior probabilities over the latent classes.
        random_state : int, RandomState instance or None, default=None
            Controls the random seed given to the method chosen to initialize the parameters.
            Pass an int for reproducible output across multiple function calls.

        """
        self.check_parameters()
        # Currently unused, for future random initializations
        self.random_state = self.check_random_state(random_state)

        # Measurement and structural models are initialized by running their M-step on the initial log responsibilities
        # obtained via kmeans or sampled uniformly (See LCA._initialize_parameters)
        self.m_step(X, resp)

    def get_parameters(self):
        """Get a copy of model parameters.

        Returns
        -------
        parameters: dict
            Copy of model parameters.

        """
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        """Set current parameters.

        Parameters
        -------
        parameters: dict
            Model parameters. Should be the same format as the dict returned by self.get_parameters.
        """
        self.parameters = parameters

    @abstractmethod
    def m_step(self, X, resp):
        """Update model parameters via maximum likelihood using the current responsilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data for this emission model.
        resp : ndarray of shape (n_samples, n_components)
            Responsibilities, i.e., posterior probabilities over the latent classes of each point in X.
        """
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(self, X):
        """Return the log-likelihood of the input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data for this emission model.

        Returns
        -------
        ll : ndarray of shape (n_samples, n_components)
            Log-likelihood of the input data conditioned on each component.

        """
        raise NotImplementedError


class Bernoulli(Emission):
    """Bernoulli (binary) emission model."""

    def m_step(self, X, resp):
        pis = X.T @ resp
        pis /= resp.sum(axis=0, keepdims=True)
        pis = np.clip(pis, 1e-15, 1 - 1e-15)  # avoid probabilities 0 or 1
        self.parameters['pis'] = pis

    def log_likelihood(self, X):
        # compute log emission probabilities
        pis = np.clip(self.parameters['pis'], 1e-15, 1 - 1e-15)  # avoid probabilities 0 or 1
        log_eps = X @ np.log(pis) + (1 - X) @ np.log(1 - pis)

        return log_eps


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

    def get_parameters(self):
        return dict(means=self.means_.copy(), covariances=self.covariances_.copy(),
                    precisions_cholesky=self.precisions_cholesky_.copy())

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


class Covariate(Emission):
    """Covariate model with simple gradient update.

    TODO: Does not currently use a reference category. All classes have a coefficient.
    TODO: Consider fancier optimizer. This implementation only seems to work in the high separation case.

    """
    def __init__(self, iter=1, lr=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.iter = iter
        self.lr = lr

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((X, intercept))

    def _forward(self, X):
        return softmax(X @ self.parameters['coef'], axis=1)

    def initialize(self, X, resp, random_state=None):
        n, D = X.shape
        _, K = resp.shape

        self.check_parameters()
        random_state = self.check_random_state(random_state)

        # Parameter initialization
        # D + 1 for intercept
        self.parameters['coef'] = random_state.normal(0, 2, size=(D + 1, K))

    def m_step(self, X, resp):
        n, D = X.shape
        _, K = resp.shape

        # Add intercept
        X = self._add_intercept(X)

        for _ in range(self.iter):
            output = self._forward(X)

            # CE/Softmax gradient
            grad = output - resp
            grad = X.T @ grad/n

            # Update parameters
            self.parameters['coef'] -= self.lr * grad

    def log_likelihood(self, X):
        X = self._add_intercept(X)
        prob = np.clip(self._forward(X), 1e-15, 1-1e-15)
        return np.log(prob)

    def predict(self, X):
        X = self._add_intercept(X)
        prob = self._forward(X)
        return prob.argmax(axis=1)


class Covariate_sk(Emission):
    # Sklearn-based covariate model.
    # I don't think we will keep this. We need more flexibility
    def m_step(self, X, resp):
        # Max assignment
        y = resp.argmax(axis=1)

        # Check which class is represented
        is_pred = np.unique(y)

        # Fit a Logistic regression
        if not hasattr(self, 'model_'):
            self.model_ = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                             warm_start=True, max_iter=1000)

        self.model_.fit(X, y)
        self.parameters['coef'] = self.model_.coef_
        self.parameters['intercept'] = self.model_.intercept_
        self.parameters['is_pred'] = is_pred
        self.parameters['n_classes'] = resp.shape[1]

    def log_likelihood(self, X):
        prob_pred = self.model_.predict_log_proba(X)
        return prob_pred

    def predict(self, X):
        return self.model_.predict(X)


EMISSION_DICT = {
    'gaussian_unit': GaussianUnit,
    'gaussian_full': GaussianFull,
    'gaussian_spherical': GaussianSpherical,
    'gaussian_diag': GaussianDiag,
    'gaussian_tied': GaussianTied,
    'bernoulli': Bernoulli,
    'binary': Bernoulli,
    'covariate': Covariate,
}
