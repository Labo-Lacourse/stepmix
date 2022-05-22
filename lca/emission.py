"""Emission models.

Encapsulate the M-step and log-likelihood computations of different conditional emission models."""
from abc import ABC, abstractmethod
import copy

import numpy as np
from sklearn.utils.validation import check_random_state
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky, _estimate_gaussian_parameters
from scipy.stats import multivariate_normal
from scipy.special import softmax

from .utils import check_in, check_int, check_positive, check_nonneg


class Emission(ABC):
    """Abstract class for Emission models.

    Emission models can be used by the LCA class for both the structural and the measurement model. Emission instances
    encapsulate maximum likelihood computations for a given model.

    All model parameters should be values of the self.parameters dict attribute. See the Bernoulli and GaussianUnit
    implementations for reference.

    To add an emission model, you must :
        - Inherit from Emission.
        - Implement the m_step, log_likelihood and sample methods.
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
        self.random_state = self.check_random_state(random_state)

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


    @abstractmethod
    def sample(self, class_no, n_samples):
        """Sample n_samples conditioned on the given class_no.

        Parameters
        ----------
        class_no : int
            Class int.
        n_samples : int
            Number of samples.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_features)
            Samples

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

    def sample(self, class_no, n_samples):
        feature_weights = self.parameters['pis'][:, class_no].reshape((1, -1))
        c = feature_weights.shape[1]
        X = (self.random_state.uniform(size=(n_samples, c)) < feature_weights).astype(int)
        return X


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
            X = self.means_[class_no] + self.random_state.standard_normal(size=(n_samples, n_features)) * np.sqrt(self.covariances_[class_no])
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


class Covariate(Emission):
    """Covariate model with simple gradient update.
    TODO: Add better stopping criterion
    """
    def __init__(self, iter=1, lr=1e-3, intercept=True, method='gradient', **kwargs):
        super().__init__(**kwargs)
        self.iter = iter
        self.lr = lr
        self.intercept = intercept
        self.method = method

    def check_parameters(self):
        super().check_parameters()
        check_in(["gradient", "newton-raphson"], method=self.method)

    #the full matrix (with column of 1s if self.intercept=True) is assumed to be given here for rapidity concerns
    def _forward(self, X_full):
        return softmax(X_full @ self.parameters['beta'], axis=1)

    def initialize(self, X, resp, random_state=None):
        n, D = X.shape
        D += self.intercept
        _, K = resp.shape

        self.check_parameters()
        random_state = self.check_random_state(random_state)

        # Parameter initialization
        # if self.intercept: beta[0,:]=intercept and beta[1:,:] = coefficients
        # Note: initial coefficients must be close to 0 for NR to be relatively stable
        self.parameters['beta'] = random_state.normal(0, 1e-3, size=(D, K))

    def get_full_matrix(self, X):
        n, _ = X.shape
        if self.intercept:
            return np.concatenate((np.ones((n, 1)), X), axis=1)
        else:
            return X

    # m-step using Newton-Raphson instead or gradient descent
    # Adapted from code by Thalles Silva
    def m_step(self, X, resp):
        X_full = self.get_full_matrix(X)
        n, D = X_full.shape
        _, K = resp.shape
        beta_shape = self.parameters['beta'].shape

        for _ in range(self.iter):
            logits = self._forward(X_full)

            if self.method == 'newton-raphson':
                HT = np.zeros((D, K, D, K))
                # calculate the hesssian
                for i in range(K):
                    for j in range(K):
                        r = np.multiply(logits[:, i], ((i == j) - logits[:, j]))
                        HT[:, i, :, j] = np.dot(np.multiply(X_full.T, r), X_full)
                H = np.reshape(HT, (D*K, D*K))

            # gradient of the cross-entropy
            G = np.dot(X_full.T, (logits - resp))

            if self.method == 'newton-raphson':
                # Newton's update
                self.parameters['beta'] = self.parameters['beta'].reshape(-1) - np.dot(np.linalg.pinv(H), G.reshape(-1))
                self.parameters['beta'] = np.reshape(self.parameters['beta'], beta_shape)
            else:
                # follow the gradient with GD
                self.parameters['beta'] = self.parameters['beta'] - self.lr * G/n

    def log_likelihood(self, X):
        X_full = self.get_full_matrix(X)
        n, D = X_full.shape
        prob = np.clip(self._forward(X_full), 1e-15, 1-1e-15)
        return np.log(prob)

    def predict(self, X):
        X_full = self.get_full_matrix(X)
        n, D = X_full.shape
        prob = self._forward(X_full)
        return prob.argmax(axis=1)

    def sample(self, class_no, n_samples):
        raise NotImplementedError


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
