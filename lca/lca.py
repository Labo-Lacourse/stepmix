"""EM for multi-step estimation of latent class models with structural variables.

Please note the class weights rho are now referred to as 'weights' and the assignments tau are known as
'responsibilities' or resp to match the sklearn stack terminology.
"""
import warnings
import numpy as np

from scipy.special import logsumexp, softmax
from sklearn.mixture._base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_random_state, check_is_fitted
from sklearn.cluster import KMeans

from . import utils
from .emission import EMISSION_DICT


class LCA(BaseEstimator):
    def __init__(
            self,
            n_components=2,
            *,
            n_steps=1,
            measurement="gaussian",
            structural="bernoulli",
            measurement_params=dict(),
            structural_params=dict(),
            tol=1e-3,
            max_iter=100,
            n_init=1,
            init_params="random",
            random_state=None,
            verbose=0,
            verbose_interval=10,
            assignment="soft",
            correction=None,
    ):
        # Attributes of the base LCA class
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.n_steps = n_steps

        # Additional attributes for 3-step estimation
        self.assignment = assignment
        self.correction = correction

        # Additional attributes to specify the measurement and structural models
        self.measurement = measurement
        self.measurement_params = measurement_params
        self.structural = structural
        self.structural_params = structural_params

    def _check_initial_parameters(self, X):
        """Validate class attributes.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        Raises
        ------
        ValueError : unacceptable choice of parameters

        """
        utils.check_type(int, n_components=self.n_components, max_iter=self.max_iter, n_init=self.n_init,
                         verbose=self.verbose, verbose_interval=self.verbose_interval)
        utils.check_positive(n_components=self.n_components, max_iter=self.max_iter, n_init=self.n_init,
                             verbose_interval=self.verbose_interval)
        utils.check_nonneg(tol=self.tol, verbose=self.verbose)
        utils.check_in([1, 2, 3], n_steps=self.n_steps)
        utils.check_in(["kmeans", "random"], init_params=self.init_params)
        utils.check_in(["modal", "soft"], init_params=self.assignment)
        utils.check_in([None, "BCH", "ML"], init_params=self.correction)
        utils.check_in(EMISSION_DICT.keys(), measurement=self.measurement)
        utils.check_in(EMISSION_DICT.keys(), structural=self.structural)
        utils.check_type(dict, measurement_params=self.measurement_params, structural_params=self.structural_params)

    def _initialize_parameters(self, X, random_state):
        """Initialize the weights and measurement model parameters.

        We do not initialize the structural model here, since the LCA class can be used without one.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        Raises
        ------
        ValueError : illegal self.init_params parameter
        """
        n_samples, _ = X.shape

        # Initialize responsibilities
        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components))
            label = (
                KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                    .fit(X)
                    .labels_
            )
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = random_state.uniform(size=(n_samples, self.n_components))
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError(
                f"Unimplemented initialization method {self.init_params}."
            )

        # Save log responsibilities
        self.log_resp_ = np.log(np.clip(resp, 1e-15, 1 - 1e-15))

        # Uniform class weights initialization
        self.weights_ = np.ones((self.n_components,)) / self.n_components

        # Initialize measurement model
        self._initialize_parameters_measurement(X, random_state)

    def _initialize_parameters_measurement(self, X, random_state=None):
        """Initialize parameters of measurement model.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
        """
        # Initialize measurement model
        if not hasattr(self, '_mm'):
            self._mm = EMISSION_DICT[self.measurement](n_components=self.n_components,
                                                       random_state=self.random_state,
                                                       **self.structural_params)
        # Use the provided random_state instead of self.random_state to ensure we have a different init every run
        self._mm.initialize(X, np.exp(self.log_resp_), random_state)

    def _initialize_parameters_structural(self, X, random_state=None):
        """Initialize parameters of structural model.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
        """
        # Initialize structural model
        if not hasattr(self, '_sm'):
            self._sm = EMISSION_DICT[self.structural](n_components=self.n_components,
                                                      random_state=self.random_state,
                                                      **self.structural_params)
        # Use the provided random_state instead of self.random_state to ensure we have a different init every run
        self._sm.initialize(X, np.exp(self.log_resp_), random_state)

    def fit(self, X, Y=None):
        """Fit LCA measurement model and optionally the structural model.

        Setting Y=None will fit the measurement model only. Providing both X and Y will fit the full model following
        the self.n_steps argument.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point of the measurement model.
        Y : array-like of shape (n_samples, n_structural), default = None
            List of n_structural-dimensional data points. Each row
            corresponds to a single data point of the structural model.

        """
        if Y is None:
            # No structural data. Simply fit the measurement data
            self.em(X)

        elif self.n_steps == 1:
            # One-step estimation
            # 1) Maximum likelihood with both measurement and structural models
            self.em(X, Y)

        elif self.n_steps == 2:
            # Two-step estimation
            # 1) Fit the measurement model
            self.em(X)
            # 2) Fit the the structural model by keeping the parameters of the measurement model fixed
            self.em(X, Y, freeze_measurement=True)

        elif self.n_steps == 3 and self.correction is None:
            # Three-step estimation
            # 1) Fit the measurement model
            self.em(X)
            # 2) Assign class probabilities
            soft_resp = self.predict_proba(X)

            # Modal assignment (clipped for numerical reasons)
            # Else we simply keep the assignment as is (soft)
            if self.assignment == 'modal':
                resp = utils.modal(soft_resp, clip=True)
            else:
                resp = soft_resp

            # 3) M-step on the structural model
            self.m_step_structural(resp, Y)

        elif self.n_steps == 3 and self.correction == 'BCH':
            # Three-step estimation with BCH correction
            # 1) Fit the measurement model
            self.em(X)

            # 2) Assign class probabilities
            soft_resp = self.predict_proba(X)

            # Modal assignment (clipped for numerical reasons)
            # Else we simply keep the assignment as is (soft)
            if self.assignment == 'modal':
                resp = utils.modal(soft_resp)
            else:
                resp = soft_resp

            # Apply BCH correction
            _, D_inv = compute_bch_matrix(soft_resp)
            resp = resp @ D_inv

            # 3) M-step on the structural model
            self.m_step_structural(resp, Y)

        elif self.n_steps == 3 and self.correction == 'ML':
            # Three-step estimation with ML correction
            # 1) Fit the measurement model
            self.em(X)

            # 2) Assign class probabilities
            soft_resp = self.predict_proba(X)

            # Compute D
            D, _ = compute_bch_matrix(soft_resp)

            # Compute log_emission_pm
            log_emission_pm = compute_log_emission_pm(soft_resp.argmax(axis=1), D)

            # 3) M-step on the structural model
            self.em(X, Y, freeze_measurement=True, log_emission_pm=log_emission_pm)

    def em(self, X, Y=None, freeze_measurement=False, log_emission_pm=None):
        """EM algorithm to fit the weights, measurement parameters and structural parameters.

        Adapted from the fit_predict method of the sklearn BaseMixture class to include (optional) structural model
        computations.

        Setting Y=None will run EM on the measurement model only. Providing both X and Y will run EM on the complete
        model, unless otherwise specified by freeze_measurement.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point of the measurement model.
        Y : array-like of shape (n_samples, n_structural), default = None
            List of n_structural-dimensional data points. Each row
            corresponds to a single data point of the structural model.
        freeze_measurement : bool, default =False
            Run EM on the complete model, but do not update measurement model parameters. Useful for two-step estimation.
        log_emission_pm : array-like of shape (n, n_components), default=None
            Log probabilities of the predicted class given the true latent class for ML correction.
        """
        # First validate the input and the class attributes
        n_samples, _ = X.shape
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if Y is not None:
            Y = self._validate_data(Y, dtype=[np.float64, np.float32], ensure_min_samples=2)

        if n_samples < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_initial_parameters(X)

        # Set up useful values for optimization
        random_state = check_random_state(self.random_state)
        max_lower_bound = -np.inf
        self.converged_ = False

        # Run multiple restarts
        for init in range(self.n_init):
            # self._print_verbose_msg_init_beg(init)

            if not freeze_measurement:
                self._initialize_parameters(X, random_state)  # Measurement model

            if Y is not None:
                self._initialize_parameters_structural(Y, random_state)  # Structural Model

            lower_bound = -np.inf

            # EM iterations
            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                # E-step
                log_prob_norm, log_resp = self._e_step(X, Y=Y, log_emission_pm=log_emission_pm)

                # M-step
                self._m_step(X, np.exp(log_resp), Y, freeze_measurement=freeze_measurement)

                # Likelihood & stopping criterion
                lower_bound = log_prob_norm
                change = lower_bound - prev_lower_bound

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn(
                "Initializations did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data.",
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

    def _e_step(self, X, Y=None, log_emission_pm=None):
        # Measurement log-likelihood
        if log_emission_pm is not None:
            # Use log probabilities of the predicted class given the true latent class for ML correction
            log_resp = log_emission_pm.copy()
        else:
            # Standard Measurement Log likelihood
            log_resp = self._mm.log_likelihood(X)

        # Add class prior probabilities
        log_resp += np.log(self.weights_).reshape((1, -1))

        # Add structural model likelihood (if structural data is provided)
        if Y is not None:
            log_resp += self._sm.log_likelihood(Y)

        # Log-likelihood
        log_prob_norm = logsumexp(log_resp, axis=1)

        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp -= log_prob_norm.reshape((-1, 1))

        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, resp, Y=None, freeze_measurement=False):
        if not freeze_measurement:
            # Update measurement model parameters
            self.weights_ = resp.mean(axis=0)
            self._mm.m_step(X, resp)

        if Y is not None:
            # Update structural model parameters
            self._sm.m_step(Y, resp)

    def m_step_structural(self, resp, Y):
        # For the third step of the 3-step approach
        if not hasattr(self, '_sm'):
            self._initialize_parameters_structural(Y)
        self._sm.m_step(Y, resp)

    def score(self, X, Y=None):
        avg_ll, _ = self._e_step(X, Y)
        return avg_ll

    def predict(self, X, Y=None):
        """Predict the labels for the data samples in X using the measurement model.

        Optionally, an array-like Y can be provided to predict the labels based on both the measurement and structural
        models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point of the measurement model.
        Y : array-like of shape (n_samples, n_features), default=None
            List of n_features-dimensional data points. Each row
            corresponds to a single data point of the structural model.
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        return self.predict_proba(X, Y).argmax(axis=1)

    def predict_proba(self, X, Y=None):
        """Predict the class probabilities for the data samples in X using the measurement model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point of the measurement model.
        Y : array-like of shape (n_samples, n_features), default=None
            List of n_features-dimensional data points. Each row
            corresponds to a single data point of the structural model.
        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Density of each component for each sample in X.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        if Y is not None:
            Y = self._validate_data(Y, reset=False)
        _, log_resp = self._e_step(X, Y)
        return np.exp(log_resp)

    def _get_parameters(self):
        params = dict(weights=self.weights_, measurement=self._mm.get_parameters())
        if hasattr(self, '_sm'):
            params['structural'] = self._sm.get_parameters()
        return params

    def get_parameters(self):
        return self._get_parameters()

    def _set_parameters(self, params):
        self.weights_ = params['weights']
        self._mm.set_parameters(params['measurement'])
        if 'structural' in params.keys():
            self._sm.set_parameters(params['structural'])


def compute_bch_matrix(resp):
    """Compute the probability D[c,s] = P(X_pred=s | X=c) of predicting class latent class s given
    that a point belongs to latent class c.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)
        Class responsibilities.

    Returns
    ----------
     D: array-like of shape (n_components, n_components)
        Matrix of conditional probabilities D[c,s] = P(X_pred=s | X=c)
     D_inv: array-like of shape (n_components, n_components)
        Pseudo-inverse of D.
    """
    # Dimensions
    n = resp.shape[0]
    X_pred = utils.modal(resp)
    weights = resp.mean(axis=0)

    # BCH correction (based on the empirical distribution: eq (6))
    D = (resp.T @ X_pred / weights.reshape((-1, 1))) / n
    D_inv = np.linalg.pinv(D)

    return D, D_inv


def compute_log_emission_pm(X_pred_idx, D):
    """(Log) probabilities of the predicted class given the true latent classes.

    Used for ML correction.

    Inputs:
        X_pred_idx: array-like of size (n,) 
            Predicted classes. X_pred[i]=argmax{c} p(X[i]=c|Y[i];rho,pis).
        D: array_like of size (n_components, n_components). 
            Matrix of (previously) estimated probabilities D[c,s] = p(X_pred=s|X=c) for pairs of classes c,s

    Returns:
        log_eps: ndarray of size (n, n_components)
            Matrix of log emission probabilities: log p(X_pred_idx[i]|X[i]=c)
    """
    # number of units
    n = X_pred_idx.size

    # number of latent classes
    C = D.shape[0]

    # compute log emission probabilities
    log_eps = np.zeros((n, C))
    log_D = np.log(np.clip(D, 1e-15, 1 - 1e-15))  # avoid probabilities 0 or 1

    for s in range(C):
        indexes_pred_s = np.where(X_pred_idx == s)
        log_eps[indexes_pred_s, :] = log_D[:, s]
    return log_eps