"""EM for multi-step estimation of latent class models with structural variables.

Please note the class weights rho are now referred to as 'weights' and the soft assignments tau are known as
'responsibilities' or resp to match the sklearn stack terminology.
"""
import warnings
import numpy as np

from scipy.special import logsumexp
from sklearn.mixture._base import BaseMixture
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_random_state

from .utils import check_in
from .emission import EMISSION_DICT


class LCA(BaseMixture):
    def __init__(
            self,
            n_components=2,
            *,
            measurement='gaussian',
            structural='bernoulli',
            measurement_params=dict(),
            structural_params=dict(),
            tol=1e-3,
            max_iter=100,
            n_init=1,
            init_params="random",
            random_state=None,
            verbose=0,
            verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=0,  # reg_covar is implemented in the Gaussian emission model
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=False,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        self.measurement = measurement
        self.measurement_params = measurement_params
        self.structural = structural
        self.structural_params = structural_params

    def _check_parameters(self, X):
        check_in(EMISSION_DICT.keys(), measurement=self.measurement)
        check_in(EMISSION_DICT.keys(), structural=self.structural)

    def _initialize_parameters(self, X, random_state):
        # Initialize latent class weights
        super()._initialize_parameters(X, random_state)

        # Uniform class weights initialization
        self.weights = np.ones((self.n_components,)) / self.n_components

        # Initialize measurement model
        if not hasattr(self, '_mm'):
            self._mm = EMISSION_DICT[self.measurement](n_components=self.n_components,
                                                       random_state=self.random_state,
                                                       **self.measurement_params)
        self._mm.initialize(X, self.resp)

    def _initialize_parameters_structural(self, Y, random_state):
        # Initialize structural model
        if not hasattr(self, '_sm'):
            self._sm = EMISSION_DICT[self.structural](n_components=self.n_components,
                                                      random_state=self.random_state,
                                                      **self.structural_params)
        self._sm.initialize(Y, self.resp)

    def _initialize(self, X, resp):
        self.resp = resp

    def fit(self, X, Y=None, freeze_measurement=False):
        """Adapted from the fit_predict method of the BaseMixture class to include (optional) structural model
        computations.

        Setting Y=None will run EM on the measurement model only. Providing both X and Y will run EM on the complete
        model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Y : array-like of shape (n_samples, n_structural), default = None
            List of n_structural-dimensional data points.
        freeze_measurement : bool, default =False
            Run EM on the complete model, but freeze measurement model parameters. Useful for two-step estimation.
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if Y is not None:
            Y = self._validate_data(Y, dtype=[np.float64, np.float32], ensure_min_samples=2)

        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        # do_init = not (self.warm_start and hasattr(self, "converged_"))
        do_init = True
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                if not freeze_measurement:
                    self._initialize_parameters(X, random_state) # Measurement model
                if Y is not None:
                    self._initialize_parameters_structural(Y, random_state)  # Structural Model

            lower_bound = -np.inf if do_init else self.lower_bound_

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X, Y)
                self._m_step(X, log_resp, Y, freeze_measurement=freeze_measurement)
                lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

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

    def fit_1_step(self, X, Y):
        self.fit(X, Y)  # Fit complete model

    def fit_2_step(self, X, Y):
        self.fit(X)  # Fit measurement model
        self.fit(X, Y, freeze_measurement=True)  # Fit complete model, but keep measurement model parameters fixed

    def fit_3_step(self, X, Y):
        self.fit(X)  # Fit measurement
        resp = self.predict_proba(X)  # Soft class assignment
        self.m_step_structural(resp, Y)  # Fit structural model

    def avg_log_likelihood(self, X, Y=None):
        avg_ll, _ = self._e_step(X, Y)
        return avg_ll

    def _e_step(self, X, Y=None):
        # Measurement likelihood
        log_resp = self._estimate_log_prob(X) + self._estimate_log_weights().reshape((1, -1))

        # Add structural model likelihood (if structural data is provided)
        if Y is not None:
            log_resp += self._sm.log_likelihood(Y)

        # Likelihood
        log_prob_norm = logsumexp(log_resp, axis=1)

        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp -= log_prob_norm.reshape((-1, 1))

        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp, Y=None, freeze_measurement=False):
        if not freeze_measurement:
            # Update measurement model parameters
            self.weights = np.exp(log_resp).mean(axis=0)
            self._mm.m_step(X, log_resp)

        if Y is not None:
            # Update structural model parameters
            self._sm.m_step(Y, log_resp)

    def m_step_structural(self, resp, Y):
        # For the third step of the 3-step approach
        if not hasattr(self, '_sm'):
            self._initialize_parameters_structural(Y, self.random_state)
        self._sm.m_step(Y, np.log(resp))

    def _estimate_log_weights(self):
        return np.log(self.weights)

    def _estimate_log_prob(self, X):
        return self._mm.log_likelihood(X)
    
    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):
        params = dict(weights=self.weights, measurement=self._mm.get_parameters())
        if hasattr(self, '_sm'):
            params['structural'] = self._sm.get_parameters()
        return params

    def get_parameters(self):
        return self._get_parameters()

    def _set_parameters(self, params):
        self.weights = params['weights']
        self._mm.set_parameters(params['measurement'])
        if 'structural' in params.keys():
            self._sm.set_parameters(params['structural'])
