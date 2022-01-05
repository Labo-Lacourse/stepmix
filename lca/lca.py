"""EM for 1-step, 2-step and 3-step estimation"""
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

    def _initialize_parameters(self, X, Y, random_state):
        # Initialize latent class weights
        super()._initialize_parameters(X, random_state)

        # Initialize class weights
        self.rho = np.ones((self.n_components,))/self.n_components

        # Initialize measurement model
        self._mm = EMISSION_DICT[self.measurement](n_components=self.n_components,
                                                   random_state=self.random_state,
                                                   **self.measurement_params)
        self._mm.initialize(X, self.resp)

        # Initialize structural model
        self._sm = EMISSION_DICT[self.structural](n_components=self.n_components,
                                                   random_state=self.random_state,
                                                   **self.structural_params)
        self._sm.initialize(Y, self.resp)

    def _initialize(self, X, resp):
        self.resp = resp

    def fit_predict(self, X, Y):
        """Estimate model parameters using X and predict the labels for X.
        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.
        .. versionadded:: 0.20
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Y : array-like of shape (n_samples, n_structural)
            List of n_structural-dimensional data points.
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        Y = self._validate_data(Y, dtype=[np.float64, np.float32], ensure_min_samples=2)

        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, Y, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            for n_iter in range(1, self.max_iter + 1):
                print(n_iter)
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X, Y)
                self._m_step(X, Y, log_resp)
                lower_bound = log_prob_norm
                # lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

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
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X, Y)

        return log_resp.argmax(axis=1)

    def _e_step(self, X, Y):
        # Full model maximum likelihood
        log_resp = self._estimate_log_prob(X, Y) + np.log(self.rho).reshape((1, -1))

        # Likelihood
        log_prob_norm = logsumexp(log_resp, axis=1)

        # Normalize log responsibilities
        log_resp -= log_prob_norm.reshape((-1, 1))

        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, Y, log_resp):
        self.rho = np.exp(log_resp).mean(axis=0)

        # Full model maximum likelihood
        self._mm.m_step(X, log_resp)
        self._sm.m_step(Y, log_resp)

    def _estimate_log_weights(self):
        pass

    def _estimate_log_prob(self, X, Y):
        return self._mm.log_likelihood(X) + self._sm.log_likelihood(Y)

    def _get_parameters(self):
        return dict(rho=self.rho, mm_params=self._mm.get_parameters(), sm_params=self._sm.get_parameters())

    def _set_parameters(self, params):
        self.rho = params['rho']
        self._mm.set_parameters(params['mm_params'])
        self._sm.set_parameters(params['sm_params'])
