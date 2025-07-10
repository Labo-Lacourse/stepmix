"""EM for stepwise estimation of latent class models with structural variables.

Please note the class weights rho are now referred to as 'weights' and the assignments tau are known as
'responsibilities' or resp to match the sklearn stack terminology.
"""

# Author: Sacha Morin <morin.sacha@gmail.com>
# Author: Robin Legault <robin.legault@umontreal.ca>
# License:

import warnings
import copy

import pandas as pd
import numpy as np

from scipy.special import logsumexp
from sklearn.mixture._base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import (
    check_random_state,
    check_is_fitted,
    _check_sample_weight,
)
from sklearn.cluster import KMeans
import tqdm

from . import utils
from .corrections import compute_bch_matrix, compute_log_emission_pm
from .emission.build_emission import EMISSION_DICT, build_emission
from .bootstrap import bootstrap


class StepMix(BaseEstimator):
    """StepMix estimator for Latent Class Analysis.

    Multi-step EM estimation of latent class models with measurement and structural models. The measurement and
    structural models can be fit together (1-step) or sequentially (2-step and 3-step). This estimator implements
    the BCH and ML bias correction methods for 3-step estimation.

    The measurement and structural models can be any of those defined in stepmix.emission. The measurement model
    can be used alone to effectively fit a latent mixture model.

    This class was adapted from the scikit-learn BaseMixture and GaussianMixture classes.

    .. versionadded:: 0.00

    Parameters
    ----------
    n_components : int, default=2
        The number of latent classes.
    n_steps : {1, 2, 3}, default=1
        Number of steps in the estimation. Must be one of :

        - 1: run EM on both the measurement and structural models.
        - 2: first run EM on the measurement model, then on the complete model, but keep the measurement parameters\
        fixed for the second step. See *Bakk, 2018*.
        - 3: first run EM on the measurement model, assign class probabilities, then fit the structural model via\
        maximum likelihood. See the correction parameter for bias correction.

    measurement : {'bernoulli', 'bernoulli_nan', 'binary', 'binary_nan', 'categorical', 'categorical_nan', 'continuous', \
    'continuous_nan', 'covariate',\
    'gaussian', 'gaussian_nan', 'gaussian_unit', 'gaussian_unit_nan', 'gaussian_spherical', 'gaussian_spherical_nan',\
    'gaussian_tied', 'gaussian_diag', 'gaussian_diag_nan', 'gaussian_full', 'multinoulli', 'multinoulli_nan', dict},\
    default='bernoulli'
        String describing the measurement model. Must be one of:

        - 'bernoulli': the observed data consists of n_features bernoulli (binary) random variables.
        - 'bernoulli_nan': the observed data consists of n_features bernoulli (binary) random variables. Supports missing values.
        - 'binary': alias for bernoulli.
        - 'binary_nan': alias for bernoulli_nan.
        - 'categorical': alias for multinoulli.
        - 'categorical_nan': alias for multinoulli_nan.
        - 'continuous': alias for gaussian_diag.
        - 'continuous_nan': alias for gaussian_diag_nan. Supports missing values.
        - 'covariate': covariate model where class probabilities are a multinomial logistic model of the features.
        - 'gaussian': alias for gaussian_unit.
        - 'gaussian_nan': alias for gaussian_unit. Supports missing values.
        - 'gaussian_unit': each gaussian component has unit variance. Only fit the mean.
        - 'gaussian_unit_nan': each gaussian component has unit variance. Only fit the mean. Supports missing values.
        - 'gaussian_spherical': each gaussian component has its own single variance.
        - 'gaussian_spherical_nan': each gaussian component has its own single variance. Supports missing values.
        - 'gaussian_tied': all gaussian components share the same general covariance matrix.
        - 'gaussian_diag': each gaussian component has its own diagonal covariance matrix.
        - 'gaussian_diag_nan': each gaussian component has its own diagonal covariance matrix. Supports missing values.
        - 'gaussian_full': each gaussian component has its own general covariance matrix.
        - 'multinoulli': the observed data consists of n_features multinoulli (categorical) random variables.
        - 'multinoulli_nan': the observed data consists of n_features multinoulli (categorical) random variables. Supports missing values.

        Models suffixed with ``_nan`` support missing values, but may be slower than their fully observed counterpart.

        Alternatively accepts a dict to define a nested model, e.g., 3 gaussian features and 2 binary features. Please
        refer to :class:`stepmix.emission.nested.Nested` for details.

    structural : {'bernoulli', 'bernoulli_nan', 'binary', 'binary_nan', 'categorical', 'categorical_nan', 'continuous', \
    'continuous_nan', 'covariate',\
    'gaussian', 'gaussian_nan', 'gaussian_unit', 'gaussian_unit_nan', 'gaussian_spherical', 'gaussian_spherical_nan',\
    'gaussian_tied', 'gaussian_diag', 'gaussian_diag_nan', 'gaussian_full', 'multinoulli', 'multinoulli_nan', dict},\
    default='bernoulli'
        String describing the structural model. Same options as those for the measurement model.
    assignment : {'soft', 'modal'}, default='modal'
        Class assignments for 3-step estimation.
        Must be one of:

            - 'soft': keep class responsibilities (posterior probabilities) as is.
            - 'modal': assign 1 to the class with max probability, 0 otherwise (one-hot encoding).

    correction : {None, 'BCH', 'ML'}, default=None
        Bias correction for 3-step estimation.
        Must be one of:

            - None : No correction. Run Naive 3-step.
            - 'BCH' : Apply the empirical BCH correction from *Vermunt, 2004*.
            - 'ML' : Apply the ML correction from *Vermunt, 2010; Bakk et al., 2013*.

    abs_tol : float, default=1e-10
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.
    rel_tol : float, default=0.00
        The convergence threshold. EM iterations will stop when the
        relative lower bound average gain is below this threshold.
    max_iter : int, default=1000
        The number of EM iterations to perform.
    n_init : int, default=1
        The number of initializations to perform. The best results are kept.
    save_param_init : bool, default=False
        Save the estimated parameters of all initializations to self.param_buffer\_.
    init_params : {'kmeans', 'random'}, default='random'
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of:

            - 'kmeans' : responsibilities are initialized using kmeans.
            - 'random' : responsibilities are initialized randomly.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters. Pass an int for reproducible output across multiple function calls.
    verbose : int, default=0
        Enable verbose output. If 1, will print detailed report of the model and the performance metrics after fitting.
    progress_bar : int, default=1
        Display a tqdm progress bar during fitting.

            - 0 : No progress bar.
            - 1 : Progress bar for initializations.
            - 2 : Progress bars for initializations and iterations. This requires a nested tqdm bar and may not work\
            properly in some terminals.

    measurement_params: {dict, None}, default=None
        Additional params passed to the measurement model class.  Particularly useful to specify optimization parameters
        for :class:`stepmix.emission.covariate.Covariate`. Ignored if the measurement descriptor is a nested object
        (see :class:`stepmix.emission.nested.Nested`).
    structural_params: {dict, None}, default=None
        Additional params passed to the structural model class.  Particularly useful to specify optimization parameters
        for :class:`stepmix.emission.covariate.Covariate`. Ignored if the structural descriptor is a nested object
        (see :class:`stepmix.emission.nested.Nested`).

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
        The weights of each mixture components.
    _mm : stepmix.emission.Emission
        Measurement model, including parameters and estimation methods.
    _sm : stepmix.emission.Emission
        Structural model, including parameters and estimation methods.
    log_resp_ : ndarray of shape (n_samples, n_components)
        Initial log responsibilities.
    measurement_in_: int
        Number of features in the measurement model.
    structural_in_: int
        Number of features in the structural model.
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.
    lower_bound_ : float
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best fit of EM.
    lower_bound_buffer_ : float
        Lower bound values on the log-likelihood (of the training data with
        respect to the model) of all EM initializations.
    param_buffer_ : list
        Final parameters of all initializations. Only updated if save_param_init=True.
    x_names_ : list
        If input is a DataFrame, column names of X.
    y_names_: list
        If input is a DataFrame, column names of Y.

    Notes
    -----

    References
    ----------

    Bolck, A., Croon, M., and Hagenaars, J. Estimating latent structure models with categorical variables: One-step
    versus three-step estimators. Political analysis, 12(1): 3–27, 2004.

    Vermunt, J. K. Latent class modeling with covariates: Two improved three-step approaches. Political analysis,
    18 (4):450–469, 2010.

    Bakk, Z., Tekle, F. B., and Vermunt, J. K. Estimating the association between latent class membership and external
    variables using bias-adjusted three-step approaches. Sociological Methodology, 43(1):272–311, 2013.

    Bakk, Z. and Kuha, J. Two-step estimation of models between latent classes and external variables. Psychometrika,
    83(4):871–892, 2018

    Examples
    --------
    .. code-block:: python

        from stepmix.datasets import data_bakk_response
        from stepmix.stepmix import StepMix
        # Soft 3-step
        X, Y, _ = data_bakk_response(n_samples=1000, sep_level=.7, random_state=42)
        model = StepMix(n_components=3, n_steps=3, measurement='bernoulli', structural='gaussian_unit', random_state=42, assignment='soft')
        model.fit(X, Y)
        model.score(X, Y)  # Average log-likelihood

        # Equivalently, each step can be performed individually. See the code of the fit method for details.
        model = StepMix(n_components=3, measurement='bernoulli', structural='gaussian_unit', random_state=42)
        model.em(X) # Step 1
        probs = model.predict_proba(X) # Step 2
        model.m_step_structural(probs, Y) # Step 3
        model.score(X, Y)  # Average log-likelihood
    """

    def __init__(
        self,
        n_components=2,
        *,
        n_steps=1,
        measurement="bernoulli",
        structural="gaussian_unit",
        assignment="modal",
        correction=None,
        abs_tol=1e-10,
        rel_tol=0.00,
        max_iter=1000,
        n_init=1,
        save_param_init=False,
        init_params="random",
        random_state=None,
        verbose=0,
        progress_bar=1,
        measurement_params=None,
        structural_params=None,
    ):
        # Attributes of the base StepMix class
        self.n_components = n_components
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.save_param_init = save_param_init
        self.init_params = init_params
        self.random_state = random_state
        self.verbose = verbose
        self.progress_bar = progress_bar
        self.n_steps = n_steps

        # Additional attributes for 3-step estimation
        self.assignment = assignment
        self.correction = correction

        # Additional attributes to specify the measurement and structural models
        self.measurement = measurement
        self.measurement_params = measurement_params
        self.structural = structural
        self.structural_params = structural_params

    ########################################################################################################################
    # INPUT VALIDATION, INITIALIZATIONS AND PARAMETER MANAGEMENT
    def _check_initial_parameters(self, X):
        """Validate class attributes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Raises
        ------
        ValueError : unacceptable choice of parameters

        """
        utils.check_type(
            int,
            n_components=self.n_components,
            max_iter=self.max_iter,
            n_init=self.n_init,
            verbose=self.verbose,
        )
        utils.check_positive(
            n_components=self.n_components,
            max_iter=self.max_iter,
            n_init=self.n_init,
        )
        utils.check_nonneg(
            abs_tol=self.abs_tol, rel_tol=self.rel_tol, verbose=self.verbose
        )
        utils.check_in([1, 2, 3], n_steps=self.n_steps)
        utils.check_in([0, 1, 2], progress_bar=self.progress_bar)
        utils.check_in(["kmeans", "random"], init_params=self.init_params)
        utils.check_in(["modal", "soft"], assignment=self.assignment)
        utils.check_in([None, "BCH", "ML"], correction=self.correction)
        utils.check_in([False, True], save_param_init=self.save_param_init)
        utils.check_descriptor(self.measurement, keys=EMISSION_DICT.keys())
        utils.check_descriptor(self.structural, keys=EMISSION_DICT.keys())
        utils.check_type(
            (dict, type(None)),
            measurement_params=self.measurement_params,
            structural_params=self.structural_params,
        )

        # Check if models support missing values
        # This is passed to sklearn.utils.check_array
        self._force_all_finite_mm = (
            "allow-nan" if utils.check_descriptor_nan(self.measurement) else True
        )
        self._force_all_finite_sm = (
            "allow-nan" if utils.check_descriptor_nan(self.structural) else True
        )

        # Buffer to save the likelihoods of different inits for debugging
        self.lower_bound_buffer_ = list()

        # Buffer to save params of different inits
        self.param_buffer_ = list()

        # Covariate models have special constraints. Check them.
        self._is_covariate = utils.check_covariate(self.measurement, self.structural)

        # Covariate models use a different conditional likelihood (See Bakk and Kuha, 2018), which should
        # not include the marginal likelihood over the latent classes in the E-step
        self._conditional_likelihood = self._is_covariate

    def _initialize_parameters(self, X, random_state):
        """Initialize the weights and measurement model parameters.

        We do not initialize the structural model here, since the StepMix class can be used without one.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

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
            raise ValueError(f"Unimplemented initialization method {self.init_params}.")

        # Save log responsibilities
        self.log_resp_ = np.log(np.clip(resp, 1e-15, 1 - 1e-15))

        # Uniform class weights initialization
        self.weights_ = np.ones((self.n_components,)) / self.n_components

        # Initialize measurement model
        self._initialize_parameters_measurement(X, random_state)

    def _initialize_parameters_measurement(
        self, X, random_state=None, init_emission=True
    ):
        """Initialize parameters of measurement model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        # Initialize measurement model
        if not hasattr(self, "_mm"):
            mm_params = (
                self.measurement_params
                if self.measurement_params is not None
                else dict()
            )
            self._mm = build_emission(
                self.measurement,
                n_components=self.n_components,
                random_state=random_state,
                **mm_params,
            )
        if init_emission:
            self._mm.initialize(X, np.exp(self.log_resp_), random_state)

    def _initialize_parameters_structural(
        self, Y, random_state=None, init_emission=True
    ):
        """Initialize parameters of structural model.

        Parameters
        ----------
        Y : array-like of shape  (n_samples, n_features_structural)
            Structural data.
        """
        # Initialize structural model
        if not hasattr(self, "_sm"):
            sm_params = (
                self.structural_params if self.structural_params is not None else dict()
            )
            self._sm = build_emission(
                self.structural,
                n_components=self.n_components,
                random_state=random_state,
                **sm_params,
            )
        if init_emission:
            self._sm.initialize(Y, np.exp(self.log_resp_), random_state)

    @property
    def n_parameters(self):
        """Get number of free parameters."""
        check_is_fitted(self)

        # Only include class weights dof if they are used for likelihood computations
        # Class weights are not used in the conditional perspective (e.g., covariate models)
        n = 0.0 if self._conditional_likelihood else (self.n_components - 1)

        # Measurement parameters
        n += self._mm.n_parameters

        # Optional structural parameters
        if hasattr(self, "_sm"):
            n += self._sm.n_parameters
        return n

    def _check_x_y(self, X=None, Y=None, reset=False):
        """Input validation function.

        Set reset=True to memorize input sizes for future validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None
        reset : bool, default=False
            Reset input sizes for future validation.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features) or None
            Validated measurement data or None if not provided.

        Y : ndarray of shape (n_samples, n_features_structural) or None
            Validated structural data or None if not provided.

        """
        # We use reset True since we take care of dimensions in this class (and not in the parent)
        if X is not None:
            X_names = utils.extract_column_names(X)
            X = self._validate_data(
                X,
                dtype=[np.float64, np.float32],
                reset=True,
                force_all_finite=self._force_all_finite_mm,
            )
        if Y is not None:
            # Handle 1D Y array
            Y_names = utils.extract_column_names(Y)
            Y = self._validate_data(
                Y,
                dtype=[np.float64, np.float32],
                reset=True,
                ensure_2d=False,
                force_all_finite=self._force_all_finite_sm,
            )

            # Force a matrix format
            if Y.ndim == 1:
                Y = Y.reshape((-1, 1))

        if reset:
            if X is not None:
                self.measurement_in_ = X.shape[1]
                self.n_features_in_ = self.measurement_in_  # Keep sklearn happy
                self.x_names_ = X_names
            if Y is not None:
                self.structural_in_ = Y.shape[1]
                self.y_names_ = Y_names
        else:
            if X is not None and X.shape[1] != self.measurement_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but StepMix is expecting {self.measurement_in_} measurement"
                    f" features as input."
                )

            if (
                Y is not None
                and hasattr(self, "structural_in_")
                and Y.shape[1] != self.structural_in_
            ):
                raise ValueError(
                    f"Y has {Y.shape[1]} features, but StepMix is expecting {self.structural_in_} structural "
                    f"features as input."
                )

        return X, Y

    def get_parameters(self):
        """Get model parameters as a Python dictionary.

        Returns
        -------
        params: dict,
            Nested dict {'weights': Current class weights,\
                         'measurement': dict of measurement params,\
                         'structural': dict of structural params,\
                         'measurement_in': number of measurements,\
                         'structural_in': number of structural features,\
                         }.
        """
        check_is_fitted(self)
        params = dict(
            weights=self.weights_,
            measurement=self._mm.get_parameters(),
            measurement_in=self.measurement_in_,
        )
        if hasattr(self, "_sm"):
            params["structural"] = self._sm.get_parameters()
            params["structural_in"] = self.structural_in_
        return params

    def get_parameters_df(self, x_names=None, y_names=None):
        """Get model parameters as a long-form DataFrame.

        Parameters
        ----------
        x_names : List of str
            Column names of X.
        y_names : List of str
            Column names of Y.

        Returns
        -------
        params: pd.DataFrame
        """
        check_is_fitted(self)

        # Dataframe of the measurement model
        if x_names is None:
            x_names = self.x_names_
        df_mm = self._mm.get_parameters_df(x_names)
        df_mm["model"] = "measurement"
        list_df = [df_mm]

        if not self._conditional_likelihood:
            # No class weights if we are using the conditional perspective
            # Create a dataframe for class weights
            df_class = list()
            for class_no, w in enumerate(self.weights_):
                df_class.append(
                    dict(
                        model="measurement",
                        model_name="class_weights",
                        param="class_weights",
                        class_no=class_no,
                        variable="nan",
                        value=w,
                    )
                )

            df_class = pd.DataFrame.from_records(df_class)
            list_df.append(df_class)

        if hasattr(self, "_sm"):
            if y_names is None:
                y_names = self.y_names_
            df_sm = self._sm.get_parameters_df(y_names)
            df_sm["model"] = "structural"
            list_df.append(df_sm)

        df = pd.concat(list_df)

        return df.set_index(["model", "model_name", "param", "class_no", "variable"])

    def get_cw_df(self, x_names=None, y_names=None):
        """Get class weights as DataFrame with classes as columns.

        Parameters
        ----------
        x_names : List of str
            Column names of X.
        y_names : List of str
            Column names of Y.

        Returns
        -------
        params: pd.DataFrame
        """
        if self._conditional_likelihood:
            raise ValueError(
                "Class weights are not defined when using the conditional likelihood perspective (covariates)."
            )
        df = self.get_parameters_df(x_names, y_names).loc[
            "measurement", "class_weights"
        ]
        return self._pivot_cw(df, aggfunc=np.mean)  # Mean of one value is okay

    def get_mm_df(self, x_names=None, y_names=None):
        """Get measurement model parameters as DataFrame with classes as columns.

        Parameters
        ----------
        x_names : List of str
            Column names of X.
        y_names : List of str
            Column names of Y.

        Returns
        -------
        params: pd.DataFrame
        """
        df = (
            self.get_parameters_df(x_names, y_names)
            .loc["measurement"]
            .drop("class_weights", level=0, errors="ignore")
        )
        return self._pivot_param(df, aggfunc=np.mean)  # Mean of one value is okay

    def get_sm_df(self, x_names=None, y_names=None):
        """Get structural model parameters as DataFrame with classes as columns.

        Parameters
        ----------
        x_names : List of str
            Column names of X.
        y_names : List of str
            Column names of Y.

        Returns
        -------
        params: pd.DataFrame
        """
        if not hasattr(self, "_sm"):
            raise ValueError("No structural model fitted.")
        df = self.get_parameters_df(x_names, y_names).loc["structural"]
        return self._pivot_param(df, aggfunc=np.mean)  # Mean of one value if okay

    def set_parameters(self, params):
        """Set parameters.

        Parameters
        ----------
        params: dict,
            Same format as self.get_parameters().

        """
        self.weights_ = params["weights"]
        random_state = check_random_state(self.random_state)

        if not hasattr(self, "_mm"):
            self._initialize_parameters_measurement(
                None, random_state=random_state, init_emission=False
            )
        self._mm.set_parameters(params["measurement"])
        self.measurement_in_ = params["measurement_in"]

        if "structural" in params.keys():
            if not hasattr(self, "_sm"):
                # Init model without random initializations (we will provide one)
                self._initialize_parameters_structural(
                    None, random_state=random_state, init_emission=False
                )
            self._sm.set_parameters(params["structural"])
            self.structural_in_ = params["structural_in"]

    def permute_classes(self, perm):
        """Permute the latent class and associated parameters of this estimator.

        Effectively remaps latent classes.

        Parameters
        ----------
        perm : ndarray of shape  (n_classes,)
            Integer array representing the target permutation. Should be a permutation of np.arange(n_classes).
        """
        check_is_fitted(self)
        self.weights_ = self.weights_[perm]

        # Permute classes of measurement model
        self._mm.permute_classes(perm)

        # Permute classes of structural model (if any)
        if hasattr(self, "_sm"):
            self._sm.permute_classes(perm)

    #######################################################################################################################
    # ESTIMATION AND EM METHODS
    def fit(self, X, Y=None, sample_weight=None, y=None):
        """Fit StepMix measurement model and optionally the structural model.

        Setting Y=None will fit the measurement model only. Providing both X and Y will fit the full model following
        the self.n_steps argument.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points to fit the measurement model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        Y : array-like of shape (n_samples, n_features_structural), default=None
            List of n_features-dimensional data points to fit the structural model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        y: array-like of shape (n_samples, n_features), default=None
            Alias for Y. Ignored if Y is provided.
        sample_weight : array-like of shape(n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        """
        if y is not None and Y is None:
            Y = y

        if Y is None:
            # No structural data. Simply fit the measurement data
            self.em(X, sample_weight=sample_weight)

        elif self.n_steps == 1:
            # One-step estimation
            # 1) Maximum likelihood with both measurement and structural models
            self.em(X, Y, sample_weight=sample_weight)

        elif self.n_steps == 2:
            # Two-step estimation
            # 1) Fit the measurement model
            self.em(X, sample_weight=sample_weight)
            # 2) Fit the structural model by keeping the parameters of the measurement model fixed
            self.em(X, Y, sample_weight=sample_weight, freeze_measurement=True)

        elif self.n_steps == 3 and self.correction is None:
            # Three-step estimation
            # 1) Fit the measurement model
            self.em(X, sample_weight=sample_weight)
            # 2) Assign class probabilities
            soft_resp = self.predict_proba_class(X)

            # Modal assignment (clipped for numerical reasons)
            # Else we simply keep the assignment as is (soft)
            resp = (
                utils.modal(soft_resp, clip=True)
                if self.assignment == "modal"
                else soft_resp
            )

            # 3) M-step on the structural model
            self.m_step_structural(resp, Y)

        elif self.n_steps == 3 and self.correction == "BCH":
            # Three-step estimation with BCH correction
            # 1) Fit the measurement model
            self.em(X)

            # 2) Assign class probabilities
            soft_resp = self.predict_proba_class(X)

            # Apply BCH correction
            _, D_inv = compute_bch_matrix(soft_resp, self.assignment)

            # Modal assignment (clipped for numerical reasons) or assignment as is (soft)
            resp = (
                utils.modal(soft_resp, clip=True)
                if self.assignment == "modal"
                else soft_resp
            )
            resp = resp @ D_inv

            # 3) M-step on the structural model
            self.m_step_structural(resp, Y)

        elif self.n_steps == 3 and self.correction == "ML":
            # Three-step estimation with ML correction
            # 1) Fit the measurement model
            self.em(X)

            # 2) Assign class probabilities
            soft_resp = self.predict_proba_class(X)

            # Compute log_emission_pm
            log_emission_pm = compute_log_emission_pm(soft_resp, self.assignment)

            # 3) Fit the structural model by keeping the parameters of the measurement model fixed
            self.em(X, Y, freeze_measurement=True, log_emission_pm=log_emission_pm)

        # Print report if required
        if self.verbose == 1:
            self.report(X, Y, sample_weight=sample_weight)

        return self

    def report(self, X, Y=None, sample_weight=None):
        """Print detailed report of the model and performance metrics.


        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None
        sample_weight : array-like of shape(n_samples,), default=None
        """
        x_names = self.x_names_ if hasattr(self, "x_names_") else None
        y_names = self.y_names_ if hasattr(self, "y_names_") else None
        utils.print_report(self, X, Y, sample_weight, x_names, y_names)

    def em(
        self,
        X,
        Y=None,
        sample_weight=None,
        freeze_measurement=False,
        log_emission_pm=None,
    ):
        """EM algorithm to fit the weights, measurement parameters and structural parameters.

        Adapted from the fit_predict method of the sklearn BaseMixture class to include (optional) structural model
        computations.

        Setting Y=None will run EM on the measurement model only. Providing both X and Y will run EM on the complete
        model, unless otherwise specified by freeze_measurement.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None
        sample_weight : array-like of shape(n_samples,), default=None
        freeze_measurement : bool, default =False
            Run EM on the complete model, but do not update measurement model parameters.
            Useful for 2-step estimation and 3-step with ML correction.
        log_emission_pm : ndarray of shape (n, n_components), default=None
            Log probabilities of the predicted class given the true latent class for ML correction.
        """
        self._check_initial_parameters(X)

        # First validate the input and the class attributes
        X, Y = self._check_x_y(X, Y, reset=True)
        n_samples = X.shape[0]

        # If sample weights exist, convert them to array (support for lists)
        # and check length
        # Otherwise set them to 1 for all examples
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)

        if n_samples < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )

        # Set up useful values for optimization
        random_state = check_random_state(self.random_state)
        max_lower_bound = -np.inf
        self.converged_ = False

        # Run multiple restarts
        if self.progress_bar:
            print("Fitting StepMix...")
        if self.progress_bar == 2:
            print(
                "The Iteration bar may update too quickly to be visualized depending on dataset size and StepMix settings.\n"
            )
        tqdm_init = tqdm.trange(
            self.n_init, disable=not self.progress_bar, desc="Initializations (n_init) "
        )
        for init in tqdm_init:
            if not freeze_measurement:
                self._initialize_parameters(X, random_state)  # Measurement model

            if Y is not None:
                self._initialize_parameters_structural(
                    Y, random_state
                )  # Structural Model

            lower_bound = -np.inf

            # EM iterations
            tqdm_iter = tqdm.tqdm(
                range(1, self.max_iter + 1),
                disable=self.progress_bar < 2,
                desc="Iterations (max_iter)    ",
                leave=False,
            )
            for n_iter in tqdm_iter:
                prev_lower_bound = lower_bound

                # E-step
                log_prob_norm, log_resp = self._e_step(
                    X, Y=Y, sample_weight=sample_weight, log_emission_pm=log_emission_pm
                )

                # M-step
                self._m_step(
                    X,
                    np.exp(log_resp),
                    Y,
                    sample_weight=sample_weight,
                    freeze_measurement=freeze_measurement,
                )

                # Likelihood & stopping criterion
                lower_bound = log_prob_norm
                change = lower_bound - prev_lower_bound
                rel_change = change / lower_bound

                # if both an absolute and a relative tolerance threshold are given, the EM algorithm stops
                # as soon as one of them is respected
                if abs(change) < self.abs_tol or abs(rel_change) < self.rel_tol:
                    self.converged_ = True
                    break

                # Ask tqdm to display current max lower bound
                ll = (
                    lower_bound * np.sum(sample_weight)
                    if sample_weight is not None
                    else lower_bound * n_samples
                )
                tqdm_iter.set_postfix(avg_LL=lower_bound, LL=ll)

            # Update buffers
            current_parameters = self.get_parameters()

            if (
                lower_bound > max_lower_bound
                or max_lower_bound == -np.inf
                or np.isnan(max_lower_bound)
            ):
                max_lower_bound = lower_bound
                best_params = current_parameters
                best_n_iter = n_iter

            if self.save_param_init:
                self.param_buffer_.append(copy.deepcopy(current_parameters))

            # Save lower bound
            ll, _ = self._e_step(
                X, Y=Y, sample_weight=sample_weight, log_emission_pm=log_emission_pm
            )
            self.lower_bound_buffer_.append(ll)

            # Ask tqdm to display current max lower bound
            max_ll = (
                max_lower_bound * np.sum(sample_weight)
                if sample_weight is not None
                else max_lower_bound * n_samples
            )
            tqdm_init.set_postfix(max_avg_LL=max_lower_bound, max_LL=max_ll)

        if not self.converged_:
            warnings.warn(
                "Initializations did not converge. "
                "Try different init parameters, "
                "or increase max_iter, abs_tol, rel_tol "
                "or check for degenerate data.",
                ConvergenceWarning,
            )

        self.set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

    def _e_step(self, X, Y=None, sample_weight=None, log_emission_pm=None):
        """E-step of the EM algorithm to compute posterior probabilities.

        Setting Y=None will ignore the structural likelihood.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None
        sample_weight : array-like of shape(n_samples,), default=None
        log_emission_pm : ndarray of shape (n, n_components), default=None
            Log probabilities of the predicted class given the true latent class for ML correction. If provided, the
            measurement model likelihood is ignored and this is used instead.

        Returns
        ----------
        avg_ll: float,
            Weighted average log likelihood over samples .
        log_resp: ndarray of shape (n_samples, n_components)
            Log responsibilities, i.e., log posterior probabilities over the latent classes.
        """
        # Measurement log-likelihood
        if log_emission_pm is not None:
            # Use log probabilities of the predicted class given the true latent class for ML correction
            log_resp = log_emission_pm.copy()
        else:
            # Standard Measurement Log likelihood
            log_resp = self._mm.log_likelihood(X)

        # If Y is not provided, we are only using the MM. Use the full likelihood
        # Also use the full likelihood if the model is not conditional
        if Y is None or not self._conditional_likelihood:
            # Add class prior probabilities
            log_resp += np.log(self.weights_).reshape((1, -1))

        # Add structural model likelihood (if structural data is provided)
        if Y is not None:
            log_resp += self._sm.log_likelihood(Y)

        # Log-likelihood
        ll = logsumexp(log_resp, axis=1)

        # Normalization
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp -= ll.reshape((-1, 1))

        return np.average(ll, weights=sample_weight), log_resp

    def _m_step(self, X, resp, Y=None, sample_weight=None, freeze_measurement=False):
        """M-step of the EM algorithm to compute maximum likelihood estimators

        Update parameters of self._mm (measurement) and optionally self._sm (structural).

        Setting Y=None will ignore the structural likelihood. freeze_measurement allows to only update the
        structural model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        resp : ndarray of shape (n_samples, n_components)
            Responsibilities, i.e., posterior probabilities over the latent classes.
        Y : array-like of shape (n_samples, n_features_structural), default=None
        sample_weight : array-like of shape(n_samples,), default=None
        freeze_measurement : bool, default =False
            Do not update the parameters of the measurement model.

        """
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)

        if not freeze_measurement:
            # Update measurement model parameters
            self.weights_ = np.clip(
                np.average(resp, weights=sample_weight, axis=0), 1e-15, 1 - 1e-15
            )
            self._mm.m_step(X, resp * sample_weight[:, np.newaxis])

        if Y is not None:
            # Update structural model parameters
            self._sm.m_step(Y, resp * sample_weight[:, np.newaxis])

    def m_step_structural(self, resp, Y, sample_weight=None):
        """M-step for the structural model only.

        Handy for 3-step estimation.

        Parameters
        ----------
        resp : ndarray of shape (n_samples, n_components)
            Responsibilities, i.e., posterior probabilities over the latent classes of each point in Y.
        Y : array-like of shape (n_samples, n_features_structural), default=None
        sample_weight : array-like of shape(n_samples,), default=None

        """
        check_is_fitted(self)
        _, Y = self._check_x_y(None, Y, reset=True)

        sample_weight = _check_sample_weight(sample_weight, Y, dtype=Y.dtype, copy=True)

        # For the third step of the 3-step approach
        random_state = check_random_state(self.random_state)
        self._initialize_parameters_structural(Y, random_state=random_state)
        self._sm.m_step(Y, resp * sample_weight[:, np.newaxis])

    def bootstrap(
        self,
        X,
        Y=None,
        n_repetitions=1000,
        sample_weight=None,
        parametric=False,
        sampler=None,
        identify_classes=True,
        progress_bar=True,
        random_state=None,
    ):
        """Parametric or Non-parametric boostrap of this estimator.

        Fit n_repetitions clones of the estimator on resampled datasets.

        If identify_classes=True, repeated parameter estimates are aligned with the class order of the main estimator using
        a permutation search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None
        sample_weight : array-like of shape(n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight. Ignored if parametric=True.
        n_repetitions: int
            Number of repetitions to fit.
        parametric: bool, default=False
            Use parametric bootstrap instead of non-parametric. Data will be generated by sampling the estimator.
        sampler: bool, default=None
            Another fitted estimator to use for sampling instead of the main estimator. Only used for parametric
            bootstrapping.
        identify_classes: bool, default=True
            Run a permutation test to align the classes of the repetitions to the classes of the main estimator. This is
            required if inference on the model parameters is needed, but can be turned off if only the likelihood needs
            to be bootstrapped to save computations.
            progress_bar : bool, default=True
            Display a tqdm progress bar for repetitions.
        progress_bar : bool, default=True
            Display a tqdm progress bar for repetitions.
        random_state : int, default=None
            If none, use self.random_state.
        Returns
        ----------
        samples: DataFrame
            Parameter DataFrame for all repetitions. Follows the convention of StepMix.get_parameters_df() with an additional
            'rep' column.
        rep_stats: DataFrame
            Likelihood statistics of each repetition.
        """
        check_is_fitted(self)
        if random_state is None:
            random_state = self.random_state

        return bootstrap(
            self,
            X,
            Y,
            n_repetitions=n_repetitions,
            sample_weight=sample_weight,
            parametric=parametric,
            sampler=sampler,
            identify_classes=identify_classes,
            progress_bar=progress_bar,
            random_state=random_state,
        )

    def bootstrap_stats(
        self,
        X,
        Y=None,
        n_repetitions=1000,
        sample_weight=None,
        parametric=False,
        progress_bar=True,
    ):
        """Bootstrapping of a StepMix estimator. Obtain boostrapped parameters and some statistics
        (mean and standard deviation).

        If a covariate model is used in the structural model, the output keys "cw_mean" and "cw_std" are omitted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None
        sample_weight : array-like of shape(n_samples,), default=None
        n_repetitions: int
            Number of repetitions to fit.
        parametric: bool, default=False
            Use parametric bootstrap instead of non-parametric. Data will be generated by sampling the estimator.
        progress_bar : bool, default=True
            Display a tqdm progress bar for repetitions.
        Returns
        ----------
        bootstrap_and_stats: dict,
            Dictionary of dataframes {
            'samples': Parameters estimated by self.boostrap in a long-form DataFrame,\
            'rep_stats': Likelihood statistics of each repetition provided by self.boostrap,\
            'cw_mean': Bootstrapped means of the class weights,\
            'cw_std': Bootstrapped standard deviations of the class weights,\
            'mm_mean': Bootstrapped means of the measurement model parameters,\
            'mm_std': Bootstrapped standard deviations of the measurement model parameters,\
            'sm_mean': Bootstrapped means of the structural model parameters,\
            'sm_std': Bootstrapped standard deviations of the structural model parameters,\
            }.
        """
        bootstrap_df, bootstrap_stats = self.bootstrap(
            X=X,
            Y=Y,
            n_repetitions=n_repetitions,
            sample_weight=sample_weight,
            parametric=parametric,
            progress_bar=progress_bar,
        )

        mm_data = bootstrap_df.loc["measurement"].drop(
            index="class_weights", level=0, errors="ignore"
        )

        result = dict()
        result["samples"] = bootstrap_df
        result["rep_stats"] = bootstrap_stats
        result["mm_mean"] = self._pivot_param(mm_data, np.mean)
        result["mm_std"] = self._pivot_param(mm_data, np.std)

        if hasattr(self, "_sm"):
            sm_data = bootstrap_df.loc["structural"]
            result["sm_mean"] = self._pivot_param(sm_data, np.mean)
            result["sm_std"] = self._pivot_param(sm_data, np.std)

        if not self._conditional_likelihood:
            cw_data = bootstrap_df.loc["measurement", "class_weights"]
            result["cw_mean"] = self._pivot_cw(cw_data, np.mean)
            result["cw_std"] = self._pivot_cw(cw_data, np.std)

        return result

    def _pivot_param(self, df, aggfunc=np.mean):
        # Standard pivot function that we reuse for bootstrapping
        return pd.pivot_table(
            df,
            columns="class_no",
            values="value",
            index=["model_name", "param", "variable"],
            aggfunc=aggfunc,
        )

    def _pivot_cw(self, df, aggfunc=np.std):
        # Standard pivot function that we reuse for bootstrapping
        return pd.pivot_table(
            df, columns="class_no", values="value", index=["param"], aggfunc=aggfunc
        )

    ########################################################################################################################
    # INFERENCE
    def score(self, X, Y=None, sample_weight=None):
        """Compute the average log-likelihood over samples.

        Setting Y=None will ignore the structural likelihood.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points to fit the measurement model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        Y : array-like of shape (n_samples, n_features_structural), default=None
            List of n_features-dimensional data points to fit the structural model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        sample_weight : array-like of shape(n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        ----------
        avg_ll: float
            Average log likelihood over samples.
        """
        check_is_fitted(self)
        X, Y = self._check_x_y(X, Y)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)

        avg_ll, _ = self._e_step(X, Y=Y, sample_weight=sample_weight)
        return avg_ll

    def bic(self, X, Y=None):
        """Bayesian information criterion for the current model on the measurement data X and optionally the structural
        data Y.

        Adapted from https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d6dd034403370fea552b21a6776bef18/sklearn/mixture/_base.py

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None

        Returns
        -------
        bic : float
            The lower the better.
        """
        return -2 * self.score(X, Y) * X.shape[0] + self.n_parameters * np.log(
            X.shape[0]
        )

    def aic(self, X, Y=None):
        """Akaike information criterion for the current model on the measurement data X and optionally the structural
        data Y.

        Adapted from https://github.com/scikit-learn/scikit-learn/blob/baf0ea25d6dd034403370fea552b21a6776bef18/sklearn/mixture/_base.py

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X, Y) * X.shape[0] + 2 * self.n_parameters

    def entropy(self, X, Y=None):
        """Entropy of the posterior over latent classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None

        Returns
        -------
        entropy : float
        """
        check_is_fitted(self)
        resp = self.predict_proba_class(X, Y)
        resp = np.clip(resp, 1e-15, 1 - 1e-15)

        return -1 * np.sum(resp * np.log(resp))

    def relative_entropy(self, X, Y=None):
        """Scaled Relative Entropy of the posterior over latent classes.

        Ramaswamy et al., 1993.

        1 - entropy / (n_samples * log(n_components))

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None

        Returns
        -------
        relative_entropy : float
        """
        entropy = self.entropy(X, Y)
        n_samples = X.shape[0]

        rel_entropy = (
            1 - entropy / (n_samples * np.log(self.n_components))
            if self.n_components > 1
            else np.nan
        )
        return rel_entropy

    def sabic(self, X, Y=None):
        """Sample-Sized Adjusted BIC.

        References
        ----------
        Sclove SL. Application of model-selection criteria to some problems in multivariate analysis. Psychometrika. 1987;52(3):333–343.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None

        Returns
        -------
        ssa_bic : float
        """
        n = X.shape[0]

        return -2 * self.score(X, Y) * n + self.n_parameters * np.log(
            n * ((n + 2) / 24)
        )

    def caic(self, X, Y=None):
        """Consistent AIC.

        References
        ----------
        Bozdogan, H. 1987. Model selection and Akaike’s information criterion (AIC):
        The general theory and its analytical extensions. Psychometrika 52: 345–370.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_features_structural), default=None

        Returns
        -------
        caic : float
            The lower the better.
        """
        n = X.shape[0]
        return -2 * self.score(X, Y) * n + self.n_parameters * (np.log(n) + 1)

    def predict_class(self, X, Y=None):
        """Predict the cluster/latent class/component labels for the data samples in X.

        Optionally, an array-like Y can be provided to predict the labels based on both the measurement and structural
        models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points to fit the measurement model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        Y : array-like of shape (n_samples, n_features_structural), default=None
            List of n_features-dimensional data points to fit the structural model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        return self.predict_proba_class(X, Y).argmax(axis=1)

    def predict_proba_class(self, X, Y=None):
        """Predict the latent class probabilities for the data samples in X using the measurement model.

        Optionally, an array-like Y can be provided to predict the posterior based on both the measurement and structural
        models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points to fit the measurement model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        Y : array-like of shape (n_samples, n_features_structural), default=None
            List of n_features-dimensional data points to fit the structural model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            P(class|X, Y) for each sample in X.
        """
        check_is_fitted(self)
        X, Y = self._check_x_y(X, Y)

        _, log_resp = self._e_step(X, Y=Y)
        return np.exp(log_resp)

    def predict_Y(self, X):
        """Call the predict method of the structural model to predict argmax P(Y|X) (Supervised prediction).

        Inference over Y is only supported if the structural model (sm) is set to 'binary', 'binary_nan', 'categorical', or 'categorical_nan'.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points to fit the measurement model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        Returns
        -------
        predictions : array, shape (n_samples, n_columns)
            Y predictions.
        """
        if not hasattr(self, "_sm"):
            raise ValueError("Calling predict_Y requires a structural model.")
        check_is_fitted(self)
        X, _ = self._check_x_y(X, None)

        _, log_resp = self._e_step(X, Y=None)

        return self._sm.predict(log_resp)

    def predict_proba_Y(self, X):
        """Call the predict method of the structural model to predict the full conditional P(Y|X).

        Inference over Y is only supported if the structural model (sm) is set to 'binary', 'binary_nan', 'categorical', or 'categorical_nan'.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points to fit the measurement model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).

        Returns
        -------
        conditional : array, shape (n_samples, n_columns)
            P(Y|X).
        """
        if not hasattr(self, "_sm"):
            raise ValueError("Calling predict_proba_Y requires a structural model.")
        check_is_fitted(self)
        X, _ = self._check_x_y(X, None)

        _, log_resp = self._e_step(X, Y=None)

        return self._sm.predict_proba(log_resp)

    def predict(self, X, Y=None):
        """Predict the cluster/latent class/component labels for the data samples in X.

        Optionally, an array-like Y can be provided to predict the labels based on both the measurement and structural
        models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points to fit the measurement model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        Y : array-like of shape (n_samples, n_features_structural), default=None
            List of n_features-dimensional data points to fit the structural model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        return self.predict_class(X, Y)

    def predict_proba(self, X, Y=None):
        """Predict the latent class probabilities for the data samples in X using the measurement model.

        Optionally, an array-like Y can be provided to predict the posterior based on both the measurement and structural
        models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points to fit the measurement model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        Y : array-like of shape (n_samples, n_features_structural), default=None
            List of n_features-dimensional data points to fit the structural model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            P(class|X, Y) for each sample in X.
        """
        return self.predict_proba_class(X, Y)

    def sample(self, n_samples, labels=None):
        """Sample method for fitted StepMix model.

        Adapted from the sklearn BaseMixture sample method.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        labels : ndarray of shape (n_samples,)
            Predetermined class labels. Will ignore class weights if provided.

        Returns
        -------
        X : array-like of shape (n_samples, n_columns)
            Measurement samples.
        Y : array-like of shape (n_samples, n_columns_structural)
            Structural samples.
        labels : ndarray of shape (n_samples,)
            Ground truth class membership.
        """
        check_is_fitted(self)

        # Validate n_samples argument
        utils.check_type(int, n_samples=n_samples)
        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        # Covariate sampling is not supported
        # You need to first sample input data, then apply the covariate model to infer weights
        if self.structural == "covariate":
            raise NotImplementedError(
                "Sampling for the covariate model is not implemented."
            )

        # Sample
        # Use measurement model generator
        rng = self._mm.random_state
        if labels is None:
            n_samples_comp = rng.multinomial(n_samples, self.weights_)
        else:
            classes, n_samples_comp = np.unique(labels, return_counts=True)

        X = np.vstack(
            [self._mm.sample(c, int(sample)) for c, sample in enumerate(n_samples_comp)]
        )

        if hasattr(self, "_sm"):
            Y = np.vstack(
                [
                    self._sm.sample(c, int(sample))
                    for c, sample in enumerate(n_samples_comp)
                ]
            )
        else:
            Y = None

        # Also return labels
        labels_ret = []
        for i, n in enumerate(n_samples_comp):
            labels_ret += [i] * n
        labels_ret = np.array(labels_ret)

        if labels is not None:
            # Reorder samples according to provided labels
            X_new = np.zeros_like(X)

            if Y is not None:
                # Optional structural data
                Y_new = np.zeros_like(Y)

            for i, c in enumerate(classes):
                mask = labels_ret == i
                mask_labels = labels == c
                X_new[mask_labels] = X[mask]

                if Y is not None:
                    # Optional structural data
                    Y_new[mask_labels] = Y[mask]

            labels_ret = labels
            X = X_new
            if Y is not None:
                # Optional structural data
                Y = Y_new
        else:
            # Shuffle everything
            shuffle_mask = rng.permutation(X.shape[0])
            X, labels_ret = (
                X[shuffle_mask],
                labels_ret[shuffle_mask],
            )

            if Y is not None:
                # Optional structural data
                Y = Y[shuffle_mask]

        return X, Y, labels_ret


class StepMixClassifier(StepMix):
    """StepMix Supervised Classifier

    Identical to a StepMix estimator, but we remap predict and predict_proba
    to perform inference over Y instead of the latent class. This follows the supervised
    learning convention of sklearn.

    We call this a classifier since inference over Y is only supported if the structural
    model (sm) is set to 'binary', 'binary_nan', 'categorical', or 'categorical_nan'.

    Also works with the aliases 'bernoulli', 'bernoulli_nan', 'multinoulli' and 'multinoulli_nan'.
    """

    def _check_initial_parameters(self, X):
        utils.check_in(
            [
                "binary",
                "binary_nan",
                "bernoulli",
                "bernoulli_nan",
                "categorical",
                "categorical_nan",
                "multinoulli",
                "multinoulli_nan",
            ],
            structural=self.structural,
        )
        super()._check_initial_parameters(X)

    def predict(self, X):
        """Call the predict method of the structural model to predict argmax P(Y|X) (Supervised prediction).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points to fit the measurement model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).
        Returns
        -------
        predictions : array, shape (n_samples, n_columns)
            Y predictions.
        """
        return self.predict_Y(X)

    def predict_proba(self, X):
        """Call the predict method of the structural model to predict the full conditional P(Y|X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points to fit the measurement model. Each row
            corresponds to a single data point. If the data is categorical, by default it should be
            0-indexed and integer encoded (not one-hot encoded).

        Returns
        -------
        conditional : array, shape (n_samples, n_columns)
            P(Y|X).
        """
        return self.predict_proba_Y(X)
