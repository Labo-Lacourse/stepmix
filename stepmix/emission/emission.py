"""Emission models.

Encapsulate the M-step and log-likelihood computations of different conditional emission models.
"""

from abc import ABC, abstractmethod
import copy

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from stepmix.utils import check_int, check_positive


class Emission(ABC):
    """Abstract class for Emission models.

    Emission models can be used by the StepMix class for both the structural and the measurement model. Emission instances
    encapsulate maximum likelihood computations for a given model.

    All model parameters should be values of the self.parameters dict attribute. See the Bernoulli and GaussianUnit
    implementations for reference.

    Model parameters should be ndarrays of shape (n_components, ...).
    In other words, the FIRST AXIS should always correspond to the latent class.

    You can save other things to self.parameters, but they must NOT be ndarrays.

    To add an emission model, you must :
        - Inherit from Emission.
        - Implement the m_step, log_likelihood and sample methods.
        - Add a corresponding string in the EMISSION_DICT at the end of emission.py.
        - Update the StepMix docstring for the measurement and structural arguments!

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
    self.n_parameters : int
        Number of free parameters in the model.

    """

    def __init__(self, n_components, random_state):
        self.n_components = n_components
        self.random_state = check_random_state(random_state)

        # Dict including all parameters for estimation
        self.parameters = dict()

        # String describing model
        self.model_str = "emission"

    def check_parameters(self):
        """Validate class attributes."""
        check_int(n_components=self.n_components)
        check_positive(n_components=self.n_components)

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
        self.random_state = check_random_state(random_state)

        # Measurement and structural models are initialized by running their M-step on the initial log responsibilities
        # obtained via kmeans or sampled uniformly (See StepMix._initialize_parameters)
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
        """Update model parameters via maximum likelihood using the current responsibilities.

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
        X : ndarray of shape (n_samples, n_columns)
            Input data for this emission model.

        Returns
        -------
        ll : ndarray of shape (n_samples, n_components)
            Log-likelihood of the input data conditioned on each component.

        """
        raise NotImplementedError

    def predict_proba(self, log_resp):
        """Compute the conditional probabilities P(Y|X) given the log responsibilities P(Z|X).

        This will only be used if the emission model is used as a structural model. X therefore represents the input
        and Y the output for supervised predictions.

        Parameters
        ----------
        log_resp : ndarray of shape (n_samples, n_components)
            Logarithm of the posterior probabilities P(Z|X) (or responsibilities) of each sample.

        Returns
        -------
        resp : ndarray of shape (n_samples, n_columns)
            Conditional probabilities P(Y|X) of each sample.
        """
        raise NotImplementedError("This emission model does not support predictions.")

    def predict(self, log_resp):
        """Compute argmax P(Y|X) given the log responsibilities P(Z|X) for supervised predictions.

        This will only be used if the emission model is used as a structural model. X therefore represents the input
        and Y the output for supervised predictions.

        Parameters
        ----------
        log_resp : ndarray of shape (n_samples, n_components)
            Logarithm of the posterior probabilities P(Z|X) (or responsibilities) of each sample.

        Returns
        -------
        resp : ndarray of shape (n_samples, n_columns)
            Argmax P(Y|X) of each sample.
        """
        raise NotImplementedError("This emission model does not support predictions.")

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
        samples : ndarray of shape (n_samples, n_columns)
            Samples

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_parameters(self):
        """Number of free parameters in the model."""
        raise NotImplementedError

    def permute_classes(self, perm):
        """Permute the latent class and associated parameters of this estimator.

        Effectively remaps latent classes.

        Parameters
        ----------
        perm : ndarray of shape  (n_classes,)
            Integer array representing the target permutation. Should be a permutation of np.arange(n_classes).
        axis: int
            Axis to use for permuting the parameters.

        """
        for key, item in self.parameters.items():
            if isinstance(item, np.ndarray):
                self.parameters[key] = item[perm]

    def print_parameters(
        self,
        indent=1,
        feature_names=None,
        index=["param", "variable"],
        columns=["model_name", "class_no"],
        model_name=None,
    ):
        """Print parameters with nice formatting.

        This method works well for emission models
        where self.parameters[key_0] is a ndarray of shape (n_components, n_features) and
        key_0 is the only key.

        Parameters
        ----------
        indent : int
            Add indent to print.
        features_names: List of str
            Variable names.
        index: List of str
            Column names in self.get_parameters_df to use as index in the displayed dataframe.
        columns: List of str
            Column names in self.get_parameters_df to use as columns in the displayed dataframe.
        model_name: str
            str to display as model name.
        """
        indent_string = "     " * indent
        df = self.get_parameters_df(feature_names)
        if model_name is not None:
            df["model_name"] = model_name
        df = pd.pivot_table(df, index=index, columns=columns, values="value")
        print(
            indent_string + df.round(4).to_string().replace("\n", "\n" + indent_string)
        )

    def get_parameters_df(self, feature_names=None):
        """Return self.parameters into a long dataframe.

        Columns should be ["model_name", "param", "class_no", "variable", "value"].

        Call self._to_df or implement custom method."""
        return self._to_df(
            param_dict=self.parameters,
            keys=list(self.parameters.keys()),
            feature_names=feature_names,
        )

    def get_default_feature_names(self, n_features):
        feature_names = [f"feature_{i}" for i in range(n_features)]
        return feature_names

    def _to_df(self, param_dict, keys, feature_names=None):
        """Unpack param_dict into a long dataframe.

        This is a generic method that can be used for all emission models
        where the values in param_dict are ndarrays of shape (n_components, n_features).

        Other emission models should implement their own method, but still return a dataframe
        with the same columns.

        Parameters
        ----------
        param_dict: dict
            Dict of parameters as structured in self.parameters.
        keys : list of str
            Keys to process in self.parameters.
        feature_names : list of str, default=None
            Variable names.

        Returns
        -------
        params : pd.DataFrame

        """
        n_features = param_dict[keys[0]].shape[1]
        if feature_names is None:
            feature_names = self.get_default_feature_names(n_features)

        params = list()
        for key in keys:
            for k in range(self.n_components):
                for n_i in range(n_features):
                    params.append(
                        dict(
                            model_name=self.model_str,
                            param=key,
                            class_no=k,
                            variable=feature_names[n_i],
                            value=param_dict[key][k, n_i],
                        )
                    )

        return pd.DataFrame.from_records(params)
