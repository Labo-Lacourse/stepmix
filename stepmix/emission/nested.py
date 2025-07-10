"""Nested emission model with support for multiple random variables."""

import copy

import numpy as np
import pandas as pd

from .emission import Emission


class Nested(Emission):
    """Nested emission model.

    The descriptor must be a dict of dicts, where the nested dicts hold arguments for nested models. Each nested dict
    is expected to have a model key referring to a valid emission model as well as
    an n_columns key describing the number of columns (i.e. features for univariate variables or
    features*n_outcomes for one-hot encoded variables) associated with that model.
    For example, a model where the first 3 features are gaussian with unit variance, the next 3 are multinoulli
    with 5 possible outcomes (for a total of 3*5=15 columns) and the last 4 are covariates would be described likeso :

    .. code-block:: python

        descriptor = {
           'model_1': {
                   'model': 'gaussian_unit',
                   'n_columns':3
            },
           'model_2': {
                   'model': 'multinoulli',
                   'n_columns': 15,
                   'n_outcomes': 5
            },
           'model_3': {
                   'model': 'covariate',
                   'n_columns': 4,
                   'method': "newton-raphson",
                   'lr': 1e-3,
            }
        }

    The above model would then expect an n_samples x 22 matrix as input (3 + 15 + 4 = 22) where columns follow the same
    order of declaration (i.e., the columns of model_1 are first, columns of model_2 come after etc.).

    As demonstrated by the covariate argument, additional arguments can be specified and are passed to the
    associated Emission class. Particularly useful to specify optimization parameters for
    :class:`stepmix.emission.covariate.Covariate`.
    """

    def __init__(self, descriptor, emission_dict, n_components, random_state, **kwargs):
        super(Nested, self).__init__(n_components, random_state)
        descriptor = copy.deepcopy(
            descriptor
        )  # Make sure we copy descriptor to avoid affecting original
        self.models = dict()
        self.columns_per_model = list()
        self.n_components = n_components
        self.random_state = random_state

        # Build the nested models
        for key, item in descriptor.items():
            # Read in model type and the number of features. Other keys are used as arguments
            model = item.pop("model")
            n_columns = item.pop("n_columns")

            # Build model
            m = emission_dict[model](
                n_components=self.n_components, random_state=self.random_state, **item
            )

            # Save model and features
            self.models[key] = m
            self.columns_per_model.append(n_columns)

    def initialize(self, X, resp, random_state=None):
        i = 0
        for m, range_ in zip(self.models.values(), self.columns_per_model):
            # Slice columns to call the m-step only on the appropriate features
            m.initialize(X[:, i : i + range_], resp, random_state)
            i += range_

    def m_step(self, X, resp):
        i = 0
        for m, range_ in zip(self.models.values(), self.columns_per_model):
            # Slice columns to call the m-step only on the appropriate features
            m.m_step(X[:, i : i + range_], resp)
            i += range_

    def log_likelihood(self, X):
        i = 0
        log_eps = np.zeros((X.shape[0], self.n_components))
        for m, range_ in zip(self.models.values(), self.columns_per_model):
            # Slice columns to compute the log-likelihood only on the appropriate columns
            log_eps += m.log_likelihood(X[:, i : i + range_])
            i += range_

        return log_eps

    def sample(self, class_no, n_samples):
        acc = list()
        for m in self.models.values():
            acc.append(m.sample(class_no, n_samples))

        return np.hstack(acc)

    def get_parameters(self):
        parameters = dict()
        for key, m in self.models.items():
            parameters[key] = m.get_parameters()
        return parameters

    def set_parameters(self, parameters):
        for key, item in parameters.items():
            self.models[key].set_parameters(item)

    def print_parameters(self, indent=1, feature_names=None):
        if feature_names is None:
            n_columns = sum(self.columns_per_model)
            feature_names = self.get_default_feature_names(n_columns)

        i = 0
        for name, m, range_ in zip(
            self.models.keys(), self.models.values(), self.columns_per_model
        ):
            # Slice parameter names to get the right column names for this submodel
            f_i = feature_names[i : i + range_]
            m.print_parameters(indent, model_name=name, feature_names=f_i)
            i += range_
            print("\n")

    @property
    def n_parameters(self):
        n = 0
        for m in self.models.values():
            n += m.n_parameters
        return n

    def permute_classes(self, perm, axis=0):
        for key, item in self.models.items():
            self.models[key].permute_classes(perm)

    def get_parameters_df(self, feature_names=None):
        df_list = list()
        if feature_names is None:
            n_columns = sum(self.columns_per_model)
            feature_names = self.get_default_feature_names(n_columns)

        i = 0
        for name, m, range_ in zip(
            self.models.keys(), self.models.values(), self.columns_per_model
        ):
            # Slice parameter names to get the right column names for this submodel
            f_i = feature_names[i : i + range_]
            df_i = m.get_parameters_df(f_i)
            df_i["model_name"] = name
            df_list.append(df_i)
            i += range_

        return pd.concat(df_list)
