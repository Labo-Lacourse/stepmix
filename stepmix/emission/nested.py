import numpy as np

from .emission import Emission


class Nested(Emission):
    """Nested emission model.

    The descriptor must be a dict of dicts, where the nested dicts hold arguments for nested models. Each nested dict
    is expected to have a model key referring to a valid emission model as well as
    an n_features key describing the number of features associated with that model.
    For example, a model where the first 3 features are gaussian with unit variance, the next 2 are binary and
    the last 4 are covariates would be described likeso :
    >>> descriptor = {
    >>>    'model_1': {
    >>>            'model': 'gaussian_unit',
    >>>            'n_features':3
    >>>     },
    >>>    'model_2': {
    >>>            'model': 'binary',
    >>>            'n_features': 2
    >>>     },
    >>>    'model_3': {
    >>>            'model': 'covariate',
    >>>            'n_features': 4,
    >>>            'method': "newton-raphson",
    >>>            'lr': 1e-3,
    >>>     }
    >>> }

    The above model would then expect an n_samples x 9 matrix as input (9 = 3 + 2 + 4) where columns follow the same
    order of declaration (i.e., the columns of model_1 are first, columns of model_2 come after etc.).

    As demonstrated by the covariate argument, additional arguments can be specified and are passed to the
    associated Emission class.
    """

    def __init__(self, descriptor, emission_dict, n_components, random_state, **kwargs):
        super(Nested, self).__init__(n_components, random_state)
        self.models = dict()
        self.features_per_model = list()
        self.n_components = n_components
        self.random_state = random_state

        # Build the nested models
        for key, item in descriptor.items():
            # Read in model type and the number of features. Other keys are used as arguments
            model = item.pop('model')
            n_features = item.pop('n_features')

            # Build model
            m = emission_dict[model](n_components=self.n_components, random_state=self.random_state, **item)

            # Save model and features
            self.models[key] = m
            self.features_per_model.append(n_features)

    def m_step(self, X, resp):
        i = 0
        for m, range_ in zip(self.models.values(), self.features_per_model):
            # Slice columns to call the m-step only on the appropriate features
            m.m_step(X[:, i:i + range_], resp)
            i += range_

    def log_likelihood(self, X):
        i = 0
        log_eps = np.zeros((X.shape[0], self.n_components))
        for m, range_ in zip(self.models.values(), self.features_per_model):
            # Slice columns to compute the log-likelihood only on the appropriate columns
            log_eps += m.log_likelihood(X[:, i:i + range_])
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

    def print_parameters(self, indent=1):
        for m in self.models.values():
            m.print_parameters(indent)

    @property
    def n_parameters(self):
        n = 0
        for m in self.models.values():
            n = m.n_parameters
        return n



















































