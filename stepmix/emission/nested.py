import numpy as np

from .emission import Emission


class Nested(Emission):
    """Nested emission model.

    The descriptor must be a dict of string-int pairs. For example, a model where
    the first 3 columns are gaussian with unit variance, the next 2 are binary and
    the last 4 are covariates would be described likeso :
    >>> param = {
    >>>'gaussian': 3,
    >>>'binary': 2,
    >>>'covariate':4
    >>> }

    Alternatively, the integer can be replaced with a nested dict to specify some emission arguments. In this
    case, the dict is expected to have an n_columns key to specify the number of associated features.

    """

    def __init__(self, descriptor, emission_dict, n_components, random_state, **kwargs):
        super(Nested, self).__init__(n_components, random_state)
        self.models = list()
        self.features_per_model = list()
        self.n_components = n_components
        self.random_state = random_state

        # Build the nested models
        for key, item in descriptor.items():
            # Check if the item is a dict. If so, seek
            # the number of features for this model
            if isinstance(item, dict):
                n_features = item.pop('n_columns')

                # Other arguments are passed to the nested model
                args = item
            else:
                # Assume item is an int describing the number of features
                n_features = item
                args = dict()

            # Build model
            m = emission_dict[key](n_components=self.n_components, random_state=self.random_state, **args)

            # Save model and features
            self.models.append(m)
            self.features_per_model.append(n_features)

    def m_step(self, X, resp):
        i = 0
        for m, range_ in zip(self.models, self.features_per_model):
            # Slice columns to call the m-step only on the appropriate features
            m.m_step(X[:, i:i + range_], resp)
            i += range_

    def log_likelihood(self, X):
        i = 0
        log_eps = np.zeros((X.shape[0], self.n_components))
        for m, range_ in zip(self.models, self.features_per_model):
            # Slice columns to compute the log-likelihood only on the appropriate columns
            log_eps += m.log_likelihood(X[:, i:i + range_])
            i += range_

        return log_eps

    def sample(self, class_no, n_samples):
        acc = list()
        for m in self.models:
            acc.append(m.sample(class_no, n_samples))

        return np.hstack(acc)

    def get_parameters(self):
        parameters = dict()
        for i, m in enumerate(self.models):
            parameters[i] = m.get_parameters()
        return parameters

    def set_parameters(self, parameters):
        for key, item in parameters.items():
            self.models[key].set_parameters(item)

    def print_parameters(self, indent):
        for m in self.models:
            m.print_parameters(indent)

    @property
    def n_parameters(self):
        n = 0
        for m in self.models:
            n = m.n_parameters
        return n



















































