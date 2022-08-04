import numpy as np
import copy

import pytest

from stepmix.stepmix import StepMix
from stepmix.emission.build_emission import EMISSION_DICT


@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning")  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning")  # Ignore convergence warnings for same reason
@pytest.mark.parametrize("model", EMISSION_DICT.keys())
def test_emissions(data, kwargs, model):
    """Fit all emission models with verbose output (to test their print statements).

    The data may not make sense for the model. We therefore do not test a particular output here."""
    X, Y = data

    # Use gaussians in the structural model, all other models are tested on the measurement data
    if model.startswith('gaussian'):
        kwargs['measurement'] = 'binary'
        kwargs['structural'] = model
    else:
        kwargs['measurement'] = model
        kwargs['structural'] = 'gaussian_unit'

    model_1 = StepMix(n_steps=1, **kwargs)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)  # Average log-likelihood
    preds_1 = model_1.predict(X, Y)  # Class predictions

    if model == 'covariate':
        # We do not expect covariate to have a sampling method
        with pytest.raises(NotImplementedError) as e_info:
            model_1.sample(100)
    else:
        # Test sampling
        model_1.sample(100)


@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("method", ["gradient", "newton-raphson"])
def test_covariate(data_covariate, kwargs, method, intercept):
    X, Z = data_covariate

    # Replace measurement model with the emission model we aim to test
    kwargs['structural'] = 'covariate'

    model_1 = StepMix(n_steps=1, structural_params=dict(method=method, intercept=intercept, lr=1e-2, max_iter=1000),
                      **kwargs)
    model_1.fit(X, Z)
    ll_1 = model_1.score(X, Z)  # Average log-likelihood


def test_nested(data, kwargs):
    """Test verbose output and sampling of nested model."""
    X, Y = data

    # For this test, ignore the measurement and structural keys in kwargs
    kwargs.pop('measurement')
    kwargs.pop('structural')

    # Binary + Gaussian measurement
    descriptor = {
        'model_1': {
            'model': 'bernoulli',
            'n_features': 6
        },
        'model_2': {
            'model': 'gaussian_unit',
            'n_features': 1
        }
    }

    # Merge data in single matrix
    Z = np.hstack((X, Y))

    # We use the same measurement and structural models, both nested
    model_3 = StepMix(measurement=copy.deepcopy(descriptor), structural=copy.deepcopy(descriptor), **kwargs)
    model_3.fit(Z, Z)
    ll_3 = model_3.score(Z, Z)
    pred_3 = model_3.predict(Z, Z)

    # Test sampling
    model_3.sample(100)
