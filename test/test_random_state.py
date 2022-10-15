"""Test seeding."""
import pytest

import numpy as np

from stepmix.stepmix import StepMix


@pytest.mark.parametrize("n_steps", [1, 2, 3])
def test_random_state(data_covariate, kwargs_covariate, n_steps):
    """Check that results are reproducible when fixing the random_state.

    Two separate instances with the same seed should yield the same result."""
    X, Y = data_covariate

    # Instance 1
    model_1 = StepMix(n_steps=n_steps, **kwargs_covariate)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)  # Average log-likelihood

    # Instance 2
    model_2 = StepMix(n_steps=n_steps, **kwargs_covariate)
    model_2.fit(X, Y)
    ll_2 = model_2.score(X, Y)

    assert ll_1 == ll_2


@pytest.mark.parametrize("n_steps", [1, 2, 3])
def test_random_state_same(data_covariate, kwargs_covariate, n_steps):
    """Check that results are reproducible when fixing the random_state.

    Two calls to the fit method of a seeded estimator should yield the same result."""
    X, Y = data_covariate

    # First try
    model_1 = StepMix(n_steps=n_steps, **kwargs_covariate)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)  # Average log-likelihood

    # Second try
    model_1.fit(X, Y)
    ll_2 = model_1.score(X, Y)

    assert ll_1 == ll_2


def test_random_state_none(data, kwargs):
    """Make sure random_state=None does not raise error (yes, this happened)."""
    X, Y = data

    kwargs.pop("random_state")

    # Instance 1
    model_1 = StepMix(n_steps=1, random_state=None, **kwargs)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)  # Average log-likelihood


@pytest.mark.parametrize("n_init_1,n_init_2", [(1, 2), (1, 10), (1, 100)])
def test_inits(data_covariate, kwargs_covariate, n_init_1, n_init_2):
    """Check that log likelihood is >= when increasing n_inits."""
    X, Y = data_covariate

    # For this test, ignore default n_inits
    kwargs_covariate.pop("n_init")

    model_1 = StepMix(n_steps=1, n_init=n_init_1, **kwargs_covariate)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)  # Average log-likelihood

    model_2 = StepMix(n_steps=1, n_init=n_init_2, **kwargs_covariate)
    model_2.fit(X, Y)
    ll_2 = model_2.score(X, Y)  # Average log-likelihood

    assert ll_1 <= ll_2


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
def test_diff_inits(data_covariate, kwargs_covariate):
    """Make sure that different inits of the same call to fit are indeed different.

    This could happen if the model resets the rng in between inits."""
    X, Y = data_covariate

    # For this test, ignore default n_inits
    kwargs_covariate.pop("n_init")
    kwargs_covariate[
        "max_iter"
    ] = 1  # Few iterations to make convergence to same point unlikely

    model_1 = StepMix(n_steps=1, n_init=1, **kwargs_covariate)
    model_1.fit(X, Y)
    ll_1 = np.array(model_1.lower_bound_buffer_)

    model_2 = StepMix(n_steps=1, n_init=2, **kwargs_covariate)
    model_2.fit(X, Y)
    ll_2 = np.array(model_2.lower_bound_buffer_)

    model_3 = StepMix(n_steps=1, n_init=5, **kwargs_covariate)
    model_3.fit(X, Y)
    ll_3 = np.array(model_3.lower_bound_buffer_)

    # Make sure the first init of all models is the same (properly seeded)
    assert ll_1[0] == ll_2[0] == ll_3[0]

    # Make sure that the different inits of model_2 and model_3 have different likelihoods, i.e., each init is
    # a bit different
    assert not np.all(ll_2 == ll_2[0])
    assert not np.all(ll_3 == ll_3[0])
