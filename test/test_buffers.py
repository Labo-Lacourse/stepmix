"""Test likelihood and parameter buffers of the main estimator."""

import pytest
from stepmix.stepmix import StepMix


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
def test_buffers(data_large, kwargs_large):
    X, Y = data_large
    kwargs_large["n_init"] = 20
    kwargs_large["max_iter"] = 5

    model_1 = StepMix(n_steps=1, save_param_init=True, **kwargs_large)
    model_1.fit(X, Y)

    params = model_1.param_buffer_
    lls = model_1.lower_bound_buffer_

    assert len(params) == 20
    assert len(lls) == 20

    for ll, p in zip(lls, params):
        model_1.set_parameters(p)
        ll_i = model_1.score(X, Y)
        assert ll == ll_i
