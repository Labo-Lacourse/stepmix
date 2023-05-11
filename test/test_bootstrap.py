import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from stepmix import StepMix
from stepmix.emission.build_emission import EMISSION_DICT
from stepmix.bootstrap import find_best_permutation, bootstrap, plot_all_parameters_CI


def test_find_best_permutation():
    ref = np.zeros((3, 3))
    ref[:, 0] = 1
    target = np.zeros((3, 3))
    target[:, 1] = 1
    perm_1 = find_best_permutation(ref, ref)
    perm_2 = find_best_permutation(ref, target)

    assert np.all(perm_1 == np.arange(3))
    assert np.all(perm_2 == np.array([1, 0, 2]))


def fit_and_test_three_permutations(estimator, X, Y):
    estimator.fit(X, Y)
    ll_1 = estimator.score(X, Y)  # Average log-likelihood
    preds_1 = estimator.predict_proba(X, Y)  # Class predictions

    estimator.permute_classes(np.array([2, 0, 1]))

    ll_2 = estimator.score(X, Y)  # Average log-likelihood
    preds_2 = estimator.predict_proba(X, Y)  # Class predictions

    estimator.permute_classes(np.array([1, 2, 0]))

    ll_3 = estimator.score(X, Y)  # Average log-likelihood
    preds_3 = estimator.predict_proba(X, Y)  # Class predictions

    assert ll_1 == ll_2 == ll_3  # Likelihoods should be invariant
    assert not np.all(
        preds_1 == preds_2
    )  # Second permutation should have different posterior
    assert np.all(
        preds_1 == preds_3
    )  # Third permutation should be back to original class order


@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
@pytest.mark.parametrize("model", EMISSION_DICT.keys())
def test_permutation(data, kwargs, model):
    """Fit all emission models, then permute latent classes and check if we still output the same likelihood.

    The data may not make sense for the model. We therefore do not test a particular output here."""
    X, Y = data

    # Use gaussians in the structural model, all other models are tested on the measurement data
    if model.startswith("gaussian") or model.startswith("continuous"):
        kwargs["measurement"] = "binary"
        kwargs["structural"] = model
    else:
        kwargs["measurement"] = model
        kwargs["structural"] = "gaussian_unit"

    model_1 = StepMix(n_steps=1, **kwargs)
    fit_and_test_three_permutations(model_1, X, Y)


def test_nested_permutation(data_nested, kwargs_nested):
    """Test permutation of nested model."""
    # We use the same measurement and structural models, both nested
    model_3 = StepMix(n_steps=1, **kwargs_nested)

    fit_and_test_three_permutations(model_3, data_nested, data_nested)


@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
@pytest.mark.parametrize("model", EMISSION_DICT.keys())
def test_bootstrap(data, kwargs, model):
    """Call the boostrap procedure on all models and make sure they don't raise errors.

    The data may not make sense for the model. We therefore do not test a particular output here."""
    X, Y = data

    # Use gaussians in the structural model, all other models are tested on the measurement data
    if model.startswith("gaussian") or model.startswith("continuous"):
        kwargs["measurement"] = "binary"
        kwargs["structural"] = model
    else:
        kwargs["measurement"] = model
        kwargs["structural"] = "gaussian_unit"

    model_1 = StepMix(n_steps=1, **kwargs)
    model_1, params = bootstrap(model_1, X, Y, n_repetitions=3)

    # Also test plots
    if model != "gaussian_full":
        # Plotting not supported for gaussian_full
        figures = plot_all_parameters_CI(model_1.get_parameters(), params)
        for f in figures:
            plt.close(f)


def test_nested_bootstrap(data_nested, kwargs_nested):
    """Call bootstrap procedure on a nested model and make sure it doesn't raise errors."""
    model_1 = StepMix(**kwargs_nested)
    model, params = bootstrap(model_1, data_nested, data_nested, n_repetitions=3)
    assert isinstance(params["measurement"]["model_1"]["pis"][0], np.ndarray)

    # Also test plots
    figures = plot_all_parameters_CI(model.get_parameters(), params)
    for f in figures:
        plt.close(f)


def test_bootstrap_df(data_nested, kwargs_nested):
    """Call bootstrap procedure on a DataFrame."""
    data_nested = pd.DataFrame(data_nested)
    model_1 = StepMix(**kwargs_nested)
    model, params = bootstrap(model_1, data_nested, data_nested, n_repetitions=3)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_bootstrap_plots_n_parameters(data_nested, kwargs_nested, n):
    """Make sure the 3-column grid in bootstrap plots does not raise errors with 1, 2, 3, 4, 5, 6 plots."""
    # Iterate over different number of parameters in the binary model
    kwargs_nested["measurement"]["model_1"]["n_columns"] = n
    kwargs_nested["structural"]["model_1"]["n_columns"] = n

    # Pick only n features for measurement and the last for structural
    data_nested = data_nested[:, list(range(n)) + [-1]]

    model_1 = StepMix(**kwargs_nested)
    model, params = bootstrap(model_1, data_nested, data_nested, n_repetitions=3)
    figures = plot_all_parameters_CI(model.get_parameters(), params)
    for f in figures:
        plt.close(f)
