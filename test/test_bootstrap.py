import numpy as np
import pandas as pd
import pytest

from stepmix import StepMix
from stepmix.emission.build_emission import EMISSION_DICT
from stepmix.bootstrap import find_best_permutation, blrt_sweep


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

    The data may not make sense for the model. We therefore do not test a particular output here.
    """
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
@pytest.mark.parametrize("parametric", [False, True])
def test_bootstrap(data, kwargs, model, parametric):
    """Call the boostrap procedure on all models and make sure they don't raise errors.

    The data may not make sense for the model. We therefore do not test a particular output here.
    """
    X, Y = data

    # Use gaussians in the structural model, all other models are tested on the measurement data
    if model.startswith("gaussian") or model.startswith("continuous"):
        kwargs["measurement"] = "binary"
        kwargs["structural"] = model
    else:
        kwargs["measurement"] = model
        kwargs["structural"] = "gaussian_unit"

    model_1 = StepMix(n_steps=1, **kwargs)
    model_1.fit(X, Y)

    if parametric and model == "covariate":
        # Parametric bootstrap is not implemented for covariate model
        with pytest.raises(NotImplementedError) as e_info:
            model_1.bootstrap_stats(X, Y, parametric=parametric, n_repetitions=3)
    else:
        model_1.bootstrap_stats(X, Y, parametric=parametric, n_repetitions=3)


def test_nested_bootstrap(data_nested, kwargs_nested):
    """Call bootstrap procedure on a nested model and make sure it doesn't raise errors."""
    model_1 = StepMix(**kwargs_nested)
    model_1.fit(data_nested, data_nested)
    model_1.bootstrap(data_nested, data_nested, n_repetitions=3)


def test_bootstrap_df(data_nested, kwargs_nested):
    """Call bootstrap procedure on a DataFrame."""
    data_nested = pd.DataFrame(data_nested)
    model_1 = StepMix(**kwargs_nested)
    model_1.fit(data_nested, data_nested)
    model_1.bootstrap(data_nested, data_nested, n_repetitions=3)


@pytest.mark.parametrize("parametric", [True, False])
def test_bootstrap_seed(data_nested, kwargs_nested, parametric):
    """Call bootstrap twice and make sure the results are the same if seeded.

    Identify classes should not affect stats either."""
    data_nested = pd.DataFrame(data_nested)
    kwargs_nested["verbose"] = 0

    model_1 = StepMix(**kwargs_nested)  # Base model is seeded

    model_1.fit(data_nested, data_nested)

    _, stats = model_1.bootstrap(
        data_nested,
        data_nested,
        n_repetitions=10,
        parametric=parametric,
        identify_classes=True,
    )

    _, stats2 = model_1.bootstrap(
        data_nested,
        data_nested,
        n_repetitions=10,
        parametric=parametric,
        identify_classes=False,
    )

    assert np.all(stats == stats2)


def _test_bootstrap_seed_sampler(data_nested, kwargs_nested):
    """Call bootstrap twice with sampler and make sure the results are the same if seeded."""
    data_nested = pd.DataFrame(data_nested)

    model_1 = StepMix(**kwargs_nested)  # Base model is seeded
    model_1.fit(data_nested, data_nested)

    kwargs_nested["n_components"] = 2
    sampler = StepMix(**kwargs_nested)
    sampler.fit(data_nested, data_nested)

    _, stats0 = model_1.bootstrap(
        data_nested, data_nested, n_repetitions=10, parametric=True, sampler=model_1
    )

    _, stats1 = model_1.bootstrap(
        data_nested, data_nested, n_repetitions=10, parametric=True, sampler=sampler
    )

    _, stats2 = model_1.bootstrap(
        data_nested, data_nested, n_repetitions=10, parametric=True, sampler=sampler
    )

    assert not np.all(stats0 == stats1)
    assert np.all(stats1 == stats2)


@pytest.mark.parametrize("parametric", [True, False])
def test_bootstrap_mm(data_nested, kwargs_nested, parametric):
    """Call bootstrap with mm only and make sure it does not raise errors."""
    data_nested = pd.DataFrame(data_nested)

    model_1 = StepMix(**kwargs_nested)

    model_1.fit(data_nested)

    model_1.bootstrap(data_nested, n_repetitions=10, parametric=parametric)


def test_bootstrap_mm_sampler(data_nested, kwargs_nested):
    """Call sampler bootstrap with mm only and make sure it does not raise errors."""
    data_nested = pd.DataFrame(data_nested)

    model_1 = StepMix(**kwargs_nested)

    model_1.fit(data_nested)

    kwargs_nested["n_components"] = 2
    sampler = StepMix(**kwargs_nested)
    sampler.fit(data_nested)

    model_1.bootstrap(data_nested, n_repetitions=10, parametric=True, sampler=sampler)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_blrt(data, kwargs):
    X, Y = data
    model = StepMix(**kwargs)
    df = blrt_sweep(model, X, Y, n_repetitions=10, random_state=42, low=1, high=4)
    sign = np.array(df["p"]) < 0.05

    assert np.all(sign == np.array([True, True, False]))


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_blrt_2(data_g_binary, kwargs_data_g_binary):
    X, Y = data_g_binary
    model = StepMix(**kwargs_data_g_binary)
    df = blrt_sweep(model, X, None, n_repetitions=10, random_state=42, low=2, high=5)
    sign = np.array(df["p"]) < 0.05

    assert np.all(sign == np.array([True, True, False]))
