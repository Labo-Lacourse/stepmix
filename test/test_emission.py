import numpy as np
import copy

import pandas as pd
import pytest

from stepmix.stepmix import StepMix
from stepmix.emission.build_emission import EMISSION_DICT
from stepmix.utils import max_one_hot, get_mixed_descriptor


@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
@pytest.mark.parametrize("model", EMISSION_DICT.keys())
def test_emissions(data, kwargs, model):
    """Fit all emission models with verbose output (to test their print statements).

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
    ll_1 = model_1.score(X, Y)  # Average log-likelihood
    preds_1 = model_1.predict(X, Y)  # Class predictions

    if model == "covariate":
        # We do not expect covariate to have a sampling method
        with pytest.raises(NotImplementedError) as e_info:
            model_1.sample(100)
    else:
        # Test sampling
        model_1.sample(100)


@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
@pytest.mark.parametrize("model", EMISSION_DICT.keys())
def test_emissions_no_sm(data, kwargs, model):
    """Fit all emission models with verbose output (to test their print statements).

    Only with a measurement model."""
    X, _ = data

    # Use gaussians in the structural model, all other models are tested on the measurement data
    if model.startswith("gaussian") or model.startswith("continuous"):
        kwargs["measurement"] = "binary"
    else:
        kwargs["measurement"] = model

    model_1 = StepMix(n_steps=1, **kwargs)
    model_1.fit(X)
    ll_1 = model_1.score(X)  # Average log-likelihood
    preds_1 = model_1.predict(X)  # Class predictions

    if model == "covariate":
        # We do not expect covariate to have a sampling method
        with pytest.raises(NotImplementedError) as e_info:
            model_1.sample(100)
    else:
        # Test sampling
        model_1.sample(100)


@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
def test_nested_with_cat_data():
    """Fit a mixed features model with different datasets for training and prediction."""
    df = pd.DataFrame(
        {
            # continuous
            "A": np.random.normal(0, 1, 100),
            # categorical encoded as integers
            "B": np.random.choice([0, 1, 2, 3, 4, 5], 100),
            # binary
            "C": np.random.choice([0, 1], 100),
        }
    )

    mixed_data, mixed_descriptor = get_mixed_descriptor(
        dataframe=df,
        continuous=["A"],
        categorical=["B"],
        binary=["C"],
    )

    X_train, X_test = mixed_data[:80], mixed_data[80:]

    model = StepMix(
        n_components=3, measurement=mixed_descriptor, verbose=1, random_state=123
    )

    model.fit(X_train)

    preds = model.predict(X_test)

    assert preds.shape == (20,)


@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("method", ["gradient", "newton-raphson"])
def test_covariate(data_covariate, kwargs, method, intercept):
    X, Z = data_covariate

    # Replace measurement model with the emission model we aim to test
    kwargs["structural"] = "covariate"

    model_1 = StepMix(
        n_steps=1,
        structural_params=dict(
            method=method, intercept=intercept, lr=1e-2, max_iter=1000
        ),
        **kwargs,
    )
    model_1.fit(X, Z)
    ll_1 = model_1.score(X, Z)  # Average log-likelihood


def test_illegal_covariate(data_covariate, kwargs_nested):
    X, Z = data_covariate

    # Make sure we can't have two covariate models.
    with pytest.raises(ValueError) as e_info:
        model_1 = StepMix(
            n_steps=1,
            measurement="covariate",
            structural="covariate",
        )
        model_1.fit(X, Z)

    kwargs_covariate = copy.deepcopy(kwargs_nested)
    kwargs_covariate["measurement"]["model_1"]["model"] = "covariate"
    with pytest.raises(ValueError) as e_info:
        model_1 = StepMix(
            n_steps=1,
            measurement=kwargs_covariate["measurement"],
            structural="covariate",
        )
        model_1.fit(X, Z)
    with pytest.raises(ValueError) as e_info:
        model_1 = StepMix(
            n_steps=1,
            measurement="covariate",
            structural=kwargs_covariate["measurement"],
        )
        model_1.fit(X, Z)

    kwargs_covariate["measurement"]["model_2"]["model"] = "covariate"
    with pytest.raises(ValueError) as e_info:
        model_1 = StepMix(
            n_steps=1,
            measurement=kwargs_covariate["measurement"],
        )
        model_1.fit(X)


def test_nested(data_nested, kwargs_nested):
    """Test verbose output and sampling of nested model."""

    # We use the same measurement and structural models, both nested
    model_3 = StepMix(**kwargs_nested)

    # Test fit and inference
    model_3.fit(data_nested, data_nested)
    ll_3 = model_3.score(data_nested, data_nested)
    pred_3 = model_3.predict(data_nested, data_nested)

    # Test sampling
    model_3.sample(100)


def test_nested_no_sm(data_nested, kwargs_nested):
    """Test verbose output and sampling of nested model with measurement model only."""

    # We use the same measurement and structural models, both nested
    model_3 = StepMix(**kwargs_nested)

    # Test fit and inference
    model_3.fit(data_nested)
    ll_3 = model_3.score(data_nested)
    pred_3 = model_3.predict(data_nested)

    # Test sampling
    model_3.sample(100)


def test_nested_twice(data_nested, kwargs_nested):
    """Test verbose output and sampling of nested model.

    Once ran into a bug where nested parameters would disappear after calling fit."""

    # We use the same measurement and structural models, both nested
    model_3 = StepMix(**kwargs_nested)

    # Test fit and inference
    model_3.fit(data_nested, data_nested)
    model_3.fit(data_nested, data_nested)


def test_get_descriptor(kwargs):
    """Test get_mixed_descriptor."""
    target = {
        "binary": {"model": "binary", "n_columns": 3},
        "continuous": {"model": "continuous", "n_columns": 1},
        "categorical": {"model": "categorical", "n_columns": 1},
    }

    df = pd.DataFrame(
        np.random.randint(0, 10, size=(100, 10)),
        columns=[f"col_{i}" for i in range(10)],
    )

    data, descriptor = get_mixed_descriptor(
        df,
        binary=["col_0", "col_1", "col_7"],
        continuous=["col_4"],
        categorical=["col_8"],
    )

    assert np.all(
        data.columns == np.array(["col_0", "col_1", "col_7", "col_4", "col_8"])
    )
    assert descriptor == target


def test_max_one_hot(data, kwargs):
    """Test the max_one_hot method in utils."""
    a = np.array(
        [
            [0, 3],
            [1, 0],
            [2, 1],
            [2, 2],
        ]
    )

    target = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )

    onehot, max_n_outcomes, total_outcomes = max_one_hot(a)

    assert max_n_outcomes == 4
    assert total_outcomes == 7
    assert np.all(onehot == target)


def test_max_one_hot_nan(data, kwargs):
    """Test the max_one_hot method in utils with NaNs"""
    a = np.array(
        [
            [0, 3],
            [1, np.nan],
            [np.nan, 1],
            [2, 2],
        ]
    )

    target = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )

    onehot, max_n_outcomes, total_outcomes = max_one_hot(a)

    assert max_n_outcomes == 4
    assert total_outcomes == 7
    assert np.allclose(onehot, target, equal_nan=True)


def test_categorical_encoding(kwargs):
    # Ignore base measurement and structural for this step
    kwargs.pop("measurement")
    kwargs.pop("structural")

    # Declare data
    data_int = np.array(
        [
            [0, 3],
            [1, 0],
            [2, 1],
            [2, 2],
        ]
    )

    data_one_hot = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )

    # Model on integer codes
    # In this case, max_n_outcomes and total_outcomes are inferred from the data
    model_1 = StepMix(
        measurement="categorical",
        measurement_params=dict(
            integer_codes=True, max_n_outcomes=None, total_outcomes=None
        ),
        **kwargs,
    )
    model_1.fit(data_int)
    param_1 = model_1.get_parameters()["measurement"]["pis"]

    # Model on one-hot codes
    # In this case, we need to specify max_n_outcomes and total_outcomes
    model_2 = StepMix(
        measurement="categorical",
        measurement_params=dict(
            integer_codes=False, max_n_outcomes=4, total_outcomes=7
        ),
        **kwargs,
    )
    model_2.fit(data_one_hot)
    param_2 = model_2.get_parameters()["measurement"]["pis"]

    # Check if parameters are the same
    assert np.all(param_1 == param_2)

    # Check if n_parameters are the same
    assert model_1.n_parameters == model_2.n_parameters

    # Check n_parameters
    n_parameters = (3 - 1) + 3 * (2 + 3)
    assert model_1.n_parameters == n_parameters


def test_categorical_less_categories_in_test():
    train = np.random.choice([0, 1, 2, 3], 100).reshape((-1, 1))
    test = np.random.choice([0, 1, 2], 100).reshape((-1, 1))  # no class 3

    model = StepMix(
        n_components=3, measurement="categorical", verbose=1, random_state=123
    )

    model.fit(train)

    preds = model.predict(test)
