import numpy as np
import pytest

from stepmix import StepMix


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings
@pytest.mark.parametrize("model", ["binary", "binary_nan"])
def test_binary_n_parameters(model):
    """Test number of parameters of a simple binary mixture."""
    rng = np.random.default_rng(42)
    data = rng.choice(a=[0, 1], size=(100, 7))

    model = StepMix(
        n_components=3,
        measurement=model,
        random_state=42,
        verbose=0,
        max_iter=1,
        n_init=1,
    )

    model.fit(data)

    assert model.n_parameters == 23


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings
@pytest.mark.parametrize("model", ["categorical", "categorical_nan"])
def test_categorical_n_parameters(model):
    """Test number of parameters of a simple categorical mixture."""
    rng = np.random.default_rng(42)
    data = rng.choice(a=[0, 1, 2], size=(200,)).reshape(-1, 1)

    model = StepMix(
        n_components=3,
        measurement=model,
        random_state=42,
        verbose=0,
        max_iter=1,
        n_init=1,
    )

    model.fit(data)

    assert model.n_parameters == 8


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings
@pytest.mark.parametrize("model", ["categorical", "categorical_nan"])
def test_categorical_n_parameters_max(model):
    """Test number of parameters of a categorical mixture where some categorical features have fewer outcomes."""
    rng = np.random.default_rng(42)
    data_1 = rng.choice(a=[0, 1, 2, 3], size=300).reshape(-1, 1)
    data_2 = rng.choice(a=[0, 1, 2], size=300).reshape(-1, 1)
    data_3 = rng.choice(a=[0, 1], size=300).reshape(-1, 1)
    data = np.hstack((data_1, data_2, data_3))

    model = StepMix(
        n_components=4,
        measurement=model,
        random_state=42,
        verbose=0,
        max_iter=1,
        n_init=1,
    )

    model.fit(data)

    n_parameters = (4 - 1) + 4 * (1 + 2 + 3)

    assert model.n_parameters == n_parameters


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings
@pytest.mark.parametrize(
    "model,n_parameters",
    [
        ("gaussian_unit", 19),
        ("gaussian_diag", 35),
        ("gaussian_unit_nan", 19),
        ("gaussian_diag_nan", 35),
        ("gaussian_spherical", 23),
        ("gaussian_spherical_nan", 23),
        ("gaussian_tied", 29),
        ("gaussian_full", 59),
    ],
)
def test_gaussian_n_parameters(model, n_parameters):
    """Test number of parameters of a simple gaussian mixture."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 4))

    model = StepMix(
        n_components=4,
        measurement=model,
        random_state=42,
        verbose=0,
        max_iter=1,
        n_init=1,
    )

    model.fit(data)

    assert model.n_parameters == n_parameters


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings
@pytest.mark.parametrize("intercept,n_parameters", [(False, 10), (True, 12)])
def test_covariate_n_parameters(intercept, n_parameters):
    """Test number of parameters of a simple covariate mixture."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(100, 2))

    opt_params = {
        "method": "newton-raphson",  # Can also be "gradient",
        "intercept": intercept,
        "max_iter": 1,  # Number of opt. step each time we update the covariate model
    }

    model = StepMix(
        n_components=3,
        measurement="gaussian_unit",
        structural="covariate",
        structural_params=opt_params,
        random_state=42,
        verbose=0,
        max_iter=1,
        n_init=1,
    )

    model.fit(data, data)

    assert model.n_parameters == n_parameters


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings
def test_nested_n_parameters():
    """Test number of parameters of a simple nested model."""
    rng = np.random.default_rng(42)
    data_continuous = rng.normal(size=(100, 2))
    data_binary = rng.choice(a=[0, 1], size=(100, 1))
    data_categorical = rng.choice(a=[0, 1, 2], size=(100, 1))
    data = np.hstack((data_continuous, data_binary, data_categorical))

    model_desc = {
        "continuous": {
            "model": "gaussian_diag",
            "n_columns": 2,
        },
        "binary": {
            "model": "binary",
            "n_columns": 1,
        },
        "category": {
            "model": "categorical",
            "n_columns": 1,
        },
    }

    model = StepMix(
        n_components=3,
        measurement=model_desc,
        random_state=42,
        verbose=0,
        max_iter=1,
        n_init=1,
    )

    model.fit(data)

    assert model.n_parameters == 23
