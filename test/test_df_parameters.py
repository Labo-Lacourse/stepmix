import pytest
import numpy as np
import pandas as pd

from stepmix import StepMix
from stepmix.emission.build_emission import EMISSION_DICT
from stepmix.utils import get_mixed_descriptor


@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
@pytest.mark.parametrize("model", EMISSION_DICT.keys())
def test_emissions_df(data, kwargs, model):
    """Call the df method of all emission models."""
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
    df = model_1.get_parameters_df()

@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
def test_nested_df():
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


    model = StepMix(
        n_components=3, measurement=mixed_descriptor, verbose=1, random_state=123
    )

    model.fit(mixed_data)

    df = model.get_parameters_df()

    model2 = StepMix(
        n_components=3, measurement=mixed_descriptor, structural=mixed_descriptor, verbose=1, random_state=123
    )

    model2.fit(mixed_data, mixed_data)

    df2 = model2.get_parameters_df()

@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
def test_df_names():
    """Check that feature names in param df are the same as in input df."""
    df = pd.DataFrame(
        {
            # continuous
            "A": np.random.normal(0, 1, 100),
            "B": np.random.normal(0, 1, 100),
            "C": np.random.normal(0, 1, 100),
        }
    )


    model = StepMix(
        n_components=3, measurement="gaussian_unit", structural="gaussian_unit", verbose=1, random_state=123
    )

    model.fit(df, df)

    params = model.get_parameters_df()
    cols = list(params["variable"].unique())
    cols.remove("nan")
    cols.sort()

    input_cols = list(df.columns)
    input_cols.sort()

    assert cols == input_cols

@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
def test_series_names():
    """Check that feature names in param df are the same as in input df."""
    df = pd.DataFrame(
        {
            # continuous
            "A": np.random.normal(0, 1, 100),
        }
    )


    model = StepMix(
        n_components=3, measurement="gaussian_unit", structural="gaussian_unit", verbose=1, random_state=123
    )

    model.fit(df[["A"]], df["A"])

    params = model.get_parameters_df()
    cols = list(params["variable"].unique())
    cols.remove("nan")
    cols.sort()

    input_cols = list(df.columns)
    input_cols.sort()

    assert cols == input_cols

