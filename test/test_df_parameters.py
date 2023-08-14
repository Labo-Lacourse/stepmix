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
    model_1.get_parameters_df()
    model_1.get_mm_df()
    model_1.get_sm_df()

    # Test class weights
    if model == "covariate":
        with pytest.raises(ValueError) as e_info:
            model_1.get_cw_df()
    else:
        model_1.get_cw_df()


@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
def test_nested_df():
    """Test parameter df of a nested model."""
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
        n_components=3,
        measurement=mixed_descriptor,
        structural=mixed_descriptor,
        verbose=1,
        random_state=123,
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
        n_components=3,
        measurement="gaussian_unit",
        structural="gaussian_unit",
        verbose=1,
        random_state=123,
    )

    model.fit(df, df)

    params = model.get_parameters_df().reset_index()
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
    """Check that feature names in param series are the same as in input series."""
    df = pd.DataFrame(
        {
            # continuous
            "A": np.random.normal(0, 1, 100),
        }
    )

    model = StepMix(
        n_components=3,
        measurement="gaussian_unit",
        structural="gaussian_unit",
        verbose=1,
        random_state=123,
    )

    model.fit(df[["A"]], df["A"])

    params = model.get_parameters_df().reset_index()
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
def test_nested_df_model_names():
    """Fit nested model with different model names but same model type."""
    desc = {
        "binary_1": {"model": "binary", "n_columns": 1},
        "binary_2": {"model": "binary", "n_columns": 1},
        "binary_3": {"model": "binary", "n_columns": 1},
    }

    data = pd.DataFrame(
        np.random.randint(0, 2, size=(100, 3)),
        columns=[f"col_{i}" for i in range(3)],
    )

    model = StepMix(n_components=3, measurement=desc, verbose=1, random_state=123)

    model.fit(data)

    df = model.get_parameters_df()
    model_names_input = list(desc.keys())
    model_names_input.sort()
    model_names_output = list(df.reset_index()["model_name"].unique())
    model_names_output.remove("class_weights")
    model_names_output.sort()

    assert model_names_input == model_names_output
