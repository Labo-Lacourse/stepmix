import pytest
import pandas as pd
import numpy as np

from stepmix.stepmix import StepMix
from stepmix.emission.build_emission import EMISSION_DICT


@pytest.mark.parametrize(
    "n_steps,corr", [(1, None), (2, None), (3, None), (3, "BCH"), (3, "ML")]
)
def test_dataframe(data, kwargs, n_steps, corr):
    X, Y = data
    X_df, Y_df = pd.DataFrame(X), pd.DataFrame(Y)

    kwargs["n_steps"] = n_steps
    kwargs["correction"] = corr

    # Test on numpy arrays
    model_1 = StepMix(**kwargs)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)  # Average log-likelihood

    # Test on dataframes
    model_2 = StepMix(**kwargs)
    model_2.fit(X_df, Y_df)
    ll_2 = model_1.score(X_df, Y_df)  # Average log-likelihood

    assert ll_1 == ll_2


@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # Ignore most numerical errors since we do not run the emission models on appropriate data
@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)  # Ignore convergence warnings for same reason
@pytest.mark.parametrize("model", EMISSION_DICT.keys())
def test_nan(data, kwargs, model):
    X, Y = data
    kwargs["measurement"] = model
    kwargs["structural"] = model

    # Test on numpy arrays
    model_1 = StepMix(n_steps=1, **kwargs)

    supports_nan = model.endswith("_nan")

    # Make sure vanilla models raise an Error if a missing value is found in the input
    X = X.astype(float)
    X_nan = X.copy()
    X_nan[0, 0] = np.nan

    if supports_nan:
        # Should fit without error
        model_1.fit(X_nan, Y)
    else:
        # Should raise error
        with pytest.raises(ValueError) as e_info:
            model_1.fit(X_nan, Y)

    Y_nan = Y.copy()
    Y_nan[0, 0] = np.nan
    if supports_nan:
        # Should fit without error
        model_1.fit(X_nan, Y)
    else:
        # Should raise error
        with pytest.raises(ValueError) as e_info:
            model_1.fit(X, Y_nan)
