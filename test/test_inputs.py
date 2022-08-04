import pytest
import pandas as pd
import numpy as np

from stepmix.stepmix import StepMix


def test_dataframe(data, kwargs):
    X, Y = data
    X_df, Y_df = pd.DataFrame(X), pd.DataFrame(Y)

    # Test on numpy arrays
    model_1 = StepMix(n_steps=1, **kwargs)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)  # Average log-likelihood

    # Test on dataframes
    model_2 = StepMix(n_steps=1, **kwargs)
    model_2.fit(X_df, Y_df)
    ll_2 = model_1.score(X_df, Y_df)  # Average log-likelihood

    assert ll_1 == ll_2


def test_nan(data, kwargs):
    X, Y = data

    # Test on numpy arrays
    model_1 = StepMix(n_steps=1, **kwargs)

    # Make sure vanilla models raise an Error if a missing value is found in the input
    X = X.astype(float)
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError) as e_info:
        model_1.fit(X_nan, Y)

    Y_nan = Y.copy()
    Y_nan[0, 0] = np.nan
    with pytest.raises(ValueError) as e_info:
        model_1.fit(X, Y_nan)
