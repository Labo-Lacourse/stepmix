import pandas as pd

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