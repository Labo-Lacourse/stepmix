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


def test_vector_1d(data, kwargs):
    X, Y = data

    # Inputs are vectors
    X = X[:, 0]
    Y = Y[:, 0]

    # Test on numpy arrays
    model_1 = StepMix(n_steps=1, **kwargs)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)  # Average log-likelihood

    # Inputs are matrices
    X_col = X.reshape((-1, 1))
    Y_col = Y.reshape((-1, 1))
    model_2 = StepMix(n_steps=1, **kwargs)
    model_2.fit(X_col, Y_col)
    ll_2 = model_1.score(Y_col, Y_col)  # Average log-likelihood

    assert ll_1 == ll_2
