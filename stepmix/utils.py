"""Utils."""

import numbers
import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted


# Check parameters utils copied from the PHATE library
def check_positive(**params):
    """Check that parameters are positive as expected
    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Number) or params[p] <= 0:
            raise ValueError("Expected {} > 0, got {}".format(p, params[p]))


def check_nonneg(**params):
    """Check that parameters are non-negative as expected
    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Number) or params[p] < 0:
            raise ValueError("Expected {} >= 0, got {}".format(p, params[p]))


def check_int(**params):
    """Check that parameters are integers as expected
    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Integral):
            raise ValueError("Expected {} integer, got {}".format(p, params[p]))


def check_if_not(x, *checks, **params):
    """Run checks only if parameters are not equal to a specified value
    Parameters
    ----------
    x : excepted value
        Checks not run if parameters equal x
    checks : function
        Unnamed arguments, check functions to be run
    params : object
        Named arguments, parameters to be checked
    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] is not x and params[p] != x:
            [check(**{p: params[p]}) for check in checks]


def check_in(choices, **params):
    """Checks parameters are in a list of allowed parameters
    Parameters
    ----------
    choices : array-like, accepted values
    params : object
        Named arguments, parameters to be checked
    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] not in choices:
            raise ValueError(
                "{} value {} not recognized. Choose from {}".format(
                    p, params[p], choices
                )
            )


def check_type(type, **params):
    """Checks parameters are of a given type.
    Parameters
    ----------
    type : Accepted type or typle of accepted types.
    params : object
        Named arguments, parameters to be checked
    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], type):
            raise ValueError(
                "{} value {} not recognized. Choose from {}".format(p, params[p], p)
            )


def check_between(v_min, v_max, **params):
    """Checks parameters are in a specified range
    Parameters
    ----------
    v_min : float, minimum allowed value (inclusive)
    v_max : float, maximum allowed value (inclusive)
    params : object
        Named arguments, parameters to be checked
    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] < v_min or params[p] > v_max:
            raise ValueError(
                "Expected {} between {} and {}, "
                "got {}".format(p, v_min, v_max, params[p])
            )


def check_descriptor(descriptor, keys):
    """Check if the emission descriptor is valid.

    A string describes a homogeneous model (e.g., all binary, all gaussian).

    A dict will trigger a nested model. Please refer to stepmix.emission.nested.Nested for details.

    Parameters
    ----------
    descriptor: str or dict, parameter description.
    keys: list, list of valid emission strings.

    Raises
    ------
    ValueError : illegal model descriptor.
    """
    if isinstance(descriptor, str):
        check_in(keys, emission=descriptor)
    elif isinstance(descriptor, dict):
        for key, item in descriptor.items():
            if isinstance(item, dict):
                if "model" not in item or "n_columns" not in item:
                    raise ValueError(
                        f"Nested dict descriptors should include at least a model key and an n_columns "
                        f"key."
                    )

                # Check that n_columns is a positive int
                check_in(keys, emission=item["model"])
            else:
                raise ValueError(
                    f"Items in a nested model description should be dicts."
                )

    else:
        raise ValueError(f"Emission descriptor should be either a string or a dict.")


def check_covariate(measurement_descriptor, structural_descriptor):
    """Check if measurement or structural models include a covariate model.

    Only one covariate model should be used in an entire StepMix estimator, be it in the measurement or the
    structural model and including in nested models.

    Parameters
    ----------
    measurement_descriptor: str or dict, measurement parameter description.
    structural_descriptor: str, dict or None, structural parameter description.

    Returns
    -------
    is_covariate: bool, indicating if measurement or structural model includes a covariate model.

    Raises
    ------
    ValueError : illegal use of a covariate model.
    """

    def is_covariate(desc):
        if isinstance(desc, str):
            return desc == "covariate"
        elif isinstance(desc, dict):
            # Iterate through all nested models to identify covariate models
            covariate_count = 0
            for key, value in desc.items():
                if value["model"] == "covariate":
                    covariate_count += 1
            if covariate_count > 1:
                raise ValueError(
                    "Only one covariate model should be used in an entire StepMix estimator, be it in "
                    "the measurement or the structural model and including in nested models."
                )
            return covariate_count == 1
        elif desc is None:
            return False
        else:
            raise ValueError(
                "Wrong model descriptor. Model descriptors should be dictionaries or string."
            )

    m_covariate = is_covariate(measurement_descriptor)
    s_covariate = is_covariate(structural_descriptor)

    # Models can't both be covariate models
    if m_covariate and s_covariate:
        raise ValueError(
            f"Only the structural or measurement model can be a covariate model, not both."
        )

    return m_covariate or s_covariate


def check_descriptor_nan(descriptor):
    """Check if the provided descriptor describes a model supporting missing values.

    Parameters
    ----------
    descriptor: str or dict, parameter description.

    Returns
    -------
    is_valid: bool, indicating a valid parameter description.

    """
    if isinstance(descriptor, str):
        # Models supporting missing values end with nan
        return descriptor.endswith("nan")
    elif isinstance(descriptor, dict):
        return any([v["model"].endswith("nan") for v in descriptor.values()])
    else:
        raise ValueError(f"Emission descriptor should be either a string or a dict.")


def identify_coef(coef):
    """Find a reference configuration of the coefficients.

    Pick whichever coefficient is closest to 0 in the first colum of coef. Subtract the associated
    row from all coefficients. This will give us a reference class with 0 coefficients everywhere.
    Pick index of second largest coefficient in the slope coefficients (second column). Subtract


    Parameters
    ----------
    coef: np.ndarray, Current coefficient estimates.

    Returns
    -------
    coef: np.ndarray, Corrected coefficient estimates with a null reference class.

    """
    second_coeff = coef[:, 1].argsort()[1]
    coef -= coef[second_coeff, :].reshape((1, -1))
    return coef


def modal(resp, clip=False):
    """Takes in class probabilities and performs modal assignment.

    Will return a one-hot encoding. The clip argument can also be used to clip the results to the (1e-15, 1-1e-15)
    range.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)
        Class probabilities.
    clip : bool, default=False
        Clip the probabilities to the range (1e-15, 1-1e-15).

    Returns
    -------
    modal_resp : array, shape (n_samples, n_components)
        Modal class assignment.

    """
    preds = resp.argmax(axis=1)
    modal_resp = np.zeros(resp.shape)
    modal_resp[np.arange(resp.shape[0]), preds] = 1

    if clip:
        modal_resp = np.clip(modal_resp, 1e-15, 1 - 1e-15)

    return modal_resp


def print_report(model, X, Y=None, sample_weight=None, x_names=None, y_names=None):
    """Print detailed output for the model.

    Parameters
    ----------
    model: stepmix.StepMix
        Fitted StepMix instance.
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_features_structural), default=None
    sample_weight : array-like of shape(n_samples,), default=None
    x_names : List of str, default=None
        Column names of X.
    y_names : List of str, default=None
        Column names of Y.
    """
    check_is_fitted(model)
    n_classes = model.n_components
    n_samples = X.shape[0]
    n_parameters = model.n_parameters
    avg_ll = model.score(X, Y, sample_weight=sample_weight)
    ll = (
        avg_ll * np.sum(sample_weight)
        if sample_weight is not None
        else avg_ll * n_samples
    )
    minus2ll = -2 * ll

    bic = model.bic(X, Y)
    aic = model.aic(X, Y)
    entropy = model.entropy(X, Y)
    relative_entropy = model.relative_entropy(X, Y)
    sabic = model.sabic(X, Y)
    caic = model.caic(X, Y)

    print("=" * 80)
    print("MODEL REPORT")
    print("=" * 80)
    print("    " + "=" * 76)
    print(f"    Measurement model parameters")
    print("    " + "=" * 76)
    model._mm.print_parameters(indent=2, feature_names=x_names)

    if hasattr(model, "_sm"):
        print("    " + "=" * 76)
        print(f"    Structural model parameters")
        print("    " + "=" * 76)
        model._sm.print_parameters(indent=2, feature_names=y_names)

    print("    " + "=" * 76)
    print(f"    Class weights")
    print("    " + "=" * 76)
    for i, w in enumerate(model.weights_):
        print(f"        Class {i + 1} : {w:.2f}")

    print("    " + "=" * 76)
    print(f"    Fit for {n_classes} latent classes")
    print("    " + "=" * 76)
    print(f"    Estimation method             : {model.n_steps}-step")
    if model.n_steps == 3:
        print(f"    Correction method             : {model.correction}")
        print(f"    Assignment method             : {model.assignment}")
    print(f"    Number of observations        : {n_samples}")
    print(f"    Number of latent classes      : {n_classes}")
    print(f"    Number of estimated parameters: {n_parameters}")
    print(f"    Log-likelihood (LL)           : {ll:.4f}")
    print(f"    -2LL                          : {minus2ll:.4f}")
    print(f"    Average LL                    : {avg_ll:.4f}")
    print(f"    AIC                           : {aic:.2f}")
    print(f"    BIC                           : {bic:.2f}")
    print(f"    CAIC                          : {caic:.2f}")
    print(f"    Sample-Size Adjusted BIC      : {sabic:.2f}")
    print(f"    Entropy                       : {entropy:.4f}")
    print(f"    Scaled Relative Entropy       : {relative_entropy:.4f}")


def max_one_hot(array, max_n_outcomes=None, total_outcomes=None):
    """Multiple categorical one-hot encoding.

    Takes an n_samples x n_features array of integer-encoded categorical features and returns an
    n_samples x (n_features x max_n_outcomes) array where max_n_outcomes is the number of outcomes in the
    categorical feature with the most categories. Categories are one-hot encoded and categories with
    fewer than max_n_outcomes simply have unused extra columns.

    Handles missing values.

    Examples
    --------
    .. code-block:: python
        arr = np.array(
            [
                [0, 3],
                [1, 0],
                [2, 1],
                [2, 2],
            ]
        )
        b, max_n_outcomes, total_outcomes = max_one_hot(a)
        print(b)

        # Should yield
        # [[1. 0. 0. 0. 0. 0. 0. 1.]
        #  [0. 1. 0. 0. 1. 0. 0. 0.]
        #  [0. 0. 1. 0. 0. 1. 0. 0.]
        #  [0. 0. 1. 0. 0. 0. 1. 0.]]

    Parameters
    ----------
    array : ndarray of shape  (n_samples, n_features)
        Integer-encoded categorical data. Will be float due to sklearn casting. We'll cast back to ints.
    max_n_outcomes : int, default=None
        Maximum number of outcomes for a single categorical feature.
        Each column in the input will have max_n_outcomes associated columns in the output.
        If None, will be inferred from the data.
    total_outcomes : int, default=None
        Total outcomes over all features. E.g., if we provide a categorical variable with two outcomes and another
        with 4 outcomes, total_outcomes = 6.
        If None, will be inferred from the data.

    Returns
    -------
    one_hot : ndarray of shape (n_samples, n_features * max_n_outcomes)
        One-hot encoded categories.

    max_n_outcomes : int
        Maximum number of outcomes for a single categorical feature.
        Each column in the input will have max_n_outcomes associated columns in the output.

    total_outcomes : int
        Total outcomes over all features. E.g., if we provide a categorical variable with two outcomes and another
        with 4 outcomes, total_outcomes = 6.

    """
    n_samples = array.shape[0]
    n_features = array.shape[1]

    # Get maximal number of outcomes
    if max_n_outcomes is None or total_outcomes is None:
        outcomes = np.nanmax(array, axis=0) + 1
        total_outcomes = int(np.sum(outcomes))
        max_n_outcomes = int(np.nanmax(outcomes))

    # Create one-hot encoding
    one_hot = np.zeros((n_samples, array.shape[1] * max_n_outcomes))

    for c in range(n_features):
        integer_codes = array[:, c]
        not_observed = np.isnan(integer_codes)
        integer_codes = np.nan_to_num(integer_codes, nan=0).astype(int)
        one_hot[np.arange(n_samples), integer_codes + c * max_n_outcomes] = 1.0

        # Now reapply NaNs
        one_hot[not_observed, c * max_n_outcomes : (c + 1) * max_n_outcomes] = np.nan

    return one_hot, max_n_outcomes, total_outcomes


def get_mixed_descriptor(dataframe, **kwargs):
    """Simpler API to build the mixed model descriptor from a dataframe.

    Mixed models can combine multiple datatypes, such as binary or continuous.
    Please refer to :class:`stepmix.emission.nested.Nested` for details on the output descriptor.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with the input data.
    kwargs : list of strings
        Each key represents a model type and the provided list will consist of columns in dataframe.

    Returns
    -------
    data : pd.DataFrame
        Dataframe with selected columns in proper order.

    descriptor : dict
        Model description. Can be provided to the measurement or structural arguments of stepmix.StepMix.

    """
    descriptor = dict()
    columns = list()

    for key, value in kwargs.items():
        columns += value
        descriptor[key] = dict(model=key, n_columns=len(value))

    data = dataframe[columns]

    return data, descriptor


def cov_np_to_df(cov, feature_names, model_str):
    """Unpack an ndarray of covariances into a long form dataframe."""
    feature_names_cov = ["cov_" + f for f in feature_names]
    params = list()
    for class_no in range(cov.shape[0]):
        for row, param_name in enumerate(feature_names_cov):
            for col, var_name in enumerate(feature_names):
                params.append(
                    dict(
                        model_name=model_str,
                        param=param_name,
                        class_no=class_no,
                        variable=var_name,
                        value=cov[class_no, row, col],
                    )
                )

    return pd.DataFrame.from_records(params)


def extract_column_names(X):
    """Extract variable names based on X type."""
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    elif isinstance(X, pd.Series):
        return [X.name]
    else:
        return None
