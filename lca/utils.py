import numbers
import numpy as np


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
    choices : array-like, accepted values
    params : object
        Named arguments, parameters to be checked
    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], type):
            raise ValueError(
                "{} value {} not recognized. Choose from {}".format(
                    p, params[p], p
                )
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


def check_emission_param(param, keys):
    """Check if the emission descriptor is valid.

    A string describes a homogeneous model (e.g., all binary, all gaussian).

    Otherwise param must be a dict of string-int pairs. For example, a model where
    the first 3 columns are gaussian with unit variance, the next 2 are binary and
    the last 4 are covariates would be described likeso :
    >>> param = {
    >>>'gaussian': 3,
    >>>'binary': 2,
    >>>'covariate':4
    >>> }

    Alternatively, the integer can be replaced with a nested dict to specify some emission arguments. In this
    case, the dict is expected to have an n_columns key to specify the number of associated features.

    Parameters
    ----------
    param: str or list, parameter description.
    param: list, list of valid emission strings.

    Returns
    -------
    is_valid: bool, indicating a valid parameter description.

    """
    if isinstance(param, str):
        check_in(keys, emission=param)
    elif isinstance(param, dict):
        for key, item in param.items():
            check_in(keys, emission=key)
            if isinstance(item, dict):
                if 'n_columns' not in item:
                    raise ValueError(f'You have specified arguments for a nested {key} model. The dict'
                                     f'should include an integer describing the number of columns.')
            elif not isinstance(item, int):
                raise ValueError(f'Items in a nested model description should be integers or dicts.')

    else:
        raise ValueError(f'Emission descriptor should be either a string or a dict.')


def identify_coef(coef):
    """Find a reference configuration of the coefficients.

    Pick whichever coefficient is closest to 0 in the first row of coef. Subtract the associated
    column from all coefficients. This will give us a reference class with 0 coefficients everywhere.


    Parameters
    ----------
    coef: np.ndarray, Current coefficient estimates.

    Returns
    -------
    coef: np.ndarray, Corrected coefficient estimates with a null reference class.

    """
    closest_id = np.argsort(coef[0])[1]
    coef -= coef[:, closest_id].reshape((-1, 1))
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
