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
