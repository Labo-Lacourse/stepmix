"""Compatibility layer for different scikit-learn versions."""

try:
    from sklearn.utils.validation import validate_data
except ImportError:

    def validate_data(estimator, X="no_validation", y="no_validation", **kwargs):
        return estimator._validate_data(X, y, **kwargs)


try:
    from sklearn.utils.validation import check_array as _check_array

    _check_array([[0]], ensure_all_finite=True)
    _USES_ENSURE_ALL_FINITE = True
except TypeError:
    _USES_ENSURE_ALL_FINITE = False


def _finite_param(ensure_all_finite):
    """Return the correct kwarg dict for the current sklearn version."""
    if _USES_ENSURE_ALL_FINITE:
        return {"ensure_all_finite": ensure_all_finite}
    return {"force_all_finite": ensure_all_finite}
