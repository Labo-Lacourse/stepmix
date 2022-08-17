"""Check compatibility with scikit-learn API"""
from sklearn.utils.estimator_checks import parametrize_with_checks
from stepmix.stepmix import StepMix


@parametrize_with_checks([StepMix(n_components=1, measurement='gaussian_unit', structural='gaussian_unit', random_state=123456)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
