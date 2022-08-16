"""Check compatibility with scikit-learn API"""
from sklearn.utils.estimator_checks import parametrize_with_checks
from stepmix.stepmix import StepMix


@parametrize_with_checks([StepMix()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
