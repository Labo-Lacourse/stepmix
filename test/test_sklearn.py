"""Check compatibility with scikit-learn API"""

from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.model_selection import GridSearchCV
from stepmix.stepmix import StepMix


@parametrize_with_checks(
    [
        StepMix(
            n_components=1,
            measurement="gaussian_unit",
            structural="gaussian_unit",
            random_state=123456,
        )
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_gridsearch(data, kwargs):
    """Quick check with the scikit-learn GridSearchCV object"""
    X, Y = data

    # Grid search with cross-validation
    model_1 = StepMix(n_steps=1, **kwargs)
    gs = GridSearchCV(estimator=model_1, param_grid=dict(n_components=[2, 3, 4, 5]))
    gs.fit(X, Y)
