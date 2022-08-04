"""Benchmarks to doublecheck expected behavior on data.

E.g., we expect 1-step to achieve a higher likelihood on the Bakk data than 3-step.
"""
import pytest
from stepmix.stepmix import StepMix


@pytest.mark.parametrize(
    "cov_1,cov_2", [("unit", "spherical"), ("spherical", "diag"), ("diag", "full")]
)
def test_gaussians(data_gaussian, kwargs_gaussian, cov_1, cov_2):
    """Test gaussian models on data with non isotropic clusters.

    Increasing the number of covariance parameters should increase the likelihood."""
    X, Y = data_gaussian

    # Ignore structural key for this test
    kwargs_gaussian.pop("structural")

    model_1 = StepMix(n_steps=1, structural="gaussian_" + cov_1, **kwargs_gaussian)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)

    model_2 = StepMix(n_steps=1, structural="gaussian_" + cov_2, **kwargs_gaussian)
    model_2.fit(X, Y)
    ll_2 = model_2.score(X, Y)

    assert ll_1 < ll_2


@pytest.mark.parametrize(
    "cov_1,cov_2", [("unit_nan", "spherical_nan"), ("spherical_nan", "diag_nan")]
)
def test_gaussians_nan(data_gaussian_nan, kwargs_gaussian_nan, cov_1, cov_2):
    """Test gaussian models on data with non spherical clusters.

    There are missing values in both the measurement and structural models.
    Increasing the number of covariance parameters should increase the likelihood."""
    X, Y = data_gaussian_nan

    # Ignore structural key for this test
    kwargs_gaussian_nan.pop("structural")

    model_1 = StepMix(n_steps=1, structural="gaussian_" + cov_1, **kwargs_gaussian_nan)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)

    model_2 = StepMix(n_steps=1, structural="gaussian_" + cov_2, **kwargs_gaussian_nan)
    model_2.fit(X, Y)
    ll_2 = model_2.score(X, Y)

    assert ll_1 < ll_2


@pytest.mark.parametrize("n_steps_1,n_steps_2", [(3, 2), (2, 1)])
def test_steps_ll(data_large, kwargs_large, n_steps_1, n_steps_2):
    """Test binary measurements + gaussian unit structural on Bakk data with multiple steps.

    We expect 1-step > 2-step > 3-step in terms of final likelihood."""
    X, Y = data_large

    model_1 = StepMix(n_steps=n_steps_1, **kwargs_large)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)

    model_2 = StepMix(n_steps=n_steps_2, **kwargs_large)
    model_2.fit(X, Y)
    ll_2 = model_2.score(X, Y)

    assert ll_1 < ll_2


@pytest.mark.parametrize("corr_1,corr_2", [(None, "BCH"), (None, "ML")])
def test_corrections_ll(data_large, kwargs_large, corr_1, corr_2):
    """Test binary measurements + gaussian unit structural on Bakk data with 3-step model and different corrections.

    We expect BCH > No correction and ML > No correction in terms of final likelihood."""
    X, Y = data_large

    model_1 = StepMix(n_steps=3, correction=corr_1, **kwargs_large)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)

    model_2 = StepMix(n_steps=3, correction=corr_2, **kwargs_large)
    model_2.fit(X, Y)
    ll_2 = model_2.score(X, Y)

    assert ll_1 < ll_2
