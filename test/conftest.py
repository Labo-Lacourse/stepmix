import pytest

from stepmix.datasets import (
    data_bakk_response,
    data_bakk_covariate,
    data_generation_gaussian,
    data_gaussian_diag,
)


# Fixtures for Bakk data
@pytest.fixture
def data():
    X, Y, _ = data_bakk_response(n_samples=100, sep_level=0.9, random_state=42)
    return X, Y


@pytest.fixture
def data_covariate():
    X, Z, _ = data_bakk_covariate(n_samples=100, sep_level=0.9, random_state=42)
    return X, Z


@pytest.fixture
def kwargs():
    # Default estimator arguments
    kwargs = dict(
        n_components=3,
        measurement="bernoulli",
        structural="gaussian_unit",
        random_state=42,
        abs_tol=1e-5,
        n_init=2,
        max_iter=200,
        verbose=1,
    )
    return kwargs


@pytest.fixture
def kwargs_covariate():
    # Default estimator arguments
    kwargs_covariate = dict(
        n_components=3,
        measurement="bernoulli",
        structural="covariate",
        random_state=42,
        abs_tol=1e-5,
        n_init=2,
        max_iter=200,
        verbose=1,
        structural_params=dict(method="newton-raphson"),
    )
    return kwargs_covariate


@pytest.fixture
def data_large():
    X, Y, _ = data_bakk_response(n_samples=3000, sep_level=0.7, random_state=42)
    return X, Y


@pytest.fixture
def data_covariate_large():
    X, Z, _ = data_bakk_covariate(n_samples=1000, sep_level=0.7, random_state=42)
    return X, Z


@pytest.fixture
def kwargs_large():
    # Default estimator arguments
    kwargs = dict(
        n_components=3,
        measurement="bernoulli",
        structural="gaussian_unit",
        random_state=42,
        abs_tol=1e-5,
        n_init=10,
        max_iter=300,
        verbose=0,
    )
    return kwargs


# Fixtures for Gaussian Hwk4 data
@pytest.fixture
def data_gaussian():
    X, Y, _ = data_generation_gaussian(n_samples=1000, sep_level=0.9, random_state=42)
    return X, Y


@pytest.fixture
def kwargs_gaussian():
    # Default estimator arguments
    kwargs = dict(
        n_components=4,
        measurement="bernoulli",
        structural="gaussian_unit",
        random_state=42,
        abs_tol=1e-5,
        n_init=10,
        max_iter=300,
        verbose=0,
    )
    return kwargs


# Fixtures for Diagonal Gaussian data with missing values
@pytest.fixture
def data_gaussian_nan():
    X, Y, _ = data_gaussian_diag(
        n_samples=1000, sep_level=0.9, random_state=42, nan_ratio=0.2
    )
    return X, Y


@pytest.fixture
def kwargs_gaussian_nan():
    # Default estimator arguments
    kwargs = dict(
        n_components=3,
        measurement="bernoulli_nan",
        structural="gaussian_unit_nan",
        random_state=42,
        abs_tol=1e-5,
        n_init=10,
        max_iter=300,
        verbose=0,
    )
    return kwargs
