"""High level tests where we call all estimation and correction methods."""

import pytest

import numpy as np
from stepmix.stepmix import StepMix
from stepmix.utils import modal


class TestStepAPI:
    """Test that the high level call and the decomposed steps yield the same results"""

    def test_1_step(self, data, kwargs):
        X, Y = data

        # API-level 1-step
        model_1 = StepMix(n_steps=1, **kwargs)
        model_1.fit(X, Y)
        ll_1 = model_1.score(X, Y)  # Average log-likelihood

        # Equivalently, each step can be performed individually. See the code of the fit method for details.
        model_2 = StepMix(**kwargs)
        model_2.em(X, Y)  # Step 1
        ll_2 = model_2.score(X, Y)

        assert ll_1 == ll_2

    def test_2_step_soft(self, data, kwargs):
        X, Y = data

        # API-level 2-step
        model_1 = StepMix(n_steps=2, assignment="soft", **kwargs)
        model_1.fit(X, Y)
        ll_1 = model_1.score(X, Y)  # Average log-likelihood

        # Equivalently, each step can be performed individually. See the code of the fit method for details.
        model_2 = StepMix(**kwargs)
        model_2.em(X)  # Step 1
        model_2.em(X, Y, freeze_measurement=True)  # Step 2
        ll_2 = model_2.score(X, Y)

        assert ll_1 == ll_2

    @pytest.mark.parametrize("assignment", ["modal", "soft"])
    def test_3_step_soft(self, data, kwargs, assignment):
        X, Y = data

        # API-level 3-step
        model_1 = StepMix(n_steps=3, assignment=assignment, **kwargs)
        model_1.fit(X, Y)
        ll_1 = model_1.score(X, Y)  # Average log-likelihood

        # Equivalently, each step can be performed individually. See the code of the fit method for details.
        model_2 = StepMix(**kwargs)
        model_2.em(X)  # Step 1
        probs = model_2.predict_proba(X)  # Step 2
        if assignment == "modal":
            probs = modal(probs, clip=True)  # Hard assignment
        model_2.m_step_structural(probs, Y)  # Step 3
        ll_2 = model_2.score(X, Y)

        assert ll_1 == ll_2


@pytest.mark.parametrize("correction", ["BCH", "ML", None])
@pytest.mark.parametrize("assignment", ["modal", "soft"])
def test_corrections(data, kwargs, correction, assignment):
    """Run BCH and ML corrections with modal and hard assignments.

    We do not check for a particular result here. We simply run the code."""
    X, Y = data

    model_1 = StepMix(n_steps=3, assignment=assignment, correction=correction, **kwargs)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)  # Average log-likelihood


def test_2_v_3(data, kwargs):
    """Test that 2-step and 3-step end up with the same measurement model."""
    X, Y = data

    model_1 = StepMix(n_steps=2, **kwargs)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X)  # Average log-likelihood of measurement model

    model_2 = StepMix(n_steps=3, **kwargs)
    model_2.fit(X, Y)
    ll_2 = model_2.score(X)  # Average log-likelihood of measurement model

    assert ll_1 == ll_2


def test_1_step_sym(data, kwargs):
    """Check 3 equivalent definition of a 1-step estimator.

    Switching the measurement and structural data in the case of 1-step should not affect results.
    Likewise, defining a single nested model that includes both measurement and structural data should yield the
    same results."""
    # Data generation
    X, Y = data

    # For this test, ignore the measurement and structural keys in kwargs
    kwargs.pop("measurement")
    kwargs.pop("structural")

    # Common arguments
    kwargs = dict(
        n_steps=1, n_components=3, abs_tol=1e-5, n_init=2, max_iter=200, random_state=42
    )

    # Binary measurements, gaussian structural
    model_1 = StepMix(measurement="bernoulli", structural="gaussian_unit", **kwargs)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)

    # Gaussian measurement, binary structural
    model_2 = StepMix(measurement="gaussian_unit", structural="bernoulli", **kwargs)
    model_2.fit(Y, X)
    ll_2 = model_2.score(Y, X)

    # Binary + Gaussian measurement
    descriptor = {
        "model_1": {"model": "bernoulli", "n_columns": 6},
        "model_2": {"model": "gaussian_unit", "n_columns": 1},
    }

    # Merge data in single matrix
    Z = np.hstack((X, Y))

    model_3 = StepMix(measurement=descriptor, **kwargs)
    model_3.fit(Z)
    ll_3 = model_3.score(Z)

    # Assert
    assert np.allclose(ll_1, ll_2, atol=1e-12)  # Strict equality fails with Python 3.7
    assert np.allclose(ll_2, ll_3, atol=1e-12)  # Strict equality fails with Python 3.7
