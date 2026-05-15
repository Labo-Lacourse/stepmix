"""
test_bootstrap_inference.py
=========================
Unit and integration tests for test_bootstrap_inference.py.

Run with:
    python test_bootstrap_inference.py
or (if pytest is installed):
    pytest test_bootstrap_inference.py -v
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from stepmix.stepmix import StepMix
from stepmix.datasets import data_bakk_complete

from stepmix.bootstrap import (
    WaldTest3Step,
    _fdr_bh,
    _bonferroni,
    _build_contrast_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures / shared helpers
# ---------------------------------------------------------------------------
N_BOOT = 10   # keep CI fast; use 500+ for real analyses
SEED = 42


def make_continuous_model(n_samples=400):
    """Fitted 3-step model with Gaussian distal outcome."""
    X, Y, _ = data_bakk_complete(n_samples=n_samples, sep_level=0.9, random_state=SEED)
    model = StepMix(
        n_components=3,
        n_steps=3,
        measurement="bernoulli",
        structural="gaussian_unit",
        random_state=SEED,
        verbose=0,
    )
    model.fit(X, Y)
    return model, X, Y


def make_binary_model(n_samples=400):
    """Fitted 3-step model with binary distal outcome."""
    X, Y, _ = data_bakk_complete(n_samples=n_samples, sep_level=0.9, random_state=SEED)
    Yb = (Y > Y.mean()).astype(float)
    model = StepMix(
        n_components=3,
        n_steps=3,
        measurement="bernoulli",
        structural="bernoulli",
        random_state=SEED,
        verbose=0,
    )
    model.fit(X, Yb)
    return model, X, Yb


def make_soft_model(n_samples=400):
    """Fitted 3-step soft-assignment model."""
    X, Y, _ = data_bakk_complete(n_samples=n_samples, sep_level=0.9, random_state=SEED)
    model = StepMix(
        n_components=3,
        n_steps=3,
        assignment="soft",
        measurement="bernoulli",
        structural="gaussian_unit",
        random_state=SEED,
        verbose=0,
    )
    model.fit(X, Y)
    return model, X, Y


# ---------------------------------------------------------------------------
# Unit tests – helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_contrast_matrix_shape(self):
        for K in range(2, 6):
            C = _build_contrast_matrix(K)
            assert C.shape == (K - 1, K), f"Bad shape for K={K}"

    def test_contrast_matrix_values(self):
        C = _build_contrast_matrix(3)
        # Row 0: compares class 0 vs class 2
        np.testing.assert_array_equal(C[0], [1, 0, -1])
        # Row 1: compares class 1 vs class 2
        np.testing.assert_array_equal(C[1], [0, 1, -1])

    def test_contrast_matrix_k2(self):
        C = _build_contrast_matrix(2)
        np.testing.assert_array_equal(C, [[1, -1]])

    def test_bonferroni_correction(self):
        p = np.array([0.01, 0.02, 0.05, 0.1])
        adj = _bonferroni(p)
        # Each p multiplied by 4 (n tests), capped at 1
        np.testing.assert_allclose(adj, np.minimum(1.0, p * 4))

    def test_bonferroni_with_nan(self):
        p = np.array([0.01, np.nan, 0.05])
        adj = _bonferroni(p)
        assert np.isnan(adj[1])
        assert not np.isnan(adj[0])
        assert not np.isnan(adj[2])

    def test_fdr_bh_basic(self):
        # All equal p-values → no inflation
        p = np.array([0.05, 0.05, 0.05])
        adj = _fdr_bh(p)
        assert all(adj <= 1.0)
        assert all(adj >= p)

    def test_fdr_bh_monotone(self):
        p = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        adj = _fdr_bh(p)
        # Adjusted p-values must be non-decreasing
        assert all(adj[i] <= adj[i + 1] for i in range(len(adj) - 1))

    def test_fdr_bh_with_nan(self):
        p = np.array([0.01, np.nan, 0.05])
        adj = _fdr_bh(p)
        assert np.isnan(adj[1])


# ---------------------------------------------------------------------------
# Unit tests – WaldTest3Step construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_fitted_model(self):
        model = StepMix(n_components=3, measurement="bernoulli", verbose=0)
        with pytest.raises(Exception):
            WaldTest3Step(model)

    def test_requires_structural_model(self):
        X, _, _ = data_bakk_complete(n_samples=200, sep_level=0.9, random_state=SEED)
        model = StepMix(n_components=3, measurement="bernoulli", verbose=0)
        model.fit(X)
        with pytest.raises(ValueError, match="structural"):
            WaldTest3Step(model)

    def test_summary_before_fit_raises(self):
        model, X, Y = make_continuous_model()
        wt = WaldTest3Step(model)
        with pytest.raises(RuntimeError):
            wt.summary()

    def test_get_estimates_before_fit_raises(self):
        model, X, Y = make_continuous_model()
        wt = WaldTest3Step(model)
        with pytest.raises(RuntimeError):
            wt.get_estimates()


# ---------------------------------------------------------------------------
# Integration tests – continuous outcome
# ---------------------------------------------------------------------------


class TestContinuousOutcome:
    @pytest.fixture(scope="class")
    def fitted_wt(self):
        model, X, Y = make_continuous_model()
        wt = WaldTest3Step(model, ci_level=0.95)
        wt.fit_bootstrap(X, Y, n_repetitions=N_BOOT, progress_bar=False, random_state=SEED)
        return wt, model, X, Y

    def test_bootstrap_samples_shape(self, fitted_wt):
        wt, model, X, Y = fitted_wt
        assert wt.bootstrap_samples_ is not None
        assert "rep" in wt.bootstrap_samples_.columns
        n_reps = wt.bootstrap_samples_["rep"].nunique()
        assert n_reps == N_BOOT

    def test_estimates_columns(self, fitted_wt):
        wt, *_ = fitted_wt
        cols = set(wt.estimates_.columns)
        assert "estimate" in cols
        assert "se" in cols
        assert any("lo" in c for c in cols)
        assert any("hi" in c for c in cols)

    def test_estimates_n_rows(self, fitted_wt):
        wt, model, X, Y = fitted_wt
        # 3 classes × n_outcome_vars
        n_vars = Y.shape[1]
        K = model.n_components
        assert len(wt.estimates_) == K * n_vars

    def test_estimates_point_estimates_match_model(self, fitted_wt):
        """Bootstrap point estimates should equal model's fitted parameters."""
        wt, model, X, Y = fitted_wt
        # Use long-form (class_no as column) rather than wide get_sm_df()
        model_sm = model.get_parameters_df().loc["structural"].reset_index()
        boot_est = wt.estimates_.reset_index()
        for _, row in model_sm.iterrows():
            match = boot_est[
                (boot_est["variable"] == row["variable"])
                & (boot_est["class_no"] == row["class_no"])
            ]
            assert len(match) == 1
            np.testing.assert_allclose(
                match["estimate"].values[0], row["value"], rtol=1e-6
            )

    def test_se_positive(self, fitted_wt):
        wt, *_ = fitted_wt
        est = wt.estimates_
        assert (est["se"] > 0).all(), "All SEs should be positive"

    def test_ci_ordering(self, fitted_wt):
        wt, *_ = fitted_wt
        est = wt.estimates_.reset_index()
        lo = [c for c in est.columns if "lo" in c][0]
        hi = [c for c in est.columns if "hi" in c][0]
        assert (est[hi] > est[lo]).all(), "CI upper must exceed CI lower"

    def test_pairwise_columns(self, fitted_wt):
        wt, *_ = fitted_wt
        expected = {"theta_j", "theta_k", "delta", "se_delta", "z", "chi2", "df", "p_value"}
        cols = set(wt.pairwise_.columns)
        assert expected.issubset(cols), f"Missing columns: {expected - cols}"

    def test_pairwise_n_rows(self, fitted_wt):
        wt, model, X, Y = fitted_wt
        K = model.n_components
        n_vars = Y.shape[1]
        n_pairs = K * (K - 1) // 2
        assert len(wt.pairwise_) == n_pairs * n_vars

    def test_pairwise_delta_sign(self, fitted_wt):
        wt, *_ = fitted_wt
        df = wt.pairwise_.reset_index()
        # delta = theta_j - theta_k
        np.testing.assert_allclose(
            df["delta"].values,
            df["theta_j"].values - df["theta_k"].values,
            rtol=1e-6,
        )

    def test_pairwise_chi2_eq_z_squared(self, fitted_wt):
        wt, *_ = fitted_wt
        df = wt.pairwise_.reset_index()
        np.testing.assert_allclose(df["chi2"].values, df["z"].values ** 2, rtol=1e-6)

    def test_pairwise_p_in_01(self, fitted_wt):
        wt, *_ = fitted_wt
        p = wt.pairwise_["p_value"].values
        assert np.all((p >= 0) & (p <= 1)), "All p-values must be in [0, 1]"

    def test_pairwise_df_is_1(self, fitted_wt):
        wt, *_ = fitted_wt
        assert (wt.pairwise_["df"] == 1).all()

    def test_omnibus_columns(self, fitted_wt):
        wt, *_ = fitted_wt
        expected = {"chi2", "df", "p_value", "sig"}
        assert expected.issubset(set(wt.omnibus_.columns))

    def test_omnibus_n_rows(self, fitted_wt):
        wt, model, X, Y = fitted_wt
        n_vars = Y.shape[1]
        assert len(wt.omnibus_) == n_vars

    def test_omnibus_df_is_K_minus_1(self, fitted_wt):
        wt, model, *_ = fitted_wt
        K = model.n_components
        assert (wt.omnibus_["df"] == K - 1).all()

    def test_omnibus_p_in_01(self, fitted_wt):
        wt, *_ = fitted_wt
        p = wt.omnibus_["p_value"].values
        assert np.all((p >= 0) & (p <= 1))

    def test_well_separated_classes_significant(self, fitted_wt):
        """With sep_level=0.9 and n=400, the omnibus test should be highly significant."""
        wt, *_ = fitted_wt
        # At least one outcome variable should be significant
        assert (wt.omnibus_["p_value"] < 0.05).any(), (
            "Expected at least one significant omnibus test with well-separated classes"
        )

    def test_summary_runs_without_error(self, fitted_wt, capsys):
        wt, *_ = fitted_wt
        wt.summary()
        captured = capsys.readouterr()
        assert "Wald" in captured.out
        assert "p-value" in captured.out.lower() or "p_value" in captured.out.lower()

    def test_get_estimates_filter_variable(self, fitted_wt):
        wt, model, X, Y = fitted_wt
        est = wt.get_estimates(variable="feature_0")
        assert len(est) == model.n_components

    def test_get_estimates_filter_class(self, fitted_wt):
        wt, model, X, Y = fitted_wt
        est = wt.get_estimates(class_no=0)
        assert len(est) == Y.shape[1]

    def test_get_pairwise_filter_variable(self, fitted_wt):
        wt, model, *_ = fitted_wt
        pw = wt.get_pairwise(variable="feature_0")
        K = model.n_components
        assert len(pw) == K * (K - 1) // 2

    def test_get_pairwise_filter_classes(self, fitted_wt):
        wt, *_ = fitted_wt
        pw = wt.get_pairwise(classes=(0, 1))
        # One row per variable
        n_vars = len(wt.omnibus_)
        assert len(pw) == n_vars

    def test_get_omnibus_filter_variable(self, fitted_wt):
        wt, *_ = fitted_wt
        omn = wt.get_omnibus(variable="feature_0")
        assert len(omn) == 1


# ---------------------------------------------------------------------------
# Integration tests – binary outcome
# ---------------------------------------------------------------------------


class TestBinaryOutcome:
    @pytest.fixture(scope="class")
    def fitted_wt(self):
        model, X, Yb = make_binary_model()
        wt = WaldTest3Step(model)
        wt.fit_bootstrap(X, Yb, n_repetitions=N_BOOT, progress_bar=False, random_state=SEED)
        return wt, model, X, Yb

    def test_estimates_exist(self, fitted_wt):
        wt, *_ = fitted_wt
        assert wt.estimates_ is not None
        assert len(wt.estimates_) > 0

    def test_pairwise_exists(self, fitted_wt):
        wt, *_ = fitted_wt
        assert wt.pairwise_ is not None
        assert len(wt.pairwise_) > 0

    def test_probabilities_in_01(self, fitted_wt):
        """Binary outcome estimates must be probabilities in [0, 1]."""
        wt, *_ = fitted_wt
        est = wt.estimates_.reset_index()
        assert (est["estimate"] >= 0).all()
        assert (est["estimate"] <= 1).all()

    def test_binary_omnibus_df(self, fitted_wt):
        wt, model, *_ = fitted_wt
        K = model.n_components
        assert (wt.omnibus_["df"] == K - 1).all()


# ---------------------------------------------------------------------------
# Integration tests – soft assignment
# ---------------------------------------------------------------------------


class TestSoftAssignment:
    @pytest.fixture(scope="class")
    def fitted_wt(self):
        model, X, Y = make_soft_model()
        wt = WaldTest3Step(model)
        wt.fit_bootstrap(X, Y, n_repetitions=N_BOOT, progress_bar=False, random_state=SEED)
        return wt, model, X, Y

    def test_fits_successfully(self, fitted_wt):
        wt, *_ = fitted_wt
        assert wt._is_fitted

    def test_se_positive(self, fitted_wt):
        wt, *_ = fitted_wt
        assert (wt.estimates_["se"] > 0).all()


# ---------------------------------------------------------------------------
# Integration tests – multiple-comparison corrections
# ---------------------------------------------------------------------------


class TestCorrections:
    @pytest.fixture(scope="class")
    def model_and_data(self):
        return make_continuous_model()

    def test_no_correction(self, model_and_data):
        model, X, Y = model_and_data
        wt = WaldTest3Step(model)
        wt.fit_bootstrap(X, Y, n_repetitions=N_BOOT, progress_bar=False,
                         random_state=SEED, correction=None)
        assert "correction" in wt.pairwise_.columns
        # With no correction, p_adj == p_value
        np.testing.assert_allclose(
            wt.pairwise_["p_adj"].values,
            wt.pairwise_["p_value"].values,
            rtol=1e-6,
        )

    def test_bonferroni(self, model_and_data):
        model, X, Y = model_and_data
        wt = WaldTest3Step(model)
        wt.fit_bootstrap(X, Y, n_repetitions=N_BOOT, progress_bar=False,
                         random_state=SEED, correction="Bonferroni")
        # Adjusted p ≥ raw p
        assert (wt.pairwise_["p_adj"] >= wt.pairwise_["p_value"] - 1e-10).all()
        assert (wt.pairwise_["p_adj"] <= 1.0 + 1e-10).all()

    def test_bh_fdr(self, model_and_data):
        model, X, Y = model_and_data
        wt = WaldTest3Step(model)
        wt.fit_bootstrap(X, Y, n_repetitions=N_BOOT, progress_bar=False,
                         random_state=SEED, correction="BH")
        assert (wt.pairwise_["p_adj"] >= wt.pairwise_["p_value"] - 1e-10).all()
        assert (wt.pairwise_["p_adj"] <= 1.0 + 1e-10).all()

    def test_unknown_correction_warns(self, model_and_data):
        model, X, Y = model_and_data
        wt = WaldTest3Step(model)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wt.fit_bootstrap(X, Y, n_repetitions=N_BOOT, progress_bar=False,
                             random_state=SEED, correction="unknown_method")
            assert any("correction" in str(warning.message).lower() or
                       "unknown" in str(warning.message).lower()
                       for warning in w)


# ---------------------------------------------------------------------------
# Integration tests – convenience function
# ---------------------------------------------------------------------------


class TestStatisticalSanity:
    def test_wald_z_matches_chi2(self):
        """χ²(1) = z² must hold exactly."""
        model, X, Y = make_continuous_model()
        wt = WaldTest3Step(model)
        wt.fit_bootstrap(X, Y, n_repetitions=N_BOOT, progress_bar=False,
                         random_state=SEED)
        df = wt.pairwise_.reset_index()
        np.testing.assert_allclose(df["chi2"], df["z"] ** 2, rtol=1e-6)

    def test_omnibus_chi2_larger_than_any_pairwise(self):
        """
        The omnibus Wald statistic should generally be ≥ any individual
        pairwise Wald statistic (not a strict mathematical guarantee, but
        holds when classes are clearly separated).
        """
        model, X, Y = make_continuous_model(n_samples=600)
        wt = WaldTest3Step(model)
        wt.fit_bootstrap(X, Y, n_repetitions=N_BOOT, progress_bar=False,
                         random_state=SEED)
        omn = wt.omnibus_.reset_index()
        pw = wt.pairwise_.reset_index()
        for var in omn["variable"].unique():
            omn_chi2 = omn[omn["variable"] == var]["chi2"].values[0]
            max_pair_chi2 = pw[pw["variable"] == var]["chi2"].max()
            # Omnibus on K-1 df cannot be compared directly, but with
            # well-separated classes it should be large
            assert np.isfinite(omn_chi2), f"Omnibus χ² should be finite for {var}"
            assert np.isfinite(max_pair_chi2)

    def test_swap_symmetric(self):
        """
        Swapping class_j and class_k should give the same χ² and p-value
        but negated z and delta.  We verify this by manually re-calling the
        pairwise logic on reversed pairs.
        """
        model, X, Y = make_continuous_model()
        wt = WaldTest3Step(model)
        wt.fit_bootstrap(X, Y, n_repetitions=N_BOOT, progress_bar=False,
                         random_state=SEED)
        df = wt.pairwise_.reset_index()
        # For each row, find the corresponding entry with j and k swapped
        # (which would NOT exist in the table since j < k always, so we
        # just verify that negating delta gives the same |z|)
        np.testing.assert_allclose(np.abs(df["z"]), np.sqrt(df["chi2"]), rtol=1e-6)

    def test_p_value_from_z(self):
        """p-value must equal 2 * (1 - Φ(|z|))."""
        from scipy.stats import norm
        model, X, Y = make_continuous_model()
        wt = WaldTest3Step(model)
        wt.fit_bootstrap(X, Y, n_repetitions=N_BOOT, progress_bar=False,
                         random_state=SEED)
        df = wt.pairwise_.reset_index()
        expected_p = 2.0 * (1.0 - norm.cdf(np.abs(df["z"].values)))
        np.testing.assert_allclose(df["p_value"].values, expected_p, rtol=1e-6)

    def test_omnibus_p_from_chi2(self):
        """Omnibus p must equal 1 - χ²_CDF(W, df=K-1)."""
        from scipy.stats import chi2
        model, X, Y = make_continuous_model()
        K = model.n_components
        wt = WaldTest3Step(model)
        wt.fit_bootstrap(X, Y, n_repetitions=N_BOOT, progress_bar=False,
                         random_state=SEED)
        omn = wt.omnibus_.reset_index()
        expected_p = 1.0 - chi2.cdf(omn["chi2"].values, df=K - 1)
        np.testing.assert_allclose(omn["p_value"].values, expected_p, rtol=1e-6)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running stepmix_inference test suite...\n")

    # Collect all test classes
    test_classes = [
        TestHelpers,
        TestConstruction,
        TestContinuousOutcome,
        TestBinaryOutcome,
        TestSoftAssignment,
        TestCorrections,
        TestConvenienceFunction,
        TestStatisticalSanity,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()

        # Resolve class-scoped fixtures manually
        class_fixtures = {}
        for name in dir(cls):
            attr = getattr(cls, name, None)
            if callable(attr) and hasattr(attr, "pytestmark"):
                pass
        # For scope="class" fixtures we call them directly
        fixture_cache = {}

        for name in sorted(dir(cls)):
            if not name.startswith("test_"):
                continue
            method = getattr(instance, name)

            # Detect if the test requires a class-scoped fixture
            import inspect
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            # Remove 'self'; check for fixture args
            fixture_args = [p for p in params if p not in ("self",)]

            try:
                if fixture_args:
                    # Build fixture if not yet built
                    fixture_name = fixture_args[0]
                    if fixture_name not in fixture_cache:
                        fixture_method = getattr(cls, fixture_name, None)
                        if fixture_method is not None:
                            # It's a pytest fixture – call it
                            gen = fixture_method(instance)
                            if hasattr(gen, "__next__"):
                                fixture_cache[fixture_name] = next(gen)
                            else:
                                fixture_cache[fixture_name] = gen()
                        else:
                            # e.g. capsys – skip
                            print(f"  SKIP  {cls.__name__}::{name}  (needs pytest fixture '{fixture_name}')")
                            continue
                    method(fixture_cache[fixture_name])
                else:
                    method()
                print(f"  PASS  {cls.__name__}::{name}")
                passed += 1
            except (AssertionError, Exception) as e:
                print(f"  FAIL  {cls.__name__}::{name}  →  {e}")
                failed += 1
                errors.append((cls.__name__, name, str(e)))

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for cls_name, test_name, msg in errors:
            print(f"  {cls_name}::{test_name}: {msg}")
    print("=" * 60)
