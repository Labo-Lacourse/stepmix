"""Demo script."""
import numpy as np
from stepmix.stepmix import StepMix
from stepmix.datasets import data_bakk_response, data_generation_gaussian


def print_results(log_likelihoods, means):
    # Helper function to quickly print experiment results
    for i, ll in enumerate(log_likelihoods):
        print(f"{i + 1}-step log-likelihood : {ll:.3f}")
    for i, mean in enumerate(means):
        print(f"{i + 1}-step means : {np.sort(mean.flatten())}")


########################################################################################################################
# Various stepwise methods are available
print("Bakk experiment...")
X, Y, _ = data_bakk_response(n_samples=3000, sep_level=0.7, random_state=42)

ll_list = []
means_list = []

# Run experiment for 1-step, 2-step and 3-step
for n_steps in [1, 2, 3]:
    m = StepMix(
        n_steps=n_steps,
        n_components=3,
        measurement="bernoulli",
        structural="gaussian_unit",
        abs_tol=1e-5,
        n_init=10,
        random_state=42,
        max_iter=200,
        verbose=1,
    )
    m.fit(X, Y)

    means_list.append(m.get_parameters()["structural"]["means"])
    ll_list.append(m.score(X, Y))

# Report ll and means
print_results(ll_list, means_list)

########################################################################################################################
# Multi-step methods can also be equivalently decomposed into individual steps
# Calling em(X) fits the measurement model. em(X, Y) fits the complete model. An additional freeze_measurement
# argument can be passed to perform the second step of the Two-step approach.
# Other methods allow to run three-step estimation.
print("\n\nBakk experiment (explicit step decomposition)...")

# ONE-STEP
m_1 = StepMix(
    n_components=3,
    measurement="bernoulli",
    structural="gaussian_unit",
    abs_tol=1e-5,
    n_init=10,
    random_state=42,
    max_iter=200,
)
# 1) Maximum likelihood with both measurement and structural models
m_1.em(X, Y)

# TWO-STEP
m_2 = StepMix(
    n_components=3,
    measurement="bernoulli",
    structural="gaussian_unit",
    abs_tol=1e-5,
    n_init=10,
    random_state=42,
    max_iter=200,
)
# 1) Fit the measurement model
# 2) Fit the structural model by keeping the parameters of the measurement model fixed
m_2.em(X)
m_2.em(X, Y, freeze_measurement=True)

# THREE-STEP
m_3 = StepMix(
    n_components=3,
    measurement="bernoulli",
    structural="gaussian_unit",
    abs_tol=1e-5,
    n_init=10,
    random_state=42,
    max_iter=200,
)
# 1) Fit the measurement model
# 2) Assign class probabilities
# 3) M-step on the structural model
m_3.em(X)
resp = m_3.predict_proba(X)
m_3.m_step_structural(resp, Y)

# Report ll and means
ll_list = []
means_list = []
for m in [m_1, m_2, m_3]:
    means_list.append(m.get_parameters()["structural"]["means"])
    ll_list.append(m.score(X, Y))
print_results(ll_list, means_list)

########################################################################################################################
# Measurement and structural models can be specified
# Here we use the gaussian data as measurements and the bernoulli data for the structural model
# Notice the first experiment here should yield the same result as previous 1-step results
print("\n\nBakk experiment (gaussian measurement, structural bernoulli)...")
ll_list = []
means_list = []

# Run experiment for 1-step, 2-step and 3-step
for n_steps in [1, 2, 3]:
    m = StepMix(
        n_steps=n_steps,
        n_components=3,
        measurement="gaussian_unit",
        structural="bernoulli",
        abs_tol=1e-5,
        n_init=10,
        random_state=42,
        max_iter=200,
    )
    m.fit(Y, X)

    means_list.append(m.get_parameters()["measurement"]["means"])
    ll_list.append(m.score(Y, X))

# Report ll and means
print_results(ll_list, means_list)

########################################################################################################################
# Gaussian emission models with various covariance estimators are available. The code is based on the
# sklearn GaussianMixture class
print("\n\nGaussian experiment...")
for cov_string in ["unit", "spherical", "tied", "diag", "full"]:
    m = StepMix(
        n_steps=1,
        n_components=4,
        measurement="bernoulli",
        structural="gaussian_" + cov_string,
        abs_tol=1e-5,
        n_init=10,
        random_state=42,
        max_iter=200,
    )
    X, Y, c = data_generation_gaussian(
        n_samples=3000, random_state=42, sep_level=0.9
    )  # Bernoulli measurements, Gaussian responses

    m.fit(X, Y)
    pr = f"Log-likelihood of Gaussian Structural Model with {cov_string} covariance"
    print(f"{pr:<70} : {m.score(X, Y):.3f}")

########################################################################################################################
# 3-step estimation supports modal and soft assignments.
X, Y, _ = data_bakk_response(n_samples=10000, sep_level=0.9, random_state=42)
print("\n\nBakk experiment with different 1-step and 2-step...")
for step in [1, 2, 3]:
    # Run experiment for 1-step, 2-step and 3-step
    m = StepMix(
        n_steps=step,
        n_components=3,
        measurement="bernoulli",
        structural="gaussian_unit",
        abs_tol=1e-5,
        n_init=10,
        random_state=42,
        max_iter=200,
    )
    m.fit(X, Y)

    # Get mu_2 estimation (which we assume is the max parameter)
    # The target value if 1
    mu_2 = m.get_parameters()["structural"]["means"].max()

    pr = f"Mean bias of {step}-step estimation"
    print(f"{pr:<75} : {mu_2 - 1:.3f}")

print("\nBakk experiment with different 3-step flavors...")
for assignment in ["modal", "soft"]:
    for correction in [None, "BCH", "ML"]:
        # Run experiment for 1-step, 2-step and 3-step
        m = StepMix(
            n_steps=3,
            n_components=3,
            measurement="bernoulli",
            structural="gaussian_unit",
            abs_tol=1e-5,
            n_init=10,
            random_state=42,
            max_iter=1000,
            assignment=assignment,
            correction=correction,
        )
        m.fit(X, Y)

        # Get mu_2 estimation (which we assume is the max parameter)
        # The target value if 1
        mu_2 = m.get_parameters()["structural"]["means"].max()

        pr = f"Mean bias of 3-step estimation with {assignment} assignments and {correction} correction"
        print(f"{pr:<75} : {mu_2 - 1:.3f}")
