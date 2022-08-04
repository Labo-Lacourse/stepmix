"""Demo script for FIML models with missing values."""
import numpy as np
from stepmix.stepmix import StepMix
from stepmix.datasets import data_gaussian_diag

# Specify covariance here. 'unit', 'spherical' or 'diag'
COV = "diag"
N_SAMPLES = 1000


def print_results(log_likelihoods):
    # Helper function to quickly print experiment results
    def get_exp_name(i):
        if i == 0:
            return "Fully observed (full implementation)"
        if i == 1:
            return "Fully observed  (fast nan implementation)"
        if i == 2:
            return "2/3 observed    (fast nan implementation)"
        if i == 3:
            return "2/3 observed    (debug nan implementation)"

    for i, ll in enumerate(log_likelihoods):
        print(f"{get_exp_name(i)} log-likelihood : {ll:.3f}")


np.random.seed(42)
print("Experiment on full data with the full implementation...")
X, Y, c = data_gaussian_diag(
    n_samples=N_SAMPLES, sep_level=0.9, random_state=42, nan_ratio=0.00
)

ll_list = []
means_list = []
cov_list = []

# Run experiment for 1-step on full data with the 'full' implementations
m = StepMix(
    n_steps=1,
    n_components=3,
    measurement="bernoulli",
    structural=f"gaussian_{COV}",
    rel_tol=1e-5,
    n_init=10,
    random_state=42,
    max_iter=100,
)
m.fit(X, Y)

means_list.append(m.get_parameters()["structural"]["means"])
cov_list.append(m.get_parameters()["structural"]["covariances"])
ll_list.append(m.score(X, Y))

# Run experiment for 1-step on full data with the 'nan' implementations
print("\nExperiment on full data with the nan implementation...")
m = StepMix(
    n_steps=1,
    n_components=3,
    measurement="bernoulli_nan",
    structural=f"gaussian_{COV}_nan",
    rel_tol=1e-5,
    n_init=10,
    random_state=42,
    max_iter=100,
)
m.fit(X, Y)

means_list.append(m.get_parameters()["structural"]["means"])
cov_list.append(m.get_parameters()["structural"]["covariances"])
ll_list.append(m.score(X, Y))

# Run experiment for 1-step on partial data with the 'nan' implementations
print("\nExperiment with 2/3 of observed data with the fast nan implementation...")
X_partial, Y_partial, c = data_gaussian_diag(
    n_samples=N_SAMPLES, sep_level=0.9, random_state=42, nan_ratio=1 / 3
)

# Run experiment for 1-step on partial data
m = StepMix(
    n_steps=1,
    n_components=3,
    measurement="bernoulli_nan",
    structural=f"gaussian_{COV}_nan",
    rel_tol=1e-5,
    n_init=10,
    random_state=42,
    max_iter=100,
)
m.fit(X_partial, Y_partial)

means_list.append(m.get_parameters()["structural"]["means"])
cov_list.append(m.get_parameters()["structural"]["covariances"])

# Score the complete dataset
ll_list.append(m.score(X, Y))

# Run experiment for 1-step on partial data using the debug likelihood
print(
    "\nExperiment with 2/3 of observed data with the debug nan implementation (this is slow)..."
)
m = StepMix(
    n_steps=1,
    n_components=3,
    measurement="bernoulli_nan",
    structural=f"gaussian_{COV}_nan",
    rel_tol=1e-5,
    n_init=10,
    random_state=42,
    max_iter=100,
    structural_params=dict(debug_likelihood=True),
)
m.fit(X_partial, Y_partial)

means_list.append(m.get_parameters()["structural"]["means"])
cov_list.append(m.get_parameters()["structural"]["covariances"])

# Score the complete dataset
ll_list.append(m.score(X, Y))

# Report ll
print("\n\nExperiment summary...")
print_results(ll_list)

print("\n\nStructural means of fully observed implementation...")
print(means_list[0].round(2))

print("\n\nStructural means of fast nan implementation on all data...")
print(means_list[1].round(2))

print("\n\nStructural means of fast nan implementation on 2/3 of data...")
print(means_list[2].round(2))

print("\n\nStructural covariances of fully observed implementation...")
print(cov_list[0].round(2))

print("\n\nStructural covariances of fast nan implementation on all data...")
print(cov_list[1].round(2))

print("\n\nStructural covariances of fast nan implementation on 2/3 of data...")
print(cov_list[2].round(2))
