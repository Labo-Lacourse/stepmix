"""Demo script for FIML models with missing values."""
import numpy as np
from lca.lca import LCA
from lca.datasets import data_gaussian_diag


def print_results(log_likelihoods, means):
    # Helper function to quickly print experiment results
    def get_exp_name(i):
        if i == 0:
            return 'Fully observed (full implementation)'
        if i == 1:
            return 'Fully observed  (fast nan implementation)'
        if i == 2:
            return '2/3 observed    (fast nan implementation)'
        if i == 3:
            return '2/3 observed    (debug nan implementation)'

    for i, ll in enumerate(log_likelihoods):
        print(f'{get_exp_name(i)} log-likelihood : {ll:.3f}')
    for i, mean in enumerate(means):
        print(f'{get_exp_name(i)} means : {np.sort(mean.flatten()).round(2)}')


np.random.seed(42)
print('Experiment on full data with the full implementation...')
X, Y, c = data_gaussian_diag(n_samples=1000, sep_level=.9, random_state=42, nan_ratio=.00)

ll_list = []
means_list = []

# Run experiment for 1-step on full data with the 'full' implementations
m = LCA(n_steps=1, n_components=3, measurement='bernoulli', structural='gaussian_unit', tol=1e-5, n_init=10,
        random_state=42, max_iter=100)
m.fit(X, Y)

print(m.get_parameters()['measurement']['pis'].round(2))
means_list.append(m.get_parameters()['structural']['means'])
ll_list.append(m.score(X, Y))

# Run experiment for 1-step on full data with the 'nan' implementations
print('\nExperiment on full data with the nan implementation...')
m = LCA(n_steps=1, n_components=3, measurement='bernoulli_nan', structural='gaussian_unit_nan', tol=1e-5, n_init=10,
        random_state=42, max_iter=100)
m.fit(X, Y)

print(m.get_parameters()['measurement']['pis'].round(2))
means_list.append(m.get_parameters()['structural']['means'])
ll_list.append(m.score(X, Y))

# Run experiment for 1-step on partial data with the 'nan' implementations
print('\nExperiment with 2/3 of observed data with the fast nan implementation...')
X_partial, Y_partial, c = data_gaussian_diag(n_samples=1000, sep_level=.9, random_state=42, nan_ratio=1/4)

# Run experiment for 1-step on partial data
m = LCA(n_steps=1, n_components=3, measurement='bernoulli_nan', structural='gaussian_nan', tol=1e-5, n_init=10,
        random_state=42, max_iter=100)
m.fit(X_partial, Y_partial)
print(m.get_parameters()['measurement']['pis'].round(2))
means_list.append(m.get_parameters()['structural']['means'])

# Score the complete dataset
ll_list.append(m.score(X, Y))

# Run experiment for 1-step on partial data using the debug likelihood
print('\nExperiment with 2/3 of observed data with the debug nan implementation...')
m = LCA(n_steps=1, n_components=3, measurement='bernoulli_nan', structural='gaussian_nan', tol=1e-5, n_init=10,
        random_state=42, max_iter=100, structural_params=dict(debug_likelihood=True))
m.fit(X_partial, Y_partial)
print(m.get_parameters()['measurement']['pis'].round(2))
means_list.append(m.get_parameters()['structural']['means'])

# Score the complete dataset
ll_list.append(m.score(X, Y))

# Report ll and means
print("\n\nExperiment summary...")
print_results(ll_list, means_list)
