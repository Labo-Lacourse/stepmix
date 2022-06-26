"""Demo script for FIML models with missing values."""
import numpy as np
from lca.lca import LCA
from lca.datasets import data_bakk_response, data_generation_gaussian


def print_results(log_likelihoods, means):
    # Helper function to quickly print experiment results
    def get_exp_name(i):
        if i == 0:
            return 'Fully observed (full implementation)'
        if i == 1:
            return 'Fully observed  (nan implementation)'
        if i == 2:
            return '2/3 observed    (nan implementation)'

    for i, ll in enumerate(log_likelihoods):
        print(f'{get_exp_name(i)} log-likelihood : {ll:.3f}')
    for i, mean in enumerate(means):
        print(f'{get_exp_name(i)} means : {np.sort(mean.flatten())}')


np.random.seed(42)
print('Bakk experiment on full data with the full implementation...')
X, Y, c = data_bakk_response(n_samples=3000, sep_level=.7, random_state=42)

ll_list = []
means_list = []

# Run experiment for 1-step on full data with the 'full' implementations
m = LCA(n_steps=1, n_components=3, measurement='bernoulli', structural='gaussian_unit', tol=1e-5, n_init=10,
        random_state=42, max_iter=200)
m.fit(X, Y)

print(m.get_parameters()['measurement']['pis'])
means_list.append(m.get_parameters()['structural']['means'])
ll_list.append(m.score(X, Y))

# Run experiment for 1-step on full data with the 'nan' implementations
print('\nBakk experiment on full data with the nan implementation...')
m = LCA(n_steps=1, n_components=3, measurement='bernoulli_nan', structural='gaussian_unit_nan', tol=1e-5, n_init=10,
        random_state=42, max_iter=200)
m.fit(X, Y)

print(m.get_parameters()['measurement']['pis'])
means_list.append(m.get_parameters()['structural']['means'])
ll_list.append(m.score(X, Y))

# Run experiment for 1-step on partial data with the 'nan' implementations
print('\nBakk experiment with 2/3 of observed data...')
X_partial, Y_partial = X.copy().astype(float), Y.copy().astype(float)
X_partial[:1000, [0, 1]] = np.nan
X_partial[1000:2000, [2, 3]] = np.nan
X_partial[2000:, [4, 5]] = np.nan
Y_partial[::3] = np.nan

# Run experiment for 1-step on partial data
m = LCA(n_steps=1, n_components=3, measurement='bernoulli_nan', structural='gaussian_nan', tol=1e-5, n_init=10,
        random_state=42, max_iter=200)
m.fit(X_partial, Y_partial)
print(m.get_parameters()['measurement']['pis'])
means_list.append(m.get_parameters()['structural']['means'])
ll_list.append(m.score(X, Y))

# Report ll and means
print("\n\nExperiment summary...")
print_results(ll_list, means_list)
