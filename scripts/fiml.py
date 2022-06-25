"""Demo script for FIML models with missing values."""
import numpy as np
from lca.lca import LCA
from lca.datasets import data_bakk_response, data_generation_gaussian


def print_results(log_likelihoods, means):
    # Helper function to quickly print experiment results
    def get_exp_name(i):
        return 'Fully observed' if i == 0 else '2/3 observed  '
    for i, ll in enumerate(log_likelihoods):
        print(f'{get_exp_name(i)} log-likelihood : {ll:.3f}')
    for i, mean in enumerate(means):
        print(f'{get_exp_name(i)} means : {np.sort(mean.flatten())}')


np.random.seed(42)
print('Bakk experiment on full data...')
X, Y, c = data_bakk_response(n_samples=3000, sep_level=.9, random_state=42)

# Shuffle data so the missing patterns are randomly assigned between classes
shuffle_mask = np.random.permutation(X.shape[0])
X, Y, c = X[shuffle_mask], Y[shuffle_mask], c[shuffle_mask]

ll_list = []
means_list = []

# Run experiment for 1-step on full data
m = LCA(n_steps=1, n_components=3, measurement='bernoulli', structural='gaussian_unit', tol=1e-5, n_init=10,
        random_state=42, max_iter=200)
m.fit(X, Y)

print(m.get_parameters()['measurement']['pis'])
means_list.append(m.get_parameters()['structural']['means'])
ll_list.append(m.score(X, Y))

print('Bakk experiment with 1/3 of data missing in the measurement...')
X_partial, Y_partial = X.copy().astype(float), Y.copy().astype(float)
X_partial[:1000, [0, 1]] = np.nan
X_partial[1000:2000, [2, 3]] = np.nan
X_partial[2000:, [4, 5]] = np.nan

X_partial[0] = np.nan

# Run experiment for 1-step on partial data
m = LCA(n_steps=1, n_components=3, measurement='bernoulli_nan', structural='gaussian_unit', tol=1e-5, n_init=10,
        random_state=42, max_iter=200)
m.fit(X_partial, Y_partial)
print(m.get_parameters()['measurement']['pis'])
means_list.append(m.get_parameters()['structural']['means'])
ll_list.append(m.score(X, Y))

# Report ll and means
print("\n\nExperiment summary...")
print_results(ll_list, means_list)
