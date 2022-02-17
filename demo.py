"""Demo script."""
import numpy as np
from lca.lca import LCA
from lca.datasets import data_generation_Bakk, data_generation_Hwk4


def print_results(log_likelihoods, means):
    # Helper function to quickly print experiment results
    for i, ll in enumerate(log_likelihoods):
        print(f'{i + 1}-step log-likelihood : {ll:.3f}')
    for i, mean in enumerate(means):
        print(f'{i + 1}-step means : {np.sort(mean.flatten())}')


########################################################################################################################
# Various multi-step methods are available
print('Bakk experiment...')
X, Y = data_generation_Bakk(sample_size=3000, sep_level=.7, random_state=42)

ll_list = []
means_list = []

# Run experiment for 1-step, 2-step and 3-step
for n_steps in [1, 2, 3]:
    m = LCA(n_steps=n_steps, n_components=3, measurement='bernoulli', structural='gaussian_unit', tol=1e-5, n_init=10,
            random_state=42, max_iter=200)
    m.fit(X, Y)

    means_list.append(m.get_parameters()['structural']['means'])
    ll_list.append(m.score(X, Y))

# Report ll and means
print_results(ll_list, means_list)

########################################################################################################################
# Multi-step methods can also be equivalently decomposed into individual steps
# Calling em(X) fits the measurement model. em(X, Y) fits the complete model. An additional freeze_measurement
# argument can be passed to perform the second step of the Two-step approach.
# Other methods allow to run three-step estimation.
print('\n\nBakk experiment (explicit step decomposition)...')

# ONE-STEP
m_1 = LCA(n_components=3, measurement='bernoulli', structural='gaussian_unit', tol=1e-5, n_init=10,
          random_state=42, max_iter=200)
# 1) Maximum likelihood with both measurement and structural models
m_1.em(X, Y)

# TWO-STEP
m_2 = LCA(n_components=3, measurement='bernoulli', structural='gaussian_unit', tol=1e-5, n_init=10,
          random_state=42, max_iter=200)
# 1) Fit the measurement model
# 2) Fit the structural model by keeping the parameters of the measurement model fixed
m_2.em(X)
m_2.em(X, Y, freeze_measurement=True)

# THREE-STEP
m_3 = LCA(n_components=3, measurement='bernoulli', structural='gaussian_unit', tol=1e-5, n_init=10,
          random_state=42, max_iter=200)
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
    means_list.append(m.get_parameters()['structural']['means'])
    ll_list.append(m.score(X, Y))
print_results(ll_list, means_list)

########################################################################################################################
# Measurement and structural models can be specified
# Here we use the gaussian data as measurements and the bernoulli data for the structural model
# Notice the first experiment here should yield the same result as previous 1-step results
print('\n\nBakk experiment (gaussian measurement, structural bernoulli)...')
ll_list = []
means_list = []

# Run experiment for 1-step, 2-step and 3-step
for n_steps in [1, 2, 3]:
    m = LCA(n_steps=n_steps, n_components=3, measurement='gaussian_unit', structural='bernoulli', tol=1e-5, n_init=10,
            random_state=42, max_iter=200)
    m.fit(Y, X)

    means_list.append(m.get_parameters()['measurement']['means'])
    ll_list.append(m.score(Y, X))

# Report ll and means
print_results(ll_list, means_list)

########################################################################################################################
# Gaussian emission models with various covariance estimators are available. The code is based on the
# sklearn GaussianMixture class
print('\n\nHmwk4 experiment...')
for cov_string in ["unit", "spherical", "tied", "diag", "full"]:
    m = LCA(n_steps=1, n_components=4, measurement='bernoulli', structural='gaussian_' + cov_string,
            tol=1e-5, n_init=10, random_state=42, max_iter=200)
    X, Y = data_generation_Hwk4(sample_size=3000, random_state=42,
                                sep_level=.9)  # Bernoulli measurements, Gaussian responses
    m.fit(X, Y)
    pr = f'Log-likelihood of Gaussian Structural Model with {cov_string} covariance'
    print(f'{pr:<70} : {m.score(X, Y):.3f}')
