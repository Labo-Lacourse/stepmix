"""Demo script."""
import numpy as np
from lca.lca import LCA
from lca.datasets import data_generation_Bakk, data_generation_Hwk4

########################################################################################################################
# Various fit_N_step methods are available
print('\n\nBakk experiment...')
X, Y = data_generation_Bakk(sample_size=3000, sep_level=.7, random_state=42)
m = LCA(n_components=3, measurement='bernoulli', structural='gaussian', tol=1e-5, n_init=10, random_state=42, max_iter=200)

# One step
m.fit_1_step(X, Y)
means_1 = m.get_parameters()['structural']['means']
print(f'1-step log-likelihood : {m.avg_log_likelihood(X, Y):.3f}')

# Two step
m.fit_2_step(X, Y)
means_2 = m.get_parameters()['structural']['means']
print(f'2-step log-likelihood : {m.avg_log_likelihood(X, Y):.3f}')

# Three step
m.fit_3_step(X, Y)
means_3 = m.get_parameters()['structural']['means']
print(f'3-step log-likelihood : {m.avg_log_likelihood(X, Y):.3f}')

# Report means
print(f'\n1-step means : {np.sort(means_1.flatten())}')
print(f'2-step means : {np.sort(means_2.flatten())}')
print(f'3-step means : {np.sort(means_3.flatten())}')

########################################################################################################################
# Multi-step methods can also be equivalently decomposed into individual steps
# Calling fit(X) fits the measurement model. fit(X, Y) fits the complete model. An additional freeze_measurement
# argument can be passed to perform the second step of the Two-step approach.
# Other methods allow to run three-step estimation.
print('\n\nBakk experiment (repeat)...')
m = LCA(n_components=3, measurement='bernoulli', structural='gaussian', tol=1e-5, n_init=10, random_state=42, max_iter=200)

# One step
m.fit(X, Y)
means_1 = m.get_parameters()['structural']['means']
print(f'1-step log-likelihood : {m.avg_log_likelihood(X, Y):.3f}')

# Two step
m.fit(X)  # Fit measurement
m.fit(X, Y, freeze_measurement=True)  # Fit structural
means_2 = m.get_parameters()['structural']['means']
print(f'2-step log-likelihood : {m.avg_log_likelihood(X, Y):.3f}')

# Three step
m.fit(X)  # Fit measurement
resp = m.predict_proba(X)  # Soft assignments
m.m_step_structural(resp, Y)  # Fit structural
means_3 = m.get_parameters()['structural']['means']
print(f'3-step log-likelihood : {m.avg_log_likelihood(X, Y):.3f}')

# Report means
print(f'\n1-step means : {np.sort(means_1.flatten())}')
print(f'2-step means : {np.sort(means_2.flatten())}')
print(f'3-step means : {np.sort(means_3.flatten())}')


########################################################################################################################
# Measurement and structural models can be specified
# Here we use the gaussian data as measurements and the bernoulli data for the structural model
print('\n\nBakk experiment (flipped)...')
m = LCA(n_components=3, measurement='gaussian', structural='bernoulli', tol=1e-5, n_init=10, random_state=42, max_iter=200)

# One step
m.fit_1_step(Y, X)  # Should be equivalent to the previous experiments
means_1 = m.get_parameters()['measurement']['means']
print(f'1-step log-likelihood : {m.avg_log_likelihood(Y, X):.3f}')

# Two step
m.fit_2_step(Y, X)
means_2 = m.get_parameters()['measurement']['means']
print(f'2-step log-likelihood : {m.avg_log_likelihood(Y, X):.3f}')

# Three step
m.fit_3_step(Y, X)
means_3 = m.get_parameters()['measurement']['means']
print(f'3-step log-likelihood : {m.avg_log_likelihood(Y, X):.3f}')

# Report means
print(f'\n1-step means : {np.sort(means_1.flatten())}')
print(f'2-step means : {np.sort(means_2.flatten())}')  # 2-step and 3-step should have the same means
print(f'3-step means : {np.sort(means_3.flatten())}')


########################################################################################################################
# Gaussian emission models with various covariance estimators are available. The code is based on the
# sklearn GaussianMixture class
print('\n\nHmwk4 experiment...')
for cov_string in ["spherical", "tied", "diag", "full"]:
    m = LCA(n_components=4, measurement='bernoulli', structural='gaussian_' + cov_string,
            tol=1e-5, n_init=10, random_state=42, max_iter=200)
    X, Y = data_generation_Hwk4(sample_size=3000, random_state=42, sep_level=.9)  # Bernoulli measurements, Gaussian responses
    m.fit_1_step(X, Y)
    pr = f'Log-likelihood of Gaussian Structural Model with {cov_string} covariance'
    print(f'{pr:<70} : {m.avg_log_likelihood(X, Y):.3f}')

