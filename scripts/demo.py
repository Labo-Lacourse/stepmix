"""Demo script."""
import numpy as np
from lca.lca import LCA
from lca.datasets import data_generation_Bakk, data_generation_gaussian, data_generation_Hwk4

print('\n\nBakk experiment...')
m = LCA(n_components=3, measurement='bernoulli', structural='gaussian', tol=1e-5, n_init=10, random_state=42, max_iter=200)
X, Y = data_generation_Bakk(sample_size=3000, sep_level=.7, random_state=42)

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
# Equivalently one can use the various fit_N_step methods
print('\n\nBakk experiment...')
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
# Emission models can be chosen (currently gaussian and bernoulli are implemented)
print('\n\nGaussian experiment...')
m = LCA(n_components=3, measurement='gaussian', structural='bernoulli', tol=1e-5, n_init=10, random_state=42, max_iter=200)
X, Y = data_generation_gaussian(sample_size=3000, random_state=42)  # Gaussian measurements, bernoulli responses

# One step
m.fit_1_step(X, Y)
print(f'1-step log-likelihood : {m.avg_log_likelihood(X, Y):.3f}')

# Two step
m.fit_2_step(X, Y)
print(f'2-step log-likelihood : {m.avg_log_likelihood(X, Y):.3f}')

# Three step
m.fit_3_step(X, Y)
print(f'3-step log-likelihood : {m.avg_log_likelihood(X, Y):.3f}')

########################################################################################################################
# Emission models can be configured via the measurement_params and structural_params arguments
print('\n\nHmwk4 experiment...')
for cov in ["spherical", "tied", "diag", "full"]:
    structural_params = dict(covariance_type=cov)
    m = LCA(n_components=3, measurement='bernoulli', structural='gaussian',
            tol=1e-5, n_init=10, random_state=42, max_iter=200, structural_params=structural_params)
    X, Y = data_generation_Hwk4(sample_size=3000, random_state=42, sep_level=.9)  # Bernoulli measurements, Gaussian responses
    m.fit_1_step(X, Y)
    pr = f'Log-likelihood of Gaussian Structural Model with {cov} covariance'
    print(f'{pr:<70} : {m.avg_log_likelihood(X, Y):.3f}')

