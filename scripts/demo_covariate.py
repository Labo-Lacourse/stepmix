import numpy as np
from lca.datasets import data_bakk_covariate
from lca.lca import LCA
from sklearn.metrics import adjusted_rand_score
from lca.utils import identify_coef


# Helper function to quickly print experiment results
def print_results(log_likelihoods, rand_scores, means, intercepts):
    for i, ll in enumerate(log_likelihoods):
        print(f'{i + 1}-step log-likelihood : {ll:.3f}')
    for i, rs in enumerate(rand_scores):
        print(f'{i + 1}-step Rand score : {rs:.3f}')
    for i, mean in enumerate(means):
        print(f'{i + 1}-step means : {mean.flatten()}')
    for i, inter in enumerate(intercepts):
        print(f'{i + 1}-step intercepts : {inter.flatten()}')


# Various multi-step methods are available
print('Bakk experiment...')
X, Y, c = data_bakk_covariate(sample_size=5000, sep_level=.9, random_state=42)

ll_list = []
rand_score = []
means_list = []
intercepts_list = []

# Run experiment for 1-step, 2-step and 3-step
for n_steps in [1, 2, 3]:
    m = LCA(n_steps=n_steps, n_components=3, measurement='bernoulli', structural='covariate', n_init=10,
            random_state=42, max_iter=1000, tol=1e-5, structural_params=dict(lr=1e-3,
                                                                             iter=1 if n_steps < 3 else 1000))
    m.fit(X, Y)

    # Model estimates all K coefficients. Apply translation to have a a null reference category
    coef = identify_coef(m.get_parameters()['structural']['coef'].copy())

    # Likelihood and paramters
    ll_list.append(m.score(X, Y))
    means_list.append(coef[0])
    intercepts_list.append(coef[1])

    # Compare clustering quality vs. ground truth
    pred = m.predict(X, Y)
    rand_score.append(adjusted_rand_score(c, pred))

# Report ll, rand score and parameters
print_results(ll_list, rand_score, means_list, intercepts_list)
