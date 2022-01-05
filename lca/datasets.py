import numpy as np
from sklearn.utils.validation import check_random_state


def data_generation_Bakk(sample_size, sep_level, random_state=None):
    # Credit : Robin Legault
    rng = check_random_state(random_state)

    C = 3  # Number of latent classes
    K = 6  # Number of observed indicators
    D = 1  # Dimensions of the response variable Zo

    # True parameters
    # rho[c] = p(X=c)
    rho = np.ones(C) / C

    # mus[k] = E[Z0|X=c]
    mus = np.array([-1, 1, 0])

    # unit variance for all gaussians
    sigmas = np.array([1, 1, 1])

    # pis[k,c] = p(Yk=1|X=c)
    # sep_level = 0.9 #0.9->high, 0.8->medium, 0.7->low
    pis = np.zeros((K, C))
    pis[:, 0] = sep_level
    pis[:int(K / 2), 1] = sep_level
    pis[int(K / 2):, 1] = 1 - sep_level
    pis[:, 2] = 1 - sep_level

    # Data generation
    X = rng.multinomial(1, rho, size=sample_size)  # one-hot: X[i,c]=1 iff X[i]=c
    Xidx = np.dot(np.array([c for c in range(C)]), X.T)  # index: Xidx[i]=c
    Y = np.array([0 + (rng.rand() < pis[k, Xidx[i]]) for i in range(sample_size) for k in range(K)]).reshape(
        sample_size, K)
    Z = np.array([rng.normal(mus[Xidx[i]], sigmas[Xidx[i]]) for i in range(sample_size)]).reshape(sample_size, D)

    return Y, Z
