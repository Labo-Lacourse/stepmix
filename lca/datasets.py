import numpy as np
from sklearn.utils.validation import check_random_state
from scipy.special import softmax


def data_bakk_measurement(sample_size, sep_level, Xidx, K, C, rng):
    # Credit : Robin Legault
    # pis[k,c] = p(Yk=1|X=c)
    # sep_level = 0.9 #0.9->high, 0.8->medium, 0.7->low
    pis = np.zeros((K, C))
    pis[:, 0] = sep_level
    pis[:int(K / 2), 1] = sep_level
    pis[int(K / 2):, 1] = 1 - sep_level
    pis[:, 2] = 1 - sep_level

    Y = np.array([0 + (rng.rand() < pis[k, Xidx[i]]) for i in range(Xidx.shape[0]) for k in range(K)]).reshape(
        sample_size, K)

    return Y


def data_bakk_response(sample_size, sep_level, random_state=None):
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

    # Data generation
    X = rng.multinomial(1, rho, size=sample_size)  # one-hot: X[i,c]=1 iff X[i]=c
    Xidx = np.dot(np.array([c for c in range(C)]), X.T)  # index: Xidx[i]=c

    Z = np.array([rng.normal(mus[Xidx[i]], sigmas[Xidx[i]]) for i in range(sample_size)]).reshape(sample_size, D)
    Y = data_bakk_measurement(sample_size, sep_level, Xidx, K, C, rng)

    # Also return ground truth class labels

    return Y, Z, Xidx


def data_bakk_covariate(sample_size, sep_level, random_state=None):
    rng = check_random_state(random_state)

    C = 3  # Number of latent classes
    K = 6  # Number of observed indicators
    D = 1  # Dimensions of the response variable Zp

    beta = np.array([0, -1, 1])

    # Intercepts are adjusted so that class sizes are roughly equal
    intercepts = np.array([0, 2.3, -3.65])

    # Sample covariate
    # Uniform integer 1-5 (see p.15 Bakk 2017)
    Z = rng.randint(low=1, high=6, size=(sample_size, D))

    # Scores
    logits = Z * beta.reshape((1, -1)) + intercepts

    # Probabilities
    pis = softmax(logits, axis=1)

    # Predicted latent class
    # TODO : Double check this. Paper does not discuss the specifics of class assignment. Modal assignment leads to class imbalance
    # TODO : We could also sample following the distribution of each row of pi
    Xidx = pis.argmax(axis=1)

    # Gen measurements
    Y = data_bakk_measurement(sample_size, sep_level, Xidx, K, C, rng)

    # Also return ground truth class labels
    c = pis.argmax(axis=1)

    return Y, Z, c


def data_generation_gaussian(sample_size, random_state=None):
    rng = check_random_state(random_state)

    C = 4  # Number of latent classes
    K = 2  # Dimension of the gaussian measurement
    D = 2  # Dimension of the response variable Z_o

    # True parameters
    rho = np.ones(C) / C  # Uniform prior on latent class

    # Evenly spaced gaussian on a high dimensional diagonal
    mus = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]]) * .35

    # unit variance for all gaussians
    # sigmas = np.ones((K,))
    pis = np.array([
        [.1, .9],
        [.9, .1],
        [.9, .1],
        [.1, .9],
    ]).T

    # Data generation
    X = rng.multinomial(1, rho, size=sample_size)  # one-hot: X[i,c]=1 iff X[i]=c
    Xidx = np.dot(np.array([c for c in range(C)]), X.T)  # index: Xidx[i]=c
    Z = np.array([0 + (rng.rand() < pis[d, Xidx[i]]) for i in range(sample_size) for d in range(D)]).reshape(
        sample_size, D)
    Y = np.array([rng.multivariate_normal(mus[Xidx[i]], np.eye(K)) for i in range(sample_size)])

    return Y, Z


# Data generation: Simulated problems from IFT6269 Hwk 4
def data_generation_Hwk4(sample_size, sep_level, random_state=None):
    random = check_random_state(random_state)

    C = 4  # Number of latent classes
    K = 6  # Number of observed indicators
    D = 2  # Dimensions of the response variable Zo

    # True parameters
    # rho[c] = p(X=c)
    rho = np.ones(C) / C

    # mus[k] = E[Z0|X=c]
    # mus[k] = E[Z0|X=c]
    mus = np.array([[-2.0344, 4.1726], \
                    [3.9779, 3.7735], \
                    [3.8007, -3.7972], \
                    [-3.0620, -3.5345]])

    # sigmas[k] = V[Z0|X=c]
    sigmas = np.array([[[2.9044, 0.2066], [0.2066, 2.7562]], \
                       [[0.2104, 0.2904], [0.2904, 12.2392]], \
                       [[0.9213, 0.0574], [0.0574, 1.8660]], \
                       [[6.2414, 6.0502], [6.0502, 6.1825]]])

    # pis[k,c] = p(Yk=1|X=c)
    # sep_level = 0.9 #0.9->high, 0.8->medium, 0.7->low
    pis = np.zeros((K, C))
    pis[:, 0] = sep_level
    pis[:int(K / 2), 1] = sep_level
    pis[int(K / 2):, 1] = 1 - sep_level
    pis[:int(K / 2), 2] = 1 - sep_level
    pis[int(K / 2):, 2] = sep_level
    pis[:, 3] = 1 - sep_level

    # Data generation
    X = random.multinomial(1, rho, size=sample_size)  # one-hot: X[i,c]=1 iff X[i]=c
    Xidx = np.dot(np.array([c for c in range(C)]), X.T)  # index: Xidx[i]=c
    Y = np.array([0 + (random.rand() < pis[k, Xidx[i]]) for i in range(sample_size) for k in range(K)]).reshape(
        sample_size, K)
    Z = np.array([random.multivariate_normal(mus[Xidx[i]], sigmas[Xidx[i]]) for i in range(sample_size)]).reshape(
        sample_size, D)

    return Y, Z
