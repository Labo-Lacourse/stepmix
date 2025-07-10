"""Various synthetic datasets."""

import warnings

import numpy as np
from sklearn.utils.validation import check_random_state
from scipy.special import softmax
import pandas as pd

from .stepmix import StepMix


def random_nan(X, Y, nan_ratio, random_state=None):
    """Randomly replace values in X and Y with NaNs with probability nan_ratio."""
    (n_samples, n_mm), n_sm = X.shape, Y.shape[1]
    rng = np.random.default_rng(random_state)

    observed_mask = rng.random((n_samples, n_mm + n_sm)) > nan_ratio

    if not observed_mask.sum(axis=1).all():
        warnings.warn(
            "Some samples are completely unobserved. This will likely result in downstream errors. Reduce the nan_ratio or try another"
            " seed."
        )

    mm_mask, sm_mask = observed_mask[:, :n_mm], observed_mask[:, n_mm:]

    X = np.where(mm_mask, X, np.nan)
    Y = np.where(sm_mask, Y, np.nan)

    return X, Y


def bakk_measurements(n_classes, n_mm, sep_level):
    """Binary measurement parameters in Bakk 2018.

    Parameters
    ----------
    n_classes: int
        Number of latent classes. Use 3 for the paper simulation.
    n_mm: int
        Number of features in the measurement model. Use 6 for the paper simulation.
    sep_level : float
        Separation level in the measurement data. Use .7, .8 or .9 for the paper simulation.

    Returns
    -------
    pis : ndarray of shape (n_mm, n_classes)
        Conditional bernoulli probabilities.

    """
    pis = np.zeros((n_mm, n_classes))
    pis[:, 0] = sep_level
    pis[: int(n_mm / 2), 1] = sep_level
    pis[int(n_mm / 2) :, 1] = 1 - sep_level
    pis[:, 2] = 1 - sep_level

    return pis


def data_bakk_response(n_samples, sep_level, n_classes=3, n_mm=6, random_state=None):
    """Simulated data for the response simulations in Bakk 2018.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    sep_level : float
        Separation level in the measurement data. Use .7, .8 or .9 for the paper simulation.
    n_classes: int
        Number of latent classes. Use 3 for the paper simulation.
    n_mm: int
        Number of features in the measurement model. Use 6 for the paper simulation.
    random_state: int
        Random state.

    Returns
    -------
    X : ndarray of shape (n_samples, n_mm)
        Binary measurement samples.
    Y : ndarray of shape (n_samples, 1)
        Response structural samples.
    labels : ndarray of shape (n_samples,)
        Ground truth class membership.

    References
    ----------
    Bakk, Z. and Kuha, J. Two-step estimation of models between latent classes and external variables. Psychometrika,
    83(4):871–892, 2018

    """
    n_sm = 1  # Always 1 structural feature

    # Measurement probabilities
    pis = bakk_measurements(n_classes, n_mm, sep_level)

    # Structural means
    means = [[-1], [1], [0]]

    # Model parameters
    params = dict(
        weights=np.ones(n_classes) / n_classes,
        measurement=dict(pis=pis.T),
        structural=dict(means=np.array(means)),
        measurement_in=n_mm,
        structural_in=n_sm,
    )

    # Sample data
    generator = StepMix(
        n_components=n_classes,
        measurement="bernoulli",
        structural="gaussian_unit",
        random_state=random_state,
    )
    generator.set_parameters(params)
    X, Y, labels = generator.sample(n_samples)

    return X, Y, labels


def data_bakk_covariate(n_samples, sep_level, n_mm=6, random_state=None):
    """Simulated data for the covariate simulations in Bakk 2018.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    sep_level : float
        Separation level in the measurement data. Use .7, .8 or .9 for the paper simulation.
    n_mm: int
        Number of features in the measurement model. Use 6 for the paper simulation.
    random_state: int
        Random state.

    Returns
    -------
    X : ndarray of shape (n_samples, n_mm)
        Binary measurement samples.
    Y : ndarray of shape (n_samples, 1)
        Covariate structural samples.
    labels : ndarray of shape (n_samples,)
        Ground truth class membership.

    References
    ----------
    Bakk, Z. and Kuha, J. Two-step estimation of models between latent classes and external variables. Psychometrika,
    83(4):871–892, 2018

    """
    rng = check_random_state(random_state)
    n_sm = 1  # Always one structural feature
    n_classes = 3  # Always 3 latent classes

    # Regression parameters
    beta = np.array([0, -1, 1])

    # Intercepts are adjusted so that class sizes are equal in average
    intercepts = np.array([0.0, 2.34459467, -3.65540533])

    # Sample covariate
    # Uniform integer 1-5 (see p.15 Bakk 2017)
    Y = rng.randint(low=1, high=6, size=(n_samples, n_sm))

    # Scores
    logits = Y * beta.reshape((1, -1)) + intercepts

    # Latent class (Ground truth class membership)
    # probabilistic assignment ( realizations from the distribution of the r.v label_i|Y_i )
    probas = softmax(
        logits, axis=1
    )  # probas[n,c] = probability for unit n to be assigned to class c
    cumul_probas = np.cumsum(probas, axis=1)
    bool_tab = (cumul_probas - np.tile(rng.rand(n_samples), (n_classes, 1)).T) >= 0
    labels = -np.sum(bool_tab, axis=1) + n_classes
    # labels = probas.argmax(axis=1)  # Old deterministic assignment

    # Measurement probabilities
    pis = bakk_measurements(n_classes, n_mm, sep_level)

    # Measurement model parameters
    params = dict(
        weights=np.ones(n_classes) / n_classes,  # Spoof. Will be ignored
        measurement=dict(pis=pis.T),
        measurement_in=n_mm,
    )

    # Sample data
    generator = StepMix(
        n_components=n_classes, measurement="bernoulli", random_state=random_state
    )
    generator.set_parameters(params)
    X, _, labels_new = generator.sample(n_samples, labels=labels)

    return X, Y, labels


def data_bakk_complete(n_samples, sep_level, n_mm=6, random_state=None, nan_ratio=0.0):
    """Stitch together data_bakk_covariate and data_bakk_response to get a complete model."""

    # Get covariate data
    X, Y_p, labels = data_bakk_covariate(n_samples, sep_level, n_mm, random_state)

    # Now use the labels to conditionally sample from a response model
    means = [[-1], [1], [0]]

    # Model parameters
    params = dict(
        weights=np.ones(3) / 3,
        measurement=dict(means=np.array(means)),
        measurement_in=1,
    )

    # Sample data
    generator = StepMix(
        n_components=3,
        measurement="gaussian_unit",
        random_state=random_state,
    )
    generator.set_parameters(params)

    Y_o, _, _ = generator.sample(n_samples, labels=labels)

    # Drop some values in measurements and response
    if nan_ratio:
        X, Y_o = random_nan(X, Y_o, nan_ratio, random_state=random_state)

    return X, np.hstack((Y_p, Y_o)), labels


def data_bakk_complex(n_samples, sep_level, random_state=None, nan_ratio=0.0):
    """Build a simulated example with mixed data and missing values.

    Measurements: 3 binary variables + 1 continuous variable.

    Structural: 3 binary response variables + 1 continuous response variable + 1 covariate.

    Missing values everywhere except in the covariate.

    Return data as a dataframe.
    """

    # Get covariate data
    X, Y_p, labels = data_bakk_covariate(n_samples, sep_level, 6, random_state)

    # Now use the labels to conditionally sample from a response model
    means = [[-1, 2], [1, -2], [0, 0]]

    # Model parameters
    params = dict(
        weights=np.ones(3) / 3,
        measurement=dict(means=np.array(means)),
        measurement_in=2,
    )

    # Sample data
    generator = StepMix(
        n_components=3,
        measurement="gaussian_unit",
        random_state=random_state,
    )
    generator.set_parameters(params)

    Y_o, _, _ = generator.sample(n_samples, labels=labels)

    # Drop some values in measurements and response
    if nan_ratio:
        X, Y_o = random_nan(X, Y_o, nan_ratio, random_state=random_state)

    # Dispatch binary variables in measurement and structural models
    X_mixed = np.hstack((X[:, :3], Y_o[:, [1]]))
    Y_mixed = np.hstack((X[:, 3:], Y_o[:, [0]], Y_p))

    # Return as dataframes
    X_mixed = pd.DataFrame(
        X_mixed,
        columns=[
            "Binary Measurement 1",
            "Binary Measurement 2",
            "Binary Measurement 3",
            "Continuous Measurement 1",
        ],
    )
    Y_mixed = pd.DataFrame(
        Y_mixed,
        columns=[
            "Binary Outcome 1",
            "Binary Outcome 2",
            "Binary Outcome 3",
            "Continuous Outcome 1",
            "Covariate 1",
        ],
    )

    return X_mixed, Y_mixed, labels


# Data generation: Simulated problems from IFT6269 Hwk 4
def data_generation_gaussian(n_samples, sep_level, n_mm=6, random_state=None):
    """Bakk binary measurement model with more complex gaussian structural model.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    sep_level : float
        Separation level in the measurement data. Use .7, .8 or .9 for the paper simulation.
    n_mm: int
        Number of features in the measurement model. Use 6 for the paper simulation.
    random_state: int
        Random state.

    Returns
    -------
    X : ndarray of shape (n_samples, n_mm)
        Binary Measurement samples.
    Y : ndarray of shape (n_samples, 2)
        Gaussian Structural samples.
    labels : ndarray of shape (n_samples,)
        Ground truth class membership.

    """
    n_classes = 4  # Number of latent classes
    n_sm = 2  # Dimensions of the response variable Zo

    # True parameters
    # rho[c] = p(X=c)
    rho = np.ones(n_classes) / n_classes

    # mus[k] = E[Z0|X=c]
    # mus[k] = E[Z0|X=c]
    mus = np.array(
        [[-2.0344, 4.1726], [3.9779, 3.7735], [3.8007, -3.7972], [-3.0620, -3.5345]]
    )

    # sigmas[k] = V[Z0|X=c]
    sigmas = np.array(
        [
            [[2.9044, 0.2066], [0.2066, 2.7562]],
            [[0.2104, 0.2904], [0.2904, 12.2392]],
            [[0.9213, 0.0574], [0.0574, 1.8660]],
            [[6.2414, 6.0502], [6.0502, 6.1825]],
        ]
    )

    # pis[k,c] = p(Yk=1|X=c)
    # sep_level = 0.9 #0.9->high, 0.8->medium, 0.7->low
    pis = bakk_measurements(n_classes, n_mm, sep_level)

    # Model parameters
    params = dict(
        weights=rho,
        measurement=dict(pis=pis.T),
        structural=dict(means=mus, covariances=sigmas),
        measurement_in=n_mm,
        structural_in=n_sm,
    )

    # Sample data
    generator = StepMix(
        n_components=n_classes,
        measurement="bernoulli",
        structural="gaussian_full",
        random_state=random_state,
    )
    generator.set_parameters(params)
    X, Y, labels = generator.sample(n_samples)

    return X, Y, labels


def data_gaussian_binary(n_samples, random_state=None):
    """Full Gaussian measurement model with 2 binary responses.

    The data has 4 latent classes.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    random_state: int
        Random state.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Gaussian Measurement samples.
    Y : ndarray of shape (n_samples, 2)
        Binary Structural samples.
    labels : ndarray of shape (n_samples,)
        Ground truth class membership.

    """
    n_classes = 4  # Number of latent classes

    # True parameters
    # rho[c] = p(X=c)
    rho = np.ones(n_classes) / n_classes

    # mus[k] = E[Z0|X=c]
    # mus[k] = E[Z0|X=c]
    mus = np.array(
        [[-2.0344, 4.1726], [3.9779, 3.7735], [3.8007, -3.7972], [-3.0620, -3.5345]]
    )

    # sigmas[k] = V[Z0|X=c]
    sigmas = np.array(
        [
            [[2.9044, 0.2066], [0.2066, 2.7562]],
            [[0.2104, 0.2904], [0.2904, 12.2392]],
            [[0.9213, 0.0574], [0.0574, 1.8660]],
            [[6.2414, 6.0502], [6.0502, 6.1825]],
        ]
    )

    pis = np.array([[0.1, 0.9, 0.1, 0.9], [0.9, 0.9, 0.1, 0.1]])

    # Model parameters
    params = dict(
        weights=rho,
        measurement=dict(pis=pis.T),
        structural=dict(means=mus, covariances=sigmas),
        measurement_in=2,
        structural_in=2,
    )

    # Sample data
    generator = StepMix(
        n_components=n_classes,
        measurement="bernoulli",
        structural="gaussian_full",
        random_state=random_state,
    )
    generator.set_parameters(params)
    Y, X, labels = generator.sample(n_samples)

    return X, Y, labels


def data_gaussian_categorical(n_samples, random_state=None):
    """Full Gaussian measurement model with 2 categorical responses.

    The data has 4 latent classes.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    random_state: int
        Random state.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Gaussian Measurement samples.
    Y : ndarray of shape (n_samples, 2)
        Categorical Structural samples.
    labels : ndarray of shape (n_samples,)
        Ground truth class membership.

    """
    n_classes = 4  # Number of latent classes

    # True parameters
    # rho[c] = p(X=c)
    rho = np.ones(n_classes) / n_classes

    # mus[k] = E[Z0|X=c]
    # mus[k] = E[Z0|X=c]
    mus = np.array(
        [[-2.0344, 4.1726], [3.9779, 3.7735], [3.8007, -3.7972], [-3.0620, -3.5345]]
    )

    # sigmas[k] = V[Z0|X=c]
    sigmas = np.array(
        [
            [[2.9044, 0.2066], [0.2066, 2.7562]],
            [[0.2104, 0.2904], [0.2904, 12.2392]],
            [[0.9213, 0.0574], [0.0574, 1.8660]],
            [[6.2414, 6.0502], [6.0502, 6.1825]],
        ]
    )

    # 4 classes x (2 categorical variables x 3 outcomes) = 4 x 6
    pis = np.array(
        [
            [0.8, 0.1, 0.1, 0.9, 0.1, 0.0],
            [0.1, 0.1, 0.8, 0.9, 0.0, 0.1],
            [0.1, 0.8, 0.1, 0.1, 0.9, 0.0],
            [0.1, 0.1, 0.8, 0.1, 0.0, 0.9],
        ]
    )

    # Model parameters
    params = dict(
        weights=rho,
        measurement=dict(means=mus, covariances=sigmas),
        structural=dict(pis=pis, max_n_outcomes=3, total_outcomes=2),
        measurement_in=2,
        structural_in=2,
    )

    # Sample data
    generator = StepMix(
        n_components=n_classes,
        measurement="gaussian_full",
        structural="categorical",
        random_state=random_state,
    )
    generator.set_parameters(params)
    X, Y, labels = generator.sample(n_samples)

    return X, Y, labels


def data_gaussian_diag(n_samples, sep_level, n_mm=6, random_state=None, nan_ratio=0.0):
    """Bakk binary measurement model with 2D diagonal gaussian structural model.

    Optionally, a random proportion of values can be replaced with missing values to test FIML models.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    sep_level : float
        Separation level in the measurement data. Use .7, .8 or .9 for the paper simulation.
    n_mm: int
        Number of features in the measurement model. Use 6 for the paper simulation.
    random_state: int
        Random state.
    nan_ratio: float
        Ratio of values to replace with missing values.

    Returns
    -------
    X : ndarray of shape (n_samples, n_mm)
        Binary ,easurement samples.
    Y : ndarray of shape (n_samples, 2)
        Gaussian structural samples.
    labels : ndarray of shape (n_samples,)
        Ground truth class membership.

    """
    n_classes = 3  # Number of latent classes
    n_sm = 2  # Dimensions of the response variable Zo

    # True parameters
    # rho[c] = p(X=c)
    rho = np.ones(n_classes) / n_classes

    # mus[k] = E[Z0|X=c]
    # mus[k] = E[Z0|X=c]
    mus = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, -5.0]])

    # sigmas[k] = V[Z0|X=c]
    sigmas = np.array(
        [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [0.0, 3.0]]]
    )

    # pis[k,c] = p(Yk=1|X=c)
    # sep_level = 0.9 #0.9->high, 0.8->medium, 0.7->low
    pis = bakk_measurements(n_classes, n_mm, sep_level)

    # Model parameters
    params = dict(
        weights=rho,
        measurement=dict(pis=pis.T),
        structural=dict(means=mus, covariances=sigmas),
        measurement_in=n_mm,
        structural_in=n_sm,
    )

    # Sample data
    generator = StepMix(
        n_components=n_classes,
        measurement="bernoulli",
        structural="gaussian_full",
        random_state=random_state,
    )
    generator.set_parameters(params)
    X, Y, labels = generator.sample(n_samples)

    # Drop some values
    if nan_ratio:
        rng = np.random.default_rng(random_state)

        observed_mask = rng.random((n_samples, n_mm + n_sm)) > nan_ratio

        if not observed_mask.sum(axis=1).all():
            warnings.warn(
                "Some samples are completely unobserved. This will likely result in downstream errors. Reduce the nan_ratio or try another"
                "seed."
            )

        mm_mask, sm_mask = observed_mask[:, :n_mm], observed_mask[:, n_mm:]

        X = np.where(mm_mask, X, np.nan)
        Y = np.where(sm_mask, Y, np.nan)

    return X, Y, labels
