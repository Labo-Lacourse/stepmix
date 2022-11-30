"""Utility functions for model bootstrapping and confidence intervals."""
import copy
import itertools

import numpy as np

from sklearn.base import clone
from sklearn.utils.validation import check_random_state


def mse(x, y):
    return np.square(np.subtract(x, y)).mean()


def find_best_permutation(reference, target, criterion=mse):
    """Find the best permutation of the columns in target to minimize some
    criterion comparing to reference.

    Parameters
    ----------
    reference : ndarray of shape  (n_samples, n_columns)
        Reference array.
    target : ndarray of shape  (n_samples, n_columns)
        Target array.
    criterion: Callable returning a scalar used to find the permutation.
    """
    n_samples, n_columns = reference.shape

    score = np.inf
    best_perm = None

    permutations = itertools.permutations(np.arange(n_columns))

    for p in permutations:
        score_p = criterion(reference, target[:, np.array(p)])
        if score_p < score:
            score = score_p
            best_perm = p

    return np.array(best_perm)


def stack_stepmix_parameters(params):
    """Transforms a list of StepMix parameters dict into a single StepMix parameters dict.

     Parameters are aggregated along a new axis."""
    base = copy.deepcopy(params[0])

    base_keys = ["measurement"]
    if "structural" in params[0]:
        base_keys.append("structural")

    for key_i in base_keys:
        for key_j in params[0][key_i].keys():
            is_nested = not isinstance(base[key_i][key_j], np.ndarray)
            if is_nested:
                # Nested parameters have another level of dictionnaries
                for key_k in params[0][key_i][key_j].keys():
                    base[key_i][key_j][key_k] = np.stack([p[key_i][key_j][key_k] for p in params])
            else:
                base[key_i][key_j] = np.stack([p[key_i][key_j] for p in params])

    return base


def bootstrap(estimator, X, Y=None, n_repetitions=1000):
    """Non-parametric boostrap of StepMix estimator.

    Fit the estimator on X,Y then fit n_repetitions on resampled datasets.

    Repetition parameters are aligned with the class order of the main estimator.

    Parameters
    ----------
    estimator : StepMix instance
        Estimator to use for bootstrapping.
    X : array-like of shape (n_samples, n_columns)
        Measurement data.
    Y : array-like of shape (n_samples, n_columns_structural), default=None
        Structural data.
    n_repetitions: int
        Number of repetitions to fit.
    Returns
    ----------
    estimator: StepMix
        Fitted instance of the estimator.
    parameters: ndarray
        StepMix parameter dictionary. Follows the same convention as the parameters of the StepMix
        object. An additional axis of size (n_repetitions,) is added at position 0 of each parameter array.
    """
    n_samples = X.shape[0]

    # First fit the base estimator and get class probabilities
    estimator.fit(X, Y)
    ref_class_probabilities = estimator.predict_proba(X, Y)

    # Not fit n_repetitions estimator with resampling and save parameters
    rng = check_random_state(estimator.random_state)
    parameters = list()

    for _ in range(n_repetitions):
        # Resample data
        rep_samples = rng.choice(n_samples, size=(n_samples,), replace=True)
        X_rep = X[rep_samples]
        Y_rep = Y[rep_samples] if Y is not None else None

        # Fit estimator on resample data
        estimator_rep = clone(estimator)
        estimator_rep.fit(X_rep, Y_rep)

        # Class ordering may be different. Reorder based on best permutation of class probabilites
        rep_class_probabilities = estimator_rep.predict_proba(X, Y)  # Inference on original samples
        perm = find_best_permutation(ref_class_probabilities, rep_class_probabilities)
        estimator_rep.permute_classes(perm)

        # Save parameters
        parameters.append(estimator_rep.get_parameters())

    return estimator, stack_stepmix_parameters(parameters)