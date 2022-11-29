"""Utility functions for model bootstrapping and confidence intervals."""
import itertools

import numpy as np


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
