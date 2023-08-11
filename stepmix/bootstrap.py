"""Utility functions for model bootstrapping and confidence intervals."""
import itertools
import pandas as pd
import warnings

import numpy as np
import tqdm

from sklearn.base import clone
from sklearn.utils.validation import check_random_state, check_is_fitted


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


def bootstrap(
    estimator, X, Y=None, n_repetitions=1000, sample_weight=None, progress_bar=True
):
    """Non-parametric boostrap of StepMix estimator.

    Fit the estimator on X,Y then fit n_repetitions on resampled datasets.

    Repetition parameters are aligned with the class order of the main estimator.

    Parameters
    ----------
    estimator : StepMix instance
        A fitted StepMix estimator. Used as a template to clone bootstrap estimator.
    X : array-like of shape (n_samples, n_features)
    Y : array-like of shape (n_samples, n_features_structural), default=None
    n_repetitions: int
        Number of repetitions to fit.
    sample_weight : array-like of shape(n_samples,), default=None
    progress_bar : bool, default=True
        Display a tqdm progress bar for repetitions.
    Returns
    ----------
    parameters: DataFrame
        DataFrame of all repetitions. Follows the convention of StepMix.get_parameters_df() with an additional
        'rep' column.
    stats: DataFrame
        Various statistics of bootstrapped estimators.
    """
    check_is_fitted(estimator)
    n_samples = X.shape[0]
    x_names = estimator.x_names_
    y_names = estimator.y_names_ if hasattr(estimator, "y_names") else None

    # Use the estimator built-in method to check the input
    # This will ensure that X and Y are numpy arrays for the rest of the bootstrap procedure
    X, Y = estimator._check_x_y(X, Y, reset=False)

    # Get class probabilities of main estimator as reference
    ref_class_probabilities = estimator.predict_proba(X, Y)

    # Raise warning if trying to permute too many columns
    if estimator.n_components > 6:
        warnings.warn(
            "Bootstrapping requires permuting latent classes. Permuting latent classes with n_components > 6 may be slow."
        )

    # Now fit n_repetitions estimator with resampling and save parameters
    rng = check_random_state(estimator.random_state)
    parameters = list()
    avg_ll_buffer = list()
    ll_buffer = list()

    if progress_bar:
        print("\nBootstrapping estimator...")

    tqdm_rep = tqdm.trange(
        n_repetitions, disable=not progress_bar, desc="Bootstrap Repetitions    "
    )
    for rep in tqdm_rep:
        # Resample data
        rep_samples = rng.choice(n_samples, size=(n_samples,), replace=True)
        X_rep = X[rep_samples]
        Y_rep = Y[rep_samples] if Y is not None else None
        sample_weight_rep = (
            sample_weight[rep_samples] if sample_weight is not None else None
        )

        # Fit estimator on resample data
        estimator_rep = clone(estimator)

        # Disable printing for repeated estimators and fit
        estimator_rep.verbose = 0
        estimator_rep.progress_bar = 0
        estimator_rep.fit(X_rep, Y_rep, sample_weight=sample_weight_rep)

        # Class ordering may be different. Reorder based on best permutation of class probabilites
        rep_class_probabilities = estimator_rep.predict_proba(
            X, Y
        )  # Inference on original samples
        perm = find_best_permutation(ref_class_probabilities, rep_class_probabilities)
        estimator_rep.permute_classes(perm)

        # Save parameters
        df_i = estimator_rep.get_parameters_df(x_names, y_names)
        df_i["rep"] = rep
        parameters.append(df_i)

        # Save likelihood
        avg_ll = estimator_rep.score(X_rep, Y_rep, sample_weight=sample_weight_rep)
        ll = (
            avg_ll * np.sum(sample_weight)
            if sample_weight is not None
            else avg_ll * n_samples
        )
        avg_ll_buffer.append(avg_ll)
        ll_buffer.append(ll)

        # Ask tqdm to display current max likelihood
        tqdm_rep.set_postfix(
            median_LL=np.median(ll_buffer),
            # min_avg_LL=np.min(avg_ll_buffer),
            # max_avg_LL=np.max(avg_ll_buffer),
            min_LL=np.min(ll_buffer),
            max_LL=np.max(ll_buffer),
        )

    return_df = pd.concat(parameters)
    return_df.sort_index(inplace=True)

    # Add likelihoods statistics
    stats = {"LL": np.array(ll_buffer), "avg_LL": np.array(avg_ll_buffer)}

    return return_df, pd.DataFrame.from_dict(stats)
