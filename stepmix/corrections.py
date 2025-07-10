"""Utility functions to compute 3-step bias corrections."""

import numpy as np

from stepmix import utils


def compute_bch_matrix(resp, assignment="modal"):
    """Compute the probability D[c,s] = P(X_pred=s | X=c) of predicting class latent class s given
    that a point belongs to latent class c.

    Used for empirical BCH correction.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)
           Class responsibilities (i.e. soft probabilities)
    assignment : {'soft', 'modal'}, default='modal'. Class assignments for 3-step estimation.

    Returns
    ----------
     D: array-like of shape (n_components, n_components)
        Matrix of conditional probabilities D[c,s] = P(X_pred=s | X=c)
     D_inv: array-like of shape (n_components, n_components)
        Pseudo-inverse of D.
    """
    # Dimensions
    n = resp.shape[0]
    resp_pred = utils.modal(resp) if (assignment == "modal") else resp
    weights = resp.mean(axis=0)

    # BCH correction (based on the empirical distribution: eq (6))
    D = (resp.T @ resp_pred / weights.reshape((-1, 1))) / n
    D_inv = np.linalg.pinv(D)

    return D, D_inv


def compute_log_emission_pm(resp, assignment="modal"):
    """(Log) probabilities of the predicted class given the true latent classes.

    Used for ML correction.

    Parameters
    ----------
        resp : array-like of shape (n_samples, n_components)
           Class responsibilities (i.e. soft probabilities)
        assignment : {'soft', 'modal'}, default='modal'. Class assignments for 3-step estimation.

    Returns
    ----------
        log_eps: ndarray of size (n, n_components)
            Matrix of log emission probabilities: log p(X_pred_idx[i]|X[i]=c)
    """

    # compute log emission probabilities
    D, _ = compute_bch_matrix(resp, assignment)
    log_D = np.log(np.clip(D, 1e-15, 1 - 1e-15))  # avoid probabilities 0 or 1
    resp_pred = utils.modal(resp) if (assignment == "modal") else resp
    log_eps = resp_pred @ log_D.T
    return log_eps
