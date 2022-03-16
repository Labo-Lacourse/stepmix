"""Utility functions to compute 3-step bias corrections.

Adapted from code by Robin Legault."""
import numpy as np

from lca import utils


def compute_bch_matrix(resp):
    """Compute the probability D[c,s] = P(X_pred=s | X=c) of predicting class latent class s given
    that a point belongs to latent class c.

    Used for empirical BCH correction.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)
        Class responsibilities.

    Returns
    ----------
     D: array-like of shape (n_components, n_components)
        Matrix of conditional probabilities D[c,s] = P(X_pred=s | X=c)
     D_inv: array-like of shape (n_components, n_components)
        Pseudo-inverse of D.
    """
    # Dimensions
    n = resp.shape[0]
    X_pred = utils.modal(resp)
    weights = resp.mean(axis=0)

    # BCH correction (based on the empirical distribution: eq (6))
    D = (resp.T @ X_pred / weights.reshape((-1, 1))) / n
    D_inv = np.linalg.pinv(D)

    return D, D_inv


def compute_log_emission_pm(X_pred_idx, D):
    """(Log) probabilities of the predicted class given the true latent classes.

    Used for ML correction.

    Parameters
    ----------
        X_pred_idx: array-like of size (n,)
            Predicted classes. X_pred[i]=argmax{c} p(X[i]=c|Y[i];rho,pis).
        D: array_like of size (n_components, n_components). 
            Matrix of (previously) estimated probabilities D[c,s] = p(X_pred=s|X=c) for pairs of classes c,s

    Returns
    ----------
        log_eps: ndarray of size (n, n_components)
            Matrix of log emission probabilities: log p(X_pred_idx[i]|X[i]=c)
    """
    # number of units
    n = X_pred_idx.size

    # number of latent classes
    C = D.shape[0]

    # compute log emission probabilities
    log_eps = np.zeros((n, C))
    log_D = np.log(np.clip(D, 1e-15, 1 - 1e-15))  # avoid probabilities 0 or 1

    for s in range(C):
        indexes_pred_s = np.where(X_pred_idx == s)
        log_eps[indexes_pred_s, :] = log_D[:, s]
    return log_eps