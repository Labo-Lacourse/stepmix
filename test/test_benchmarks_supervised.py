"""Benchmark supervised classifier performance on datasets."""

import numpy as np

from sklearn.metrics import accuracy_score
from stepmix.stepmix import StepMixClassifier


def test_data_gaussian_binary(data_g_binary, kwargs_data_g_binary):
    X, Y = data_g_binary

    clf = StepMixClassifier(**kwargs_data_g_binary)
    clf.fit(X, Y)

    preds = clf.predict(X)

    assert accuracy_score(Y[:, 0], preds[:, 0]) > 0.85
    assert accuracy_score(Y[:, 1], preds[:, 1]) > 0.85


def test_data_gaussian_binary_nan(data_g_binary, kwargs_data_g_binary):
    kwargs_data_g_binary["structural"] = "binary_nan"

    X, Y = data_g_binary

    # Mask out 50% of Y
    Y_train = Y.astype(float)
    rng = np.random.default_rng(42)
    mask = rng.choice([False, True], size=Y.shape, p=[0.5, 0.5])
    Y_train[mask] = np.nan

    clf = StepMixClassifier(**kwargs_data_g_binary)
    clf.fit(X, Y_train)

    preds = clf.predict(X)

    assert accuracy_score(Y[:, 0], preds[:, 0]) > 0.85
    assert accuracy_score(Y[:, 1], preds[:, 1]) > 0.85


def test_data_gaussian_categorical(data_g_cat, kwargs_data_g_cat):
    X, Y = data_g_cat

    clf = StepMixClassifier(**kwargs_data_g_cat)
    clf.fit(X, Y)

    preds = clf.predict(X)

    assert accuracy_score(Y[:, 0], preds[:, 0]) > 0.80
    assert accuracy_score(Y[:, 1], preds[:, 1]) > 0.80


def test_data_gaussian_categorical_nan(data_g_cat, kwargs_data_g_cat):
    kwargs_data_g_cat["structural"] = "categorical_nan"

    X, Y = data_g_cat

    # Mask out 50% of Y
    Y_train = Y.astype(float)
    rng = np.random.default_rng(42)
    mask = rng.choice([False, True], size=Y.shape, p=[0.5, 0.5])
    Y_train[mask] = np.nan

    clf = StepMixClassifier(**kwargs_data_g_cat)
    clf.fit(X, Y_train)

    preds = clf.predict(X)

    assert accuracy_score(Y[:, 0], preds[:, 0]) > 0.80
    assert accuracy_score(Y[:, 1], preds[:, 1]) > 0.80
