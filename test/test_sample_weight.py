"""Test sample weights."""
import numpy as np
import pytest
from stepmix.stepmix import StepMix

def test_default_weights(data, kwargs):
    """Running StepMix without sample weights should be the same as running StepMix with unit weights."""
    X, Y = data

    model_1 = StepMix(**kwargs)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)

    model_2 = StepMix(**kwargs)
    sample_weight = np.ones(X.shape[0])
    model_2.fit(X, Y, sample_weight=sample_weight)
    ll_2 = model_2.score(X, Y, sample_weight=sample_weight)

    assert ll_1 == ll_2

def test_different_weights(data, kwargs):
    """Running StepMix with different weights should result in a different likelihood."""
    X, Y = data

    model_1 = StepMix(**kwargs)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)

    model_2 = StepMix(**kwargs)
    sample_weight = np.ones(X.shape[0])
    sample_weight[0] = 100
    model_2.fit(X, Y, sample_weight=sample_weight)
    ll_2 = model_2.score(X, Y, sample_weight=sample_weight)

    assert ll_1 != ll_2
