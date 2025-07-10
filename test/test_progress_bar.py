"""Test progress bars."""

import pytest

from stepmix.stepmix import StepMix


@pytest.mark.parametrize("progress_mode", [0, 1, 2])
def test_progress_bar(data, kwargs, progress_mode):
    """Run StepMix with and without progress bars."""
    X, Y = data

    model_1 = StepMix(**kwargs, progress_bar=progress_mode)
    model_1.fit(X, Y)
