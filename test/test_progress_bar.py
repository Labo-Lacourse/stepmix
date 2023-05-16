"""Test sample weights."""
from stepmix.stepmix import StepMix

def test_progress_bar(data, kwargs):
    """Run StepMix with and without progress bars."""
    X, Y = data

    model_1 = StepMix(**kwargs, progress_bar=True)
    model_1.fit(X, Y)

    model_2 = StepMix(**kwargs, progress_bar=False)
    model_2.fit(X, Y)