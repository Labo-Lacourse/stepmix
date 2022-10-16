import copy
import pytest

from stepmix.stepmix import StepMix
from stepmix.emission.build_emission import EMISSION_DICT

# Check for fiml models with a _nan prefix. Save (vanilla name, fiml name) tuples
fiml_models = [
    (model[:-4], model) for model in EMISSION_DICT.keys() if model.endswith("_nan")
]


@pytest.mark.parametrize("vanilla,fiml", fiml_models)
def test_fiml_fully_observed(data, kwargs, vanilla, fiml):
    """FIML models (handling missing values) should yield the same estimates as the vanilla models when run
    on fully observed data."""
    X, Y = data  # Data has no missing values

    kwargs_fiml = copy.deepcopy(kwargs)

    # Use gaussians in the structural model, all other models are used on the measurement data
    if vanilla.startswith("gaussian") or vanilla.startswith("continuous"):
        kwargs["measurement"] = "binary"
        kwargs["structural"] = vanilla
        kwargs_fiml["measurement"] = "binary"
        kwargs_fiml["structural"] = fiml
    else:
        kwargs["measurement"] = vanilla
        kwargs["structural"] = "gaussian_unit"
        kwargs_fiml["measurement"] = fiml
        kwargs_fiml["structural"] = "gaussian_unit"

    # Vanilla model
    model_1 = StepMix(n_steps=1, **kwargs)
    model_1.fit(X, Y)
    ll_1 = model_1.score(X, Y)  # Average log-likelihood

    # FIML models
    model_2 = StepMix(n_steps=1, **kwargs_fiml)
    model_2.fit(X, Y)
    ll_2 = model_2.score(X, Y)

    assert ll_1 == pytest.approx(ll_2)
