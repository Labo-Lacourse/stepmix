from .categorical import Bernoulli, BernoulliNan, Multinoulli, MultinoulliNan
from .gaussian import (
    GaussianUnit,
    GaussianSpherical,
    GaussianDiag,
    GaussianTied,
    GaussianFull,
    GaussianUnitNan,
    GaussianDiagNan,
    GaussianSphericalNan,
)
from .covariate import Covariate
from .nested import Nested

EMISSION_DICT = {
    "gaussian": GaussianUnit,
    "gaussian_unit": GaussianUnit,
    "gaussian_full": GaussianFull,
    "gaussian_spherical": GaussianSpherical,
    "gaussian_diag": GaussianDiag,
    "continuous": GaussianDiag,
    "gaussian_tied": GaussianTied,
    "bernoulli": Bernoulli,
    "binary": Bernoulli,
    "multinoulli": Multinoulli,
    "categorical": Multinoulli,
    "covariate": Covariate,
    "gaussian_nan": GaussianUnitNan,
    "gaussian_unit_nan": GaussianUnitNan,
    "gaussian_spherical_nan": GaussianSphericalNan,
    "gaussian_diag_nan": GaussianDiagNan,
    "continuous_nan": GaussianDiagNan,
    "bernoulli_nan": BernoulliNan,
    "binary_nan": BernoulliNan,
    "multinoulli_nan": MultinoulliNan,
    "categorical_nan": MultinoulliNan,
}


def build_emission(descriptor, **kwargs):
    """Build an emission model.

    Is a simple switch between a string and a dict. A dict type
    will trigger a more complex Nested model class.

    Parameters
    ----------
    descriptor: str or dict

    Returns
    -------
    emission_model: Emission

    """
    if isinstance(descriptor, str):
        # Single homogenous model (e.g., only binary, only gaussian)
        return EMISSION_DICT[descriptor](**kwargs)
    else:
        # Nested model
        return Nested(descriptor, EMISSION_DICT, **kwargs)
