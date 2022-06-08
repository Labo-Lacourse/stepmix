from .categorical import Bernoulli
from .gaussian import GaussianUnit, GaussianSpherical, GaussianDiag, GaussianTied, GaussianFull
from .covariate import Covariate
from .nested import Nested

EMISSION_DICT = {
    'gaussian': GaussianUnit,
    'gaussian_unit': GaussianUnit,
    'gaussian_full': GaussianFull,
    'gaussian_spherical': GaussianSpherical,
    'gaussian_diag': GaussianDiag,
    'gaussian_tied': GaussianTied,
    'bernoulli': Bernoulli,
    'binary': Bernoulli,
    'covariate': Covariate,
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
