from .categorical import Bernoulli
from .gaussian import GaussianUnit, GaussianSpherical, GaussianDiag, GaussianTied, GaussianFull
from .covariate import Covariate

EMISSION_DICT = {
    'gaussian_unit': GaussianUnit,
    'gaussian_full': GaussianFull,
    'gaussian_spherical': GaussianSpherical,
    'gaussian_diag': GaussianDiag,
    'gaussian_tied': GaussianTied,
    'bernoulli': Bernoulli,
    'binary': Bernoulli,
    'covariate': Covariate,
}
