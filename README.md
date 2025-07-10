StepMix
==============================
<a href="https://pypi.org/project/stepmix/"><img src="https://badge.fury.io/py/stepmix.svg" alt="PyPI version"></a>
[![Paper](https://img.shields.io/badge/JSS-Paper-0474ac.svg)]()
[![Build](https://github.com/Labo-Lacourse/stepmix/actions/workflows/pytest.yaml/badge.svg)](https://github.com/Labo-Lacourse/stepmix/actions/workflows/pytest.yaml)
[![Documentation Status](https://readthedocs.org/projects/stepmix/badge/?version=latest)](https://stepmix.readthedocs.io/en/latest/index.html)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Downloads](https://static.pepy.tech/badge/stepmix)](https://pepy.tech/project/stepmix)
[![Downloads](https://static.pepy.tech/badge/stepmix/month)](https://pepy.tech/project/stepmix)
[![DOI](https://img.shields.io/badge/DOI-10.18637/jss.v113.i08-E66A0A.svg)](https://doi.org/10.18637%2Fjss.v113.i08)

*For StepMixR, please refer to <a href="https://github.com/Labo-Lacourse/stepmixr">this repository.</a>*

A Python package following the scikit-learn API for generalized mixture modeling. The package supports categorical 
data (Latent Class Analysis) and continuous data (Gaussian Mixtures/Latent Profile Analysis). StepMix can be used for
both clustering and supervised learning.

Additional features include:
* Support for missing values through Full Information Maximum Likelihood (FIML); 
* Multiple stepwise Expectation-Maximization (EM) estimation methods based on pseudolikelihood theory;
* Covariates and distal outcomes;
* Parametric and non-parametric bootstrapping.

![](https://drive.google.com/uc?export=view&id=1mB9-Y2N3biqHRyRVX5cvIdixBpoiyCG_)

# Reference
If you find StepMix useful, please leave a ⭐ and consider citing our [Journal of Statistical Software paper]():
```
@Article{,
  title = {{StepMix}: A {Python} Package for Pseudo-Likelihood
    Estimation of Generalized Mixture Models with External
    Variables},
  author = {Sacha Morin and Robin Legault and F{\'e}lix Lalibert{\'e}
    and Zsuzsa Bakk and Charles-{\'E}douard Gigu{\`e}re and Roxane
    {de la Sablonni{\`e}re} and {\'E}ric Lacourse},
  journal = {Journal of Statistical Software},
  year = {2025},
  volume = {113},
  number = {8},
  pages = {1--39},
  doi = {10.18637/jss.v113.i08},
}
```


# Install
You can install StepMix with pip, preferably in a virtual environment: 
```
pip install stepmix
``` 
# Quickstart
A StepMix mixture using categorical variables on a preloaded data matrix. StepMix accepts either `numpy.array`or 
`pandas.DataFrame`. Categories should be integer-encoded and 0-indexed.

```python
from stepmix.stepmix import StepMix

# Categorical StepMix Model with 3 latent classes
model = StepMix(n_components=3, measurement="categorical")
model.fit(data)

# Allow missing values
model_nan = StepMix(n_components=3, measurement="categorical_nan")
model_nan.fit(data_nan)
```
For binary data you can also use `measurement="binary"` or `measurement="binary_nan"`. For continuous data, you can fit a Gaussian Mixture with diagonal covariances using `measurement="continuous"` or `measurement="continuous_nan"`.

Set `verbose=1` for a detailed output.

Please refer to the StepMix tutorials to learn how to combine continuous and categorical data in the same model.
# Tutorials
Detailed tutorials are available in notebooks: 
1. [Generalized Mixture Models with StepMix](https://colab.research.google.com/drive/1T8017QsMCiy62z2QHOvmbzE-tCECO-w7?): 
an in-depth look at how mixture models can be defined with StepMix. The tutorial uses the Iris Dataset as an example
and covers:
   1. Gaussian Mixtures (Latent Profile Analysis);
   2. Binary Mixtures (LCA);
   3. Categorical Mixtures (LCA);
   3. Mixed Categorical and Continuous Mixtures;
   5. Missing Values through Full-Information Maximum Likelihood.
2. [Stepwise Estimation with StepMix](https://colab.research.google.com/drive/1xJB4y6eaprBMw98lB7kflWz8MfQcT2cI?usp=drive_link):
    a tutorial demonstrating how to define measurement and structural models. The tutorial discusses:
   1. LCA models with distal outcomes;
   2. LCA models with covariates; 
   3. 1-step, 2-step and 3-step estimation;
   4. Corrections (BCH or ML) and other options for 3-step estimation;
   5. Putting it All Together: A Complete Model with Missing Values
3. [Model Selection](https://colab.research.google.com/drive/1btXHCx90eCsnUlQv_yN-9AzKDhJP_JkG?usp=drive_link):
    1. Selecting the number of components in a mixture model (```n_components```) with cross-validation;
    3. Selecting the number of components with the Parametric Bootstrapped Likelihood Ratio Test (BLRT);
    2. Fit indices: AIC, BIC and other metrics.
4. [Parameters, Bootstrapping and CI](https://colab.research.google.com/drive/14DJCqFTUaYp3JtLAeAMYmGHFLCHE-r7z):
   a tutorial discussing how to:
   1. Access StepMix parameters;
   2. Bootstrap StepMix estimators;
   2. Quickly plot confidence intervals.
5. [Supervised and Semi-Supervised Learning with StepMix](https://colab.research.google.com/drive/1GKkdKkCsHWnB4ocjkx8oQdf-gUxHWjeB?usp=sharing):
   1. Binary Classification;
   1. Multiclass Classification;
   1. Semi-Supervised Learning;
   1. Cross-Validation.
6. [Deriving p-values in StepMix](https://colab.research.google.com/drive/1oaofJ68eHjahSPNty75npBzws3JuAjPO?usp=sharing): a tutorial demonstrating how to transform SM parameters into conventional regression coefficients and how to derive p-values.
   The tutorial covers models with:
   1. Continuous covariate;
   2. Binary covariate;
   3. Categorical covariate;
   4. Multiple covariates (different distributions);
   5. Binary distal outcome;


![](https://drive.google.com/uc?export=view&id=1gajwp-NTu9kSdK_7DBhpiX0SebEx5WMF)
