StepMix
==============================
<a href="https://pypi.org/project/stepmix/"><img src="https://badge.fury.io/py/stepmix.svg" alt="PyPI version"></a>
[![Build](https://github.com/Labo-Lacourse/stepmix/actions/workflows/pytest.yaml/badge.svg)](https://github.com/Labo-Lacourse/stepmix/actions/workflows/pytest.yaml)
[![Documentation Status](https://readthedocs.org/projects/stepmix/badge/?version=latest)](https://stepmix.readthedocs.io/en/latest/index.html)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

A Python package for latent class analysis and clustering of continuous and categorical data, with support for missing values. 
Various stepwise estimation methods are available for models with measurement and structural components. Largely based on [Bakk & Kuha, 2018](https://pubmed.ncbi.nlm.nih.gov/29150817/).


# Install
You can install StepMix with pip, preferably in a virtual environment : 
```
pip install stepmix
``` 
# Tutorials
Detailed tutorials are available in notebooks : 
1. [Latent Class Analysis with StepMix](https://colab.research.google.com/drive/1KAxcvxjL_vB2lAG9e47we7hrf_2fR1eK?usp=sharing) : 
an in-depth look at how latent class models can be defined with StepMix. The tutorial uses the Iris Dataset as an example
and covers :
   1. Continuous LCA models;
   2. Binary LCA models;
   3. Categorical LCA models;
   3. Mixed LCA models (continuous and categorical data);
   5. Missing Values.
2. [Stepwise Estimation with StepMix](https://colab.research.google.com/drive/1T_UObkN5Y-iFTKiun0zOkKk7LjtMeV25?usp=sharing) :
    a tutorial demonstrating how to define measurement and structural models. The tutorial discusses:
   1. LCA models with response variables;
   2. LCA models with covariates; 
   3. 1-step, 2-step and 3-step estimation;
   4. Corrections and other options for 3-step estimation.
3. [Model Selection](https://colab.research.google.com/drive/1iyFTD-D2wn88_vd-qxXkovIuWHRtU7V8?usp=sharing) :
   a short tutorial discussing:
    1. Selecting the number of latent classes (```n_components```);
    2. Comparing models with AIC and BIC.
4. [Parameters, Bootstrapping and CI](https://colab.research.google.com/drive/14Ir08HXQ3svydbVV4jlvi1HjGnfc4fc0?usp=sharing) :
   a tutorial discussing how to:
   1. Access StepMix parameters;
   2. Bootstrap StepMix estimators;
   2. Quickly plot confidence intervals.

# Quickstart
A simple example for 3-step estimation on simulated data :

```python
from stepmix.datasets import data_bakk_response
from stepmix.stepmix import StepMix

# Soft 3-step 
X, Y, _ = data_bakk_response(n_samples=1000, sep_level=.9, random_state=42)
model = StepMix(n_components=3, n_steps=3, measurement='bernoulli', structural='gaussian_unit', assignment='soft',
            random_state=42)
model.fit(X, Y)
print(model.score(X, Y))  # Average log-likelihood

# Equivalently, each step can be performed individually. See the code of the fit method for details.
model = StepMix(n_components=3, measurement='bernoulli', structural='gaussian_unit', random_state=42)
model.em(X)  # Step 1
probs = model.predict_proba(X)  # Step 2
model.m_step_structural(probs, Y)  # Step 3
print(model.score(X, Y))
```
1-step and 2-step estimation are simply a matter of changing of the `n_steps` argument. Additionally, some bias correction
methods are available for 3-step estimation.

# References
- Bolck, A., Croon, M., and Hagenaars, J. Estimating latent structure models with categorical variables: One-step
versus three-step estimators. Political analysis, 12(1): 3–27, 2004.
- Vermunt, J. K. Latent class modeling with covariates: Two improved three-step approaches. Political analysis,
18 (4):450–469, 2010.

- Bakk, Z., Tekle, F. B., and Vermunt, J. K. Estimating the association between latent class membership and external
variables using bias-adjusted three-step approaches. Sociological Methodology, 43(1):272–311, 2013.

- Bakk, Z. and Kuha, J. Two-step estimation of models between latent classes and external variables. Psychometrika,
83(4):871–892, 2018
