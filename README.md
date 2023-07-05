StepMix
==============================
<a href="https://pypi.org/project/stepmix/"><img src="https://badge.fury.io/py/stepmix.svg" alt="PyPI version"></a>
[![Build](https://github.com/Labo-Lacourse/stepmix/actions/workflows/pytest.yaml/badge.svg)](https://github.com/Labo-Lacourse/stepmix/actions/workflows/pytest.yaml)
[![Documentation Status](https://readthedocs.org/projects/stepmix/badge/?version=latest)](https://stepmix.readthedocs.io/en/latest/index.html)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Downloads](https://static.pepy.tech/badge/stepmix)](https://pepy.tech/project/stepmix)
[![Downloads](https://static.pepy.tech/badge/stepmix/month)](https://pepy.tech/project/stepmix)
[![arXiv](https://img.shields.io/badge/arXiv-2304.03853-b31b1b.svg)](https://arxiv.org/abs/2304.03853)

*For StepMixR, please refer to <a href="https://github.com/Labo-Lacourse/stepmixr">this repository.</a>*

A Python package following the scikit-learn API for model-based clustering and generalized mixture modeling (latent class/profile analysis) of continuous and categorical data. 
StepMix handles missing values through Full Information Maximum Likelihood (FIML) and provides multiple stepwise Expectation-Maximization (EM) estimation methods based on pseudolikelihood theory. 
Additional features include support for covariates and distal outcomes, various simulation utilities, and non-parametric bootstrapping, which allows inference
in semi-supervised and unsupervised settings.

# Reference
If you find StepMix useful, please consider citing our [arXiv preprint](https://arxiv.org/abs/2304.03853):
```
@article{morin2023stepmix,
  title={StepMix: A Python Package for Pseudo-Likelihood Estimation of Generalized Mixture Models with External Variables},
  author={Morin, Sacha and Legault, Robin and Bakk, Zsuzsa and Gigu{\`e}re, Charles-{\'E}douard and de la Sablonni{\`e}re, Roxane and Lacourse, {\'E}ric},
  journal={arXiv preprint arXiv:2304.03853},
  year={2023}
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
1. [Generalized Mixture Models with StepMix](https://colab.research.google.com/drive/1KAxcvxjL_vB2lAG9e47we7hrf_2fR1eK?usp=sharing): 
an in-depth look at how latent class models can be defined with StepMix. The tutorial uses the Iris Dataset as an example
and covers:
   1. Continuous LCA models (latent profile analysis/gaussian mixture model);
   2. Binary LCA models;
   3. Categorical LCA models;
   3. Mixed variables mixture models (continuous and categorical data);
   5. Missing Values through Full-Information Maximum Likelihood.
2. [Stepwise Estimation with StepMix](https://colab.research.google.com/drive/1T_UObkN5Y-iFTKiun0zOkKk7LjtMeV25?usp=sharing):
    a tutorial demonstrating how to define measurement and structural models. The tutorial discusses:
   1. LCA models with distal outcomes;
   2. LCA models with covariates; 
   3. 1-step, 2-step and 3-step estimation;
   4. Corrections (BCH or ML) and other options for 3-step estimation.
3. [Model Selection](https://colab.research.google.com/drive/1iyFTD-D2wn88_vd-qxXkovIuWHRtU7V8?usp=sharing):
   a short tutorial discussing:
    1. Selecting the number of components in a mixture model (```n_components```);
    2. Comparing models with fit indices: AIC and BIC.
4. [Parameters, Bootstrapping and CI](https://colab.research.google.com/drive/14Ir08HXQ3svydbVV4jlvi1HjGnfc4fc0?usp=sharing):
   a tutorial discussing how to:
   1. Access StepMix parameters;
   2. Bootstrap StepMix estimators;
   2. Quickly plot confidence intervals.

