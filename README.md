Latent Class Analysis
==============================
A Python package for multi-step estimation of latent class models with measurement and structural components. 
The package can also be used to fit mixture models with various observed random variables. Largely based on this [2018 work by Bakk & Kuha](https://pubmed.ncbi.nlm.nih.gov/29150817/).

# Install
You can install this repo directly with pip, preferably in a virtual environment : 
```
pip install --upgrade git+https://github.com/sachaMorin/lca.git
``` 
# Usage
A simple example for 3-step estimation on simulated data : 
```python
from lca.datasets import data_generation_Bakk
from lca.lca import LCA

# Soft 3-step 
X, Y = data_generation_Bakk(sample_size=1000, sep_level=.7, random_state=42)
model = LCA(n_components=3, n_steps=3, measurement='bernoulli', structural='gaussian_unit', assignment='soft', random_state=42)
model.fit(X, Y)
model.score(X, Y)  # Average log-likelihood

# Equivalently, each step can be performed individually. See the code of the fit method for details.
model = LCA(n_components=3, measurement='bernoulli', structural='gaussian_unit', random_state=42)
model.em(X) # Step 1
probs = model.predict_proba(X) # Step 2
model.m_step_structural(probs, Y) # Step 3
model.score(X, Y)
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
