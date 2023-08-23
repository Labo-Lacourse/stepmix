"""Single covariate simulation."""
from stepmix.datasets import data_bakk_covariate
from stepmix.stepmix import StepMix

# Simulate data
Y, Z_p, _ = data_bakk_covariate(n_samples=2000, sep_level=0.9, random_state=42)

# Specify optimization parameters for the covariate SM if needed
covariate_params = {"method": "newton-raphson", "max_iter": 1, "intercept": True}

# Fit StepMix Estimator
model = StepMix(
    n_components=3,
    measurement="binary",
    structural="covariate",
    n_steps=1,
    random_state=42,
    structural_params=covariate_params,
)
model.fit(Y, Z_p)

# Retrieve structural model coefficients as a dataframe
betas = model.get_sm_df()

# Set a reference class with null coefficients for identifiability
betas = betas.sub(betas[1], axis=0)
print(betas)
