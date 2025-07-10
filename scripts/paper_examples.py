"""All examples from Sections 5 and 6 of the paper."""

# SECTION 5.2: ESTIMATORS
# Section 5 only illustrates the API
# Importing packages. Not in the paper
from stepmix.datasets import data_bakk_response
from stepmix.stepmix import StepMix

# Generating dummy data. Not in the paper
Y, Z_o, _ = data_bakk_response(
    n_samples=2000,
    sep_level=0.9,
    random_state=42,
)

# Simple model with categorical MM only
model = StepMix(n_components=3, measurement="categorical")
model.fit(Y)

# Model with categorical MM and outcome SM
model = StepMix(
    n_components=3,
    measurement="categorical",
    structural="gaussian_unit",
    n_steps=3,
    assignment="soft",
    correction="BCH",
)
model.fit(Y, Z_o)

# Methods
model.predict(Y, Z_o)
model.predict_proba(Y, Z_o)
model.score(Y, Z_o)
model.aic(Y, Z_o)
model.bic(Y, Z_o)
model.get_mm_df()
model.get_sm_df()
model.get_cw_df()
model.sample(100)

# SECTION 5.3: NONPARAMETRIC BOOTSTRAPPING
stats_dict = model.bootstrap_stats(
    Y,
    Z_o,
    n_repetitions=10,
    progress_bar=True,
)

# SECTION 5.4: ADDITIONAL FEATURES
# Manually run 3-step with low-level method calls
model.em(Y)
soft_assignments = model.predict_proba(Y)
model.m_step_structural(soft_assignments, Z_o)

# SECTION 6.1: SINGLE OUTCOME SIMULATION
print("\n\n\nSUBSECTION 6.1: SINGLE OUTCOME SIMULATION")
print("Printing verbose output and mean parameters of the distal outcome\n")

from stepmix.datasets import data_bakk_response
from stepmix.stepmix import StepMix

Y, Z_o, _ = data_bakk_response(
    n_samples=2000,
    sep_level=0.9,
    random_state=42,
)

model = StepMix(
    n_components=3,
    measurement="binary",
    structural="gaussian_unit",
    n_steps=1,
    random_state=42,
    verbose=1,
)
model.fit(Y, Z_o)

mus = model.get_sm_df()
print(mus)

# SECTION 6.2: SINGLE COVARIATE SIMULATION
print("\n\n\nSUBSECTION 6.2: SINGLE COVARIATE SIMULATION")
print("Printing beta parameters of the covariate model\n")

from stepmix.datasets import data_bakk_covariate
from stepmix.stepmix import StepMix

Y, Z_p, _ = data_bakk_covariate(
    n_samples=2000,
    sep_level=0.9,
    random_state=42,
)

covariate_params = {
    "method": "newton-raphson",
    "max_iter": 1,
    "intercept": True,
}

model = StepMix(
    n_components=3,
    measurement="binary",
    structural="covariate",
    n_steps=1,
    random_state=42,
    structural_params=covariate_params,
)
model.fit(Y, Z_p)

betas = model.get_sm_df()

betas = betas.sub(betas[1], axis=0)
print(betas)

# SECTION 6.3: COMPLETE MODEL SIMULATION
print("\n\n\nSUBSECTION 6.3: COMPLETE MODEL SIMULATION")
print("Printing mean parameters of the distal outcome\n")

from stepmix.datasets import data_bakk_complete
from stepmix import StepMix

Y, Z, _ = data_bakk_complete(
    n_samples=2000,
    sep_level=0.8,
    nan_ratio=0.25,
    random_state=42,
)

structural_descriptor = {
    "covariate": {
        "model": "covariate",
        "n_columns": 1,
        "method": "newton-raphson",
        "max_iter": 1,
    },
    "response": {
        "model": "gaussian_unit_nan",
        "n_columns": 1,
    },
}

model = StepMix(
    n_components=3,
    measurement="binary_nan",
    structural=structural_descriptor,
    n_steps=1,
    random_state=42,
)
model.fit(Y, Z)

mus = model.get_sm_df().loc["response"]
print(mus)

# SECTION 6.4: APPLICATION EXAMPLE
print("\n\n\nSUBSECTION 6.4: APPLICATION EXAMPLE")
print("Printing verbose output\n")

import pandas as pd
from stepmix.stepmix import StepMix

data = pd.read_csv("data/StepMix_Real_Data_GSS.csv")
data = data.rename(
    columns={
        "realinc1000": "Income (1000)",
        "papres": "Father's job prestige",
        "madeg": "Mother's education",
        "padeg": "Father's education",
    }
)

data_mm, data_sm = (
    data[["Father's job prestige", "Mother's education", "Father's education"]],
    data[["Income (1000)"]],
)

model = StepMix(
    n_components=3,
    measurement="categorical_nan",
    structural="gaussian_diag_nan",
    n_steps=3,
    correction="ML",
    assignment="modal",
    random_state=123,
    max_iter=10000,
    verbose=0,
    progress_bar=True,
)
model.fit(data_mm, data_sm)
model.permute_classes([0, 2, 1])
model.report(data_mm, data_sm)
