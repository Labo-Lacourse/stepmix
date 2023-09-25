"""All examples from Section 6 of the paper."""

######### SUBSECTION 6.1: SINGLE OUTCOME SIMULATION #########
print("SUBSECTION 6.1: SINGLE OUTCOME SIMULATION")
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

######### SUBSECTION 6.2: SINGLE COVARIATE SIMULATION #########
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

######### SUBSECTION 6.3: COMPLETE MODEL SIMULATION #########
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

######### SUBSECTION 6.4: APPLICATION EXAMPLE #########
print("\n\n\nSUBSECTION 6.4: APPLICATION EXAMPLE")
print("Printing verbose output\n")

import pandas as pd
from stepmix.stepmix import StepMix

data = pd.read_csv("StepMix_Real_Data_GSS.csv")
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
