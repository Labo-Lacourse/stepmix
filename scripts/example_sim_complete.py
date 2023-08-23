"""Single complete model simulation (with outcome and covariate)."""
from stepmix.datasets import data_bakk_complete
from stepmix import StepMix

# Simulate Data
# Z includes both the covariate (first column) and
# the response (second column)
Y, Z, labels = data_bakk_complete(2000, sep_level=0.8, nan_ratio=0.25, random_state=42)

# Define the structural model
structural_descriptor = {
    # Covariate
    "covariate": {
        "model": "covariate",
        "n_columns": 1,
        "method": "newton-raphson",
        "max_iter": 1,
    },
    # Response
    "response": {"model": "gaussian_unit_nan", "n_columns": 1},  # Allow missing values
}

# Fit StepMix Estimator
model = StepMix(
    n_components=3,
    measurement="binary_nan",
    structural=structural_descriptor,
    n_steps=1,
    random_state=42,
)
model.fit(Y, Z)

# Retrieve structural model response parameters as a dataframe
params = model.get_sm_df().loc["response"]
print(params)
