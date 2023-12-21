"""Visualize StepMix verbose output on GSS data."""
import pandas as pd
from stepmix.stepmix import StepMix

# Load data
data = pd.read_csv("StepMix_Real_Data_GSS.csv")
data = data.rename(
    columns={
        "realinc1000": "Income (1000)",
        "papres": "Father's job prestige",
        "madeg": "Mother's education",
        "padeg": "Father's education",
    }
)

# The "papres", "madeg" and "padeg" variables are used as items in the measurement model.
# The "realinc1000" variable is used as the distal outcome
data_mm, data_sm = (
    data[["Father's job prestige", "Mother's education", "Father's education"]],
    data[["Income (1000)"]],
)

# Model
model = StepMix(
    n_components=3,
    measurement="categorical_nan",
    structural="gaussian_diag_nan",
    n_steps=3,
    correction="ML",
    assignment="modal",
    random_state=123,
    max_iter=10000,  # Not strictly required, but used for consistency with bootstrap structural models
    verbose=0,
    progress_bar=True,
)
model.fit(data_mm, data_sm)
model.permute_classes([0, 2, 1])  # For classes to align with Low, Medium, High
model.report(data_mm, data_sm)  # Manually print report
