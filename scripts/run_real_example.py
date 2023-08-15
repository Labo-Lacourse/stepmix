### Real data example ###

import pandas as pd
import numpy as np
from stepmix.stepmix import StepMix

from sklearn.base import clone
from scipy.stats import norm


# Define print_table function that we can reuse
def print_table(df, title):
    print(f"\n{title}")
    print(df.round(2).to_string())


# Load data
data = pd.read_csv('StepMix_Real_Data_GSS.csv')
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

### Simple LCA model
model_simple = StepMix(
    n_components=3,
    measurement="categorical_nan",
    random_state=123,
    max_iter=10000,  # Not strictly required, but used for consistency with bootstrap structural models
    verbose=0,
    progress_bar=False,
)

# Fit and print parameters of main model
model_simple.fit(data_mm)

# Table 8 : Estimated MM parameters
params_mm = model_simple.get_parameters_df().loc["measurement"]

params_mm_pivot = pd.pivot_table(
    params_mm, columns="class_no", values="value", index=["model_name", "variable"]
).reindex()

# Rename and reorder class columns
params_mm_pivot = params_mm_pivot.rename(columns={0: "Low", 1: "High", 2: "Middle"})[
    ["Low", "Middle", "High"]
]
print_table(params_mm_pivot, "Table 8 : Estimated MM parameters")


### Multi-Step Approaches
model_base = StepMix(
    n_components=3,
    measurement="categorical_nan",
    structural="gaussian_diag_nan",
    random_state=123,
    max_iter=10000,
    verbose=0,
    n_steps=1,
    progress_bar=False,
)

params = list()

# Define model as n_steps, correction and permutation to apply to classes
# None means no correction
model_params = [
    (1, None, [0, 1, 2]),  # (n_steps, correction, permutation)
    (2, None, [0, 2, 1]),
    (3, None, [0, 2, 1]),
    (3, "BCH", [0, 2, 1]),
    (3, "ML", [0, 2, 1]),
]

for n_steps, correction, perm in model_params:
    # Clone base model and set n_steps and correction
    model = clone(model_base).set_params(n_steps=n_steps, correction=correction)
    model.fit(data_mm, data_sm)
    model.permute_classes(perm)

    bootstrap_params, _ = model.bootstrap(
        data_mm, data_sm, n_repetitions=100, progress_bar=False
    )

    # Add column with model descriptor
    bootstrap_params["Method"] = (
        f"{n_steps}-step" + ("" if correction is None else f" ({correction})")
    )

    # Save parameters to main list
    means = bootstrap_params.loc["structural", "gaussian_diag_nan", "means"]
    params.append(means)

# Concat all bootstrapped parameters in one dataframe
params = (
    pd.concat(params, axis=0)
    .reset_index()
    .set_index(["class_no", "Method", "variable", "rep"])
)

# Table 9 : Bootstrapped SM parameters
params_pivot = pd.pivot_table(
    params,
    index="Method",
    columns="class_no",
    values="value",
    aggfunc=[np.mean, np.std],
).reindex()
params_pivot = params_pivot.rename(columns={0: "Low", 1: "Middle", 2: "High"})
print_table(params_pivot, "Table 9 : Estimated SM parameters")


# Table 10 : Z-scores
params_diff = params - params.loc[0]  # Remove means of class 0 from all classes
params_diff_pivot = (
    pd.pivot_table(
        params_diff,
        index=["Method", "class_no"],
        values="value",
        aggfunc=[np.mean, np.std],
    )
    .reindex()
    .droplevel(1, axis=1)
)

params_diff_pivot = params_diff_pivot.rename(index={0: "Low", 1: "Middle", 2: "High"})
params_diff_pivot = params_diff_pivot.drop(index="Low", level="class_no")
params_diff_pivot["Z"] = params_diff_pivot["mean"] / params_diff_pivot["std"]
params_diff_pivot["P(<|t|)"] = 2 * norm.cdf(-np.abs(params_diff_pivot["Z"]))
print_table(
    params_diff_pivot,
    "Table 10 : Family’s income differences between classes for each method.",
)
