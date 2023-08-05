### Real data example ###

import pandas as pd
import numpy as np
from stepmix.stepmix import StepMix
from stepmix.bootstrap import bootstrap
from tabulate import tabulate
from scipy.stats import norm

data = pd.read_csv("StepMix_Real_Data_GSS.csv")

data_mesure = data.iloc[
    :, [0, 1, 2]
]  # The "papres", "madeg" and "padeg" variables are used as items in the measurement model.
realinc1000_Dist = data[
    "realinc1000"
]  # The "realinc1000" variable is used as the distal outcome


### Simple LCA model
model_simple = StepMix(
    n_components=3,
    measurement="categorical_nan",
    random_state=123,
    max_iter=10000,  # Not strictly required, but used for consistency with bootstrap structural models
    verbose=1,
)

model_simple.fit(data_mesure)

# Table 8
params_model_simple = model_simple.get_parameters()

headers_Table8 = ["", "Low", " Middle", "High"]

Data_Table8 = [
    [
        "Class size",
        params_model_simple["weights"][0],
        params_model_simple["weights"][2],
        params_model_simple["weights"][1],
    ],
    ["Father's job prestige", "", "", ""],
    [
        "Low",
        params_model_simple["measurement"]["pis"][0][0],
        params_model_simple["measurement"]["pis"][2][0],
        params_model_simple["measurement"]["pis"][1][0],
    ],
    [
        "Middle",
        params_model_simple["measurement"]["pis"][0][1],
        params_model_simple["measurement"]["pis"][2][1],
        params_model_simple["measurement"]["pis"][1][1],
    ],
    [
        "High",
        params_model_simple["measurement"]["pis"][0][2],
        params_model_simple["measurement"]["pis"][2][2],
        params_model_simple["measurement"]["pis"][1][2],
    ],
    ["Mother’s education", "", "", ""],
    [
        "Below high school",
        params_model_simple["measurement"]["pis"][0][5],
        params_model_simple["measurement"]["pis"][2][5],
        params_model_simple["measurement"]["pis"][1][5],
    ],
    [
        "High school",
        params_model_simple["measurement"]["pis"][0][6],
        params_model_simple["measurement"]["pis"][2][6],
        params_model_simple["measurement"]["pis"][1][6],
    ],
    [
        "Junior college",
        params_model_simple["measurement"]["pis"][0][7],
        params_model_simple["measurement"]["pis"][2][7],
        params_model_simple["measurement"]["pis"][1][7],
    ],
    [
        "Bachelor",
        params_model_simple["measurement"]["pis"][0][8],
        params_model_simple["measurement"]["pis"][2][8],
        params_model_simple["measurement"]["pis"][1][8],
    ],
    [
        "Graduate",
        params_model_simple["measurement"]["pis"][0][9],
        params_model_simple["measurement"]["pis"][2][9],
        params_model_simple["measurement"]["pis"][1][9],
    ],
    ["Father’s education", "", "", ""],
    [
        "Below high school",
        params_model_simple["measurement"]["pis"][0][10],
        params_model_simple["measurement"]["pis"][2][10],
        params_model_simple["measurement"]["pis"][1][10],
    ],
    [
        "High school",
        params_model_simple["measurement"]["pis"][0][11],
        params_model_simple["measurement"]["pis"][2][11],
        params_model_simple["measurement"]["pis"][1][11],
    ],
    [
        "Junior college",
        params_model_simple["measurement"]["pis"][0][12],
        params_model_simple["measurement"]["pis"][2][12],
        params_model_simple["measurement"]["pis"][1][12],
    ],
    [
        "Bachelor",
        params_model_simple["measurement"]["pis"][0][13],
        params_model_simple["measurement"]["pis"][2][13],
        params_model_simple["measurement"]["pis"][1][13],
    ],
    [
        "Graduate",
        params_model_simple["measurement"]["pis"][0][14],
        params_model_simple["measurement"]["pis"][2][14],
        params_model_simple["measurement"]["pis"][1][14],
    ],
]

Data_Table8_rounded = [
    [item if isinstance(item, str) else round(item, 2) for item in row]
    for row in Data_Table8
]

Table8 = tabulate(Data_Table8_rounded, headers_Table8, tablefmt="fancy_grid")
print(Table8)


### LCA with distal outcome

# One-step approach
model_Dist_1step = StepMix(
    n_components=3,
    measurement="categorical_nan",
    structural="continuous_nan",
    random_state=123,
    max_iter=10000,
    verbose=0,
    n_steps=1,
)

model_Dist_1step, bootstrapped_params_1step = bootstrap(
    model_Dist_1step, data_mesure, realinc1000_Dist, n_repetitions=100
)

# Two-step approach
model_Dist_2steps = StepMix(
    n_components=3,
    measurement="categorical_nan",
    structural="continuous_nan",
    random_state=123,
    max_iter=10000,
    verbose=0,
    n_steps=2,
)

model_Dist_2steps, bootstrapped_params_2steps = bootstrap(
    model_Dist_2steps, data_mesure, realinc1000_Dist, n_repetitions=100
)

# Naive three-step approach
model_Dist_3steps_Naive = StepMix(
    n_components=3,
    measurement="categorical_nan",
    structural="continuous_nan",
    random_state=123,
    max_iter=10000,
    verbose=0,
    n_steps=3,
)

model_Dist_3steps_Naive, bootstrapped_params_3steps_Naive = bootstrap(
    model_Dist_3steps_Naive, data_mesure, realinc1000_Dist, n_repetitions=100
)

# Three-step BCH
model_Dist_3steps_BCH = StepMix(
    n_components=3,
    measurement="categorical_nan",
    structural="continuous_nan",
    random_state=123,
    max_iter=10000,
    verbose=0,
    n_steps=3,
    correction="BCH",
)

model_Dist_3steps_BCH, bootstrapped_params_3steps_BCH = bootstrap(
    model_Dist_3steps_BCH, data_mesure, realinc1000_Dist, n_repetitions=100
)

# Three-step ML
model_Dist_3steps_ML = StepMix(
    n_components=3,
    measurement="categorical_nan",
    structural="continuous_nan",
    random_state=123,
    max_iter=10000,
    verbose=0,
    n_steps=3,
    correction="ML",
)

model_Dist_3steps_ML, bootstrapped_params_3steps_ML = bootstrap(
    model_Dist_3steps_ML, data_mesure, realinc1000_Dist, n_repetitions=100
)


# Weights
print("Class prevalence one-step model. LCA model distorted")
params_1step = model_Dist_1step.get_parameters()
print(params_1step["weights"])

print("    ")

print("Class prevalence two-step model")
params_2steps = model_Dist_2steps.get_parameters()
print(params_2steps["weights"])

print("    ")

print("Class prevalence naive three-step model")
params_3steps_Naive = model_Dist_3steps_Naive.get_parameters()
print(params_3steps_Naive["weights"])

print("    ")

print("Class prevalence three-step BCH model")
params_3steps_BCH = model_Dist_3steps_BCH.get_parameters()
print(params_3steps_BCH["weights"])

print("    ")

print("Class prevalence three-step ML model")
params_3steps_ML = model_Dist_3steps_ML.get_parameters()
print(params_3steps_ML["weights"])


# Table 9
# One-step parameters
param_c0_response_1step = bootstrapped_params_1step["structural"]["means"][
    :, 0, 0
]  # average income by class
param_c1_response_1step = bootstrapped_params_1step["structural"]["means"][:, 1, 0]
param_c2_response_1step = bootstrapped_params_1step["structural"]["means"][:, 2, 0]

# Two-step parameters
param_c0_response_2steps = bootstrapped_params_2steps["structural"]["means"][:, 0, 0]
param_c1_response_2steps = bootstrapped_params_2steps["structural"]["means"][:, 1, 0]
param_c2_response_2steps = bootstrapped_params_2steps["structural"]["means"][:, 2, 0]

# Naive three-step parameters
param_c0_response_3steps_Naive = bootstrapped_params_3steps_Naive["structural"][
    "means"
][:, 0, 0]
param_c1_response_3steps_Naive = bootstrapped_params_3steps_Naive["structural"][
    "means"
][:, 1, 0]
param_c2_response_3steps_Naive = bootstrapped_params_3steps_Naive["structural"][
    "means"
][:, 2, 0]

# Three-step BCH parameters
param_c0_response_3steps_BCH = bootstrapped_params_3steps_BCH["structural"]["means"][
    :, 0, 0
]
param_c1_response_3steps_BCH = bootstrapped_params_3steps_BCH["structural"]["means"][
    :, 1, 0
]
param_c2_response_3steps_BCH = bootstrapped_params_3steps_BCH["structural"]["means"][
    :, 2, 0
]

# Three-step ML parameters
param_c0_response_3steps_ML = bootstrapped_params_3steps_ML["structural"]["means"][
    :, 0, 0
]
param_c1_response_3steps_ML = bootstrapped_params_3steps_ML["structural"]["means"][
    :, 1, 0
]
param_c2_response_3steps_ML = bootstrapped_params_3steps_ML["structural"]["means"][
    :, 2, 0
]

headers_Table9 = [
    "",
    "Low class income",
    "SE",
    " Middle class income",
    "SE",
    "High class income",
    "SE",
]
Data_Table9 = [
    [
        "One-step",
        np.mean(param_c0_response_1step),
        np.std(bootstrapped_params_1step["structural"]["means"][:, 0]),
        np.mean(param_c1_response_1step),
        np.std(bootstrapped_params_1step["structural"]["means"][:, 1]),
        np.mean(param_c2_response_1step),
        np.std(bootstrapped_params_1step["structural"]["means"][:, 2]),
    ],
    [
        "Two-step",
        np.mean(param_c0_response_2steps),
        np.std(bootstrapped_params_2steps["structural"]["means"][:, 0]),
        np.mean(param_c2_response_2steps),
        np.std(bootstrapped_params_2steps["structural"]["means"][:, 2]),
        np.mean(param_c1_response_2steps),
        np.std(bootstrapped_params_2steps["structural"]["means"][:, 1]),
    ],
    [
        "Naive three-step",
        np.mean(param_c0_response_3steps_Naive),
        np.std(bootstrapped_params_3steps_Naive["structural"]["means"][:, 0]),
        np.mean(param_c2_response_3steps_Naive),
        np.std(bootstrapped_params_3steps_Naive["structural"]["means"][:, 2]),
        np.mean(param_c1_response_3steps_Naive),
        np.std(bootstrapped_params_3steps_Naive["structural"]["means"][:, 1]),
    ],
    [
        "Three-step BCH",
        np.mean(param_c0_response_3steps_BCH),
        np.std(bootstrapped_params_3steps_BCH["structural"]["means"][:, 0]),
        np.mean(param_c2_response_3steps_BCH),
        np.std(bootstrapped_params_3steps_BCH["structural"]["means"][:, 2]),
        np.mean(param_c1_response_3steps_BCH),
        np.std(bootstrapped_params_3steps_BCH["structural"]["means"][:, 1]),
    ],
    [
        "Three-step ML",
        np.mean(param_c0_response_3steps_ML),
        np.std(bootstrapped_params_3steps_ML["structural"]["means"][:, 0]),
        np.mean(param_c2_response_3steps_ML),
        np.std(bootstrapped_params_3steps_ML["structural"]["means"][:, 2]),
        np.mean(param_c1_response_3steps_ML),
        np.std(bootstrapped_params_3steps_ML["structural"]["means"][:, 1]),
    ],
]

Data_Table9_rounded = [
    [item if isinstance(item, str) else round(item, 2) for item in row]
    for row in Data_Table9
]

Table9 = tabulate(Data_Table9_rounded, headers_Table9, tablefmt="fancy_grid")
print(Table9)


# Table 10
# Differences in the familys’ income (C=Low vs C=middle, C=Low vs C=High) for each approach
DIFF01 = (
    bootstrapped_params_1step["structural"]["means"][:, 1, 0]
    - bootstrapped_params_1step["structural"]["means"][:, 0, 0]
)
DIFF02 = (
    bootstrapped_params_1step["structural"]["means"][:, 2, 0]
    - bootstrapped_params_1step["structural"]["means"][:, 0, 0]
)

DIFF01_2steps = (
    bootstrapped_params_2steps["structural"]["means"][:, 1, 0]
    - bootstrapped_params_2steps["structural"]["means"][:, 0, 0]
)
DIFF02_2steps = (
    bootstrapped_params_2steps["structural"]["means"][:, 2, 0]
    - bootstrapped_params_2steps["structural"]["means"][:, 0, 0]
)

DIFF01_3steps_Naive = (
    bootstrapped_params_3steps_Naive["structural"]["means"][:, 1, 0]
    - bootstrapped_params_3steps_Naive["structural"]["means"][:, 0, 0]
)
DIFF02_3steps_Naive = (
    bootstrapped_params_3steps_Naive["structural"]["means"][:, 2, 0]
    - bootstrapped_params_3steps_Naive["structural"]["means"][:, 0, 0]
)

DIFF01_3steps_BCH = (
    bootstrapped_params_3steps_BCH["structural"]["means"][:, 1, 0]
    - bootstrapped_params_3steps_BCH["structural"]["means"][:, 0, 0]
)
DIFF02_3steps_BCH = (
    bootstrapped_params_3steps_BCH["structural"]["means"][:, 2, 0]
    - bootstrapped_params_3steps_BCH["structural"]["means"][:, 0, 0]
)

DIFF01_3steps_ML = (
    bootstrapped_params_3steps_ML["structural"]["means"][:, 1, 0]
    - bootstrapped_params_3steps_ML["structural"]["means"][:, 0, 0]
)
DIFF02_3steps_ML = (
    bootstrapped_params_3steps_ML["structural"]["means"][:, 2, 0]
    - bootstrapped_params_3steps_ML["structural"]["means"][:, 0, 0]
)


# One-step
Coef_table_01 = {"Est": np.mean(DIFF01), "SE": np.std(DIFF01)}
Coef_table_01["Z"] = Coef_table_01["Est"] / Coef_table_01["SE"]
Coef_table_01["P(<|t|)"] = 2 * norm.cdf(-np.abs(Coef_table_01["Z"]))
Coef_table_02 = {"Est": np.mean(DIFF02), "SE": np.std(DIFF02)}
Coef_table_02["Z"] = Coef_table_02["Est"] / Coef_table_02["SE"]
Coef_table_02["P(<|t|)"] = 2 * norm.cdf(-np.abs(Coef_table_02["Z"]))

# Two-step
Coef_table_02_2steps = {"Est": np.mean(DIFF02_2steps), "SE": np.std(DIFF02_2steps)}
Coef_table_02_2steps["Z"] = Coef_table_02_2steps["Est"] / Coef_table_02_2steps["SE"]
Coef_table_02_2steps["P(<|t|)"] = 2 * norm.cdf(-np.abs(Coef_table_02_2steps["Z"]))
Coef_table_01_2steps = {"Est": np.mean(DIFF01_2steps), "SE": np.std(DIFF01_2steps)}
Coef_table_01_2steps["Z"] = Coef_table_01_2steps["Est"] / Coef_table_01_2steps["SE"]
Coef_table_01_2steps["P(<|t|)"] = 2 * norm.cdf(-np.abs(Coef_table_01_2steps["Z"]))

# Naive three-step
Coef_table_02_3steps_Naive = {
    "Est": np.mean(DIFF02_3steps_Naive),
    "SE": np.std(DIFF02_3steps_Naive),
}
Coef_table_02_3steps_Naive["Z"] = (
    Coef_table_02_3steps_Naive["Est"] / Coef_table_02_3steps_Naive["SE"]
)
Coef_table_02_3steps_Naive["P(<|t|)"] = 2 * norm.cdf(
    -np.abs(Coef_table_02_3steps_Naive["Z"])
)
Coef_table_01_3steps_Naive = {
    "Est": np.mean(DIFF01_3steps_Naive),
    "SE": np.std(DIFF01_3steps_Naive),
}
Coef_table_01_3steps_Naive["Z"] = (
    Coef_table_01_3steps_Naive["Est"] / Coef_table_01_3steps_Naive["SE"]
)
Coef_table_01_3steps_Naive["P(<|t|)"] = 2 * norm.cdf(
    -np.abs(Coef_table_01_3steps_Naive["Z"])
)

# Three-step BCH
Coef_table_02_3steps_BCH = {
    "Est": np.mean(DIFF02_3steps_BCH),
    "SE": np.std(DIFF02_3steps_BCH),
}
Coef_table_02_3steps_BCH["Z"] = (
    Coef_table_02_3steps_BCH["Est"] / Coef_table_02_3steps_BCH["SE"]
)
Coef_table_02_3steps_BCH["P(<|t|)"] = 2 * norm.cdf(
    -np.abs(Coef_table_02_3steps_BCH["Z"])
)
Coef_table_01_3steps_BCH = {
    "Est": np.mean(DIFF01_3steps_BCH),
    "SE": np.std(DIFF01_3steps_BCH),
}
Coef_table_01_3steps_BCH["Z"] = (
    Coef_table_01_3steps_BCH["Est"] / Coef_table_01_3steps_BCH["SE"]
)
Coef_table_01_3steps_BCH["P(<|t|)"] = 2 * norm.cdf(
    -np.abs(Coef_table_01_3steps_BCH["Z"])
)

# Three-step ML
Coef_table_02_3steps_ML = {
    "Est": np.mean(DIFF02_3steps_ML),
    "SE": np.std(DIFF02_3steps_ML),
}
Coef_table_02_3steps_ML["Z"] = (
    Coef_table_02_3steps_ML["Est"] / Coef_table_02_3steps_ML["SE"]
)
Coef_table_02_3steps_ML["P(<|t|)"] = 2 * norm.cdf(-np.abs(Coef_table_02_3steps_ML["Z"]))
Coef_table_01_3steps_ML = {
    "Est": np.mean(DIFF01_3steps_ML),
    "SE": np.std(DIFF01_3steps_ML),
}
Coef_table_01_3steps_ML["Z"] = (
    Coef_table_01_3steps_ML["Est"] / Coef_table_01_3steps_ML["SE"]
)
Coef_table_01_3steps_ML["P(<|t|)"] = 2 * norm.cdf(-np.abs(Coef_table_01_3steps_ML["Z"]))


Data_Table10 = [
    ["1-step", "", "", "", ""],
    [
        "Middle",
        Coef_table_01["Est"],
        Coef_table_01["SE"],
        Coef_table_01["Z"],
        format(Coef_table_01["P(<|t|)"], "0.15f"),
    ],
    [
        "High class",
        Coef_table_02["Est"],
        Coef_table_02["SE"],
        Coef_table_02["Z"],
        format(Coef_table_02["P(<|t|)"], "0.15f"),
    ],
    ["2-step", "", "", "", ""],
    [
        "Middle",
        Coef_table_02_2steps["Est"],
        Coef_table_02_2steps["SE"],
        Coef_table_02_2steps["Z"],
        format(Coef_table_02_2steps["P(<|t|)"], "0.15f"),
    ],
    [
        "High class",
        Coef_table_01_2steps["Est"],
        Coef_table_01_2steps["SE"],
        Coef_table_01_2steps["Z"],
        format(Coef_table_01_2steps["P(<|t|)"], "0.15f"),
    ],
    ["Naive 3-step", "", "", "", ""],
    [
        "Middle",
        Coef_table_02_3steps_Naive["Est"],
        Coef_table_02_3steps_Naive["SE"],
        Coef_table_02_3steps_Naive["Z"],
        format(Coef_table_02_3steps_Naive["P(<|t|)"], "0.15f"),
    ],
    [
        "High class",
        Coef_table_01_3steps_Naive["Est"],
        Coef_table_01_3steps_Naive["SE"],
        Coef_table_01_3steps_Naive["Z"],
        format(Coef_table_01_3steps_Naive["P(<|t|)"], "0.15f"),
    ],
    ["3-step BCH", "", "", "", ""],
    [
        "Middle",
        Coef_table_02_3steps_BCH["Est"],
        Coef_table_02_3steps_BCH["SE"],
        Coef_table_02_3steps_BCH["Z"],
        format(Coef_table_02_3steps_BCH["P(<|t|)"], "0.15f"),
    ],
    [
        "High class",
        Coef_table_01_3steps_BCH["Est"],
        Coef_table_01_3steps_BCH["SE"],
        Coef_table_01_3steps_BCH["Z"],
        format(Coef_table_01_3steps_BCH["P(<|t|)"], "0.15f"),
    ],
    ["3-step ML", "", "", "", ""],
    [
        "Middle",
        Coef_table_02_3steps_ML["Est"],
        Coef_table_02_3steps_ML["SE"],
        Coef_table_02_3steps_ML["Z"],
        format(Coef_table_02_3steps_ML["P(<|t|)"], "0.15f"),
    ],
    [
        "High class",
        Coef_table_01_3steps_ML["Est"],
        Coef_table_01_3steps_ML["SE"],
        Coef_table_01_3steps_ML["Z"],
        format(Coef_table_01_3steps_ML["P(<|t|)"], "0.15f"),
    ],
]

headers_Table10 = ["Est", "SE", "Z", "P(>|Z|)"]

Data_Table10_rounded = [
    [item if isinstance(item, str) else round(item, 2) for item in row]
    for row in Data_Table10
]

Table10 = tabulate(Data_Table10_rounded, headers_Table10, tablefmt="fancy_grid")
print(Table10)
