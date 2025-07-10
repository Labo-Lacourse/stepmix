from pathlib import Path
import pandas as pd
from stepmix.stepmix import StepMix
import time

data_dir = Path("data")
results_ll = dict()
results_time = dict()

# Carcinoma
carci_DF = pd.read_csv(data_dir / "carcinoma.csv")
carci_MM = carci_DF.iloc[:, [1, 2, 3, 4, 5, 6, 7]]
carci_MM -= 1  # to obtain binary indicators

t0 = time.time()
model_simple = StepMix(
    n_components=3,
    measurement="binary",
    random_state=123,
    verbose=0,
    progress_bar=False,
)

model_simple.fit(carci_MM)
t1 = time.time()

results_ll["carcinoma"] = model_simple.score(carci_MM) * carci_MM.shape[0]
results_time["carcinoma"] = t1 - t0

# Simulated data: Distal outcome
Datasim_Dist_DF = pd.read_csv(data_dir / "datasim_Dist.csv")
Datasim_Dist_MM = Datasim_Dist_DF.iloc[:, [1, 2, 3, 4, 5, 6]]
Datasim_Dist_SM = Datasim_Dist_DF.iloc[:, [7]]

t0 = time.time()
model_Dist_SM = StepMix(
    n_components=3,
    measurement="binary",
    structural="gaussian_unit",
    random_state=123,
    n_steps=1,
    verbose=0,
    progress_bar=False,
)

model_Dist_SM.fit(Datasim_Dist_MM, Datasim_Dist_SM)
t1 = time.time()

results_ll["bakk_sim_dist"] = (
    model_Dist_SM.score(Datasim_Dist_MM, Datasim_Dist_SM) * Datasim_Dist_MM.shape[0]
)
results_time["bakk_sim_dist"] = t1 - t0


# Simulated data: Covariate
Datasim_cov_DF = pd.read_csv(data_dir / "datasim_cov.csv")
Datasim_cov_MM = Datasim_cov_DF.iloc[:, [1, 2, 3, 4, 5, 6]]
Datasim_cov_SM = Datasim_cov_DF.iloc[:, [7]]

covariate_params = {
    "method": "newton-raphson",
    "max_iter": 1,
    "intercept": True,
}

t0 = time.time()
model_cov_SM = StepMix(
    n_components=3,
    measurement="binary",
    structural="covariate",
    structural_params=covariate_params,
    random_state=123,
    n_steps=1,
    verbose=0,
    progress_bar=False,
)

model_cov_SM.fit(Datasim_cov_MM, Datasim_cov_SM)
t1 = time.time()

results_ll["bakk_sim_cov"] = (
    model_cov_SM.score(Datasim_cov_MM, Datasim_cov_SM) * Datasim_cov_MM.shape[0]
)
results_time["bakk_sim_cov"] = t1 - t0

# IRIS
Iris_DF = pd.read_csv(data_dir / "iris.csv")
Iris_MM = Iris_DF.iloc[:, [1, 2, 3, 4]]


t0 = time.time()
model_iris_MM = StepMix(
    n_components=3,
    measurement="gaussian_full",
    random_state=3,
    n_init=20,
    verbose=0,
    progress_bar=False,
)

model_iris_MM.fit(Iris_MM)
t1 = time.time()

results_ll["iris"] = model_iris_MM.score(Iris_MM) * Iris_MM.shape[0]
results_time["iris"] = t1 - t0

# Diabetes
Diabetes_DF = pd.read_csv(data_dir / "diabetes.csv")
Diabetes_MM = Diabetes_DF.iloc[:, [2, 3, 4]]

SM_param_integer = {"Normal": 0, "Chemical": 1, "Overt": 2}
Diabetes_SM = Diabetes_DF.iloc[:, 1].map(SM_param_integer)

t0 = time.time()
model_diabetes_SM = StepMix(
    n_components=3,
    measurement="continuous",
    structural="categorical",
    random_state=123,
    n_steps=1,
    verbose=0,
    progress_bar=False,
)

model_diabetes_SM.fit(Diabetes_MM, Diabetes_SM)
t1 = time.time()
results_ll["diabetes"] = (
    model_diabetes_SM.score(Diabetes_MM, Diabetes_SM) * Diabetes_MM.shape[0]
)
results_time["diabetes"] = t1 - t0

# Banknote
Banknote_DF = pd.read_csv("data/banknote.csv")
Banknote_MM = Banknote_DF.iloc[:, [2, 3, 4, 5, 6, 7]]

SM_param_integer2 = {"genuine": 0, "counterfeit": 1}
Banknote_SM = Banknote_DF.iloc[:, 1].map(SM_param_integer2)

covariate_params = {
    "method": "newton-raphson",
    "max_iter": 1,
    "intercept": True,
}

t0 = time.time()
model_banknote_SM = StepMix(
    n_components=2,
    measurement="continuous",
    structural="covariate",
    structural_params=covariate_params,
    random_state=123,
    n_steps=1,
    verbose=0,
    progress_bar=False,
)

model_banknote_SM.fit(Banknote_MM, Banknote_SM)
t1 = time.time()
results_ll["banknote"] = (
    model_banknote_SM.score(Banknote_MM, Banknote_SM) * Banknote_MM.shape[0]
)
results_time["banknote"] = t1 - t0

# Report results
print("Table 11: Fit Times (sec.)")
for key, value in results_time.items():
    print(f"{key.ljust(13)}: {value:.3f}")

print("\n")
print("Table 11: StepMix Log-Likelihoods")
for key, value in results_ll.items():
    print(f"{key.ljust(13)}: {value:.3f}")
