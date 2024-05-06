### Real data example ###

import argparse
import pandas as pd
import numpy as np
from stepmix.stepmix import StepMix

from scipy.stats import norm


def main(n_repetitions, max_iter):
    # Load data
    data = pd.read_csv("data/StepMix_Real_Data_GSS.csv")
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
        max_iter=max_iter,  # Not strictly required, but used for consistency with bootstrap structural models
        verbose=0,
        progress_bar=False,
    )

    # Fit and print parameters of main model
    model_simple.fit(data_mm)
    model_simple.permute_classes(
        [0, 2, 1]
    )  # Permute classes to ensure that 0, 1, 2 is Low, Medium, High

    # Table 8 : Estimated MM parameters
    params_mm = model_simple.get_mm_df()
    params_mm = params_mm.rename(
        columns={0: "Low", 1: "Middle", 2: "High"}
    )  # Rename classes

    print("Table 8 : Estimated MM parameters")
    print(params_mm.round(2))  # Round parameters

    class_weights = model_simple.get_cw_df()
    class_weights = class_weights.rename(
        columns={0: "Low", 1: "Middle", 2: "High"}
    )  # Rename classes
    print("\nTable 8 : Class weights")
    print(class_weights.round(2))  # Round parameters

    # Table 9 : Bootstrapped SM parameters
    # Define a function that we will call for all 5 multi-step estimators
    def fit_and_bootstrap(n_steps, correction, permutation, method_str):
        model = StepMix(
            n_steps=n_steps,
            correction=correction,
            n_components=3,
            measurement="categorical_nan",
            structural="gaussian_diag_nan",
            random_state=123,
            max_iter=max_iter,
            verbose=0,
            progress_bar=False,
        )

        # 1-step
        model.fit(data_mm, data_sm)
        model.permute_classes(
            permutation
        )  # Permute classes to ensure that 0, 1, 2 is Low, Medium, High
        stats_dict = model.bootstrap_stats(
            data_mm, data_sm, n_repetitions=n_repetitions, progress_bar=True
        )

        # Look at the means and standard deviations of the structural model.
        means = (
            stats_dict["sm_mean"].loc["gaussian_diag_nan", "means"].copy()
        )  # Means of the mean paramater
        errors = (
            stats_dict["sm_std"].loc["gaussian_diag_nan", "means"].copy()
        )  # STD of the mean parameter

        # Also get raw bootsrapped samples for Table 10
        samples = (
            stats_dict["samples"].loc["structural", "gaussian_diag_nan", "means"].copy()
        )  # Raw samples

        # Add column with model descriptor
        means["method"] = method_str
        errors["method"] = method_str
        samples["method"] = method_str

        # Check the class prevalence of all models
        print(f"\nClass prevalence for {method_str}:")
        print(model.get_cw_df().round(3))

        return means, errors, samples

    # Apply function to all 5 multi-step estimators
    means_1_step, errors_1_step, samples_1_step = fit_and_bootstrap(
        n_steps=1, correction=None, method_str="1-step", permutation=[0, 1, 2]
    )
    means_2_step, errors_2_step, samples_2_step = fit_and_bootstrap(
        n_steps=2, correction=None, method_str="2-step", permutation=[0, 2, 1]
    )
    means_3_step, errors_3_step, samples_3_step = fit_and_bootstrap(
        n_steps=3, correction=None, method_str="3-step", permutation=[0, 2, 1]
    )
    means_3_step_bch, errors_3_step_bch, samples_3_step_bch = fit_and_bootstrap(
        n_steps=3, correction="BCH", method_str="3-step (BCH)", permutation=[0, 2, 1]
    )
    means_3_step_ml, errors_3_step_ml, samples_3_step_ml = fit_and_bootstrap(
        n_steps=3, correction="ML", method_str="3-step (ML)", permutation=[0, 2, 1]
    )

    # Concat all results
    means_sm = pd.concat(
        [means_1_step, means_2_step, means_3_step, means_3_step_bch, means_3_step_ml]
    )
    stds_sm = pd.concat(
        [
            errors_1_step,
            errors_2_step,
            errors_3_step,
            errors_3_step_bch,
            errors_3_step_ml,
        ]
    )

    # Reindex and rename for nicer tables
    means_sm = means_sm.reset_index().set_index(["variable", "method"]).sort_index()
    means_sm = means_sm.rename(
        columns={0: "Low", 1: "Middle", 2: "High"}
    )  # Rename classes
    stds_sm = stds_sm.reset_index().set_index(["variable", "method"]).sort_index()
    stds_sm = stds_sm.rename(
        columns={0: "Low", 1: "Middle", 2: "High"}
    )  # Rename classes

    print("\nTable 9 : Estimated SM parameters (means)")
    print(means_sm.round(2))

    print("\nTable 9 : Estimated SM parameters (errors)")
    print(stds_sm.round(2))

    # Table 10 : Z-scores
    # Here we need to manually compute the differences High - Low and Middle - Low over all repetitions
    # First collect all bootstrap samples in a single dataframe
    samples = pd.concat(
        [
            samples_1_step,
            samples_2_step,
            samples_3_step,
            samples_3_step_bch,
            samples_3_step_ml,
        ]
    )

    # Send repetition to index (rows) and class_no to columns
    samples = pd.pivot_table(
        samples, index=["method", "variable", "rep"], columns="class_no", values="value"
    )
    samples = samples.rename(
        columns={0: "Low", 1: "Middle", 2: "High"}
    )  # Rename classes

    # Remove Low column from the Middle and High Columns
    samples["Middle"] = samples["Middle"] - samples["Low"]
    samples["High"] = samples["High"] - samples["Low"]

    # We no longer need the Low class
    samples = samples.drop("Low", axis=1)

    # Send back the class to index ("Rows")
    samples = samples.stack()

    # Now group by everything except the rep
    # This means we will be computing statistics over the repetitions
    stats = samples.groupby(["variable", "method", "class_no"]).agg(["mean", "std"])

    # Z-score
    stats["Z"] = stats["mean"] / stats["std"]

    # P-value
    stats["P(>|z|)"] = 2 * norm.cdf(-np.abs(stats["Z"]))

    stats[["mean", "std", "Z"]] = stats[["mean", "std", "Z"]].round(2)
    stats["P(>|z|)"] = stats["P(>|z|)"].round(3)
    print("\nTable 10 : Family’s income differences between classes for each method.")
    print(stats)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        description="Run StepMix example on real GSS data."
    )
    parser.add_argument(
        "--n_repetitions",
        "-r",
        help="Number of bootstrap repetitions.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_iter",
        "-m",
        help="Max number of EM iterations.",
        type=int,
        default=10000,
    )

    args = parser.parse_args()
    main(n_repetitions=args.n_repetitions, max_iter=args.max_iter)
