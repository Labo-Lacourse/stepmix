"""Reproduction of results from Bakk & Kuha (2018)"""

import argparse
import warnings

import pandas as pd
import numpy as np

from stepmix.stepmix import StepMix
from stepmix.datasets import data_bakk_response, data_bakk_covariate
from stepmix.utils import identify_coef

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

# Original simulation parameters from Bakk 2018 do not technically converge
# but the results make sense. Ignore warnings
# You should normally tune the tolerance to avoid ConvergenceWarnings
simplefilter("ignore", category=ConvergenceWarning)


def main(n_simulations=10, latex=False, covariate=False):
    # Common arguments for all models
    stepmix_args = dict(
        n_components=3,
        measurement="bernoulli",
        structural="covariate" if covariate else "gaussian_unit",
        n_init=1,
        max_iter=250,  # Latent Gold default : 250 EM iterations + 50 NR iterations
        abs_tol=1e-8,  # Latent Gold default : 1e-8
        progress_bar=0,
    )

    # Model-specific arguments
    # Specify optimization parameters if we have a covariate model
    structural_params_12 = (
        dict(method="newton-raphson", max_iter=1) if covariate else dict()
    )
    structural_params_3 = (
        dict(method="newton-raphson", max_iter=stepmix_args["max_iter"])
        if covariate
        else dict()
    )
    models = {
        "1-step": dict(n_steps=1, structural_params=structural_params_12),
        "2-step": dict(n_steps=2, structural_params=structural_params_12),
        "3-step (Naive)": dict(
            n_steps=3,
            assignment="modal",
            correction=None,
            structural_params=structural_params_3,
        ),
        "3-step (BCH)": dict(
            n_steps=3,
            assignment="modal",
            correction="BCH",
            structural_params=structural_params_3,
        ),
        "3-step (ML)": dict(
            n_steps=3,
            assignment="modal",
            correction="ML",
            structural_params=structural_params_3,
        ),
    }

    # Result collector
    results = list()

    # Loop over repetitions
    for r in range(n_simulations):
        random_state = 12345 * r

        # Loop over sample sizes
        for size in [500, 1000, 2000]:
            # Loop over separations
            for sep in [0.7, 0.8, 0.9]:
                # Generate dataset
                dataset = data_bakk_covariate if covariate else data_bakk_response
                X, Y, _ = dataset(
                    n_samples=size, sep_level=sep, random_state=random_state
                )

                # Loop over models
                for name, model_args in models.items():
                    model = StepMix(
                        **stepmix_args, **model_args, random_state=random_state
                    )
                    model.fit(X, Y)

                    # Get max mean
                    if covariate:
                        coeff = model.get_parameters()["structural"]["beta"]

                        # Model estimates all K coefficients
                        # Recenter the middle coefficient to 0 to "recreate" the reference class
                        coef = identify_coef(coeff)

                        # Pick highest slope coefficient
                        mu = coef[:, 1].max()
                    else:
                        # Get highest mean
                        coef = model.get_parameters()["structural"]["means"]
                        mu = coef.max()

                    # Save results
                    result_i = {
                        "Class Separation": sep,
                        "Sample Size": size,
                        "Model": name,
                        "mu": mu,
                    }

                    # Test if the coefficients are degenerate
                    # This tends to happen (rarely) with the BCH correction in the low separation case
                    # with a covariate structural
                    if np.absolute(coef).max() < 1000:
                        results.append(result_i)
                    else:
                        warnings.warn(
                            f"Model {name} with sample size {size} appears degenerate. Excluding run from results."
                        )

    # Use pandas to average and print results
    df = pd.DataFrame(results)

    # Performance metrics
    g = df.groupby(["Class Separation", "Sample Size", "Model"])
    df = g.apply(lambda x: (x - 1).mean())
    df = df.rename(dict(mu="Bias"), axis=1)
    df["RMSE"] = g.apply(lambda x: np.sqrt(((x - 1) ** 2).mean()))

    # Get a nice format
    df = pd.melt(df.reset_index(), id_vars=df.index.names)
    df = df.pivot(
        index=["Class Separation", "Sample Size"],
        columns=["variable", "Model"],
        values="value",
    )
    df = df.reindex(
        ["1-step", "2-step", "3-step (Naive)", "3-step (BCH)", "3-step (ML)"],
        axis=1,
        level=1,
    ).round(2)

    # Print!
    print(df.to_string())

    # Print latex results
    if latex:
        print(df.to_latex(multirow=True, multicolumn=True))


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        description="Reproduce results from Bakk & Kuha 2018."
    )
    parser.add_argument(
        "--n_simulations",
        "-s",
        help="Number of simulations to run. Results are averaged.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--latex",
        "-l",
        help="Also print a latex version of the results. Requires the optional dependency Jinja2.",
        action="store_true",
    )
    parser.add_argument(
        "--covariate",
        "-c",
        help="Run the covariate simulation. Otherwise runs the response simulation by default.",
        action="store_true",
    )

    args = parser.parse_args()
    main(n_simulations=args.n_simulations, latex=args.latex, covariate=args.covariate)
