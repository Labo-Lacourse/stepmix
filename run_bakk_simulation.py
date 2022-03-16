#!/usr/bin/env python3
"""Partial reproduction of results from Bakk & Kuha (2018)"""
import argparse

import pandas as pd
import numpy as np

from lca.lca import LCA
from lca.datasets import data_generation_Bakk


# Parser
parser = argparse.ArgumentParser(description="Reproduce the response variable results from Bakk & Kuha 2018.")
parser.add_argument('--n_simulations',
                    '-s',
                    help='Number of simulations to run. Results are averaged.',
                    type=int, 
                    default=500)
parser.add_argument('--latex',
                    '-l',
                    help='Also print a latex version of the results.',
                    action='store_true')

args = parser.parse_args()

# Common arguments for all models
lca_args = dict(n_components=3,
                measurement='bernoulli',
                structural='gaussian_unit',
                tol=1e-5,
                n_init=1,
                max_iter=1000)

# Model-specific arguments
models = {
    '1-step': dict(n_steps=1),
    '2-step': dict(n_steps=2),
    '3-step (Naive)': dict(n_steps=3, assignment='modal', correction=None),
    '3-step (BCH)': dict(n_steps=3, assignment='modal', correction='BCH'),
    '3-step (ML)': dict(n_steps=3, assignment='modal', correction='ML'),
}

# Result collector
results = list()

# Loop over repetitions
for r in range(args.n_simulations):
    random_state = 12345 * r

    # Loop over sample sizes
    for size in [500, 1000, 2000]:
        # Loop over separations
        for sep in [.7, .8, .9]:
            # Generate dataset
            X, Y = data_generation_Bakk(sample_size=size, sep_level=sep, random_state=random_state)

            # Loop over models
            for name, model_args in models.items():
                model = LCA(**lca_args, **model_args, random_state=random_state)
                model.fit(X, Y)

                # Get max mean
                mu = model.get_parameters()['structural']['means'].max()

                # Save results
                result_i = {
                    'Class Separation': sep,
                    'Sample Size': size,
                    'Model': name,
                    'mu': mu,
                }
                results.append(result_i)


# Use pandas to average and print results
df = pd.DataFrame(results)

# Performance metrics
g = df.groupby(['Class Separation', 'Sample Size', 'Model'])
df = g.apply(lambda x: (x-1).mean())
df = df.rename(dict(mu='Bias'), axis=1)
df['RMSE'] = g.apply(lambda x: np.sqrt(((x-1) ** 2).mean()))

# Get a nice format
df = pd.melt(df.reset_index(), id_vars=df.index.names)
df = df.pivot(index=['Class Separation', 'Sample Size'], columns=['variable', 'Model'], values='value')
df = df.reindex(['1-step', '2-step', '3-step (Naive)', '3-step (BCH)', '3-step (ML)'], axis=1, level=1).round(2)

# Print!
print(df.to_string())

if args.latex:
    print(df.to_latex(multirow=True, multicolumn=True))