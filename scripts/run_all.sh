#!/usr/bin/env bash
# Paper Examples
printf "SECTIONS 5 and 6: PAPER EXAMPLES\n"
python3 paper_examples.py

printf "\n\n\nNOW REPRODUCING TABLE RESULTS\n"

# Outcome Simulation
printf "\n\n\nSUBSECTION 6.1: FULL OUTCOME SIMULATION\n"
python3 run_bakk_simulation.py -s $1  # Replace 5 with 500 for the full simulation
# Covariate Simulation
printf "\n\n\nSUBSECTION 6.2: FULL COVARIATE SIMULATION\n"
python3 run_bakk_simulation.py -s $1 -c  # Replace 5 with 500 for the full simulation
# Complete Simulation
printf "\n\n\nSUBSECTION 6.3: FULL COMPLETE MODEL SIMULATION\n"
python3 run_bakk_simulation_complete.py -s $1  # Replace 5 with 500 for the full simulation
# Real Data Example
printf "\n\n\nSUBSECTION 6.4: APPLICATION EXAMPLE\n"
python3 run_real_example.py --n_repetitions $2 --max_iter 10000  # Replace 5 with 100 for the full results
# Package Comparison
printf "\n\n\nSUBSECTION 6.5: PACKAGE COMPARISON (STEPMIX ONLY)\n"
python3 run_package_comparison.py
