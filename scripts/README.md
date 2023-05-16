Computational examples
==============================
This directory provides the scripts to reproduce the simulations and tables from the Computational
examples section of the StepMix paper. Results can be reproduced with the 
following commands:

```bash
# Outcome simulation
python3 run_bakk_simulation.py -s 500
# Covariate simulation
python3 run_bakk_simulation.py -s 500 -c
# Complete simulation
python3 run_bakk_simulation_complete.py -s 500
```
All three commands should output the simulation tables from the paper. You can ignore the UserWarnings.
