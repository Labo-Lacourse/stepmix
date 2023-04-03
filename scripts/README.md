Computational examples
==============================
This directory provides the scripts to reproduce the simulations and tables from the Computational
examples section of the StepMix paper. Results can be reproduced with the 
following commands:

```bash
# Outcome simulation
python3 run_bakk_simulation.py -s 500 -l
# Covariate simulation
python3 run_bakk_simulation.py -s 500 -l -c
# Complete simulation
python3 run_bakk_simulation_complete.py -s 500 -l
```
All three commands should output the latex tables from the paper, up to formatting. You can ignore the UserWarnings.
