Simulation and Real-Data Examples
==============================================
This directory provides the scripts to reproduce the simulations and tables from the Computational examples section of the StepMix paper. The script to reproduce the application example is also accessible here.

The code blocks in the paper fit a single model and you can run them with the `example_*.py` file. The full table results can be reproduced by running the `run_*.py` files.

For `example_real_GSS.py` and `run_real_example.py`, you will need to download the [GSS dataset](https://drive.google.com/file/d/1vdSzzBe7tPxfQ_X-hUuF3qidDd6CWFiL/view?usp=drive_link) and save it in the `scripts`directory.

Simulation Examples
--------------------------
Simulation results in Tables can be reproduced with the following commands:

```bash
# Outcome simulation
python3 run_bakk_simulation.py -s 500
# Covariate simulation
python3 run_bakk_simulation.py -s 500 -c
# Complete simulation
python3 run_bakk_simulation_complete.py -s 500
```
All three commands should output the simulation tables from the paper. You can ignore  ConvergenceWarnings and UserWarnings.


Real-Data Example
-----------------------
The data come from the combined 1976 and 1977 datasets of the American General Social Survey (GSS) and were obtained 
using the R package `gssr` (https://github.com/kjhealy/gssr). The dataset is composed of the following 4 variables: 

 - `papres`: Father's occupational prestige score
    - 0="Low", 1="Medium", 2="High"
 - `madeg`: Mother's highest degree
    - 0="Less than high school", 1="High school", 2="Associate/Junior college", 3="Bachelor", 4="Graduate"
 - `padeg`: Father's highest degree
    - 0="Less than high school", 1="High school", 2="Associate/Junior college", 3="Bachelor", 4="Graduate"
 - `realinc1000`: Respondent’s family income measured in thousands of dollars

The `papres`, `madeg`, and `padeg` variables are used as items for the measurement model. The `realinc1000` variable, 
used as the distal outcome, was obtained by dividing each value of the original variable `realinc` 
(respondent’s family income in constant dollars) by 1000. Please refer to the article for more information. You can run all 
the StepMix estimators with

```bash
python3 run_real_example.py --n_repetitions 100 --max_iter 10000
```

Running the above may take a while (15 minutes on a Macbook Pro M2 Pro). You can lower the number of bootstrap repetitions for faster results.
