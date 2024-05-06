Simulation and Real-Data Examples
==============================================
This directory provides the scripts to reproduce the simulations and tables from the Computational examples section of the StepMix paper. The script to reproduce the application example is also accessible here.

You will need to download the [datasets](https://drive.google.com/file/d/1LCtU4Oe8SwqeSORbanKNcnQkz6X9TG92/view?usp=sharing) and extract them in the `scripts/data`directory.

Running Everything
--------------------------
The `run_all.sh` file allows you to run all the scripts at once. The first argument controls the number of simulations 
in the simulation examples, and the second argument controls the number of bootstrap repetitions in the real data 
example. To quickly run everything, simply run
```bash
./run_all.sh 5 5
```

To reproduce full results (this may take a while), run
```bash
./run_all.sh 500 100
```


Paper Examples
--------------------------
You can run all the code blocks in the paper with
```bash
python3 paper_examples.py
```

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
All three commands should output the simulation tables from the paper. You can get faster results by lowering
the number of simulations (e.g., `-s 50`).


Real Data Example
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

Running the above may take a while (15 minutes on a Macbook Pro M2 Pro). You can lower the number of bootstrap repetitions for faster results (e.g., `--n_repetitions 10`).
