import pandas as pd
import numpy as np
from stepmix.stepmix import StepMix
import time

#Carcinoma
carci_DF = pd.read_csv("Data/carcinoma.csv")
carci_MM = carci_DF.iloc[:, [1,2,3,4,5,6,7]]
carci_MM -= 1 #to obtain binary indicators

t0 = time.time()
model_simple = StepMix(n_components=3, 
                       measurement='binary', 
                       random_state=123,  
                       verbose=1)

model_simple.fit(carci_MM)
t1 = time.time()

total = t1-t0
round(total, 5)

#Simulated data: Distal outcome
Datasim_Dist_DF = pd.read_csv("Data/datasim_Dist.csv")
Datasim_Dist_MM = Datasim_Dist_DF.iloc[:, [1,2,3,4,5,6]]
Datasim_Dist_SM = Datasim_Dist_DF.iloc[:, [7]]

t0 = time.time()
model_Dist_SM = StepMix(n_components=3, 
                        measurement='binary',
                        structural='gaussian_unit',
                        random_state=123,
                        n_steps=1,
                        verbose=1)

model_Dist_SM.fit(Datasim_Dist_MM, Datasim_Dist_SM)
t1 = time.time()

total = t1-t0
round(total, 5)


#Simulated data: Covariate
Datasim_cov_DF = pd.read_csv("Data/datasim_cov.csv")
Datasim_cov_MM = Datasim_cov_DF.iloc[:, [1,2,3,4,5,6]]
Datasim_cov_SM = Datasim_cov_DF.iloc[:, [7]]

covariate_params = {
    "method": "newton-raphson", 
    "max_iter": 1,
    "intercept": True,
}

t0 = time.time()
model_cov_SM = StepMix(n_components=3, 
                       measurement='binary',
                       structural='covariate', 
                       structural_params=covariate_params,
                       random_state=123,
                       n_steps=1,
                       verbose=1)

model_cov_SM.fit(Datasim_cov_MM, Datasim_cov_SM)
t1 = time.time()

total = t1-t0
round(total, 5)

#IRIS
Iris_DF = pd.read_csv("Data/iris.csv")
Iris_MM = Iris_DF.iloc[:, [1,2,3,4]]


t0 = time.time()
model_iris_MM = StepMix(n_components=3, 
                        measurement='continuous', 
                        random_state=123,
                        verbose=1)

model_iris_MM.fit(Iris_MM)
t1 = time.time()

total = t1-t0
round(total, 5)

#Diabetes
Diabetes_DF = pd.read_csv("Data/diabetes.csv")
Diabetes_MM = Diabetes_DF.iloc[:, [2,3,4]]

Diabetes_SM = Diabetes_DF.iloc[:, [1]]
SM_param_integer = {"Normal": 0, "Chemical": 1, "Overt": 2}
Diabetes_SM['class'] = Diabetes_SM['class'].map(SM_param_integer)

t0 = time.time()
model_diabetes_SM = StepMix(n_components=3, 
                            measurement='continuous', 
                            structural='categorical',
                            random_state=123,
                            n_steps=1,
                            verbose=1)

model_diabetes_SM.fit(Diabetes_MM,Diabetes_SM)
t1 = time.time()

total = t1-t0
round(total, 5)

#Banknote
Banknote_DF = pd.read_csv("Data/banknote.csv")
Banknote_MM = Banknote_DF.iloc[:, [2,3,4,5,6,7]]

Banknote_SM = Banknote_DF.iloc[:, [1]]
SM_param_integer2 = {"genuine": 0, "counterfeit": 1}
Banknote_SM['Status'] = Banknote_SM['Status'].map(SM_param_integer2)

covariate_params = {
    "method": "newton-raphson", 
    "max_iter": 1,
    "intercept": True,
}

t0 = time.time()
model_banknote_SM = StepMix(n_components=2, 
                            measurement='continuous', 
                            structural='covariate',
                            structural_params=covariate_params,
                            random_state=123,
                            n_steps=1,
                            verbose=1)

model_banknote_SM.fit(Banknote_MM, Banknote_SM)
t1 = time.time()

total = t1-t0
round(total, 5)





























