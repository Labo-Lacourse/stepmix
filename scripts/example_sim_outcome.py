"""Single outcome simulation."""
from stepmix.datasets import data_bakk_response
from stepmix.stepmix import StepMix

# Simulate data
Y, Z_o, _ = data_bakk_response(n_samples=2000, sep_level=.9, 
                               random_state=42)

# Fit StepMix Estimator
model = StepMix(n_components=3, measurement='binary',
                structural='gaussian_unit', n_steps=1, random_state=42,
                verbose=1)
model.fit(Y, Z_o)

# Retrieve structural model means as a dataframe
mus = model.get_sm_df()
print(mus)