Tutorials
=========
Quickstart
----------
The following shows how to run StepMix on simulated data from Bakk and Kuha, 2018 to identify three latent groups.

``n_components`` controls
the number of latent classes while ``n_steps`` sets the stepwise estimation procedure. Homogeneous models (e.g., all binary, all gaussian) can be described with a single string by setting the
``measurement`` and ``structural`` parameters. Setting ``verbose=1`` ensures we get a detailed print statement::

   from stepmix.datasets import data_bakk_response
   from stepmix.stepmix import StepMix

   # Soft 3-step
   # X is an array of binary measurements. Y is a single continuous response
   X, Y, _ = data_bakk_response(n_samples=1000, sep_level=.7, random_state=42)
   model = StepMix(n_components=3, n_steps=3, measurement='bernoulli',
                   structural='gaussian_unit', assignment='soft', verbose=1, random_state=42)
   model.fit(X, Y)

The API allows to easily predict class memberships or probabilities::

    class_ids = model.predict(X, Y)
    class_probs = model.predict_proba(X, Y)

Input Data
----------
StepMix accepts the ``numpy.array`` and ``pandas.DataFrame`` data types. Additionally, emission models suffixed with
``_nan`` support missing values denoted by ``np.NaN``.

Advanced Usage
--------------
For all available options, please refer to the :doc:`api` documentation. [TODO : Add links to tutorial notebooks].
