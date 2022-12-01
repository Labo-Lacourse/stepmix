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
   X, Y, _ = data_bakk_response(n_samples=1000, sep_level=.9, random_state=42)
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
For all available options, please refer to the :doc:`api` documentation.
Detailed tutorials are available in notebooks :

#. `Latent Class Analysis with StepMix <https://colab.research.google.com/drive/1KAxcvxjL_vB2lAG9e47we7hrf_2fR1eK?usp=sharing>`_ : an in-depth look at how latent class models can be defined with StepMix. The tutorial uses the Iris Dataset as an example and covers

    #. Continuous LCA models;
    #. Binary LCA models;
    #. Categorical LCA models;
    #. Mixed LCA models (continuous and categorical data);
    #. Missing Values.

#. `Stepwise Estimation with StepMix <https://colab.research.google.com/drive/1T_UObkN5Y-iFTKiun0zOkKk7LjtMeV25?usp=sharing>`_ : a tutorial demonstrating how to define measurement and structural models. The tutorial discusses:

    #. LCA models with response variables;
    #. LCA models with covariates;
    #. 1-step, 2-step and 3-step estimation;
    #. Corrections and other options for 3-step estimation.

#. `Model Selection <https://colab.research.google.com/drive/1iyFTD-D2wn88_vd-qxXkovIuWHRtU7V8?usp=sharing>`_ :
   a short tutorial discussing:

    #. Selecting the number of latent classes (```n_components```);
    #. Comparing models with AIC and BIC.

#. `Parameters, Bootstrapping and CI <https://colab.research.google.com/drive/14Ir08HXQ3svydbVV4jlvi1HjGnfc4fc0?usp=sharing>`_ :
   a tutorial discussing how to:

   #. Access StepMix parameters;
   #. Bootstrap StepMix estimators;
   #. Quickly plot confidence intervals.
