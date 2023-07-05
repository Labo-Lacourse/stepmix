Tutorials
=========
Quickstart
----------
A StepMix mixture using categorical variables on a preloaded data matrix. StepMix accepts either ``numpy.array`` or ``pandas.DataFrame``. Categories should be integer-encoded and 0-indexed. ::

    from stepmix.stepmix import StepMix

    # Categorical StepMix Model with 3 latent classes
    model = StepMix(n_components=3, measurement="categorical")
    model.fit(data)

    # Allow missing values
    model_nan = StepMix(n_components=3, measurement="categorical_nan")
    model_nan.fit(data_nan)

For binary data you can also use ``measurement="binary"`` or ``measurement="binary_nan"``. For continuous data, you can fit a Gaussian Mixture with diagonal covariances using ``measurement="continuous"`` or ``measurement="continuous_nan"``.

Set ``verbose=1`` for a detailed output.

Please refer to the StepMix tutorials to learn how to combine continuous and categorical data in the same model.

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
