Tutorials
=========
Quickstart
----------
The following shows a simple StepMix mixture using the continuous variables of the Iris Dataset. ``n_components`` controls
the number of latent classes.::

    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.metrics import rand_score

    from stepmix.stepmix import StepMix

    # Load dataset in a Dataframe
    data_continuous, target = load_iris(return_X_y=True, as_frame=True)

    # Continuous StepMix Model with 3 latent classes
    model = StepMix(n_components=3, measurement="continuous", verbose=1, random_state=123)

    # Fit model and predict clusters
    model.fit(data_continuous)
    pred_continuous = model.predict(data_continuous)

    # A Rand score close to 1 indicates good alignment between clusters and flower types
    print(rand_score(pred_continuous, target))

The API allows to easily predict class memberships or probabilities::

    class_ids = model.predict(X, Y)
    class_probs = model.predict_proba(X, Y)

StepMix also provides support for categorical mixtures::

    # Create categorical data based on the Iris Dataset quantiles
    data_categorical = data_continuous.copy()
    for col in data_categorical:
       data_categorical[col] = pd.qcut(data_continuous[col], q=3).cat.codes

    # Categorical StepMix Model with 3 latent classes
    model = StepMix(n_components=3, measurement="categorical", verbose=0, random_state=123)

    # Fit model and predict clusters
    model.fit(data_categorical)
    pred_categorical = model.predict(data_categorical)

    # A Rand score close to 1 indicates good alignment between clusters and flower types
    print(rand_score(pred_categorical, target))

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
