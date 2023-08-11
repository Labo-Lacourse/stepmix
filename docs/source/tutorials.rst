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
Detailed tutorial notebooks are available in the `README <https://github.com/Labo-Lacourse/stepmix>`_.
