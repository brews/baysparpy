.. _examples:
.. currentmodule:: bayspar

########
Examples
########

To start things off, import a few basic
tools, including `bayspar`:

.. ipython:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import bayspar as bsr

There are a few options when it comes to prediction with :py:mod:`bayspar`.
Below, we use example data - included with the package - to walk through each
type of prediction.

Standard prediction
-------------------

We can access the example data with :py:func:`get_example_data` and use the
returned stream with :py:func:`pandas.read_csv` or :py:func:`numpy.genfromtxt`:

.. ipython:: python

    example_file = bsr.get_example_data('castaneda2010.csv')
    d = np.genfromtxt(example_file, delimiter=',', names=True)

This dataset (from `Castañeda et al. 2010 <https://doi.org/10.1029/2009PA001740>`_)
has two columns giving sediment age (calendar years BP) and TEX\ :sub:`86`.

.. ipython:: python

    d['age'][:5]
    d['tex86'][:5]

We can make a "standard" prediction of sea-surface temperature (SST) with :py:func:`predict_seatemp`:

.. ipython:: python

    prediction = bsr.predict_seatemp(d['tex86'], lon=34.0733, lat=31.6517,
                                     prior_std=6, temptype='sst')

The TEX\ :sub:`86` data and site position are passed to the prediction
function. We also give standard deviation for the prior
SST distribution and specify that SST are
the target variable. A :py:class:`~predict.Prediction` instance is returned,
which we can poked and prod with direct queries or using a few built-in
functions:

.. ipython:: python

    @savefig predictplot_castaneda_rough.png width=4in
    bsr.predictplot(prediction)

You might have noticed that :py:func:`predictplot` returns an
:py:class:`matplotlib.Axes` instance. This means we can catch it and then
modify the plot as needed:

.. ipython:: python

    ax = bsr.predictplot(prediction, x=d['age'], xlabel='Age',
                         ylabel='SST (°C)')
    ax.grid()
    @savefig predictplot_castaneda_pretty.png width=4in
    ax.legend()

In much the same way, we can view the distribution of the prediction prior and posterior with :py:func:`densityplot`:

.. ipython:: python

    ax = bsr.densityplot(prediction,
                         x=np.arange(1, 40, 0.1),
                         xlabel='SST (°C)')
    ax.grid(axis='x')
    @savefig densityplot_castaneda_pretty.png width=4in
    ax.legend()

We can access the full prediction ensemble with:

.. ipython:: python

    prediction.ensemble

A useful summary is available using:

.. ipython:: python

    prediction.percentile()[:10]  # Showing only the first few values.

See the :py:meth:`~predict.Prediction.percentile` method for more options.

Analog prediction
-----------------

Begin by loading data for a "Deep-Time" analog prediction:

.. ipython:: python

    example_file = bsr.get_example_data('wilsonlake.csv')
    d = np.genfromtxt(example_file, delimiter=',', names=True)

This dataset is a TEX\ :sub:`86` record from record from Wilson Lake, New Jersey
(`Zachos et al. 2006 <https://doi.org/10.1130/G22522.1>`_). The file has two
columns giving depth (m) and TEX\ :sub:`86`:

.. ipython:: python

    d['depth'][:5]
    d['tex86'][:5]

We can run the analog prediction of SST with :py:func:`predict_seatemp_analog`:

.. ipython:: python

    search_tolerance = np.std(d['tex86'], ddof=1) * 2

    prediction = bsr.predict_seatemp_analog(d['tex86'], temptype='sst',
                                            prior_mean=30, prior_std=20,
                                            search_tol=search_tolerance,
                                            nens=500)

A :py:class:`~predict.Prediction` instance is returned from the function. The arguments that we pass to the function indicate that the calibration is for SST. We also indicate a prior mean and standard deviation for the sea temperature inference. Note that we pass a search tolerance used to find analogous conditions across the global TEX\ :sub:`86` dataset. The `nens` argument is to reduce the size of the model parameter ensemble used for the inference - we're using this option because otherwise this page of documentation would take far to log to compile, so it isn't required if you're following along with these examples on your own machine. By default, a progress bar is printed to the screen. This is an optional feature that can be switched off, for example, if you are processing many cores in batch.

.. Note::

   Analog predictions are slow if many analogs are selected.



Specifically, for analog predictions, we can map the information used for the inference with :py:func:`analogmap`:

.. ipython:: python

    paleo_location = (38.2, -56.7)
    ax = bsr.analogmap(prediction, latlon=paleo_location)
    @savefig analogmap_wilson_rough.png width=4in
    ax.legend()

This maps the location of all available TEX\ :sub:`86` observations. We passed an optional paleo-location of the site to the mapping function. The map also indicates the grids that were identified as analogs for the prediction.

We can also examine our prediction as though it were a standard prediction. For example, we can plot a time series of the prediction:

.. ipython:: python

    @savefig predictplot_wilson_rough.png width=4in
    bsr.predictplot(prediction, x=d['depth'], xlabel='Depth (m)',
                    ylabel='SST (°C)')


Forward prediction
------------------

For this example, we make inferences about TEX\ :sub:`86` from SST data using a
forward-model prediction. We start by creating a SST series spanning from 0 - 40 °C:

.. ipython:: python

    sst = np.arange(0, 41)

And now plug the SST data into :py:func:`predict_tex` along with additional information.
In this case we're using the same site location as in `Standard prediction`_:

.. ipython:: python

    prediction = bsr.predict_tex(sst, lon=34.0733, lat=31.6517, temptype='sst')

As might be expected, we can use the output of the forward prediction to parse and plot:

.. ipython:: python

    ax = bsr.predictplot(prediction, x=sst,
                         xlabel='SST (°C)',
                         ylabel=r'TEX$_{86}$')
    ax.grid()
    @savefig predictplot_forward_pretty.png width=4in
    ax.legend()

