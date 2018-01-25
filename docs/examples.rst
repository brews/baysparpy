.. _examples:
.. currentmodule:: bayspar

########
Examples
########

.. warning::

   The project is under heavy development. Code and documentation are not complete.

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

We can make a "standard" prediction of sea temperature with :py:func:`predict_seatemp`:

.. ipython:: python

    prediction = bsr.predict_seatemp(d['tex86'], lon=34.0733, lat=31.6517,
                                     prior_std=6, temptype='sst')

The TEX\ :sub:`86` data and site position are passed to the prediction
function. Additionally, we give standard deviation for the prior
sea-temperature distribution. We also specify that sea-surface temperatures are
the target variable. A :py:class:`~predict.Prediction` instance is returned,
which we can poked and prod with direct queries or with a few built-in
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

In much the same way, we can view the distribution of the prediction prior and posterior with :py:func:`densityplot':

.. ipython:: python

    ax = bsr.densityplot(prediction,
                         x=np.arange(1, 40, 0.1),
                         xlabel='SST (°C)')
    ax.grid(axis='x')
    @savefig densityplot_castaneda_pretty.png width=4in
    ax.legend()


Analog prediction
-----------------

Begin by loading the example data for a "Deep-Time" analog prediction:

.. ipython:: python

    example_file = bsr.get_example_data('wilsonlake.csv')
    d = np.genfromtxt(example_file, delimiter=',', names=True)

This dataset is a TEX86 record from record from Wilson Lake, New Jersey
(`Zachos et al. 2006 <https://doi.org/10.1130/G22522.1>`_). The file has two
columns giving depth (m) and TEX\ :sub:`86`:

.. ipython:: python

    d['depth'][:5]
    d['tex86'][:5]

We can run an "analog" prediction of sea temperature with :py:func:`predict_seatemp`:

.. ipython:: python

    search_tolerance = np.std(d['tex86'], ddof=1) * 2

    prediction = bsr.predict_seatemp_analog(d['tex86'], temptype='sst',
                                            prior_mean=30, prior_std=20,
                                            search_tol=search_tolerance,
                                            nens=500)

Blah blah, what is an analog prediction, talk about the above function run. We
get a prediction object out of it. Beware (note) that it can be slow if many
analogs are selected.

We can plot a time series of the prediction:

.. ipython:: python

    @savefig predictplot_wilson_rough.png width=4in
    bsr.predictplot(prediction, x=d['depth'], xlabel='Depth (m)',
                    ylabel='SST (°C)')

We can also map the information used for the analog prediction:

.. ipython:: python

    paleo_location = (38.2, -56.7)
    ax = bsr.analogmap(prediction, latlon=paleo_location)
    @savefig analogmap_wilson_rough.png width=4in
    ax.legend()

Blah blah blah blah.

Forward prediction
------------------

Blah blah.
