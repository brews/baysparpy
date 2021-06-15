baysparpy3
==========

.. image:: https://travis-ci.org/brews/baysparpy.svg?branch=master
    :target: https://travis-ci.org/brews/baysparpy


An Open Source Python package for TEX86 calibration.

This package is based on the original BAYSPAR (BAYesian SPAtially-varying Regression) for MATLAB (https://github.com/jesstierney/BAYSPAR).

This package is the updated version of baysparpy with new module(s) such as `predict_tex_analog`
based on the original python version of BAYSPAR (https://github.com/brews/baysparpy).

Quick example
-------------

Standard prediction
~~~~~~~~~~~~~~~~~~~


First, load key packages and an example dataset:

.. code-block:: python

    import numpy as np
    import bayspar as bsr

    example_file = bsr.get_example_data('castaneda2010.csv')
    d = np.genfromtxt(example_file, delimiter=',', names=True)

This dataset (from `Castañeda et al. 2010 <https://doi.org/10.1029/2009PA001740>`_)
has two columns giving sediment age (calendar years BP) and TEX86.

We can make a "standard" prediction of sea-surface temperature (SST) with ``predict_seatemp()``:

.. code-block:: python

    prediction = bsr.predict_seatemp(d['tex86'], lon=34.0733, lat=31.6517,
                                     prior_std=6, temptype='sst')

To see actual numbers from the prediction, directly parse ``prediction.ensemble`` or use ``prediction.percentile()`` to get the 5%, 50% and 95% percentiles.

You can also plot your prediction with ``bsr.predictplot()`` or ``bsr.densityplot()``.


Analog prediction
~~~~~~~~~~~~~~~~~


Begin by loading data for a “Deep-Time” analog prediction:

This dataset is a TEX86 record from record from Wilson Lake, New Jersey (`Zachos et al. 2006 <https://doi.org/10.1130/G22522.1>`_). The file has two columns giving depth (m) and TEX86:


.. code-block:: python

    example_file = bsr.get_example_data('wilsonlake.csv')
    d = np.genfromtxt(example_file, delimiter=',', names=True)
    

We can run the analog prediction of SST with ``predict_seatemp_analog()``.
We can also examine our prediction as though it were a standard prediction. For example, we can plot a time series of the prediction:

.. code-block:: python

    search_tolerance = np.std(d['tex86'], ddof=1) * 2
    prediction = bsr.predict_seatemp_analog(d['tex86'], temptype='sst',prior_mean=30, prior_std=20,search_tol=search_tolerance,nens=500)
    ax = bsr.predictplot(prediction, x=d['depth'], xlabel='Depth (m)', ylabel='SST (°C)')
    ax.grid()
    ax.legend()


Forward prediction
~~~~~~~~~~~~~~~~~~


For this example, we make inferences about TEX86 from SST data using a forward-model prediction. We start by creating a SST series spanning from 0 - 40 °C. 

And now plug the SST data into ``predict_tex()`` along with additional information. In this case we’re using the same site location as in Standard prediction:

.. code-block:: python

    sst = np.arange(0, 41)
    prediction = bsr.predict_tex(sst, lon=34.0733, lat=31.6517, temptype='sst')

As might be expected, we can use the output of the forward prediction to parse and plot:

.. code-block:: python

    ax = bsr.predictplot(prediction, x=sst,xlabel='SST (°C)',ylabel=r'TEX$_{86}$')
    ax.grid()
    ax.legend()
    

Analog forward prediction
~~~~~~~~~~~~~~~~~~~~~~~~~


This tool will calculate forwarded TEX using given SST data. Here is an example:

.. code-block:: python

    sst = np.arange(0, 41)
    prediction = bsr.predict_tex_analog(sst, temptype = 'sst', search_tol = 5., nens=8000)
    ax = bsr.predictplot(prediction, x=sst,xlabel='SST (°C)',ylabel=r'TEX$_{86}$')
    ax.grid()
    ax.legend()


First, we make inferences about deep-time TEX86 from SST data using a forward-model analog prediction. We start by creating a SST series spanning from 0 - 40 °C.

And then plug the SST data into ``predict_tex_analog()`` along with additional information (search tolerance is 5 °C).

We can use the output of the forward prediction to parse and plot.

Read More
~~~~~~~~~


For further details, examples, and additional prediction functions, see the online documentation (https://baysparpy.readthedocs.io).


Installation
------------

To install **baysparpy** with pip, run:

.. code-block:: bash

    $ pip install baysparpy3

To install with conda, run:

.. code-block:: bash

    $ conda install baysparpy -c sbmalev

Unfortunately, **baysparpy** is not compatible with Python 2.

Support and development
-----------------------

- Documentation is available online (https://baysparpy.readthedocs.io).

- Please feel free to report bugs and issues or view the source code on GitHub (https://github.com/mingsongli/baysparpy).


License
-------

**baysparpy** is available under the Open Source GPLv3 (https://www.gnu.org/licenses).

