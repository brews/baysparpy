baysparpy
=========

.. image:: https://travis-ci.org/brews/baysparpy.svg?branch=master
    :target: https://travis-ci.org/brews/baysparpy


An Open Source Python package for TEX86 calibration.

This package is based on the original BAYSPAR (BAYesian SPAtially-varying Regression) for MATLAB (https://github.com/jesstierney/BAYSPAR).


Quick example
-------------

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

For further details, examples, and additional prediction functions, see the online documentation (https://baysparpy.readthedocs.io).


Installation
------------

To install **baysparpy** with pip, run:

.. code-block:: bash

    $ pip install git+git://github.com/brews/baysparpy.git@stable

Unfortunately, **baysparpy** is not compatible with Python 2.

Support and development
-----------------------

- Documentation is available online (https://baysparpy.readthedocs.io).

- Please feel free to report bugs and issues or view the source code on GitHub (https://github.com/brews/baysparpy).


License
-------

**baysparpy** is available under the Open Source GPLv3 (https://www.gnu.org/licenses).

