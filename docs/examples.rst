.. _examples:

Examples
========

.. warning::

   The project is under heavy development. Code and documentation are not complete.

Here are some quick examples of what you can do with **baysparpy**.

To start things off, we import a few basic tools, including `bayspar`:

.. ipython:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import bayspar as bsr

Standard prediction
-------------------

Begin by reading in an example dataset...

.. ipython:: python

    example_file = bsr.get_example_data('shevenell2011.csv')
    d = np.genfromtxt(example_file, delimiter=',', names=True)

Blah blah.

.. ipython:: python

    prediction = bsr.predict_seatemp(d['tex86'], lon=-64.2080, lat=-64.8527,
                                     prior_std=6, temptype='sst', nens=200)

Blah blah.

.. ipython:: python

    @savefig predictplot_shevenell_rough.png width=4in
    bsr.predictplot(prediction)

Blah blah blah BLAH.

.. ipython:: python

    ax = bsr.predictplot(prediction, x=d['age'], xlabel='Age',
                              ylabel='SST (°C)')
    ax.grid()
    @savefig predictplot_shevenell_pretty.png width=4in
    ax.legend()


Analog prediction
-----------------

Blah blah.

Forward prediction
------------------

Blah blah.
