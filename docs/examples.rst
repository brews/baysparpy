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

    prediction = bsr.predict_seatemp(d['tex86'], lon=34.0733, lat=31.6517,
                                     prior_std=6, temptype='subt', nens=200)

Blah blah.

.. ipython:: python

    @savefig predict_plot.png width=4in
    ax = bsr.plot.plot_series(prediction, x=d['age'], xlabel='Age',
                              ylabel='Temperature (°C)')
    ax.grid(True)


Analog prediction
-----------------

Blah blah.

"Forward" prediction
--------------------

Blah blah.
