.. _examples:

Examples
========

.. warning::

   The project is under heavy development. Code and documentation are not complete.

Here are some quick examples of what you can do with **baysparpy**.

To start things off, we import a few basic tools, including `bayspar`:

.. ipython:: python

    import numpy as np
    import bayspar as bsr
    import matplotlib.pyplot as plt

Standard prediction
-------------------

.. ipython:: python

    example_file = bsr.get_example_data('shevenell2011.csv')
    d = np.genfromtxt(example_file, delimiter=',', names=True)

Analog prediction
-----------------

