.. _installing:

############
Installation
############

.. warning::

   The project is under heavy development. Code and documentation are not complete.

Requirements
------------

- Python >= 3.4
- `numpy <http://www.numpy.org/>`_
- `scipy <https://www.scipy.org/>`_
- `matplotlib <https://matplotlib.org/>`_
- `cartopy <http://scitools.org.uk/cartopy/>`_
- `tqdm <https://pypi.python.org/pypi/tqdm>`_
- `attrs <http://www.attrs.org>`_


Instructions
------------

pip
~~~

To install **baysparpy** with pip, run::

    $ pip install baysparpy

and follow the on-screen prompts.

conda
~~~~~

If you'd rather use Anaconda/miniconda, run::

    $ conda install baysparpy -c sbmalev

and follow prompts.


Testing
-------

To run the testing suite after installation, first install `py.test <https://docs.pytest.org/en/latest/>`_. Run::

    $ pytest --pyargs bayspar

