.. _installing:

############
Installation
############


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

conda
~~~~~

To install **baysparpy** with ``conda``, run::

    $ conda install baysparpy -c sbmalev

This is usually the easiest way to install the package.

pip
~~~

To install with ``pip``, run::

    $ pip install baysparpy

and follow the on-screen prompts. We do not recommend installing with ``pip`` unless you are brave, or already have a copy of `cartopy <http://scitools.org.uk/cartopy/>`_ installed.


Testing
-------

To run the testing suite after installation, first install `py.test <https://docs.pytest.org/en/latest/>`_. Run::

    $ pytest --pyargs bayspar

