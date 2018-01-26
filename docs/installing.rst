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

pip
~~~

To install **baysparpy** with pip, run::

    $ pip install git+git://github.com/brews/baysparpy.git@stable

and follow the on-screen prompts.

Testing
-------

To run the testing suite after installation, first install `py.test <https://docs.pytest.org/en/latest/>`_. Run::

    $ pytest --pyargs bayspar

