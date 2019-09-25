from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='baysparpy',
    version='0.0.4',
    description='An Open Source Python package for TEX86 calibration',
    long_description=readme(),
    license='GPLv3',

    author='S. Brewster Malevich, Mingsong Li',
    author_email='malevich@email.arizona.edu, mul450@psu.edu',
    url='https://github.com/mingsongli/baysparpy',
    # rl='https://github.com/brews/baysparpy',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    keywords='marine paleoclimate tex86 calibration',

    packages=find_packages(exclude=['docs']),

    install_requires=['numpy', 'scipy', 'matplotlib', 'attrs', 'tqdm', 'cartopy'],
    tests_require=['pytest'],
    package_data={'bayspar': ['modelparams/Output_SpatAg_subT/*.mat',
                              'modelparams/Output_SpatAg_SST/*.mat',
                              'observations/*.mat',
                              'example_data/*.csv']},
)
