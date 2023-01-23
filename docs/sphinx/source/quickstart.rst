Getting Started
===============

First, clone the repository
::
    git clone https://github.com/MeteoSwiss-APN/finch.git

Development setup
-----------------

If you want to start working on finch itself, you probably don't want to install the package in the conventional way.
Instead, it is advised to install finch in development mode.

A conda environment is provided, containing all the build and run dependencies of finch.
You can install and activate it as follows:

.. code-block:: text

    conda env create -f environment.yml
    conda activate finch

Next, you can install finch in development mode with conda develop

.. code-block:: text

    conda-develop finch

Zebra
^^^^^

The zebra package provides C++ implementations of operators in finch.
Zebra provides a CMake setup, which can be used for development as well as installation.

.. TODO

Running experiments
^^^^^^^^^^^^^^^^^^^

Finch provides a script for running experiments, located at ``scripts/finch``.
You can run it with

::
    
    python scripts/finch

Configuration options are documented in :ref:`Run and Debug Configuration`.

Running tests
^^^^^^^^^^^^^

Finch uses `pytest` for testing.

::

    srun pytest

Installation
------------

You can either install finch with conda (recommended) or with pip.


conda / mamba
^^^^^^^^^^^^^

A conda recipe is provided which can be built and installed using ``conda-build``.
A faster alternative to conda is ``mamba``, which can be used together with `boa <https://github.com/mamba-org/boa>`_ for building the recipe.
Make sure your environment uses Python 3.10.

.. code-block:: text

    conda install mamba -c conda-forge
    mamba install conda-build boa -c conda-forge
    conda mambabuild -c conda-forge conda
    mamba install -c local -c conda-forge finch_mch

If you prefer to use ``conda`` instead of ``mamba``, you can skip the first line from above, replace the ``mamba`` commands with the same ``conda`` commands and use ``conda build`` instead of ``conda mambabuild``.

A few dependecies are only provided via PyPI. You can install them manually with ``pip``.

.. code-block:: text

    pip install wonderwords


setuptools (pip)
^^^^^^^^^^^^^^^^

Requires at least CMake 3.14 as well as a C++-Compiler with C++-20 (concepts) support.

.. code-block:: text

    pip install scikit-build "pybind11[global]"
    python setup.py install
