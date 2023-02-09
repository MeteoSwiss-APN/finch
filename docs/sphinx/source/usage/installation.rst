Installation
============

You can either install finch with conda (recommended) or with pip.
Neither the conda, nor the pip package are distributed yet, however, and you need to build them yourself.
So, first clone the repository

.. code-block:: text

    git clone https://github.com/MeteoSwiss-APN/finch.git

conda / mamba
^^^^^^^^^^^^^

A conda recipe is provided which can be built and installed using ``conda-build``.
A faster alternative to conda is ``mamba``, which can be used together with `boa <https://github.com/mamba-org/boa>`_ for building the recipe.

.. code-block:: text

    conda install mamba -c conda-forge
    mamba install conda-build boa -c conda-forge
    conda mambabuild -c conda-forge conda
    mamba install -c local -c conda-forge finch_mch

If you prefer to use ``conda`` instead of ``mamba``, run the following commands.

.. code-block:: text

    conda install conda-build -c conda-forge
    conda build -c conda-forge conda
    conda install -c local -c conda-forge finch_mch

A few dependecies are only provided via PyPI. You can install them manually with ``pip``.

.. code-block:: text

    pip install wonderwords


pip
^^^

For the pip installation, you need to make sure that you have a proper C++ setup installed.
At least CMake 3.14 as well as a C++-Compiler with C++-20 (concepts) support, such as GCC >= 10.2 is required.

You can also just use the conda development environment, as documented in :ref:`dev-setup`, which includes CMake and a proper C++ compiler.

.. code-block:: text

    pip install .
