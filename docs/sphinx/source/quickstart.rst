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

.. TODO: Maybe we can switch to `pip install -e .` at some point.
.. Curently, this does not properly install zebra however.

.. code-block:: text

    conda-develop src

Finally, build and install zebra

.. code-block:: text

    cmake -S src/zebra -B src/zebra/build -DINSTALL_GTEST=OFF
    cmake --build src/zebra/build --target zebra --config Release
    cmake --install src/zebra/build --prefix "$(pwd)/src/zebra"
    conda-develop src/zebra

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