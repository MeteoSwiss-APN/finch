.. _dev-setup:

Setup
=====

Development Environment
-----------------------

Finch provides a conda environment, which can be used for development.
It contains all the dependencies of finch and other practical tools for development.
You can install and activate the enviroment as follows.

.. code-block:: text

    conda env create -f environment.yml
    conda activate finch

Installing finch
----------------

For development, you probably don't want to re-package and install finch after every change you made.
An alternative is to install finch in development mode.

.. code-block:: text

    conda-develop src

The above command does not install the zebra extension module.
Since zebra is written in C++, you cannot install zebra in development mode, as you can with a pure Python library.
Instead, you have to build zebra manually.

.. code-block:: text

    cmake -S src/zebra -B src/zebra/build -DINSTALL_GTEST=OFF
    cmake --build src/zebra/build --target zebra --config Release
    cmake --install src/zebra/build --prefix "$(pwd)/src/zebra"
    conda-develop src/zebra
