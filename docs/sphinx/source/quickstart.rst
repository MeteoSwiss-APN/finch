Getting Started
===============

Clone the repository
::
    git clone https://github.com/MeteoSwiss-APN/finch.git

Install and activate the conda environment

::
    conda env create -f environment.yml
    conda activate finch

Install the ``zebra`` module

::
    cd zebra
    python setup.py install

Run the experiments
::
    bash srun.sh

Run the tests
::
    srun pytest
