# finch [distributed data processing for ICON data]

![alt text](https://github.com/MeteoSwiss-APN/finch/blob/main/images/201903_Zebra_finch_lo.png?raw=true)

Image distributed through Creative Commons License from https://commons.wikimedia.org/wiki/File:201903_Zebra_finch.svg

# Overview

Finch provides parallelized prototypes for ICON data processing operators for a distributed infrastructure with SLURM.
It is deisgned as an experimental library plus a run script, which runs configurable experiments.

# Installation

## setuptools

Requires at least CMake 3.14 as well as a C++-Compiler with C++-20 (concepts) support.

```
pip install -e .
```

## conda
```
conda build -c conda-forge conda
conda install -c ${CONDA_PREFIX}/conda-bld/  -c conda-forge finch_mch
```

# Run experiments
```
python run.py
```

# Run tests
```
pytest
```

# Usage

## Start a distributed scheduler on the cluster

Dask provides a dashboard for monitoring task activities.
This is a very powerful tool to get insights into how well parallelized the code actually is and where there is still potential for optimization.

The dashboard is attached to a running scheduler session.
Therefore we provide a script which runs a scheduler on a compute node, to which finch will automatically connect.
The script can be started as follows.
```
bash start_scheduler.sh
```
This opens up an interactive python session, which can be used to scale the cluster.

# Configuration

Finch uses the Python's configparser library for configuration with `.ini` files and extended interpolation.
Variables can be cross-referenced with `${section:variable}` and environment variables are available via `${VAR}`.
Finch provides a default config file `config/finch.ini`.
Custom configuration, which overrides the defaults, can be provided in `config/custom.ini` as well as in a location specified by the  environment variable `CONFIG`.

## Debug mode

To run Finch in debug mode, the `debug_mode` option in the `global` section of the configuration can be set to "true" or "false".
By default, the debug option is retrieved from the environment variable `DEBUG`.

In debug mode, Finch will run a synchronous dask scheduler and will put the log level to debug (instead of info).

## Dask configuration

For setting up the dask clusters, Finch uses the dask provided configuration to find out about the reources provided by the HPC infrastracture.
For jobqueue.slurm configurations, it interprets the configured resources (cores and memory) as resources available per node on the cluster.
The job specific resources will be overwritten.
See [here](https://docs.dask.org/en/stable/configuration.html) and [here](https://jobqueue.dask.org/en/latest/configuration-setup.html) for information on how to configure dask.
