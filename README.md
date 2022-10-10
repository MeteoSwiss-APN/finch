# finch [distributed data processing for ICON data]

![alt text](https://github.com/MeteoSwiss-APN/finch/blob/main/images/201903_Zebra_finch_lo.png?raw=true)

Image distributed through Creative Commons License from https://commons.wikimedia.org/wiki/File:201903_Zebra_finch.svg

# Overview

Finch provides parallelized prototypes for ICON data processing operators for a distributed infrastructure.
It is deisgned as an experimental library plus a run script, which runs configurable experiments.

# Setup

## Conda

An `environments.yml` is provided to setup a conda environment with all required dependencies.
```
conda env create -f environment.yml
conda activate finch
```

## Run experiments
```
python run.py
```

## Run tests
```
pytest
```

# Configuration

Finch uses the Python's configparser library for configuration with `.ini` files and extended interpolation.
Variables can be cross-referenced with `${section:variable}` and environment variables are available via `${VAR}`.
Finch provides a default config file `config/finch.ini`.
Custom configuration, which overrides the defaults, can be provided in `config/custom.ini` as well as in a location specified by the  environment variable `CONFIG`.
