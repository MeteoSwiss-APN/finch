Introduction
============

The core of finch builds a library for conveniently creating an experiment pipeline for evaluating parallelized data processing operator prototypes.
Finch can be used to better understand, how an operator prototype scales and performs with different infrastructure configurations.

In addition to the core library, finch also provides a few built-in operators.
These operators can be evaluated using a run script and a run configuration.
Current findings on how to effectively parallelize data processing operators in the MeteoSwiss environment, which are documented in the tex documentation, are based on experiments executed with this run script.
Hence, the reported experiment results can be reproduced with a proper run configuration.
