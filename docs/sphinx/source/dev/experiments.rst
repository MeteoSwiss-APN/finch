Experiments
===================

Introduction
------------

The experiment setup, mainly the function :func:`finch.measure_runtimes` can be considered the heart of finch.
This function does not run any implementation itself, neither does it load the input.
However, it puts all those individual modules together into one experiment.
As a result, finch uses a modular approach for setting up experiments.
This allows a high level of configuration for all sorts of experiments.
While finch defines a default workflow which can be used out of the box, it is also possible to overwrite the individual components.
Finally, they are being passed down to :func:`finch.measure_runtimes`, which glues them together.

Following is a list of the most important components with a brief description for each of them.

:RunConfig:
    The run configuration contains everything for setting up the environment of the experiment.
    The implementation of the operator is also specified here.
:Inputs:
    Defines which inputs are being passed to the operator to run and how the input is loaded.
    The standard way of providing inputs is via finch's built-in input management. However, this can be overwritten.
:Implementation Runner:
    The implementation runner runs the operator implementation.
    It can introduce more fine-grained runtime measurements.
    The simplest implementation runner just runs the provided implementation on the provided arguments without doing anything else.
    The :func:`finch.measure_runtimes` will measure the runtime of the input preparation and the full execution of the implementation runner on its own.


Measuring Operator Runtimes
----------------------------------

For standard finch operators, you do not have to call :func:`finch.measure_runtimes` explicitly.
Instead, finch provides a function :func:`finch.measure_operator_runtimes`.
This function will take a run configuration along with an input and some input version from finch's input management.
::
    from finch import RunConfig, measure_operator_runtimes
    from finch.data import Input, Format
    from finch.scheduler import ClusterConfig
    import finch.brn

    brn_input = finch.brn.get_brn_input()
    version = Input.Version(format=Format.GRIB)
    brn_config = RunConfig(impl=finch.brn.impl.brn_xr)

    time = measure_operator_runtimes(brn_config, brn_input, version)

If we want to create a series of runtime measurements, we can pass multiple run configurations and multiple input versions.
The output will be a list of lists of runtimes, where the outer list is for the different run configurations and the inner list for the different input versions.
It is also possible to pass a single run configuration and multiple input versions and vice-versa.
The output format will then be a single list of runtimes, as expected.
We can use :func:`finch.RunConfig.list_configs` as well as :func:`finch.Input.Version.list_configs` to easily create multiple instances.
::
    versions = Input.Version.list_configs(
        format=[Format.NETCDF, Format.ZARR],
        dim_order="xyz"
    )

    brn_configs = RunConfig.list_configs(
        impl=finch.brn.list_brn_implementations(),
        cluster_config=finch.scheduler.ClusterConfig.list_configs(
            cores_per_worker=[1,5]
        ),
        workers=[1, 10, 20]
    )

    times = measure_operator_runtimes(brn_configs, brn_input, versions)


Configuration
-------------

The configuration of the environment is specified with the :class:`finch.RunConfig` class.

.. TODO Write this section of the documentation after generalizing the run config and providing a dask specific configuration
