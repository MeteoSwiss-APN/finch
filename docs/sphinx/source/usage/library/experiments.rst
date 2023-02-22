.. _experiments:

Running Experiments
===================

The :mod:`finch.experiments` module is the core module of finch.
Most prominently, it features the :func:`finch.measure_runtimes` function, with which we run experiments.
We can configure the experiment via a :class:`finch.RunConfig` object, which we pass to :func:`finch.measure_runtimes`.
The :class:`finch.RunConfig` class is an abstract class, which can be used to launch runtime experiments for any python function.
For functions conforming to the default operator signature of finch, a specialized class :class:`finch.OperatorRunConfig` is provided, which provides more support.
Users who want to measure runtimes for different function signatures must provide their own implementation of :class:`finch.RunConfig`.


Measuring Operator Runtimes
---------------------------

There are two required ingredients when creating a :class:`finch.OperatorRunConfig` instance: an operator conforming to the default operator signature and an input with a version, as explained in :ref:`input_management`.

The default operator signature is defined by :attr:`finch.DefaultOperator`. It is a function which takes a `xarray.Dataset` and returns a `xarray.DataArray`.
Let's use the built-in :func:`finch.brn.brn` operator for our examples.
::
    import finch
    import finch.brn

    run_config = finch.OperatorRunConfig(
        impl=finch.brn.brn,
        input_obj=finch.brn.get_brn_input(),
        input_version=finch.data.Input.Version(
            format=finch.data.Format.ZARR
        )
    )

    runtimes = finch.measure_runtimes(run_config)

The above script will measure the runtime of the BRN operator when using ZARR as an input format.
By default, the execution will be repeated five times and the average runtime will be reported. Additionally, a warmup iteration is added at the beginning, whose runtime will be discarded.
We can control this behavior by setting `iterations` and `warmup` of our `run_config` object.

Also by default, the output of the operator will be stored to zarr.
This might blow up the parallel runtime of the operator unexpectedly. It can be disabled by setting `store_output=False`.

Dask Configurations
^^^^^^^^^^^^^^^^^^^

The :class:`finch.OperatorRunConfig` inherits from :class:`finch.DaskRunConfig`, which let's us control dask-specific configurations.
For example, we might want to adjust the number of workers, with which dask runs our experiment.
We can do this by setting the `workers` attribute of the `run_config` object.

Additionally, we can configure some more fine-grained properties of the dask cluster by setting `cluster_config`.
The `cluster_config` attribute takes a :class:`finch.scheduler.ClusterConfig` object.
Take a look at the class definition to find out which properties can be set.


Configuration Classes
---------------------

Finch provides the class :class:`finch.util.Config`, from which specific configuration classes inherit.
This currently includes:

- :class:`finch.OperatorRunConfig` (and its subclasses :class:`finch.DaskRunConfig` and :class:`finch.RunConfig`)
- :class:`finch.data.Input.Version`
- :class:`finch.scheduler.ClusterConfig`

The :class:`finch.util.Config` class provides the class function :func:`finch.util.Config.list_configs`.
With this function we can easily create a list of configuration objects.
We can use it the same way we use the constructor of a configuration class, but we also have the ability to provide a list for an argument instead of a single one.
The resulting list of configuration objects will be the cross product between all the argument lists which were provided.

For example, we can produce a list of run configurations with different numbers of dask workers for all possible input formats as follows.
::
    run_config = finch.OperatorRunConfig(
        impl=finch.brn.brn,
        input_obj=finch.brn.get_brn_input(),
        input_version=finch.data.Input.Version(
            format=[f for f in finch.data.Format]
        ),
        workers=[5, 10, 15, 20]
    )

We can then use this list as an argument for :func:`finch.measure_runtimes` to run them all after another, allowing us to easily setup all kinds of experiments.::
    runtimes = finch.measure_runtimes(run_config)

.. info::
    Classes which inherit the :class:`finch.util.Config` class are expected to be keyword-only dataclasses.
    Hence, you can also use `dataclass-specific features <https://docs.python.org/3/library/dataclasses.html>`_ on them.


Runtime Objects
---------------

The :func:`finch.measure_runtimes` function returns a list of :class:`finch.Runtime` objects.
The :class:`finch.Runtime` class and its derivates are dataclasses containing only `float` attributes.
These attributes are specific runtime measurements which will be populated when running :func:`finch.measure_runtimes` (more concretely, when running :func:`finch.RunConfig.measure`).
The base :class:`finch.Runtime` class has the attributes `full`, `input_loading` and `compute`.
The `full` attribute is required and captures the runtime of the full experiment, including loading and storing the data.
The `input_loading` attribute captures the runtime for loading the input while `compute` captures the runtime of the actual execution time of the operator.

.. note::
    When using dask, `input_loading` will only include the time used for setting up the input.
    Because of dask's lazy loading, the effective load time of the input will be captured in `compute`.

The :class:`finch.DaskRunConfig` and therefore also the :class:`finch.OperatorRunConfig` are implemented to return a :class:`finch.DaskRuntime` object, which contains more fine-grained dask-specific runtime measurements.
Take a look at the class specification to find out what is included.
