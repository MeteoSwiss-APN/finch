Using a Custom Operator
=======================

Finch provides some built-in operators, which can be used for running experiments, in order to get a general understanding on how we can use finch for evaluating parallel operators and finding common bottlenecks when parallelizing with dask.
However, a basic design principle of finch is to provide an interface with which we can plug in custom operators and evaluate them using the tooling available in finch.

The following steps are required to create a custom operator for finch.

- Write an implementation (or multiple) of the operator.
- Define one or multiple input objects. Each of which requires the following steps.
    - Define an input source
    - Define the source version
    - Define a name for the input
- Integrate it into your experiment script

If you want to add your operator to the built-in operators of finch, make sure to read :ref:`new_builtin_ops`.

Implementing the Operator
-------------------------

Finch defines a default function signature for operators: :py:attr:`finch.DefaultOperator`
An operator conforming to the default signature takes a single ``xarray.Dataset`` and returns a ``xarray.DataArray``.
Additional optional arguments after the dataset are allowed.

.. note::
    In principle, it is also possible to write operators which follow a custom signature.
    Finch is designed in a modular fashion and the core functionalities are generally independent of the operator signature.
    However, these core functions require operator specific arguments, which are already implemented in finch for the default operator signature.
    If you require a different signature, you will need to implement these arguments yourself.

As a running example, let's implement an averaging operator. ::

    def avg_xr(data: xr.Dataset, array="T", dim="x") -> xr.DataArray:
        return data[array].mean(dim)

This operator conforms to the default signature: The first (and only) required argument is a dataset, and it returns a data array.

Let's write a second implementation of the operator. ::

    def avg_split(data: xr.Dataset, array="T", dim="x") -> xr.DataArray:
        dim_len = data[array].sizes[dim]
        part1 = data[array].isel({dim: slice(0, dim_len//2)})
        part2 = data[array].isel({dim: slice(dim_len//2, None)})
        return (part1.mean(dim) + part2.mean(dim)) / 2

Note that the two operator implementations have the exact same signature.
This is a requirement when implementing multiple operator versions.
Additionally, it is a good custom to stick to a specific naming pattern when naming the operators.
In this example, the naming pattern is ``"avg_.*"``.


Defining Inputs
---------------

.. attention:: Finch's input management relies on the default operator signature. If you use a custom input format for your operator, you need to implement your own input management system.

Finch provides its own input management.
This allows a user to easily setup new input versions.
For example, a user might want to see how his operator performs on both GRIB and NetCDF input files and compare the performance of his operator when using the two different file formats.
Input management is documented in detail in :ref:`input-management`.

In finch, an input is an object, which contains data for an experiment run.
An input can have different versions, which describe how the data is provided.
An operator can use different inputs for different experiments.
You create a new input from the class :py:class:`finch.data.Input` ::

    avg_input = finch.data.Input(
        name="average",
        source=avg_source,
        source_version=avg_src_version
    )

The data of an input is provided by its source, which needs to be provided when creating a new input.
The source describes how the data of the input is loaded initially.
You can easily create new versions of an input and store them to disk with finch.
Finch will query the available versions of an input when a specific version is requested and can create a new version on the fly.

A source is simply a function which takes a :py:class:`finch.data.Input.Version` object and returns a ``xarray.Dataset``.
For example, if you have a NetCDF file, you can define a source as follows. ::

    def avg_source(version: finch.data.Input.Version) -> xr.Dataset:
        return xr.open_dataset("data.nc")

The :py:class:`finch.data.Input.Version` argument indicates, which version of the input was requested.
In principle, finch will ensure itself that the requested version will be returned.
However, it might be more efficient to directly load the source in a specific format than to later on reformat it.
For example, it is often more efficient to directly load the requested chunk size instead of rechunking later on. ::

    def avg_source(version: Input.Version) -> xr.Dataset:
        return xr.open_dataset("avg_data.nc", chunks=version.chunks)

Along with the source, you need to provide a source version to the constructor of :py:class:`finch.data.Input`.
The source version fully describes the source data, which is returned by default from the source.
It must be complete, i.e. no fields are allowed to be ``None``. ::

    avg_src_version = finch.data.Input.Version(
        format=finch.data.Format.NetCDF,
        dim_order="xyz",
        chunks={"x": 10, "y": 10, "z": 1},
        coords=True,
    )


Running and Evaluating Experiments
----------------------------------

We can now use our operators to run experiments.
Let's compare how well the two operators scale.
We can use :py:func:`finch.measure_operator_runtimes` to measure the runtimes of our operators. ::

    runtimes = finch.measure_operator_runtimes(run_configs, avg_input, avg_src_version, iterations=5)

The :py:func:`finch.measure_operator_runtimes` function requires a list of :py:class:`finch.RunConfig` objects, which defines our experiment configuration.
Let's use dask with a single core per worker and go up to 40 cores. ::

    run_configs = finch.DaskRunConfig.list_configs(
        cluster_config = finch.scheduler.ClusterConfig(cores_per_worker=1),
        workers = range(5, 45, 5)
    )

Our output ``runtimes`` is now a 2D-list of raw :py:class:`finch.experiments.Runtime` objects.
We could inspect them manually, but finch provides some features for evaluation.
For this purpose, we can first transform our runtime objects into a "results dataset". ::

    results = finch.eval.create_results_dataset(
        runtimes,
        run_configs,
        avg_src_version,
        avg_input,
        experiment_name = "avg_scaling"
    )

The results dataset captures our runtimes along with our experiment configurations inside a single object.
It can be used as an input for the different evaluation function of the :py:mod:`finch.eval` module.

Let's create a plot which compares the scalability of our two operators. ::

    finch.eval.create_plots(results, scaling_dims=["workers"])

The :py:func:`finch.eval.create_plots` function creates a plot per configuration attribute, for which we have selected more than a sinlge value.
In our case, this is only the "workers" attribute, for which we ask the function to create a scalability plot.
The plot will be saved inside the :confval:`plot_dir` directory.

It should look somewhat like this.

.. TODO: Insert image
