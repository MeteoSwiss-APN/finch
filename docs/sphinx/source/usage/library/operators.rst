Defining a Custom Operator
==============================

Finch provides some built-in operators, which can be used for running experiments, in order to get a general understanding on how we can use finch for evaluating parallel operators and finding common bottlenecks when parallelizing with dask.
However, a basic design principle of finch is to provide an interface with which we can plug in custom operators and evaluate them using the tooling available in finch.

The following steps are required to create a custom operator for finch.

- Define a signature for the operator. Ideally, this conforms to the default operator signature of finch.
- Write an implementation of the operator, according to the interface.
- Define one or multiple input objects. Each of which requires the following steps.
    - Define an input source
    - Define the source version
    - Define a name for the input
- Integrate it into your experiment script

If you want to add your operator to the built-in operators of finch, make sure to read :ref:`new_builtin_ops`.

Default Operator Signature
--------------------------

Finch defines a default function signature for operators: :py:attr:`finch.DefaultOperator`
An operator conforming to the default signature takes a single ``xarray.Dataset`` and returns a ``xarray.DataArray``.
Additional optional arguments after the dataset are allowed.
Other signatures can also be used in principle. In order to run experiments with custom signatures, some additional implementation steps are however necessary, as documented in :ref:`_custom_signature`.
If you have multiple implementations of the same operator, these implementations must however all use the same signature.

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

    input = Input(
        name="foo",
        source=source,
        source_version=src_version
    )

The data of an input is provided by its source, which needs to be provided when creating a new input.
The source describes how the data of the input is loaded initially.
You can easily create new versions of an input and store them to disk with finch.
Finch will query the available versions of an input when a specific version is requested and can create a new version on the fly.

A source is simply a function which takes a :py:class:`finch.data.Input.Version` object and returns a ``xarray.Dataset``.
For example, if you have a NetCDF file, you can define a source as follows. ::

    def source(version: Input.Version) -> xr.Dataset:
        return xr.open_dataset("data.nc")

The :py:class:`finch.data.Input.Version` argument indicates, which version of the input was requested.
In principle, finch will ensure itself that the requested version will be returned.
However, it might be more efficient to directly load the source in a specific format than to later on reformat it.
For example, it is often more efficient to directly load the requested chunk size instead of rechunking later on. ::

    def source(version: Input.Version) -> xr.Dataset:
        return xr.open_dataset("data.nc", chunks=version.chunks)

Along with the source, you need to provide a source version to the constructor of :py:class:`finch.data.Input`.
The source version fully describes the source data, which is returned by default from the source.
It must be complete, i.e. no fields are allowed to be ``None``. ::

    src_version = Input.Version(
        format=data.Format.NetCDF,
        dim_order="xyz",
        chunks={"x": 10, "y": 10, "z": 1},
        coords=True,
    )

.. _custom_signature:

Custom Operator Signature
-------------------------
