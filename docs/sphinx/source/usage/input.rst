Input Management
===================

Finch provides its own input management via the class :class:`finch.Input`.
An instance of this class describes a single input for some operator.
An input can have multiple versions, described by the class :class:`finch.Input.Version`.
Every such version contains the same data, but stores it differently.

Creating a new Input
--------------------

We can create a new input by creating a new instance of :class:`finch.data.Input`.
The constructor takes a source, which is a function taking a :class:`finch.data.Input.Version` object and returning a ``xarray.Dataset``.
The returned dataset serves as the input for some operator and its content must always be the same with every call to the source function.
The version argument will be passed when we want to create a new version of the input.
This allows the source function to efficiently output the requested version of the input.
It is not expected that the output of the source function matches the requested version.
However, those version properties which won't match will be transformed in some default manner, which might be less efficient than what would be possible.
For example, loading the correct chunks directly from some source file is more efficient than loading the data with some arbitrary chunking and rechunking afterwards.
Besides the actual source, we must also provide a (complete) version object, which describes the default output of the source.
::
    from finch.data import Version, Format

    def foo_source(version):
        return finch.data.load_grib("foo.grib", ["P", "T"], chunks=version.chunks)

    foo_version = Input.Version(
        format=Format.GRIB,
        dim_order="zyx",
        chunks={k:-1 for k in list("xyz")},
        coords=True
    )

    foo = Input("foo", foo_source, foo_version)


Creating new Versions
----------------------

We can easily create a new version of our input with the function :func:`finch.data.Input.add_version`.
The new version will be stored in our input's directory inside the finch input data store.
When adding a new version, finch will try to retrieve an already existing version which can be used to construct the new version.
If there is no such preexisting version, it will use the source to construct the new version.
The name of a new version will be automatically generated. However, the name is only really used for storing the data.
::
    version = Input.Version(
        format=Format.ZARR,
        dim_order="xyz",
        chunks={"x": 10, "y": -1, "z": -1},
        coords=False
    )

    foo.add_version(version)

If you don't care about certain version properties, you can omit them in the constructor.
They will then be set according to the version which was used for loading the data.
::
    version = Input.Version(
        format=Format.NETCDF
    )

    foo.add_version(version)

Additionally, if you already have an input ready which you want to add, you can provide it with the ``data`` argument.
However, keep in mind that you are responsible yourself that your data matches the version you provide.
If you provide the data yourself, the version can no longer have any unset attributes.
::
    version = Input.Version(
        format=Format.NETCDF,
        dim_order="xyz",
        chunks={"x": 10, "y": -1, "z": -1},
        coords=False
    )

    data, _ = foo.get_version(version)
    foo.add_version(version, data)