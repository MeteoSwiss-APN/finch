Input Management
===================

Finch provides its own input management via the class :class:`finch.Input`.
An instance of this class describes a single input for some operator.
An input can have multiple versions, described by the class :class:`finch.Input.Version`.
Every such version contains the same data, but stores it differently.

Creating a new Input
--------------------

We can create a new input by creating a new instance of :class:`finch.Input`.
The constructor takes a source, which is a function taking a :class:`finch.Input.Version` object and returning a ``xarray.Dataset``.
The returned dataset serves as the input for some operator and its content must always be the same with every call to the source function.
The version argument will be passed when we want to create a new version of the input.
This allows the source function to efficiently output the requested version of the input.
It is not expected that the output of the source function matches the requested version.
However, those version properties which won't match will be transformed in some default manner, which might be less efficient than what would be possible.
For example, loading the correct chunks directly from some source file is more efficient than loading the data with some arbitrary chunking and rechunking afterwards.

Besides the actual source, we must also provide a version object, which describes the default output of the source.
This source version cannot have any attributes set to ``None``.
::
    def foo_source(version):
        return finch.data.load_grib("foo.grib", ["P", "T"], chunks=version.chunks)

    foo_version = finch.Input.Version(
        format=finch.Format.GRIB,
        dim_order="zyx",
        chunks={k:-1 for k in list("xyz")},
        coords=True
    )
    
    foo = finch.Input("foo", foo_source, foo_version)