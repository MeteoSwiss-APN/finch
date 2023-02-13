import copy
import enum
import pathlib
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from glob import glob
from typing import Any, Dict, List, Literal, Union, overload

import dask.config
import numpy as np
import xarray as xr
import yaml
from deprecated.sphinx import deprecated

from . import __version__ as pkg_version
from . import cfg, util

data_config = cfg["data"]
grib_dir = data_config["grib_dir"]
"""The base directory from where to load grib files"""
netcdf_dir = data_config["netcdf_dir"]
"""The base directory from where to load and store netCDF files"""
zarr_dir = data_config["zarr_dir"]
"""The base directory from where to load and store zarr files"""
tmp_dir = cfg["global"]["tmp_dir"]
"""A directory which can be used as a temporary storage"""


class Format(enum.Enum):
    """
    Supported file formats

    Group:
        Input
    """

    GRIB = "grib"
    NETCDF = "netcdf"
    ZARR = "zarr"
    FAKE = "fake"  # special format for fake data


DimOrder = Union[str, list[str]]
"""
Type hint for dimension order.
If the dimension order is a string, the dimensions are specified by individual characters ('x', 'y', etc.).
With a list of strings, it is possible to give more descriptive names to individual dimensions.

Group:
    Data
"""

###############################
# Chunking utilities
###############################


Chunks = Mapping[Hashable, int | tuple[int, ...] | None | Literal["auto"]]
"""
A type alias for xarray chunks, capturing all the possible options.
In xarray, chunks are specified per dimension in a dictionary with the key being the dimension name.
The values can be one of the following:

``positive int``
    A chunk size for all chunks (except the last) in a dimension.
``tuple[int]``
    Excplicit chunk sizes for all chunks
``-1``
    No chunking. This is equivalent to a single chunk size of the dimension length.
``None``
    Arbitrary chunk size / no change to the current chunking.
``"auto"``
    An "ideal" chunk size.
    Can be interpreted as ``array.chunk-size`` from the dask config.


See Also:
    https://docs.dask.org/en/stable/array-chunks.html#specifying-chunk-shapes
    and
    https://docs.dask.org/en/stable/array-chunks.html#automatic-chunking

Group:
    Data
"""

auto_chunk_size: int = dask.config.get("array.chunk-size")
"""
The chunk size used for the "auto" keyword.

Group:
    Data
"""


def simplify_chunks(c: Chunks) -> Mapping[Hashable, int | tuple[int, ...]]:
    """
    Simplyfies a chunks dictionary by resolving "auto" and removing None entries.

    Group:
        Data
    """
    return {k: auto_chunk_size if v == "auto" else v for k, v in c.items() if v is not None}


def get_chunk_sizes(s: int, d: int) -> list[int]:
    """
    Returns a list of explicit chunk sizes from a single chunk size.

    Args:
        s (int): The single chunk size
        d (int): The size of the dimension.

    Group:
        Data
    """
    if s == -1:
        return [d]
    out = [s] * (d // s)
    if d % s != 0:
        out.append(d % s)
    return out


def chunk_args_equal(c1: Chunks, c2: Chunks, dim_sizes: Mapping[Hashable, int]) -> bool:
    """
    Returns whether two xarray chunk arguments are equal.
    Auto and None chunk arguments will always be equal.
    If a dimension name is not present, its size will be interpreted as `None`.

    Args:
        c1 (Chunks): The first chunk argument
        c2 (Chunks): The second chunk argument
        dim_sizes (Mapping[Hashable, int]):
            The dimension sizes of the xarray datastructure for which c1 and c2 can be applied.

    Group:
        Data
    """
    for k, v1 in c1.items():
        if k in c2:
            v2 = c2[k]
            if v1 is None or v2 is None or v1 == "auto" or v2 == "auto":
                continue
            if v1 == v2:
                continue
            if isinstance(v1, int):
                v1l: list[int] = get_chunk_sizes(v1, dim_sizes[k])
            else:
                v1l = list(v1)
            if isinstance(v2, int):
                v2l: list[int] = get_chunk_sizes(v2, dim_sizes[k])
            else:
                v2l = list(v2)
            if v1l == v2l:
                continue
            else:
                return False
    return True


def can_rechunk_no_split(c1: Chunks, c2: Chunks) -> bool:
    """
    Returns True, if `c1` can be rechunked according to `c2` without the need to split up any chunks.

    Group:
        Data
    """
    for k, v1 in c1.items():
        if k in c2:
            v2 = c2[k]
            if v1 is None or v2 is None or v1 == "auto" or v2 == "auto" or v2 == -1:
                continue
            if v1 == -1:
                return False
            if isinstance(v1, int) and isinstance(v2, int):
                if v1 > v2 or v2 % v1 != 0:
                    return False
            else:
                if isinstance(v1, int):
                    d = np.sum(v2)
                    v1l = get_chunk_sizes(v1, d)
                else:
                    v1l = list(v1)
                if isinstance(v2, int):
                    d = np.sum(v1)
                    v2l = get_chunk_sizes(v2, d)
                else:
                    v2l = list(v2)
                if len(v2l) > len(v1l):
                    return False
                cur = 0
                while v1l:
                    cur += v1l.pop()
                    if cur > v2l[-1]:
                        return False
                    elif cur == v2l[-1]:
                        cur = 0
                        v2l.pop()
                if v2l:
                    return False
    return True


def adjust_dims(dims: Iterable[Hashable], array: xr.DataArray) -> xr.DataArray:
    """
    Return a new DataArray with the same content as `array` such that
    the dimensions match `dims` in content and order.
    This is achieved with a combination of `expand_dims`, `squeeze` and `transform`.
    When trying to remove dimensions with sizes larger than 1, an error will be thrown.

    Args:
        dims (list[str]):
            A list of dimension names,
            which identify the dimensions and the dimension order of the output array.

    Returns:
        xr.DataArray: An arrary which has dimensions as specified by `dims`.

    Group:
        Data
    """
    # Performance might be improved.
    # We could check whether the dimensions in the array are in correct order.
    # If so, we wouldn't need a transpose.
    # Since we are however working with views, this might not even be faster.
    to_remove = set(array.dims).difference(dims)
    to_add = set(dims).difference(array.dims)
    return array.squeeze(to_remove).expand_dims(list(to_add)).transpose(*dims)


def get_dim_order_list(order: DimOrder) -> list[str]:
    """Transforms a dimension order into list form.

    Args:
        order (DimOrder): The dimension order to transform.

    Returns:
        list[str]: The dimension order as a list.

    Group:
        Data
    """
    if isinstance(order, str):
        return list(order)
    else:
        return order


@deprecated("Dimension index is deprecated.", version="0.0.1a1")
def translate_order(order: List[str] | str, index: Dict[str, str]) -> str | List[str]:
    """
    Translates a dimension order from compact form to verbose form or vice versa.
    A dimension order in compact form is a string where each letter represents a dimension (e.g. "xyz").
    A dimension order in verbose form is a list of dimension names (e.g. ["x", "y", "generalVerticalLayer"]).

    Args:
        order: The dimension order either in compact or verbose form.
        index: A dicitionary, mapping letter representations of the dimensions to their verbose names.

    Returns:
        The dimension order in compact form if `order` was given in verbose form or vice versa.

    Group:
        Data
    """
    if isinstance(order, str):
        return [index[x] for x in list(order)]
    else:
        rev_index = {v: k for k, v in index.items()}
        return "".join([rev_index[x] for x in order])


def load_array_grib(
    path: util.PathLike | list[util.PathLike],
    shortName: str,
    chunks: dict[str, int] | None = None,
    key_filters: dict[str, Any] = {},
    parallel: bool = True,
    cache: bool = True,
    index_path: str = "",
    load_coords: bool = True,
) -> xr.DataArray:
    """
    Loads a DataArray from a given grib file.

    Args:
        path: The path(s) to the grib file(s) from which to load the array
        shortName: The GRIB shortName of the variable to load
        chunks: A dictionary, indicating the chunk size per dimension to be loaded.
            If `None`, no chunks will be used and the data will be loaded as notmal numpy arrays instead of dask arrays.
        key_filters: A dictionary used for filtering GRIB messages.
            Only messages where the given key matches the according value in this dictionary will be loaded.
        parallel: Whether to load files in parallel. Ignored if only one file is loaded.
        index_path: The path to a cfgrib index file. Ignored if multiple files are loaded.
        load_coords: Whether to load coordinates or not.

    Group:
        Data
    """

    key_filters["shortName"] = shortName
    backend_args = {"read_keys": list(key_filters.keys()), "filter_by_keys": key_filters}

    args: dict[str, Any] = {
        "engine": "cfgrib",
        "chunks": chunks,
        "backend_kwargs": backend_args,
        # remove "paramters" from the default list.
        # Otherwise the variable would be stored under the cfVarName instead of the shortName.
        "encode_cf": ("time", "geography", "vertical"),
        "cache": cache,
    }

    if isinstance(path, list):
        dataset = xr.open_mfdataset([util.get_absolute(p, grib_dir) for p in path], parallel=parallel, **args)
    else:
        args["indexpath"] = index_path
        dataset = xr.open_dataset(util.get_absolute(path, grib_dir), **args)
    out: xr.DataArray = dataset[shortName]
    if not load_coords:
        for c in list(out.coords.keys()):
            del out[c]
    return out


def load_grib(grib_file: util.PathLike | list[util.PathLike], short_names: List[str], **kwargs: Any) -> xr.Dataset:
    """
    Convenience function for loading multiple ``xarray.DataArray``s from a grib file with
    :func:`load_array_grib` and returning them as a dataset.

    Args:
        grib_file: The location of the grib file to load
        short_names: The names of the variables to load
        kwargs: Additional arguments passed to :func:`load_array_grib`
    Returns:
        The requested variables wrapped in a ``xarray.Dataset``
    See Also:
        :func:`load_array_grib`
    Group:
        Data
    """
    arrays = [load_array_grib(grib_file, shortName=sn, **kwargs) for sn in short_names]
    return xr.merge(arrays)


class Input:
    """
    Class for managing experiment inputs on disk.

    Group:
        Input
    """

    name: str
    """The name of this input"""
    _path: pathlib.Path
    """The path where this input is stored"""

    @dataclass
    class Version(yaml.YAMLObject, util.Config):
        """A version of the input"""

        yaml_tag = "!Version"
        """The tag to use for encoding in yaml. DO NOT MODIFY"""
        finch_version = pkg_version
        """The finch version that was used to create this input version. DO NOT MODIFY"""
        format: Format | None = None
        """
        The file format of this version.
        This can be passed as a ``Format`` object or as a
        string representation of a ``Format`` item during initialization.
        """
        dim_order: DimOrder | None = None
        """
        The dimension order.
        The type of the dimension order (``str`` or ``list[str]``) must be the same across all versions of an input.
        """
        chunks: Chunks = field(default_factory=dict)
        """The chunking as a dict, mapping dimension short names to chunk sizes"""
        name: str = ""
        """The name of this version"""
        coords: bool | None = None
        """Whether this version holds coordinates"""

        def __post_init__(self) -> None:
            # allow passing the string representation of format.
            if isinstance(self.format, str):
                self.format = Format(self.format)

        def __le__(self, other: "Input.Version") -> bool:
            """Self is less or equal to other, if other can be constructed from self
            without any relevant future differences in performance when accessing the data
            and without any changes in content."""
            # Most attributes must be equal in order for self and other to be comparable
            if (
                self.format is not None
                and other.format is not None
                and self.format != other.format
                or self.dim_order is not None
                and other.dim_order is not None
                and self.dim_order != other.dim_order
                or self.coords is not None
                and other.coords is not None
                and self.coords != other.coords
            ):
                return False
            # compare chunks
            return (
                self.format != Format.ZARR
                or self.chunks is None
                or other.chunks is None
                or can_rechunk_no_split(self.chunks, other.chunks)
            )

        def impose(self, ds: xr.Dataset) -> xr.Dataset:
            """Transforms the given dataset such that it conforms to this version."""
            ds_dim_order = list(ds.dims.keys())
            if self.dim_order is not None and ds_dim_order != list(self.dim_order):
                ds = ds.transpose(*get_dim_order_list(self.dim_order))
            if self.chunks is not None:
                if not chunk_args_equal(self.chunks, ds.chunksizes, ds.sizes):
                    ds = ds.chunk(self.chunks)
            if self.coords is not None and not self.coords:
                ds = ds.drop_vars(ds.coords.keys())
            return ds

        @deprecated(
            "This is a simple filter over the chunks dictionary, which doesn't need a special method.",
            version="0.0.1a4",
        )
        def get_all_chunks(self, dims: list[str]) -> Chunks:
            """Returns a dict which holds a chunk size for every dimension provided"""
            if self.chunks is None:
                return {}
            else:
                return {d: self.chunks[d] if d in self.chunks else -1 for d in dims}

    source: Callable[[Version], xr.Dataset]
    """A function from which to get the input data"""
    source_version: Version
    """The version of the source data"""
    versions: list[Version]
    """The different versions of this input"""

    def __init__(
        self,
        name: str,
        source: Callable[[Version], xr.Dataset],
        source_version: Version,
        store_path: util.PathLike | None = None,
    ) -> None:
        """Creates a new Input for the given source.

        Args:
            name (str): The name of the input
            source (Callable[[Version], xr.Dataset]): The source of the input.
                The dataset returned by this function must always have the same content.
                It should match the attributes of the passed version as closely as possible, but is not required to.
            source_version (Version): A version object describing the source.
                This version object cannot have None fields.
                The chunks for all dimensions need to be explicitly specified.
            store_path (util.PathArg | None, optional): A path to a directory which can be used to store input versions.
                The versions will not be stored directly in this directory, but in a subdirectory according to `name`.
                If None is passed, the configured default finch input store will be used.
                Defaults to None.
        """
        self.name = name
        self.source = source
        # validity check on source version
        if any(a is None for a in util.get_class_attributes(source_version)):
            raise ValueError("Received source version with None attribute.", source_version)
        # this assertion is only for type checking (which doesn't understand the above if statement)
        assert source_version.dim_order is not None
        if any(d not in source_version.chunks for d in source_version.dim_order):
            raise ValueError("Received source version with incomplete chunk specification.", source_version)
        self.source_version = source_version
        self.source_version.name = "source"
        self.versions = [source_version]
        if store_path is None:
            store_path = data_config["input_store"]
        self._path = pathlib.Path(util.get_path(store_path, name)).absolute()

        # load existing versions
        for v in glob("*.yml", root_dir=str(self._path)):
            with open(self._path.joinpath(v)) as f:
                version: "Input.Version" = yaml.load(f, yaml.Loader)
                # backwards compatibility
                if version.finch_version is None:
                    version.finch_version = "0.0.1a1"
                version = util.add_missing_properties(version, self.source_version)
                # add version
                self.versions.append(version)

    def add_version(self, version: Version, dataset: xr.Dataset | None = None) -> Version:
        """
        Adds a new version of this input.
        If a version with the same properties already exists, nothing happens.
        If a version with the same name already exists, a value error will be raised.

        Arguments
            version: The version properties of the new version
            dataset: The data of the new version. If `None`, the data will be generated from a preexisting version.

        Returns:
            The actual newly created version
        """
        # check whether version already exists
        if self.has_version(version):
            print("This verion already exists and won't be added")
            return version

        nu_version = copy.copy(version)

        # cannot store grib
        if nu_version.format == Format.GRIB:
            raise NotImplementedError

        # handle name
        names = [v.name for v in self.versions if v.name is not None]
        if version.name == "":
            nu_version.name = util.random_entity_name(excludes=names)
        elif version.name in names:
            raise ValueError("A version with the given name exists already")

        if dataset is None:
            # create new version to be added
            dataset, nu_version = self.get_version(nu_version, add_if_not_exists=False)

        # add missing chunk sizes
        chunks = dict(nu_version.chunks)
        for d in self.source_version.chunks:
            if d not in chunks or chunks[d] is None or chunks[d] == "auto":
                chunks[d] = -1
        nu_version.chunks = chunks

        # store data
        filename = str(self._path.joinpath(nu_version.name))
        if nu_version.format == Format.NETCDF:
            dataset.to_netcdf(filename + ".nc", mode="w")
        elif nu_version.format == Format.ZARR:
            dataset.to_zarr(filename, mode="w")
        else:
            raise ValueError  # grib case was already caught, so we can raise a ValueError here

        # store yaml
        with open(self._path.joinpath(nu_version.name + ".yml"), mode="w") as f:
            yaml.dump(nu_version, f)

        # register
        self.versions.append(nu_version)
        return nu_version

    def has_version(self, version: Version) -> bool:
        """
        Returns whether a version with the given properties already exists
        """
        normal_matches = [util.has_attributes(version, v, excludes=["name", "chunks"]) for v in self.versions]
        chunks = simplify_chunks(version.chunks)
        chunk_matches = [version.chunks[k] == v for k, v in chunks.items() for version in self.versions]
        return any(n and c for n, c in zip(normal_matches, chunk_matches))

    @overload
    def get_version(
        self,
        version: Version,
        add_if_not_exists: bool = ...,
        weak_compare: bool = ...,
    ) -> tuple[xr.Dataset, Version]:
        ...

    @overload
    def get_version(
        self,
        version: Version,
        add_if_not_exists: bool = ...,
        weak_compare: bool = ...,
        create_if_not_exists: bool = ...,
    ) -> tuple[xr.Dataset, Version] | None:
        ...

    def get_version(
        self,
        version: Version,
        add_if_not_exists: bool = False,
        weak_compare: bool = True,
        create_if_not_exists: bool = True,
    ) -> tuple[xr.Dataset, Version] | None:
        """
        Returns a version with the given properties.

        Args:
            version (Version):
                The version properties
            create_if_not_exists (bool):
                Indicates whether to create and return a new version if no such version already exists.
            add_if_not_exists (bool):
                Indicates whether to directly add a newly created version to this input
            weak_compare (bool):
                If true, a smaller version can also be returned.
                The partial order between versions is defined in the Version class.
        Returns:
            None:
                If a new version would need to be created but ``create_if_not_exists`` is false.
            tuple[xr.Dataset, Version]
                The data of the new version along with its properties
        """
        # find matching preexisting version
        target = None
        source_name = None
        for v in self.versions[1:]:  # avoid loading the source version when possible
            if not weak_compare and util.has_attributes(version, v):
                target = v
                break
            elif weak_compare and v <= version and (target is None or target <= v):
                target = util.fill_none_properties(version, v)
                source_name = v.name

        if target is None:
            # create new version if desired, or if the new version can be deduced from the source version
            if (
                create_if_not_exists
                or (not weak_compare and util.has_attributes(version, self.source_version))
                or (weak_compare and self.source_version <= version)
            ):
                # fill none properties of version
                target = util.fill_none_properties(version, self.source_version)
                # load source (for targeted version)
                dataset = self.source(target)
                # impose version
                dataset = target.impose(dataset)

                # add version
                if add_if_not_exists:
                    self.add_version(target, dataset)
            else:
                return None
        else:
            # load the preexisting version
            assert source_name is not None  # for type checker
            filename = str(self._path.joinpath(source_name))
            chunks = dict(target.chunks)
            if target.format == Format.NETCDF:
                dataset = xr.open_dataset(filename + ".nc", chunks=chunks)
            elif target.format == Format.ZARR:
                dataset = xr.open_dataset(filename, chunks=chunks, engine="zarr")
            elif target.format == Format.GRIB:
                dataset = xr.open_dataset(filename, chunks=chunks, engine="cfgrib")
        return dataset, target

    def list_versions(self) -> list[Version]:
        """
        Lists the available versions for this input
        """
        return copy.deepcopy(self.versions)
