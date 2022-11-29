from dataclasses import dataclass
import enum
from glob import glob
import pathlib
from typing import List, Dict, Any, Optional, Tuple, Union
import xarray as xr
import dask.array as da
import numpy as np
from . import config
from . import util
from . import version as pkg_version
from collections.abc import Callable
import yaml
import copy
from deprecated.sphinx import deprecated

data_config = config["data"]
grib_dir = data_config["grib_dir"]
"""The base directory from where to load grib files"""
netcdf_dir = data_config["netcdf_dir"]
"""The base directory from where to load and store netCDF files"""
zarr_dir = data_config["zarr_dir"]
"""The base directory from where to load and store zarr files"""
tmp_dir = config["global"]["tmp_dir"]
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
    FAKE = "fake" # special format for fake data

DimOrder = Union[str, list[str]]
"""
Type hint for dimension order

Group:
    Data
"""

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
        return list(str)
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
    path: str | List[str], 
    shortName: str, 
    chunks: Dict[str, int] | None, 
    key_filters: Dict[str, Any] = {},
    parallel: bool = True,
    cache: bool = True,
    index_path: str = "",
    load_coords: bool = True,
    **kwargs) -> xr.DataArray:
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
    """

    key_filters["shortName"] = shortName
    backend_args = {
        "read_keys": list(key_filters.keys()),
        "filter_by_keys": key_filters
    }

    args = {
        "engine": "cfgrib",
        "chunks": chunks,
        "backend_kwargs": backend_args,
        # remove "paramters" from the default list.
        # Otherwise the variable would be stored under the cfVarName instead of the shortName.
        "encode_cf": ("time", "geography", "vertical"),
        "cache": cache
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

def load_grib(grib_file: util.PathArg | list[util.PathArg], short_names: List[str], **kwargs) -> xr.Dataset:
    """
    Convenience function for loading multiple ``xarray.DataArray``s from a grib file with :func:`load_array_grib` and returning them as a dataset.

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

@deprecated("Use ``xarray.open_dataset`` instead", version="0.0.1a1")
def load_netcdf(filename, chunks: Dict) -> List[xr.DataArray]:
    """
    Loads the content of a netCDF file into `DataArray`s.

    Args:
        filename: The location of the netcdf file to load
        chunks: A chunk argument passed to ``xarray.open_dataset``.
    Returns:
        The content of the netcdf file as a list of ``xarray.DataArray``s.
    """
    filename = util.get_absolute(filename, netcdf_dir)
    ds: xr.Dataset = xr.open_dataset(filename, chunks=chunks)
    return list(ds.data_vars.values())

@deprecated("Use ``xarray.open_dataset(..., engine='zarr'`` instead.", version="0.0.1a1")
def load_zarr(filename, dim_names: List[str], chunks: List[int] = "auto", names: List[str] = None, inline_array = True) -> List[xr.DataArray]:
    """
    Loads the content of a zarr directory into `DataArray`(s).
    """
    filename = util.get_absolute(filename, zarr_dir)
    def single_load(component):
        array = da.from_zarr(url=filename, chunks=chunks, inline_array=inline_array, component=component)
        return xr.DataArray(array, dims=dim_names, name=component)
    if names:
        return [single_load(c) for c in names]
    else:
        return single_load(None)

@deprecated("Use ``xarray.Dataset.to_netcdf`` instead.", version="0.0.1a1")
def store_netcdf(
    filename: str, 
    data: List[xr.DataArray],
    names: List[str] | None = None):
    """
    Stores a list of `DataArray`s into a netCDF file.

    Args:
        filename: The name of the netCDF file.
        data: The `DataArray`s to be stored.
        names: A list of names for the arrays to be stored.
    """
    if names is None:
        names = [x.name for x in data]
    filename = util.get_absolute(filename, netcdf_dir)
    out = xr.Dataset(dict(zip(names, data)))
    out.to_netcdf(filename)

@deprecated("Use ``xarray.Dataset.to_zarr`` instead.", version="0.0.1a1")
def store_zarr(dirname: str, arrays: List[xr.DataArray]):
    """
    Stores a list of `DataArray`s into a zarr directory.

    Args:
        filename: The name of the zarr directory.
        data: The `DataArray`s to be stored.
    """
    dirname = util.get_absolute(dirname, zarr_dir)
    if type(arrays) is not list:
        arrays = [arrays]
    for a in arrays:
        da.to_zarr(a.data, dirname, a.name, overwrite=True)

@deprecated("Use ``xarray.Dataset.transpose`` instead.", version="0.0.1a1")
def reorder_dims(data: List[xr.DataArray], dim_order: List[str]) -> List[xr.DataArray]:
    """
    Returns the given list of data arrays, where the dimensions are ordered according to `dim_order`.
    """
    orders = [[d for d in dim_order if d in x.dims] for x in data]
    data = [x.transpose(*order) for x, order in zip(data, orders)]
    return data

@deprecated("Use ``xarray.DataArray.isel`` directly instead.", version="0.0.1a1")
def split_cube(data: xr.DataArray, split_dim: str, splits: List[int], names: List[str] = None) -> List[xr.DataArray]:
    """
    Splits a single data array hypercube into multiple individual data arrays.
    
    Args:
        data: The hypercube to be split.
        split_dim: The dimension along which to split the hypercube.
        splits: The individual sizes of `split_dim` dimension of the returned data arrays.
    """
    ends = np.cumsum(splits).tolist()
    starts = [0] + ends[:-1]
    # split
    out = [data.isel({split_dim : slice(s, e)}) for s, e in zip(starts, ends)]
    # reduce dimensionality where possible
    out = [x.squeeze(split_dim) if x.sizes[split_dim] == 1 else x for x in out]
    if names:
        out = [x.rename(n) for x, n in zip(out, names)]
    return out

class Input():
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
        format: Format = None
        """The file format of this version"""
        dim_order: DimOrder = None
        """
        The dimension order.
        The type of the dimension order (``str`` or ``list[str]``) must be the same across all versions of an input.
        """
        chunks: dict[str, int] = None
        """The chunking as a dict, mapping dimension short names to chunk sizes"""
        name: str = None
        """The name of this version"""
        coords: bool = None
        """Whether this version holds coordinates"""

        def __le__(self, other):
            """Self is less or equal to other, if other can be constructed from self 
            without any relevant future differences in performance when accessing the data 
            and without any changes in content."""
            # Most attributes must be equal in order for self and other to be comparable
            if self.format is not None and other.format is not None and self.format != other.format or \
                self.dim_order is not None and other.dim_order is not None and self.dim_order != other.dim_order or \
                self.coords is not None and other.coords is not None and self.coords != other.coords:
                return False
            # compare chunks
            return self.format != Format.ZARR \
                or self.chunks is None \
                or other.chunks is None \
                or util.can_rechunk_no_split(self.chunks, other.chunks)

        def impose(self, ds: xr.Dataset) -> xr.Dataset:
            """Transforms the given dataset such that it conforms to this version."""
            if self.dim_order is not None and ds.dims != self.dim_order:
                ds = ds.transpose(*get_dim_order_list(self.dim_order))
            if self.chunks is not None:
                if not util.chunk_args_equal(self.chunks, ds.chunksizes, ds.sizes):
                    ds = ds.chunk(self.chunks)
            if self.coords is not None and not self.coords:
                ds = ds.drop_vars(ds.coords.keys())
            return ds

        def get_all_chunks(self, dims: list[str]) -> dict[str, str]:
            """Returns a dict which holds a chunk size for every dimension provided (as short name)"""
            return {d : self.chunks[d] if d in self.chunks else -1 for d in dims}

        @classmethod
        def get_class_attr(cls) -> list[str]:
            return ["format", "dim_order", "chunks", "coords"]

    source: Callable[[Optional[Version]], xr.Dataset]
    """A function from which to get the input data"""
    source_version: Version
    """The version of the source data"""
    versions: list[Version]
    """The different versions of this input"""

    def __init__(self,
        name: str, 
        source: Callable[[Version], xr.Dataset],
        source_version: Version,
        store_path: util.PathArg | None = None
    ) -> None:
        """Creates a new Input for the given source.

        Args:
            name (str): The name of the input
            source (Callable[[Version], xr.Dataset]): The source of the input.
                The dataset returned by this function must always have the same content.
                It should match the attributes of the passed version as closely as possible, but is not required to.
            source_version (Version): A version object describing the source.
                This version object cannot have None fields. The chunks for all dimensions need to be explicitly specified.
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
        if any(d not in source_version.chunks for d in source_version.dim_order):
            raise ValueError("Received source version with incomplete chunk specification.", source_version)
        self.source_version = source_version
        self.source_version.name = "source"
        self.versions = [source_version]
        if store_path is None:
            store_path = data_config["input_store"]
        self._path = util.get_path(store_path, name).absolute()

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
            The properties of the newly created version
        """
        # check whether version already exists
        vname = version.name
        version.name = None
        if self.has_version(version):
            print("This verion already exists and won't be added")
            return
        version.name = vname

        # cannot store grib
        if version.format == Format.GRIB:
            raise NotImplementedError

        # handle name
        names = [v.name for v in self.versions]
        if vname is None:
            version.name = util.random_entity_name(excludes=names)
        if vname in names:
            raise ValueError("A version with the given name exists already")

        if dataset is None:
            # create new version to be added
            dataset, version = self.get_version(version, create_if_not_exists=True, add_if_not_exists=False)

        # add missing chunk sizes
        for d in self.source_version.chunks:
            if d not in version.chunks or version.chunks[d] is None or version.chunks[d] == "auto":
                version.chunks[d] = -1

        # store data
        filename = str(self._path.joinpath(version.name))
        if version.format == Format.NETCDF:
            dataset.to_netcdf(filename + ".nc", mode="w")
        elif version.format == Format.ZARR:
            dataset.to_zarr(filename, mode="w")
        else:
            raise ValueError # grib case was already caught, so we can raise a ValueError here

        # store yaml
        with open(self._path.joinpath(version.name + ".yml"), mode="w") as f:
            yaml.dump(version, f)

        # register
        self.versions.append(version)
        return version

    def has_version(self, version: Version) -> bool:
        """
        Returns whether a version with the given properties already exists
        """
        return any(util.has_attributes(version, v) for v in self.versions)

    def get_version(self, version: Version, create_if_not_exists: bool = True, add_if_not_exists: bool = False, weak_compare: bool = True) -> Tuple[xr.Dataset, Version]:
        """
        Returns a version with the given properties.

        Args:
            version: The version properties
            create_if_not_exists: Indicates whether to create and return a new version if no such version already exists.
            add_if_not_exists: Indicates whether to directly add a newly created version to this input
            weak_compare: If true, a smaller version can also be returned.
                The partial order between versions is defined in the Version class.
        """
        # find matching preexisting version
        target = None
        source_name = None
        for v in self.versions[1:]: # avoid loading the source version when possible
            if not weak_compare and util.has_attributes(version, v):
                target = v
                break
            elif weak_compare and v <= version and (target is None or target <= v):
                target = util.fill_none_properties(version, v)
                source_name = v.name

        if target is None:
            # create new version if desired, or if the new version can be deduced from the source version
            if create_if_not_exists or \
                (not weak_compare and util.has_attributes(version, self.source_version)) or \
                (weak_compare and self.source_version <= version):
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
            filename = str(self._path.joinpath(source_name))
            chunks = target.chunks
            if target.format == Format.NETCDF:
                dataset = xr.open_dataset(filename+".nc", chunks=chunks)
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
        