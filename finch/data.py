import enum
from glob import glob
import pathlib
from typing import List, Dict, Any, Tuple
import xarray as xr
import dask.array as da
import numpy as np
from . import config
from . import util
from collections.abc import Callable
import yaml

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
    """Supported file formats"""
    GRIB = enum.auto()
    NETCDF = enum.auto()
    ZARR = enum.auto()

def translate_order(order: List[str] | str, index: Dict[str, str]) -> str | List[str]:
    """
    Translates a dimension order from compact form to verbose form or vice versa.
    A dimension order in compact form is a string where each letter represents a dimension (e.g. "xyz").
    A dimension order in verbose form is a list of dimension names (e.g. ["x", "y", "generalVerticalLayer"]).

    Arguments:
    ---
    - order: The dimension order either in compact or verbose form.
    - index: A dicitionary, mapping letter representations of the dimensions to their verbose names.

    Returns:
    ---
    The dimension order in compact form if `order` was given in verbose form or vice versa.
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
    index_path: str = tmp_dir + "/grib.idx",
    load_coords: bool = True,
    **kwargs) -> xr.DataArray:
    """
    Loads a DataArray from a given grib file.

    Arguments:
    ---
    - path: str or list of str. The path(s) to the grib file(s) from which to load the array
    - shortName: str. The GRIB shortName of the variable to load
    - chunks: dict. A dictionary, indicating the chunk size per dimension to be loaded. 
    If `None`, no chunks will be used and the data will be loaded as notmal numpy arrays instead of dask arrays.
    - key_filters: dict. A dictionary used for filtering GRIB messages.
    Only messages where the given key matches the according value in this dictionary will be loaded.
    - parallel: bool. Whether to load files in parallel. Ignored if only one file is loaded.
    - index_path: str. The path to a cfgrib index file. Ignored if multiple files are loaded.
    - load_coords: bool. Whether to load coordinates or not.
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

def load_grib(grib_file, short_names: List[str], **kwargs) -> List[xr.DataArray]:
    """
    Convenience function for loading multiple `DataArray`s from a grib file with `load_array_grib`.
    """
    return [load_array_grib(grib_file, shortName=sn, **kwargs) for sn in short_names]

def load_netcdf(filename, chunks: Dict) -> List[xr.DataArray]:
    """
    Loads the content of a netCDF file into `DataArray`s.
    """
    filename = util.get_absolute(filename, netcdf_dir)
    ds: xr.Dataset = xr.open_dataset(filename, chunks=chunks)
    return list(ds.data_vars.values())

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

def store_netcdf(
    filename: str, 
    data: List[xr.DataArray],
    names: List[str] | None = None):
    """
    Stores a list of `DataArray`s into a netCDF file.

    Arguments:
    ---
    - filename: str. The name of the netCDF file.
    - data: List[DataArray]. The `DataArray`s to be stored.
    - names: List[str]. A list of names for the arrays to be stored.
    """
    if names is None:
        names = [x.name for x in data]
    filename = util.get_absolute(filename, netcdf_dir)
    out = xr.Dataset(dict(zip(names, data)))
    out.to_netcdf(filename)

def store_zarr(dirname: str, arrays: List[xr.DataArray]):
    """
    Stores a list of `DataArray`s into a zarr directory.

    Arguments:
    ---
    - filename: str. The name of the zarr directory.
    - data: List[DataArray]. The `DataArray`s to be stored.
    """
    dirname = util.get_absolute(dirname, zarr_dir)
    if type(arrays) is not list:
        arrays = [arrays]
    for a in arrays:
        da.to_zarr(a.data, dirname, a.name, overwrite=True)

def reorder_dims(data: List[xr.DataArray], dim_order: List[str]) -> List[xr.DataArray]:
    """
    Returns the given list of data arrays, where the dimensions are ordered according to `dim_order`.
    """
    orders = [[d for d in dim_order if d in x.dims] for x in data]
    data = [x.transpose(*order) for x, order in zip(data, orders)]
    return data

def split_cube(data: xr.DataArray, split_dim: str, splits: List[int], names: List[str] = None) -> List[xr.DataArray]:
    """
    Splits a single data array hypercube into multiple individual data arrays.
    
    Arguments:
    ---
    - data: The hypercube to be split.
    - split_dim: str. The dimension along which to split the hypercube.
    - splits: List[int]. The individual sizes of `split_dim` dimension of the returned data arrays.
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
    """
    name: str
    """The name of this input"""
    _path: pathlib.Path
    """The path where this input is stored"""
    source: Callable[[], list[xr.DataArray]]
    """A function from which to get the input data"""
    dim_index: dict[str, str]
    """An index mapping dimension short names to dimension names"""
    array_names: list[str]
    """The names of the data arrays"""

    class Version():
        """A version of the input"""
        format: Format
        dim_order: str
        chunks: dict[str, int]
        name: str

        def impose(self, arrays: list[xr.DataArray], dim_index: dict[str, str]) -> list[xr.DataArray]:
            """
            Returns a new version of the given input arrays according to the attributes of this object.
            """
            out = []
            for a in arrays:
                b = a.transpose(*translate_order(self.dim_order, dim_index))

    source_version: Version
    """The version of the source data"""
    versions: list[Version]
    """The different versions of this input"""

    def __init__(self, 
        store_path: pathlib.Path, 
        name: str, 
        source: Callable[[], list[xr.DataArray]],
        source_version: Version,
        dim_index: dict[str, str],
        array_names: list[str]
    ) -> None:
        self.name = name
        self._path = store_path.joinpath(name).absolute()
        self.source = source
        self.source_version = source_version
        self.source_version.name = "source"
        self.versions = [source_version]
        self.dim_index = dim_index
        self.array_names = array_names

        # create if not exists
        self._path.mkdir(parents=True, exist_ok=True)

        # load existing versions
        for v in glob("*.yml", root_dir=str(self._path)):
            with open(v) as f:
                self.versions.append(yaml.load(f))

    def add_version(self, version: Version, arrays: list[xr.DataArray] | None = None) -> Version:
        """
        Adds a new version of this input.
        If a version with the same properties already exists, nothing happens.
        If a version with the same name already exists, a value error will be raised.

        Arguments:
        ---
        - version: The version properties of the new version
        - arrays: optional. The data of the new version. If `None`, the data will be generated from the source version.

        Returns:
        ---
        The properties of the newly created version
        """
        # check whether version already exists
        vname = version.name
        version.name = None
        if self.has_version(version):
            print("This verion already exists and won't be added")
            return
        version.name = vname

        # handle name
        names = [v.name for v in self.versions]
        if vname is None:
            version.name = util.random_entity_name(excludes=names)
        if vname in names:
            raise ValueError("A version with the given name exists already")

        if arrays is None:
            # create new version to be added
            arrays, version = self.get_version(version, create_if_not_exists=True, add_if_not_exists=False)

        # store yaml
        with open(self._path.joinpath(version.name + ".yml"), mode="w") as f:
            yaml.dump(version, f)

        # store data
        filename = str(self._path.joinpath(version.name))
        if version.format == Format.NETCDF:
            store_netcdf(filename + ".nc", arrays)
        elif version.format == Format.ZARR:
            store_zarr(filename, arrays)
        else:
            raise NotImplementedError

        # register
        self.versions.append(version)
        return version

    def has_version(self, version: Version) -> bool:
        """
        Returns whether a version with the given properties already exists
        """
        return any(util.has_attributes(version, v) for v in self.versions)

    def get_version(self, version: Version, create_if_not_exists: bool = True, add_if_not_exists: bool = False) -> Tuple[list[xr.DataArray], Version]:
        """
        Returns a version with the given properties.

        Arguments:
        ---
        - version: The version properties
        - create_if_not_exists: Indicates whether to create and return a new version if no such version already exists.
        - add_if_not_exists: Indicates whether to directly add a newly created version to this input
        """
        # find matching preexisting version
        target = None
        for v in self.versions[1:]: # avoid loading the source version when possible
            if util.has_attributes(version, v):
                target = v
                break

        if target is None:
            if create_if_not_exists or util.has_attributes(version, self.source_version):
                # load source
                arrays = self.source()
            
                # impose version properties
                target = util.fill_none_properties(version, self.source_version)
                arrays = [
                    a.transpose(translate_order(target.dim_order, self.dim_index)).
                        chunk(target.chunks)
                    for a in arrays
                ]

                # add version (if not grib)
                if add_if_not_exists and target.format != Format.GRIB:
                    self.add_version(target, arrays)
            else:
                return None
        else:
            filename = str(self._path.joinpath(target.name))
            if target.format == Format.NETCDF:
                arrays = load_netcdf(filename+".nc", target.chunks)
            elif target.format == Format.ZARR:
                arrays = load_zarr(filename, translate_order(target.dim_order, self.dim_index), names=self.array_names)
            elif target.format == Format.GRIB:
                arrays = load_grib(filename, self.array_names)

        return arrays, target
        