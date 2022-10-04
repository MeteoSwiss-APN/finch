from typing import List, Dict, Any
import xarray as xr
import dask.array as da
import numpy as np

grib_dir = '/scratch/cosuna/postproc_np_products/grib_files/cosmo-1e/'
"""The base directory from where to load grib files"""
netcdf_dir = "/scratch/thoerman/netcdf/"
"""The base directory from where to load and store netCDF files"""
zarr_dir = "/scratch/thoerman/zarr/"
"""The base directory from where to load and store zarr files"""
tmp_dir = "/scratch/thoerman/tmp"
"""A directory which can be used as a temporary storage"""

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
    index_path: str = "tmp/grib.idx",
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

    if isinstance(path, list) or '*' in path:
        dataset = xr.open_mfdataset([grib_dir + p for p in path], parallel=parallel, **args)
    else:
        args["indexpath"] = index_path
        dataset = xr.open_dataset(grib_dir + path, **args)
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
    ds: xr.Dataset = xr.open_dataset(netcdf_dir + filename, chunks=chunks)
    return list(ds.data_vars.values())

def load_zarr(filename, dim_names: List[str], chunks: List[int] = "auto", names: List[str] = None, inline_array = True) -> List[xr.DataArray]:
    """
    Loads the content of a zarr directory into `DataArray`(s).
    """
    def single_load(component):
        array = da.from_zarr(url=zarr_dir+filename, chunks=chunks, inline_array=inline_array, component=component)
        return xr.DataArray(array, dims=dim_names, name=component)
    if names:
        return [single_load(c) for c in names]
    else:
        return single_load(None)

def store_netcdf(
    filename: str, 
    data: List[xr.DataArray],
    names: List[str]):
    """
    Stores a list of `DataArray`s into a netCDF file.

    Arguments:
    ---
    - filename: str. The name of the netCDF file.
    - data: List[DataArray]. The `DataArray`s to be stored.
    - names: List[str]. A list of names for the arrays to be stored.
    """
    out = xr.Dataset(dict(zip(names, data)))
    out.to_netcdf(netcdf_dir+filename)

def store_zarr(dirname: str, arrays: List[xr.DataArray]):
    """
    Stores a list of `DataArray`s into a zarr directory.

    Arguments:
    ---
    - filename: str. The name of the zarr directory.
    - data: List[DataArray]. The `DataArray`s to be stored.
    """
    if type(arrays) is not list:
        arrays = [arrays]
    for a in arrays:
        da.to_zarr(a.data, zarr_dir+dirname, a.name, overwrite=True)

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