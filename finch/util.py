from ast import arg
from contextlib import closing
import socket
from typing import Callable, Dict, List
import dask.array as da
import xarray as xr
import numpy as np

def adjust_dims(dims: List[str], array: xr.DataArray) -> xr.DataArray:
    """
    Returns a new DataArray with the same content as `array` such that the dimensions match `dims` in content and order.
    This is achieved with a combination of `expand_dims`, `squeeze` and `transform`.
    When trying to remove dimensions with sizes larger than 1, an error will be thrown.
    """
    # Performance might be improved. We could check whether the dimensions in the array are in correct order.
    # If so, we wouldn't need a transpose. Since we are however working with views, this might not even be faster.
    to_remove = set(array.dims).difference(dims)
    to_add = set(dims).difference(array.dims)
    return array.squeeze(to_remove).expand_dims(list(to_add)).transpose(*dims)

def custom_map_blocks(f: Callable, *args: xr.DataArray, 
    name: str = None, dtype = None, template: xr.DataArray = None, f_in_out_type = np.ndarray, 
    **kwargs) -> xr.DataArray:
    """
    Custom implementation of map_blocks from dask for xarray data arrays based on dask's `blockwise` and `map_blocks` functions.

    Arguments:
    ---
    - f: Callable. The function to be executed on the chunks of `args`.
    The input / output of the chunks can be controlled with the `f_in_out_type` argument.
    - args: xarray.DataArray. The data array arguments to `f`.
    The first element will be used as a template for the output where 
    - name: str, optional. The name of the output array. Defaults to the name of `f`.
    - dtype: type, optional. The type of the elements of the output array. Defaults to the dtype of `template`.
    - template: xr.DataArray, optional. A template array used to determine some characteristics of the output array.
    The coordinates, attributes and the dimensions will be copied from the template into the output array.
    Additionally the template's data will be used to determine the type (e.g. dask array) and the dtype (e.g. np.float64) of the data of the output.
    The content of the data is irrelevant (the array can be empty).
    Defaults to using the first element of `args` as a template argument.
    - f_in_out_type: optional. The type of the input arrays for `f`.
    Can be either `numpy.ndarray`, `dask.array.Array` or `xarray.DataArray`, providing a performance penalty increasing in this order.
    Note that this parameter only serves as a convenience to support the different interfaces for data handling.
    The data chunks must still fully fit into memory.
    Defaults to `numpy.ndarray`.
    """

    if template is None:
        template = args[0]
    if dtype is None:
        dtype = template.dtype
    if name is None:
        name = f.__name__
    # identify the individual dimensions while keeping the order in which they occur
    dims = list(dict.fromkeys([d for a in args for d in a.dims]))

    if f_in_out_type is xr.DataArray:
        # we need to use map_blocks here since it supports the block_info argument, 
        # which we use to construct coordinates for the chunks
        # map_blocks expects all arguments to have the same number of dimensions
        xr_args = [adjust_dims(dims, a).data for a in args]
        # a helper function for retrieving the coordinates of a data array chunk
        def get_chunk_coords(array_location: List[slice], array: xr.DataArray) -> Dict[str, xr.DataArray]:
            dim_ind_map = {d:i for d, i in zip(array.dims, range(len(array.dims)))}
            coord_dims = set(array.dims).intersection(array.coords.keys())
            # add non-index coordinates
            out = {d : array.coords[d] for d in array.coords.keys() if d not in coord_dims}
            # add index coordinates
            for d in coord_dims:
                s = array_location[dim_ind_map[d]]
                c = array.coords[d][s]
                out[d] = c
            return out
        # This function wraps `f` such that it can accept numpy arrays.
        # It creates xr.DataArrays from the numpy arrays with the appropriate metadata before calling `f`.
        # Afterwards, the underlying numpy array is extracted from the data array.
        def xr_wrap(*chunks: np.ndarray, block_info, **kwargs):
            xr_chunks = [
                    adjust_dims( # readjust dimensions of the chunk according to the dimensions of the full array
                        a.dims,
                        xr.DataArray(c, coords=get_chunk_coords(info["array-location"], a), dims=dims, attrs=a.attrs)
                    )
                    for c, info, a in zip(chunks, block_info.values(), args)
            ]
            return np.array(f(*xr_chunks, **kwargs).data)
        # run map_blocks
        out = da.map_blocks(xr_wrap, *xr_args, name=name, dtype=dtype, meta=template.data, **kwargs)
    else:
        # we can use da.blockwise for dask and numpy arrays, which reduces some overhead compared to map_blocks
        # da.blockwise requires array-index pairs, which we can easily generate from the dimension names
        index_map = {k:v for k, v in zip(dims, range(len(dims)))}
        index = [tuple([index_map[d] for d in a.dims]) for a in args]
        out_ind = index[0]
        da_args = [a.data for a in args]
        block_args = [x for z in zip(da_args, index) for x in z]

        if f_in_out_type is da.Array:
            # wrap `f` for numpy array in- and output
            f = lambda *chunks, **kwargs: np.array(f(*[da.Array(c) for c in chunks], **kwargs))
        # run da.blockwise
        out = da.blockwise(f, out_ind, *block_args, name=name, dtype=dtype, meta=template.data, **kwargs)
    # out is now a dask array, which should be converted to an xarray data array
    return xr.DataArray(out, name=name, attrs=template.attrs, coords=template.coords, dims=template.dims)

def check_socket_open(host: str = "localhost", port: int = 80) -> bool:
    """Returns whether a port is in use / open (`True`) or not (`False`)."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0
