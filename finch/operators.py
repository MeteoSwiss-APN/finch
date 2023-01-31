"""
This module adds class-, function- and type definitions for implementating new operators.
"""

from typing import Callable, Hashable

import dask.array as da
import numpy as np
import xarray as xr

from . import data


def custom_map_blocks(
    f: Callable,
    *args: xr.DataArray | xr.Dataset,
    name: str | None = None,
    dtype=None,
    template: xr.DataArray | None = None,
    f_in_out_type=np.ndarray,
    **kwargs
) -> xr.DataArray:
    """Custom implementation of map_blocks from dask for xarray data arrays based on
    dask's ``blockwise`` and ``map_blocks`` functions.

    Args:
        f (Callable):
            The function to be executed on the chunks of `args`.
            The input / output of the chunks can be controlled
            with the `f_in_out_type` argument.
        args (xr.DataArray | xr.Dataset ...):
            The data array arguments to ``f``.
            If a dataset is passed, its non-coordinate variables
            will be extracted and used as inputs to `f`.
            The first element will be used as a template for the
            output if no explicit template was passed.
        name (str | None, optional):
            The name of the output array. Defaults to the name of `f`.
            Defaults to None.
        dtype (_type_, optional):
            The type of the elements of the output array.
            Defaults to None, which indicates the dtype of ``template``.
        template (xr.DataArray | None, optional):
            A template array used to determine some characteristics of the output array.
            The coordinates, attributes and the dimensions will be
            copied from the template into the output array.
            Additionally the template's data will be used to determine the type
            (e.g. dask array) and the dtype (e.g. np.float64) of the data of the output.
            The content of the data is irrelevant (the array can be empty).
            Defaults to None, which means using the first element of
            `args` as a template argument.
        f_in_out_type (_type_, optional):
            The type of the input arrays for ``f``.
            Can be either ``numpy.ndarray``, ``dask.array.Array`` or
            ``xarray.DataArray``,
            providing a performance penalty increasing in this order.
            Note that this parameter only serves as a convenience to
            support the different interfaces for data handling.
            The data chunks must still fully fit into memory.
            Defaults to ``np.ndarray``.

    Returns:
        xr.DataArray: The result of the blockwise computation.

    Group:
        Operators
    """

    # extract arrays
    if isinstance(args[0], xr.Dataset):
        assert len(args) == 1, "Only one dataset can be passed."
        dataset = args[0]
        arrays = list(dataset.data_vars.values())
    else:
        arrays = [a for a in args if isinstance(a, xr.DataArray)]

    # handle optional arguments
    if template is None:
        template = arrays[0].copy()
    if dtype is None:
        dtype = template.dtype
    if name is None:
        name = f.__name__
    # identify the individual dimensions while keeping the order in which they occur
    dims = list(dict.fromkeys([d for a in arrays for d in a.dims]))

    if f_in_out_type is xr.DataArray:
        # we need to use map_blocks here since it supports the block_info argument,
        # which we use to construct coordinates for the chunks
        # map_blocks expects all arguments to have the same number of dimensions
        xr_args = [data.adjust_dims(dims, a).data for a in arrays]

        # a helper function for retrieving the coordinates of a data array chunk
        def get_chunk_coords(array_location: list[slice], array: xr.DataArray) -> dict[Hashable, xr.DataArray]:
            dim_ind_map = {d: i for d, i in zip(array.dims, range(len(array.dims)))}
            coord_dims = set(array.dims).intersection(array.coords.keys())
            # add non-index coordinates
            out = {d: array.coords[d] for d in array.coords.keys() if d not in coord_dims}
            # add index coordinates
            for d in coord_dims:
                s = array_location[dim_ind_map[d]]
                c: xr.DataArray = array.coords[d][s]
                out[d] = c
            return out

        # This function wraps `f` such that it can accept numpy arrays.
        # It creates xr.DataArrays from the numpy arrays
        # with the appropriate metadata before calling `f`.
        # Afterwards, the underlying numpy array is extracted from the data array.
        def xr_wrap(*chunks: np.ndarray, block_info, **kwargs):
            xr_chunks = [
                # readjust dimensions of the chunk according
                # to the dimensions of the full array
                data.adjust_dims(
                    a.dims,
                    xr.DataArray(
                        c,
                        coords=get_chunk_coords(info["array-location"], a),
                        dims=dims,
                        attrs=a.attrs,
                    ),
                )
                for c, info, a in zip(chunks, block_info.values(), arrays)
            ]
            return np.array(f(*xr_chunks, **kwargs).data)

        # run map_blocks
        out = da.map_blocks(xr_wrap, *xr_args, name=name, dtype=dtype, meta=template.data, **kwargs)
    else:
        # we can use da.blockwise for dask and numpy arrays, which reduces some overhead compared to map_blocks
        # da.blockwise requires array-index pairs, which we can easily generate from the dimension names
        index_map = {k: v for k, v in zip(dims, range(len(dims)))}
        index = [tuple([index_map[d] for d in a.dims]) for a in args]
        out_ind = index[0]
        da_args = [a.data for a in args]
        block_args = [x for z in zip(da_args, index) for x in z]

        if f_in_out_type is da.Array:
            # wrap `f` for numpy array in- and output
            f = lambda *chunks, **kwargs: np.array(f(*[da.from_array(c) for c in chunks], **kwargs))  # noqa
        # run da.blockwise
        out = da.blockwise(f, out_ind, *block_args, name=name, dtype=dtype, meta=template.data, **kwargs)
    # out is now a dask array, which should be converted to an xarray data array
    return xr.DataArray(out, name=name, attrs=template.attrs, coords=template.coords, dims=template.dims)
