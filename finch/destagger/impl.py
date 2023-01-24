"""algorithm for destaggering a field."""

import xarray as xr
import numpy as np
from .. import util


def destagger_xr(dataset: xr.Dataset, dim: str = "x") -> xr.Dataset:
    """Xarray implementation of the destagger operator."""
    left = dataset[{dim: slice(0, -1)}]
    right = dataset[{dim: slice(1, None)}]
    return (left + right) * 0.5


def __block_destagger_np(array: np.ndarray, dim: int = 0) -> np.ndarray:
    i = [slice(None)] * array.ndim
    i[dim] = slice(-1)
    left = array[tuple(i)]
    i[dim] = slice(1, None)
    right = array[tuple(i)]
    return (left + right) * 0.5


def destagger_blocked_np(dataset: xr.Dataset, dim: str = "x") -> xr.Dataset:
    """Blockwise numpy implementation of the destagger operator."""
    array = dataset.to_array(dim="tmp")
    dim_index = array.dims.index(dim)
    out = util.custom_map_blocks(
        __block_destagger_np, array, name="destaggered", dim=dim_index
    )
    return out.to_dataset(dim="tmp")
