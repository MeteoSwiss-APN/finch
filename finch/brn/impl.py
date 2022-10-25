from math import sqrt
from typing import Callable
import xarray as xr
import dask.array as da
import dask
import numpy as np

import zebra
from .. import constants as const
from .. import util
from . import input
from .. import data

def thetav_xr(dataset: xr.Dataset) -> xr.DataArray:
    p, t, qv = [dataset[n] for n in input.thetav_array_names]

    pc_rvd = const.PC_R_V / const.PC_R_D
    pc_rdocp = const.PC_R_D/const.PC_CP_D
    pc_rvd_o = pc_rvd - 1.0

    return (const.P0 / p) ** pc_rdocp * t * (1.+(pc_rvd_o*qv / (1.-qv)))

def brn_xr(dataset: xr.Dataset) -> xr.DataArray:
    nlevels = dataset.sizes["generalVerticalLayer"]

    thetav = thetav_xr(dataset.drop_vars(input.brn_only_array_names))
    thetav_sum = thetav.isel(generalVerticalLayer=slice(None, None, -1)).cumsum(dim='generalVerticalLayer')
    nlevels_xr =xr.DataArray(data=np.arange(nlevels,0,-1), dims=["generalVerticalLayer"])

    u, v, hhl, hsurf = [dataset[n] for n in input.brn_only_array_names]

    brn_1 = const.PC_G * (hhl-hsurf)*(thetav - thetav.isel(generalVerticalLayer=-1)) * nlevels_xr
    brn_2 = (thetav_sum)*(u*u + v*v)

    brn = brn_1 / brn_2
    return brn

def block_thetav_np(
    p: np.ndarray,
    t: np.ndarray,
    qv: np.ndarray
) -> np.ndarray:
    """
    thetav implementation on numpy chunks
    """
    pc_rvd = const.PC_R_V / const.PC_R_D
    pc_rdocp = const.PC_R_D/const.PC_CP_D
    pc_rvd_o = pc_rvd - 1.0

    return (const.P0 / p) ** pc_rdocp * t * (1.+(pc_rvd_o*qv / (1.-qv)))

def block_brn_np(
    p: np.ndarray, 
    t: np.ndarray, 
    qv: np.ndarray, 
    u: np.ndarray, 
    v: np.ndarray, 
    hhl: np.ndarray, 
    hsurf: np.ndarray
) -> np.ndarray:
    """
    BRN implementation on numpy chunks.
    Required dimension order: xyz
    """
    nlevels = p.shape[2]
    hsurf = np.expand_dims(hsurf, axis=2)

    thetav = block_thetav_np(p, t, qv)
    thetav_sum = np.flip(thetav, 2).cumsum(axis=2)
    nlevels_da = np.reshape(np.arange(nlevels,0,-1), (1, 1, nlevels))

    brn_1 = const.PC_G * (hhl-hsurf)*(thetav - thetav[:, :, -1:]) * nlevels_da
    brn_2 = (thetav_sum)*(u*u + v*v)

    brn = brn_1 / brn_2
    return brn

def thetav_blocked_np(dataset: xr.Dataset) -> xr.DataArray:
    """
    thetav implementation using `custom_map_blocks` and numpy arrays
    """
    arrays = [dataset[n] for n in input.brn_array_names[:3]]
    template = xr.DataArray(arrays[0].data, coords=arrays[0].coords, dims=arrays[0].dims)
    return util.custom_map_blocks(block_thetav_np, *arrays, name="thetav", template=template)

def brn_blocked_np(dataset: xr.Dataset) -> xr.DataArray:
    """
    brn implementation using `custom_map_blocks` and numpy arrays
    """
    dataset = dataset.transpose(*data.translate_order("xyz", input.dim_index)) # ensure correct dimension order
    arrays = [dataset[n] for n in input.brn_array_names]
    template = xr.DataArray(arrays[0].data, coords=arrays[0].coords, dims=arrays[0].dims)
    return util.custom_map_blocks(block_brn_np, *arrays, name="brn", template=template)

def thetav_blocked_cpp(dataset: xr.Dataset) -> xr.DataArray:
    """
    thetav implementation using `custom_map_blocks` and numpy arrays with the zebra backend
    """
    def wrapper(p,t,qv):
        out = np.zeros_like(p)
        zebra.thetav(p, t, qv, out)
        return out
    arrays = [dataset[n] for n in input.brn_array_names[:3]]
    return util.custom_map_blocks(wrapper, *arrays, name="thetav")

def brn_blocked_cpp(dataset: xr.Dataset, reps: int = 1) -> xr.DataArray:
    """
    brn implementation using `custom_map_blocks` and numpy arrays with the zebra backend
    """
    dataset = dataset.transpose(*data.translate_order("xyz", input.dim_index)) # ensure correct dimension order
    def wrapper(*arrays):
        arrays = list(arrays)
        for _ in range(reps):
            out = np.zeros_like(arrays[0])
            zebra.brn(*arrays, out)
            arrays[0] = out
        return out
    arrays = [dataset[n] for n in input.brn_array_names]
    return util.custom_map_blocks(wrapper, *arrays, name="brn")