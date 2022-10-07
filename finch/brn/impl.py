from typing import Callable
import xarray as xr
import dask.array as da
import dask
import numpy as np
import zebra
from .. import constants as const
from .. import util
from .. import data
from . import input

def thetav_xr(
    p: xr.DataArray,
    t: xr.DataArray,
    qv: xr.DataArray
    ) -> xr.DataArray:

    pc_rvd = const.PC_R_V / const.PC_R_D
    pc_rdocp = const.PC_R_D/const.PC_CP_D
    pc_rvd_o = pc_rvd - 1.0

    return (const.P0 / p) ** pc_rdocp * t * (1.+(pc_rvd_o*qv / (1.-qv)))

def brn_xr(p: xr.DataArray, 
    t: xr.DataArray,
    qv: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    hhl: xr.DataArray,
    hsurf: xr.DataArray, 
    client: dask.distributed.Client = None
    ) -> xr.DataArray:
    
    nlevels = len(p.coords["generalVerticalLayer"])

    thetav = thetav_xr(p,t,qv)
    if client:
        thetav = client.persist(thetav)
    thetav_sum = thetav.isel(generalVerticalLayer=slice(None, None, -1)).cumsum(dim='generalVerticalLayer')

    nlevels_xr =xr.DataArray(data=np.arange(nlevels,0,-1), dims=["generalVerticalLayer"])

    brn_1 = const.PC_G * (hhl-hsurf)*(thetav - thetav.isel(generalVerticalLayer=79)) * nlevels_xr
    brn_2 = (thetav_sum)*(u*u + v*v)

    brn = brn_1 / brn_2
    return brn

def block_thetav_np(p: np.ndarray, t: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """
    thetav implementation on numpy chunks
    """
    pc_rvd = const.PC_R_V / const.PC_R_D
    pc_rdocp = const.PC_R_D/const.PC_CP_D
    pc_rvd_o = pc_rvd - 1.0

    return (const.P0 / p) ** pc_rdocp * t * (1.+(pc_rvd_o*qv / (1.-qv)))

def block_brn_np(p: np.ndarray, t, qv, u, v, hhl, hsurf) -> np.ndarray:
    """
    BRN implementation on numpy chunks.
    Required dimension order: xyz
    """
    nlevels = p.shape[2]
    hsurf = np.expand_dims(hsurf, axis=2)

    thetav = block_thetav_np(p, t, qv)
    thetav_sum = np.flip(thetav, 2).cumsum(axis=2)
    nlevels_da = np.reshape(np.arange(nlevels,0,-1), (1, 1, nlevels))

    brn_1 = const.PC_G * (hhl-hsurf)*(thetav - thetav[:, :, 79:80]) * nlevels_da
    brn_2 = (thetav_sum)*(u*u + v*v)

    brn = brn_1 / brn_2
    return brn

def thetav_blocked_np(
    p: xr.DataArray,
    t: xr.DataArray,
    qv: xr.DataArray
    ) -> xr.DataArray:
    """
    thetav implementation using `custom_map_blocks` and numpy arrays
    """
    return util.custom_map_blocks(block_thetav_np, p, t, qv, name="thetav")

def brn_blocked_np(p: xr.DataArray, 
    t: xr.DataArray,
    qv: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    hhl: xr.DataArray,
    hsurf: xr.DataArray, 
    ) -> xr.DataArray:
    """
    brn implementation using `custom_map_blocks` and numpy arrays
    """
    arrays = input.reorder_dims([p, t, qv, u, v, hhl, hsurf], "xyz") # ensure correct dimension order
    return util.custom_map_blocks(block_brn_np, *arrays, name="brn")

def thetav_blocked_cpp(
    p: xr.DataArray,
    t: xr.DataArray,
    qv: xr.DataArray
    ) -> xr.DataArray:
    """
    thetav implementation using `custom_map_blocks` and numpy arrays with the zebra backend
    """
    def wrapper(p,t,qv):
        out = np.zeros_like(p)
        zebra.thetav(p, t, qv, out)
        return out
    return util.custom_map_blocks(wrapper, p, t, qv, name="thetav")

def brn_blocked_cpp(p: xr.DataArray, 
    t: xr.DataArray,
    qv: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    hhl: xr.DataArray,
    hsurf: xr.DataArray
    ) -> xr.DataArray:
    """
    brn implementation using `custom_map_blocks` and numpy arrays with the zebra backend
    """
    def wrapper(*arrays):
        out = np.zeros_like(arrays[0])
        zebra.brn(*arrays, out)
        return out
    return util.custom_map_blocks(wrapper, p, t, qv, u, v, hhl, hsurf, name="brn")