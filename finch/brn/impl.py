import functools

import dask.array as da
import numpy as np
import xarray as xr

import zebra

from .. import constants as const
from .. import util
from . import input


def thetav_xr(dataset: xr.Dataset) -> xr.DataArray:
    """
    Basic xarray implementation of the thetav operator

    Args:
        dataset: The thetav operator input
    Returns:
        The thetav operator output

    Group:
        THETAV
    """
    p, t, qv = [dataset[n] for n in input.thetav_array_names]

    pc_rvd = const.PC_R_V / const.PC_R_D
    pc_rdocp = const.PC_R_D / const.PC_CP_D
    pc_rvd_o = pc_rvd - 1.0

    return (const.P0 / p) ** pc_rdocp * t * (1.0 + (pc_rvd_o * qv / (1.0 - qv)))


def brn_xr(dataset: xr.Dataset, reps: int = 1) -> xr.DataArray:
    """
    Basic xarray implementation of the brn operator

    Args:
        dataset (xr.Dataset): The brn operator input
        reps (int, optional): The amount of repeated BRN executions. Defaults to 1.

    Returns:
        xr.DataArray: The brn operator output

    Group:
        BRN
    """

    nlevels = dataset.sizes["z"]

    for _ in range(reps):
        thetav = thetav_xr(dataset.drop_vars(input.brn_only_array_names))
        thetav_sum = thetav.isel(z=slice(None, None, -1)).cumsum(dim="z")
        nlevels_xr = da.arange(nlevels, 0, -1, chunks=dataset.chunksizes["z"])
        nlevels_xr = xr.DataArray(data=nlevels_xr, dims=["z"])

        u, v, hhl, hsurf = [dataset[n] for n in input.brn_only_array_names]

        brn_1 = const.PC_G * (hhl - hsurf) * (thetav - thetav.isel(z=-1)) * nlevels_xr
        brn_2 = (thetav_sum) * (u * u + v * v)

        brn = brn_1 / brn_2
        dataset["P"] = brn

    return brn


def __block_thetav_np(p: np.ndarray, t: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """
    thetav implementation on numpy chunks
    """
    pc_rvd = const.PC_R_V / const.PC_R_D
    pc_rdocp = const.PC_R_D / const.PC_CP_D
    pc_rvd_o = pc_rvd - 1.0

    return (const.P0 / p) ** pc_rdocp * t * (1.0 + (pc_rvd_o * qv / (1.0 - qv)))


def __block_brn_np(
    p: np.ndarray,
    t: np.ndarray,
    qv: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    hhl: np.ndarray,
    hsurf: np.ndarray,
    reps: int,
) -> np.ndarray:
    """
    BRN implementation on numpy chunks.
    Required dimension order: xyz
    """
    nlevels = p.shape[2]
    hsurf = np.expand_dims(hsurf, axis=2)

    for _ in range(reps):
        thetav = __block_thetav_np(p, t, qv)
        thetav_sum = np.flip(thetav, 2).cumsum(axis=2)
        nlevels_da = np.reshape(np.arange(nlevels, 0, -1), (1, 1, nlevels))

        brn_1 = const.PC_G * (hhl - hsurf) * (thetav - thetav[:, :, -1:]) * nlevels_da
        brn_2 = (thetav_sum) * (u * u + v * v)

        brn = brn_1 / brn_2
        p = brn
    return brn


def thetav_blocked_np(dataset: xr.Dataset) -> xr.DataArray:
    """
    Blockwise thetav implementation on numpy arrays

    Args:
        dataset: The thetav operator input
    Returns:
        The thetav operator output

    Group:
        THETAV
    """
    arrays = [dataset[n] for n in input.brn_array_names[:3]]
    template = xr.DataArray(
        arrays[0].data, coords=arrays[0].coords, dims=arrays[0].dims
    )
    return util.custom_map_blocks(
        __block_thetav_np, *arrays, name="thetav", template=template
    )


def brn_blocked_np(dataset: xr.Dataset, reps: int = 1) -> xr.DataArray:
    """
    Blockwise brn implementation on numpy arrays

    Args:
        dataset (xr.Dataset): The brn operator input
        reps (int, optional): The amount of repeated BRN executions. Defaults to 1.

    Returns:
        xr.DataArray: The brn operator output

    Group:
        BRN
    """
    dataset = dataset.transpose(*"xyz")  # ensure correct dimension order
    arrays = [dataset[n] for n in input.brn_array_names]
    template = xr.DataArray(
        arrays[0].data, coords=arrays[0].coords, dims=arrays[0].dims
    )
    return util.custom_map_blocks(
        functools.partial(__block_brn_np, reps=reps),
        *arrays,
        name="brn",
        template=template
    )


def thetav_blocked_cpp(dataset: xr.Dataset) -> xr.DataArray:
    """
    Blockwise thetav wrapper for :func:`zebra.brn`.

    Args:
        dataset: The thetav operator input
    Returns:
        The thetav operator output

    Group:
        THETAV
    """

    def wrapper(p, t, qv):
        out = np.zeros_like(p)
        zebra.thetav(p, t, qv, out)
        return out

    arrays = [dataset[n] for n in input.brn_array_names[:3]]
    return util.custom_map_blocks(wrapper, *arrays, name="thetav")


def brn_blocked_cpp(dataset: xr.Dataset, reps: int = 1) -> xr.DataArray:
    """
    Blockwise thetav wrapper for :func:`zebra.brn`.

    Args:
        dataset (xr.Dataset): The brn operator input
        reps (int, optional): The amount of repeated BRN executions. Defaults to 1.

    Returns:
        xr.DataArray: The brn operator output

    Group:
        BRN
    """
    dataset = dataset.transpose(*"xyz")  # ensure correct dimension order

    def wrapper(*arrays):
        arrays = list(arrays)
        out = np.empty_like(arrays[0])
        for _ in range(reps):
            zebra.brn(*arrays, out)
            arrays[0] = out
        return out

    arrays = [dataset[n] for n in input.brn_array_names]
    return util.custom_map_blocks(wrapper, *arrays, name="brn")
