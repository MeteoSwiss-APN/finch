import os
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr

from .. import cfg, data

brn_array_names = ["P", "T", "QV", "U", "V", "HHL", "HSURF"]
"""The names of the brn input arrays"""
thetav_array_names = brn_array_names[:3]
"""The names of the thetav input arrays"""
brn_only_array_names = brn_array_names[3:]
"""The names of the brn input arrays which are not used for the thetav computation"""


grib_input_version = data.Input.Version(
    format=data.Format.GRIB,
    dim_order="zyx",
    chunks={"z": 1, "x": -1, "y": -1},
    coords=True,
)


def load_input_grib(version: data.Input.Version | None = None) -> xr.Dataset:
    if version is None:
        version = grib_input_version

    if version.format == data.Format.FAKE:
        # create a fake dataset
        shape = {"x": 1170, "y": 700, "z": 80}
        assert version.dim_order is not None
        dims = list(version.dim_order)
        size = [shape[d] for d in dims]
        chunks_list = [version.chunks[d] if d in version.chunks else shape[d] for d in dims]
        array = da.random.random(size, chunks_list)
        array = xr.DataArray(array, dims=dims)
        arrays: list[xr.DataArray] = [
            (array + x).rename(n) for n, x in zip(brn_array_names, np.arange(0, 0.1 * len(brn_array_names), 0.1))
        ]
        arrays[-1] = arrays[-1].loc[{"z": 0}]
        return xr.merge(arrays)

    chunks = dict(version.chunks)

    # load data from first grib file
    grib_file = "lfff00000000"
    short_names = ["P", "T", "QV", "U", "V"]
    args: dict[str, Any] = {
        "chunks": chunks,
        "index_path": os.path.join(cfg["brn"]["grib_index_dir"], grib_file + ".idx"),
        "cache": False,
        "key_filters": {"typeOfLevel": "generalVerticalLayer"},
        "load_coords": False,
    }
    out1 = data.load_grib(grib_file, short_names, **args)

    # load data from second grib file
    grib_file = "lfff00000000c"
    args["index_path"] = os.path.join(cfg["brn"]["grib_index_dir"], grib_file + ".idx")
    if "generalVerticalLayer" in chunks:
        chunks["generalVertical"] = chunks.pop("generalVerticalLayer")
    args["chunks"] = chunks
    args["key_filters"]["typeOfLevel"] = "generalVertical"
    hhl = data.load_array_grib(grib_file, "HHL", **args)
    del args["key_filters"]["typeOfLevel"]
    hsurf = data.load_array_grib(grib_file, "HSURF", **args)
    hhl = hhl.rename({"generalVertical": "generalVerticalLayer"})
    hhl = hhl[:-1, :, :]  # TODO shouldn't be necessary

    out = xr.merge([out1, hhl, hsurf])
    out.rename({"generalVerticalLayer": "z"})
    return out


def get_brn_input() -> data.Input:
    """
    Returns an input object for brn functions.

    Group:
        BRN
    """
    return data.Input(
        name="brn",
        source=load_input_grib,
        source_version=grib_input_version,
    )
