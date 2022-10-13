import os
from .. import data
from .. import config
from . import impl
from typing import List
import xarray as xr

brn_array_names = ["P", "T", "QV", "U", "V", "HHL", "HSURF"]
"""The names of the brn input arrays"""
thetav_array_names = brn_array_names[:3]
"""The names of the thetav input arrays"""
brn_only_array_names = brn_array_names[3:]
"""The names of the brn input arrays which are not used for the thetav computation"""
dim_index = {
    "x": "x",
    "y": "y",
    "z": "generalVerticalLayer"
}
"""Dictionary for translating dimension names"""

def translate_order(order):
    return data.translate_order(order, dim_index)

def reorder_dims(input, dims):
    if isinstance(dims, str):
        dims = translate_order(dims)
    return data.reorder_dims(input, dims)

def load_input_grib(chunk_size=None, horizontal_chunk_size=None) -> xr.Dataset:
    chunks = {"generalVerticalLayer": chunk_size} if chunk_size else chunk_size
    if horizontal_chunk_size:
        chunks.update({"x": horizontal_chunk_size})

    # load data from first grib file
    grib_file = "lfff00000000"
    short_names = ["P", "T", "QV", "U", "V"]
    args = {
        "chunks": chunks,
        "key_filters": {},
        "index_path": config["brn"]["grib_index_dir"] + grib_file + ".idx",
        "cache": False,
        "key_filters": {"typeOfLevel": "generalVerticalLayer"},
        "load_coords": False
    }
    out1 = data.load_grib(grib_file, short_names, **args)

    # load data from second grib file
    grib_file = "lfff00000000c"
    args["index_path"] = config["brn"]["grib_index_dir"] + grib_file + ".idx"
    if chunk_size:
        chunks["generalVertical"] = chunks.pop("generalVerticalLayer")
        args["chunks"] = chunks
    args["key_filters"]["typeOfLevel"] = "generalVertical"
    hhl = data.load_array_grib(grib_file, "HHL", **args)
    del args["key_filters"]["typeOfLevel"]
    hsurf = data.load_array_grib(grib_file, "HSURF", **args)
    hhl = hhl.rename({"generalVertical": "generalVerticalLayer"})
    hhl = hhl[:-1, :, :] # TODO shouldn't be necessary

    arrays = out1 + [hhl, hsurf]
    return xr.Dataset({n:a for n, a in zip(brn_array_names, arrays)})

brn_input = data.Input(
    config["global"]["data_store"],
    name="brn",
    source=lambda : load_input_grib(chunk_size=1),
    source_version=data.Input.Version(
        format=data.Format.GRIB,
        dim_order="zyx",
        chunks={dim_index["z"] : 1},
        name=None
    ),
    dim_index=dim_index,
    array_names=brn_array_names
)
"""
Defines the input for brn functions
"""
