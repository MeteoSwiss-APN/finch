import chunk
import os
from .. import data
from .. import config
import xarray as xr
import dask.array as da
import numpy as np
from .. import util

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

inv_dim_index = util.inverse(dim_index)

def translate_order(order):
    return data.translate_order(order, dim_index)

def reorder_dims(input, dims):
    if isinstance(dims, str):
        dims = translate_order(dims)
    return data.reorder_dims(input, dims)

grib_input_version = data.Input.Version(
    format=data.Format.GRIB,
    dim_order="zyx",
    chunks={"z" : 1, "x" : -1, "y" : -1},
    coords=True
)

def load_input_grib(version: data.Input.Version = None) -> xr.Dataset:
    if version is None:
        version = grib_input_version

    if version.format == data.Format.FAKE:
        # create a fake dataset
        shape = {"x": 1170, "y": 700, "z": 80}
        dims = list(version.dim_order)
        size = [shape[d] for d in dims]
        chunks = [version.chunks[d] if d in version.chunks else shape[d] for d in dims]
        array = da.random.random(size, chunks)
        array = xr.DataArray(array, dims=translate_order(version.dim_order))
        arrays: list[xr.DataArray] = [
            (array + x).rename(n) 
            for n, x 
            in zip(brn_array_names, np.arange(0, 0.1*len(brn_array_names), 0.1))
        ]
        arrays[-1] = arrays[-1].loc[{dim_index["z"]: 0}]
        return xr.merge(arrays)

    chunks = util.map_keys(version.chunks, dim_index)

    # load data from first grib file
    grib_file = "lfff00000000"
    short_names = ["P", "T", "QV", "U", "V"]
    args = {
        "chunks": chunks,
        "key_filters": {},
        "index_path": os.path.join(config["brn"]["grib_index_dir"], grib_file + ".idx"),
        "cache": False,
        "key_filters": {"typeOfLevel": "generalVerticalLayer"},
        "load_coords": False
    }
    out1 = data.load_grib(grib_file, short_names, **args)

    # load data from second grib file
    grib_file = "lfff00000000c"
    args["index_path"] = os.path.join(config["brn"]["grib_index_dir"], grib_file + ".idx")
    if "generalVerticalLayer" in chunks:
        chunks["generalVertical"] = chunks.pop("generalVerticalLayer")
    args["chunks"] = chunks
    args["key_filters"]["typeOfLevel"] = "generalVertical"
    hhl = data.load_array_grib(grib_file, "HHL", **args)
    del args["key_filters"]["typeOfLevel"]
    hsurf = data.load_array_grib(grib_file, "HSURF", **args)
    hhl = hhl.rename({"generalVertical": "generalVerticalLayer"})
    hhl = hhl[:-1, :, :] # TODO shouldn't be necessary

    out = xr.merge([out1, hhl, hsurf])
    return out

brn_input = data.Input(
    config["data"]["input_store"],
    name="brn",
    source=load_input_grib,
    source_version=grib_input_version,
    dim_index=dim_index
)
"""
Defines the input for brn functions
"""
