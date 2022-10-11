import os
from .. import data
from . import impl
from collections import defaultdict

# set the grib definition path. Must be done before importing xarray
grib_definitions_path = "/project/g110/spack-install/tsa/cosmo-eccodes-definitions/2.19.0.7/gcc/zcuyy4uduizdpxfzqmxg6bc74p2skdfp/cosmoDefinitions/definitions/:/project/g110/spack-install/tsa/eccodes/2.19.0/gcc/viigacbsqxbbcid22hjvijrrcihebyeh/share/eccodes/definitions/"
"""Path to the grib definitions to use."""
os.environ["GRIB_DEFINITION_PATH"] = grib_definitions_path

input_array_names = ["P", "T", "QV", "U", "V", "HHL", "HSURF"]
"""The names of the brn input arrays"""
dim_index = {
    "x": "x",
    "y": "y",
    "z": "generalVerticalLayer"
}
"""Dictionary for translating dimension names"""

from io import UnsupportedOperation
from typing import List
import xarray as xr

def translate_order(order):
    return data.translate_order(order, dim_index)

def reorder_dims(input, dims):
    if isinstance(dims, str):
        dims = translate_order(dims)
    return data.reorder_dims(input, dims)

def load_input_grib(chunk_size=None, horizontal_chunk_size=None):
    chunks = {"generalVerticalLayer": chunk_size} if chunk_size else chunk_size
    if horizontal_chunk_size:
        chunks.update({"x": horizontal_chunk_size})

    # load data from first grib file
    grib_file = "/lfff00000000"
    short_names = ["P", "T", "QV", "U", "V"]
    args = {
        "chunks": chunks,
        "key_filters": {},
        "index_path": "notebook/tmp/grib1.idx",
        "cache": False,
        "key_filters": {"typeOfLevel": "generalVerticalLayer"},
        "load_coords": False
    }
    out1 = data.load_grib(grib_file, short_names, **args)

    # load data from second grib file
    grib_file = "/lfff00000000c"
    args["index_path"] = "notebook/tmp/grib2.idx"
    if chunk_size:
        chunks["generalVertical"] = chunks.pop("generalVerticalLayer")
        args["chunks"] = chunks
    args["key_filters"]["typeOfLevel"] = "generalVertical"
    hhl = data.load_array_grib(grib_file, "HHL", **args)
    del args["key_filters"]["typeOfLevel"]
    hsurf = data.load_array_grib(grib_file, "HSURF", **args)
    hhl = hhl.rename({"generalVertical": "generalVerticalLayer"})
    hhl = hhl[:-1, :, :] # TODO shouldn't be necessary

    return out1 + [hhl, hsurf]

def load_input(format: data.Format = data.Format.GRIB, dim_order: str = "xyz", data_cube: bool = False, chunk_size: int = 30) -> List[xr.DataArray]:
    """
    Loads the input for the brn computation
    """
    dim_names = list(dim_order)
    dim_names[dim_names.index("z")] = "generalVerticalLayer"
    
    if data_cube:
        if format == data.Format.ZARR:
            cube = data.load_zarr(f"brn_data_{dim_order}_cube", dim_names=dim_names, chunks=[chunk_size, -1, -1], names=["data"], inline_array=True)[0]
        elif format == data.Format.NETCDF:
            cube = data.load_netcdf(f"brn_data_{dim_order}_cube.nc", chunks={"x": 30})[0]
        else:
            raise UnsupportedOperation()
        arrays = data.split_cube(cube, split_dim = "generalVerticalLayer", splits = [80]*6 + [1])
    else:
        if format == data.Format.ZARR:
            arrays = data.load_zarr(f"brn_data_{dim_order}_da", dim_names=dim_names, names=input_array_names[:-1], chunks=[chunk_size, -1, -1])
            arrays += data.load_zarr(f"brn_data_{dim_order}_da", dim_names=["x", "y"], names=input_array_names[-1:], chunks=[chunk_size, -1])
        elif format == data.Format.NETCDF:
            arrays = data.load_netcdf(f"brn_data_{dim_order}.nc", chunks={"x": 30})
        elif format == data.Format.GRIB:
            dim_order = "zxy"
            arrays = load_input_grib(chunk_size=1)
        else:
            raise UnsupportedOperation()
    return arrays
