import numpy as np
import dask.array as da
import xarray as xr
import os

from ..data import Input, Format, load_grib
from .. import config

destagger_array_names = ["U"]
"""The names of the destagger input arrays"""

input_version = Input.Version(
    format=Format.GRIB, dim_order="xyz", chunks={"z": 1, "x": -1, "y": -1}, coords=True
)


def load_input_grib(version: Input.Version | None = None):
    if version is None:
        version = input_version

    if version.format == Format.FAKE:
        # create a fake dataset
        shape = {"x": 1170, "y": 700, "z": 80}
        dims = list(version.dim_order)
        size = [shape[d] for d in dims]
        chunks_list = [
            version.chunks[d] if d in version.chunks else shape[d] for d in dims
        ]
        array = da.random.random(size, chunks_list)
        array = xr.DataArray(array, dims=dims)
        arrays: list[xr.DataArray] = [
            (array + x).rename(n)
            for n, x in zip(
                destagger_array_names,
                np.arange(0, 0.1 * len(destagger_array_names), 0.1),
            )
        ]
        arrays[-1] = arrays[-1].loc[{"z": 0}]
        return xr.merge(arrays)

    chunks = version.chunks

    # load data from first grib file
    grib_file = "lfff00000000"
    short_names = ["U"]
    args = {
        "chunks": chunks,
        "index_path": os.path.join(config["brn"]["grib_index_dir"], grib_file + ".idx"),
        "cache": False,
        "key_filters": {"typeOfLevel": "generalVerticalLayer"},
        "load_coords": False,
    }
    out = load_grib(grib_file, short_names, **args)
    return out


def get_input() -> Input:
    """
    Return an input for the destagger operator.

    Group:
        Destagger
    """
    return Input(name="destagger", source=load_input_grib, source_version=input_version)
