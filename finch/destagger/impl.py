"""algorithm for destaggering a field."""

import xarray as xr


def destagger_xr(dataset: xr.Dataset, dim: str = "x") -> xr.Dataset:
    """Xarray implementation of the destagger operator."""
    left = dataset[{dim: slice(0, -1)}]
    right = dataset[{dim: slice(1, None)}]
    return (left + right) * 0.5
