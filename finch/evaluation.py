from ensurepip import version
from typing import Any
from collections.abc import Callable
from . import Input
from . import util
import xarray as xr
import numpy as np
import numbers


def print_version_results(results: list[Any], versions: list[Input.Version]):
    """
    Prints the results of an experiment for different input versions.
    """
    for r, v in zip(results, versions):
        print(f"{v}\n    {r}")

def print_imp_results(results: list[list[Any]], imps: list[Callable], versions: list[Input.Version]):
    """
    Prints the results of an experiment for different implementations and input versions.
    """
    for imp, r in zip(imps, results):
        print(imp.__name__)
        print()
        print_version_results(r, versions)
        print()

def create_result_array(results: list[list[float]], imps: list[Callable], versions: list[Input.Version], experiment_name: str = None) -> xr.DataArray:
    """
    Constructs a data array from the results of an experiment.
    The dimensions or given by the attributes of the Version class plus one dimension for the different implementations.
    The coordinates are labels for the version attributes / implementations.
    This result array is can then be used as an input for different evaluation functions.
    The result array will contain zeros for every combination of version attributes, which is not listed in `versions`.
    """
    if experiment_name is None:
        experiment_name = util.random_entity_name()
    version_attr = util.get_class_attributes(Input.Version)
    # versions to coordinates
    coords = {
        a : list(set(
            v.__dict__[a]
            for v in versions
        ))
        for a in version_attr
    }
    # transform non-numeric version attributes to strings and sort numeric attributes
    for a in version_attr:
        if not isinstance(coords[a][0], numbers.Number):
            coords[a] = [str(c) for c in coords[a]]
        else:
            coords[a] = sorted(coords[a])
    # add implementations to coordinates
    coords["imp"] = [i.__name__ for i in imps]

    # initialize data
    dim_sizes = [len(coords[a]) for a in coords]
    data = np.zeros(dim_sizes, dtype=float)

    # create array
    array = xr.DataArray(data, coords, name=experiment_name)
    for result, i in zip(results, imps):
        for r, v in zip(result, versions):
            array.loc[{a : v.__dict__[a] for a in version_attr} | {"imp":i}] = r
    return array
    