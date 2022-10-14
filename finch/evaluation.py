from ensurepip import version
import pathlib
from typing import Any
from collections.abc import Callable
from . import Input
from . import util
from . import config
import xarray as xr
import numpy as np
import numbers
import pandas as pd


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
    The result array will contain NaN for every combination of version attributes, which is not listed in `versions`.
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
    data = np.full(dim_sizes, np.nan, dtype=float)

    # create array
    array = xr.DataArray(data, coords, name=experiment_name)
    for result, i in zip(results, imps):
        for r, v in zip(result, versions):
            array.loc[{a : v.__dict__[a] for a in version_attr} | {"imp":i}] = r
    return array

def create_plots(results: xr.DataArray, reduction: Callable = np.nanmean):
    """
    Creates a series of plots for the results array.
    The plot creation works as follows.
    Every plot has multiple different lines, which correspond to the different implementations.
    The y-axis indicates the value of the result, while the x-axis is a dimension of the result array.
    For every dimension of size greater than 1, except for the 'imp' dimension, a new plot will be created.
    The other dimensions will then be reduced by flattening and then reducing according to the given reduction function.

    If the coordinates of a dimension have type `str`, a bar plot will be generated. Otherwise a standard line plot will be saved.
    The plots will be stored in config's `plot_dir` in a directory according to the experiment name.
    """

    path = pathlib.Path(config["global"]["plot_dir"], results.name)
    path.mkdir(parents=True, exist_ok=True)

    for d in results.dims:
        if d != "imp" and results.sizes[d] > 1:
            to_reduce = [dd for dd in results.dims if dd != "imp" and dd != d]
            to_plot = results.reduce(reduction, to_reduce)
            df = pd.DataFrame({
                i: to_plot[i] for i in results.coords["imp"]
            })
            if isinstance(to_plot.coords[d][0], str):
                plot = df.plot(
                    kind="bar", 
                    stacked=True,
                    xticks=to_plot.coords[d]
                )
            else:
                plot = df.plot(
                    kind = "line",
                    xticks = to_plot.coords[d]
                )
            plot.get_figure().savefig(path.joinpath(d + ".svg"), format="svg")
