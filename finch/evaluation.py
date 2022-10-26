from ensurepip import version
import functools
import pathlib
from typing import Any
from collections.abc import Callable
from . import Input
from . import util
from . import config
from .experiments import RunConfig
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

def print_results(results: list[list[Any]], run_configs: list[RunConfig], versions: list[Input.Version]):
    """
    Prints the results of an experiment for different run configurations and input versions.
    """
    for rc, r in zip(run_configs, results):
        print(rc)
        print()
        print_version_results(r, versions)
        print()

def create_result_array(
    results: list[list[float]] | list[float] | float, 
    run_configs: list[RunConfig] | RunConfig, 
    versions: list[Input.Version] | Input.Version, 
    experiment_name: str = None, 
    impl_names: list[str] | None = None
) -> xr.DataArray:
    """
    Constructs a data array from the results of an experiment.
    The dimensions are given by the attributes of the Version and RunConfig classes.
    The coordinates are labels for the version and run config attributes.
    This result array can then be used as an input for different evaluation functions.
    The result array will contain NaN for every combination of version and run config attributes, which is not listed in `versions`.
    """
    # prepare arguments
    if not isinstance(run_configs, list):
        run_configs = [run_configs]
        results = [results]
    if not isinstance(versions, list):
        versions = [versions]
        results = [[r] for r in results]

    if experiment_name is None:
        experiment_name = util.random_entity_name()
    def get_attrs(entities: list, cls: type) -> list[dict[str, str]]:
        """Retrurns a list of dictionaries with string representations of the class attributes of the given entities."""
        # construct flattened attribute dictionary
        attrs: list[dict] = [
            util.flatten_dict({
                a: e.__dict__[a] 
                for a in cls.get_class_attr()
            }) 
            for e in entities
        ]
        # transform non-numeric attributes to strings
        out = []
        for attr_d in attrs:
            out_d = dict()
            for k, v in attr_d.items():
                if isinstance(v, Callable): # special case: for better readability we use the function name
                    if isinstance(v, functools.partial):
                        v_str = v.func.__name__
                        v_str += "_" + "_".join(str(a) for a in v.args)
                        v_str += "_" + "_".join(k + "=" + str(v) for k, v in v.keywords.items())
                        v = v_str
                    else:
                        v = v.__name__
                elif not isinstance(v, numbers.Number):
                    v = str(v)
                out_d[k] = v
            out.append(out_d)
        return out

    # get attributes from run configs and versions
    version_attrs = get_attrs(versions, Input.Version)
    va_keys = list(version_attrs[0].keys())
    rc_attrs = get_attrs(run_configs, RunConfig)
    if impl_names is not None:
        # set implementation names
        for a, impl_name in zip(rc_attrs, impl_names):
            a["impl"] = impl_name
    rca_keys = list(rc_attrs[0].keys())
    # construct coordinates
    coords = {
        a : list(set(va[a] for va in version_attrs))
        for a in va_keys
    }
    coords.update({
        a : list(set(ra[a] for ra in rc_attrs))
        for a in rca_keys
    })
    # sort numeric coordinates
    for a in va_keys + rca_keys:
        if isinstance(coords[a][0], numbers.Number):
            coords[a] = sorted(coords[a])

    # initialize data
    dim_sizes = [len(coords[a]) for a in coords]
    data = np.full(dim_sizes, np.nan, dtype=float)

    # create array
    array = xr.DataArray(data, coords, name=experiment_name)
    for result, rca in zip(results, rc_attrs):
        for r, va in zip(result, version_attrs):
            array.loc[va | rca] = r
    return array

def create_plots(results: xr.DataArray, reduction: Callable = np.nanmin, normalize_lines: bool = False):
    """
    Creates a series of plots for the results array.
    The plot creation works as follows.
    Every plot has multiple different lines, which correspond to the different implementations.
    The y-axis indicates the value of the result, while the x-axis is a dimension of the result array.
    For every dimension of size greater than 1, except for the 'imp' dimension, a new plot will be created.
    The other dimensions will then be reduced by flattening and then reducing according to the given reduction function.

    If the coordinates of a dimension have type `str`, a bar plot will be generated. Otherwise a standard line plot will be saved.
    The plots will be stored in config's `plot_dir` in a directory according to the experiment name.

    Arguments
    ---
    - results: xr.DataArray. The result array
    - reduction: Callable. The reduction function. (See `xarray.DataArray.reduce`)
    - normalize_lines: bool. If true, the runtimes for an implementation in a line plot will be 
    divided by the longest runtime of said implementation.
    This should be enabled if the progression of the runtime should be compared instead of the absolute runtime.
    """

    path = pathlib.Path(config["evaluation"]["plot_dir"], results.name)
    path.mkdir(parents=True, exist_ok=True)

    for d in results.dims:
        if d != "impl" and results.sizes[d] > 1:
            to_reduce = [dd for dd in results.dims if dd != "impl" and dd != d]
            to_plot = results.reduce(reduction, to_reduce)
            if normalize_lines and not isinstance(results[d].data[0], str):
                for i in results.coords["impl"].data:
                    to_plot.loc[dict(impl=i)] /= to_plot.sel(impl=i).max()
            ticks = to_plot.coords[d].data
            df = pd.DataFrame({
                i: to_plot.sel(impl=i).data for i in results.coords["impl"].data
            } | {"ticks": ticks})
            plotargs = dict(
                x="ticks",
                xlabel=d,
                ylabel="Runtime [s]"
            )
            if isinstance(ticks[0], str):
                plot = df.plot(
                    kind="bar", 
                    stacked=True,
                    **plotargs
                )
            else:
                plot = df.plot(
                    kind = "line",
                    xticks=ticks,
                    **plotargs
                )
            plot.get_figure().savefig(path.joinpath(d + ".png"), format="png")
