from cProfile import label
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
import pandas as pd
import matplotx
import matplotlib.pyplot as plt


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
    impl_names: list[str] | Callable[[Callable], str] | None = None
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

    # get attributes from run configs and versions
    version_attrs = [util.get_primitive_attrs_from_dataclass(v) for v in versions]
    va_keys = list(version_attrs[0].keys())
    rc_attrs = [util.get_primitive_attrs_from_dataclass(rc) for rc in run_configs]
    rca_keys = list(rc_attrs[0].keys())
    if impl_names is not None:
        # set implementation names
        if isinstance(impl_name, list):
            for a, impl_name in zip(rc_attrs, impl_names):
                a["impl"] = impl_name
        elif isinstance(impl_name, Callable):
            for a, rc in zip(rc_attrs, run_configs):
                a["impl"] = impl_name(rc.impl)
    # construct coordinates
    coords = {
        a : list(set(va[a] for va in version_attrs))
        for a in va_keys
    }
    coords.update({
        a : list(set(ra[a] for ra in rc_attrs))
        for a in rca_keys
    })

    # initialize data
    dim_sizes = [len(coords[a]) for a in coords]
    data = np.full(dim_sizes, np.nan, dtype=float)

    # create array
    array = xr.DataArray(data, coords, name=experiment_name)
    for result, rca in zip(results, rc_attrs):
        for r, va in zip(result, version_attrs):
            array.loc[va | rca] = r
    return array

def create_cores_dimension(
    results: xr.DataArray, 
    contributors: list[str] = [
        "workers",
        "cluster_config_cores_per_worker"
    ],
    cores_dim = "cores",
    reduction: Callable = np.min
) -> xr.DataArray:
    """
    Merges the dimensions in the results array which contribute to the total amount of cores into a single 'cores' dimension.
    The number of cores are calculated by the product of the coordinates of the individual dimensions.
    The resulting dimension is sorted in increasing core order.

    Arguments:
    ---
    - results: The results array
    - contributors: List of dimension names which contribute to the total core count
    - cores_dim: The dimension name of the new 'cores' dimension
    - reduction: How runtimes should be combined which have the same amount of cores
    """
    out = results.stack({cores_dim: contributors})
    coords = out[cores_dim] # this is now a multiindex
    coords = [np.prod(x) for x in coords.data] # calculate the number of cores
    out[cores_dim] = coords
    # reduce cores_dim to have unique values
    coords = np.unique(coords) # output is sorted
    out_cols = [out.loc[{cores_dim : c}] for c in coords]
    out_cols = [c if cores_dim in c.dims else c.expand_dims(cores_dim) for c in out_cols]
    out_cols = [c.reduce(reduction, cores_dim) for c in out_cols]
    out_cols = xr.concat(out_cols, cores_dim)
    out = xr.concat(out_cols, cores_dim)
    out[cores_dim] = coords
    return out

def find_scaling_factor(x: np.ndarray, y: np.ndarray, axis: int = None) -> float:
    """
    Returns the scaling factor for a series of runtime measurements.
    """
    # this is just linear least squares on the log features
    logx = np.log(x)
    logy = np.log(y)
    xd = logx - np.expand_dims(logx.mean(axis=axis), axis)
    yd = logy - np.expand_dims(logy.mean(axis=axis), axis)
    return - np.sum(xd*yd, axis=axis) / np.sum(xd*xd, axis=axis)


def create_plots(
    results: xr.DataArray, 
    reduction: Callable = np.nanmin, 
    scaling_dims: list[str] = ["cores"]
):
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
    - scaling dims: list[str]. Dimensions which are used for scalability plots.
    For those dimensions, a log-log plot will be created and the scaling factor calculated.
    Additionally the runtimes will be normalized to be in the interval [0, 1].
    """

    path = pathlib.Path(config["evaluation"]["plot_dir"], results.name)
    path.mkdir(parents=True, exist_ok=True)

    for d in results.dims:
        if d != "impl" and results.sizes[d] > 1:
            # reduce dimensions other than d and impl
            to_reduce = [dd for dd in results.dims if dd != "impl" and dd != d]
            to_plot = results.reduce(reduction, to_reduce)
            # sort dimensions
            to_plot = to_plot.sortby(list(to_plot.dims))
            # convert to numpy array
            to_plot = to_plot.transpose("impl", d)
            labels = to_plot.coords["impl"].data
            ticks = to_plot.coords[d].data
            to_plot = to_plot.data
            # handle scaling dimension
            if d in scaling_dims:
                # normalize
                to_plot /= np.max(to_plot, axis=1)[:, np.newaxis]
                # calculate scaling factors
                scaling_factors = find_scaling_factor(np.reshape(ticks, (1, -1)), to_plot, axis=1)
            # prepare plotting arguments
            ylabel = "Runtime [s]"
            style = matplotx.styles.duftify(matplotx.styles.dracula)
            if d in scaling_dims:
                ylabel="runtime (normalized)"
            # plot
            plt.clf()
            with plt.style.context(style):
                if isinstance(ticks[0], str):
                    # bar plot
                    # calculate the bar postions
                    group_size = len(labels) + 1
                    bar_width = 1 / group_size
                    xpos = np.arange(0, len(ticks), bar_width) - (1 - 2*bar_width)/2
                    xpos = xpos.reshape((len(ticks), len(labels))).transpose()
                    xpos = xpos[:-1, :] # remove empty bar
                    for l, rt, xp in zip(labels, to_plot, xpos):
                        plt.bar(xp, rt, label=l)
                    plt.xticks(range(len(ticks)), ticks)
                else:
                    # line plot
                    if d in scaling_dims:
                        psrt = 1/ticks
                        psrt /= psrt.max()
                        plt.plot(ticks, psrt, label=r"Perfect scaling, $\alpha=1$", linestyle="--")
                        labels = [l + r", $\alpha=" + "%.2f"%sf + r"$" for l, sf in zip(labels, scaling_factors)]
                        plt.xscale("log", base=2)
                        plt.yscale("log", base=2)
                    for l, rt in zip(labels, to_plot):
                        plt.plot(ticks, rt, label=l)
                    plt.xlabel(d)
                    plt.xticks(ticks)
                    matplotx.ylabel_top(ylabel)
                    if d in scaling_dims:
                        matplotx.line_labels()
                    else:
                        plt.legend()
                # save plot
                plt.savefig(path.joinpath(d + ".png"), format="png", bbox_inches="tight")
