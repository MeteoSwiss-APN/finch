from cProfile import label
from ensurepip import version
import functools
import pathlib
from typing import Any, Tuple
from collections.abc import Callable
from . import Input
from . import util
from . import config
from .experiments import RunConfig, Runtime
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

def create_result_dataset(
    results: list[list[Runtime]] | list[Runtime] | Runtime, 
    run_configs: list[RunConfig] | RunConfig, 
    versions: list[Input.Version] | Input.Version, 
    experiment_name: str = None, 
    impl_names: list[str] | Callable[[Callable], str] | None = None
) -> xr.Dataset:
    """
    Constructs a dataset from the results of an experiment.
    The dimensions are given by the attributes of the Version and RunConfig classes.
    The coordinates are labels for the version and run config attributes.
    The array entries in the dataset are the different runtimes which were recorded
    This result dataset can then be used as an input for different evaluation functions.
    The result dataset will contain NaN for every combination of version and run config attributes, which is not listed in `versions`.
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
        if isinstance(impl_names, list):
            for a, impl_name in zip(rc_attrs, impl_names):
                a["impl"] = impl_name
        elif isinstance(impl_names, Callable):
            for a, rc in zip(rc_attrs, run_configs):
                a["impl"] = impl_names(rc.impl)
    # construct coordinates
    coords = {
        a : list(set(va[a] for va in version_attrs))
        for a in va_keys
    }
    coords.update({
        a : list(set(ra[a] for ra in rc_attrs))
        for a in rca_keys
    })

    dim_sizes = [len(coords[a]) for a in coords]

    # create dataset
    ds = xr.Dataset(coords=coords)
    for attr in util.get_class_attributes(Runtime):
        # initialize data
        data = np.full(dim_sizes, np.nan, dtype=float)

        array = xr.DataArray(data, coords, name=attr)
        has_entries = False # indicates whether the current runtime attribute has entries
        for result, rca in zip(results, rc_attrs):
            for r, va in zip(result, version_attrs):
                entry = r.__dict__[attr] # get the runtime entry
                if entry is not None:
                    has_entries = True
                array.loc[va | rca] = entry
        if has_entries: # only add runtimes which were actually recorded
            ds[attr] = array
    ds.attrs["name"] = experiment_name
    return ds

def create_cores_dimension(
    results: xr.Dataset, 
    contributors: list[str] = [
        "workers",
        "cluster_config_cores_per_worker"
    ],
    cores_dim = "cores",
    reduction: Callable = np.min
) -> xr.Dataset:
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
    out_cols = [out.loc[{cores_dim : c}] for c in coords] # collect columns according to core size
    out_cols = [col if cores_dim in col.dims else col.expand_dims(cores_dim) for col in out_cols] # make sure there is cores_dim
    out_cols = [col.reduce(reduction, cores_dim) for col in out_cols]
    out = xr.concat(out_cols, cores_dim)
    out[cores_dim] = coords
    out.attrs = results.attrs
    return out

def speedup(runtimes: np.ndarray, axis: int = -1, base: np.ndarray = None) -> np.ndarray:
    """
    Calculates the speedup for an array of runtimes.

    Arguments:
    ---
    - runtimes: np.ndarray. The array of runtimes to convert to speedups
    - axis: int. The axis which defines a series of runtimes.
    Only relevant if `base` is not given.
    Defaults to the last dimension.
    - base: np.ndarray. An array of runtimes indicating a speedup of 1.
    Should have one dimension less than runtimes.
    By default, the base will be determined from the first element in the runtime series.
    """
    if base is None:
        first_index = [slice(x) for x in runtimes.shape]
        first_index[axis] = 0
        base = runtimes[tuple(first_index)]
    base = np.expand_dims(base, axis)
    return base / runtimes

def find_scaling(scale: np.ndarray, speedup: np.ndarray, axis: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the scaling factor and scaling rate for a series of speedups.
    This is done via regression on functions of the type $y = \alpha * x^\beta$.
    $\alpha$ indicates the scaling factor and $\beta$ the scaling rate.
    This assumes that the speedup for scale = 1 is 1.
    """
    alpha, beta = util.simple_lin_reg(np.log(scale), np.log(speedup), axis=axis)
    alpha = np.exp(alpha)
    return alpha, beta


def create_plots(
    results: xr.Dataset, 
    reduction: Callable = np.nanmin, 
    scaling_dims: list[str] = ["cores"],
    find_scaling_props: bool = True,
    plot_scaling_fits: bool = False,
    plot_scaling_baseline: bool = True,
    runtime_selection: list[str] = None
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
    For those dimensions, a log-log plot of the speedup will be created and the scaling factor calculated.
    - find_scaling_props: bool. Whether to calculate the scaling factor and scaling rate for scaling dimensions
    - plot_scaling_fits: bool. 
    Whether to plot the functions which are being fitted for calculating the scaling factor and rate.
    - plot_scaling_baseline: bool.
    Whether to plot a baseline for scaling dimensions.
    - runtime_selection: list[str]. The runtime types to plot. Defaults to all recorded runtimes.
    """

    path = pathlib.Path(config["evaluation"]["plot_dir"], results.attrs["name"])
    path.mkdir(parents=True, exist_ok=True)

    for d in results.dims:
        if d != "impl" and results.sizes[d] > 1:
            # reduce dimensions other than d and impl
            to_reduce = [dd for dd in results.dims if dd != "impl" and dd != d]
            to_plot = results.reduce(reduction, to_reduce)
            # sort dimensions
            to_plot = to_plot.sortby(list(to_plot.dims))
            # reorder dimensions
            to_plot = to_plot.transpose("impl", d)
            for runtime_type, runtime_data in to_plot.data_vars.items():
                if runtime_selection is not None and runtime_type not in runtime_selection or \
                    d in scaling_dims and runtime_type not in ["full", "compute"]:
                    continue
                # get plotting arguments
                labels = to_plot.coords["impl"].data
                ticks = to_plot.coords[d].data
                # convert to numpy array
                runtime_data = runtime_data.data
                # prepare plotting arguments
                style = matplotx.styles.duftify(matplotx.styles.dracula)
                # plot
                plt.clf()
                with plt.style.context(style):
                    if isinstance(ticks[0], str):
                        # bar plot
                        # calculate the bar postions
                        group_size = len(labels) + 1
                        bar_width = 1 / group_size
                        xpos = np.arange(0, len(ticks), bar_width) - (1 - 2*bar_width)/2
                        xpos = xpos.reshape((len(ticks), len(labels)+1)).transpose()
                        xpos = xpos[:-1, :] # remove empty bar
                        for l, rt, xp in zip(labels, runtime_data, xpos):
                            plt.bar(xp, rt, width=bar_width, label=l)
                        plt.xticks(range(len(ticks)), ticks)
                        plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
                        plt.xlabel(d)
                        matplotx.ylabel_top("Runtime [s]")
                    else:
                        # line plot
                        ylabel = "Runtime [s]"
                        # handle scaling dimension
                        if d in scaling_dims:
                            # compute speedup
                            runtime_data = speedup(runtime_data)
                            # calculate scaling rate and factor
                            if find_scaling_props:
                                scale = ticks / ticks[0]
                                alpha, beta = find_scaling(np.reshape(scale, (1, -1)), runtime_data, axis=1)
                                labels = [
                                    l + r", $\alpha=" + "%.2f"%sf + r"$, $\beta=" + "%.2f"%sr + r"$" 
                                    for l, sf, sr in zip(labels, alpha, beta)
                                ]
                            # plot baseline
                            if plot_scaling_baseline:
                                base_label = "Perfect linear scaling"
                                if find_scaling_props:
                                    base_label += r", $\alpha=1$, $\beta=1$"
                                plt.plot(ticks, scale, label=base_label, linestyle="--")
                            # plot fitted scaling functions
                            if plot_scaling_fits and find_scaling_props:
                                cycler = style["axes.prop_cycle"]
                                if plot_scaling_baseline:
                                    cycler = cycler[1:]
                                for a, b, c in zip(alpha, beta, cycler):
                                    x = np.linspace(ticks[0], ticks[-1], 100)
                                    xt = x / ticks[0]
                                    y = a*xt**b
                                    plt.plot(x, y, linestyle=":", color=c["color"])
                            # plt.xscale("log", base=2)
                            # plt.yscale("log", base=2)
                            ylabel = "Speedup"
                        for l, rt in zip(labels, runtime_data):
                            plt.plot(ticks, rt, label=l)
                        plt.xlabel(d)
                        plt.xticks(ticks)
                        matplotx.ylabel_top(ylabel)
                        if d in scaling_dims:
                            matplotx.line_labels()
                        else:
                            plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
                    # save plot
                    plt.savefig(path.joinpath(d + "_" + runtime_type + ".png"), format="png", bbox_inches="tight")

def plot_runtime_parts(
    results: xr.Dataset
):
    """
    Plots how the full runtimes are split up.
    """
    # collect data
    rt_types = list(results.data_vars.keys())
    rt_data = np.vstack([results.data_vars[rtt].data.flatten() for rtt in rt_types if rtt != "full"])
    # normalize
    rt_data /= np.sum(rt_data, axis=0)[np.newaxis, :]

    style = matplotx.styles.duftify(matplotx.styles.dracula)
    plt.clf()
    with plt.style.context(style):
        bottom = 0
        for i, rt_type in enumerate(rt_types):
            row = rt_data[i, :]
            ticks = range(len(row))
            plt.bar(ticks, row, bottom=bottom, label=rt_type)
            bottom += row
        path = pathlib.Path(config["evaluation"]["plot_dir"], results.attrs["name"])
        path.mkdir(parents=True, exist_ok=True)
        plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
        plt.xticks(ticks)
        matplotx.ylabel_top("Runtime %")
        plt.savefig(path.joinpath("runtime_parts.png"), format="png", bbox_inches="tight")
