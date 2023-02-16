import pathlib
import warnings
from collections.abc import Callable
from typing import Any, Hashable, Tuple

import matplotlib.pyplot as plt  # type: ignore
import matplotx  # type: ignore
import numpy as np
import xarray as xr
import yaml
from deprecated.sphinx import deprecated

from . import cfg, util
from .data import Input
from .experiments import RunConfig, Runtime, _RTList


def get_pyplot_grouped_bar_pos(groups: int, labels: int) -> Tuple[np.ndarray, float]:
    """
    Returns an array of bar positions when trying to create a grouped bar plot for pyplot,
    along with the width of an individual bar.
    A row in the returned array contains the bar positions for a label,
    while a column contains the bar positions for a group.

    Args:
        groups (int): The number of groups in the bar plot
        labels (int): The number of labels in the bar plot

    Group:
        Evaluation
    """
    group_size = labels + 1
    bar_width = 1 / group_size
    xpos = np.arange(0, groups, bar_width) - (1 - 2 * bar_width) / 2
    xpos = xpos.reshape(groups, group_size).transpose()
    xpos = xpos[:-1, :]  # remove empty bar
    return xpos, bar_width


def print_version_results(results: list[Any], versions: list[Input.Version]) -> None:
    """
    Prints the results of an experiment for different input versions.

    Group:
        Evaluation
    """
    for r, v in zip(results, versions):
        print(f"{v}\n    {r}")


def print_results(results: list[list[Any]], run_configs: list[RunConfig], versions: list[Input.Version]) -> None:
    """
    Prints the results of an experiment for different run configurations and input versions.

    Group:
        Evaluation
    """
    for rc, r in zip(run_configs, results):
        print(rc)
        print()
        print_version_results(r, versions)
        print()


def create_result_dataset(
    results: _RTList,
    run_configs: list[RunConfig] | RunConfig,
    versions: list[Input.Version] | Input.Version,
    input: Input,
    experiment_name: str | None = None,
    impl_names: list[str] | Callable[[Callable], str] | None = None,
) -> xr.Dataset:
    """
    Constructs a dataset from the results of an experiment.
    The dimensions are given by the attributes of the Version and RunConfig classes.
    The coordinates are labels for the version and run config attributes.
    The array entries in the dataset are the different runtimes which were recorded
    This result dataset can then be used as an input for different evaluation functions.
    The result dataset will contain NaN for every combination of version and run config attributes,
    which is not listed in `versions`.

    Group:
        Evaluation
    """
    # unify arguments
    if isinstance(run_configs, RunConfig):
        run_configs = [run_configs]
    if isinstance(versions, Input.Version):
        versions = [versions]

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
        elif callable(impl_names):
            for a, rc in zip(rc_attrs, run_configs):
                if rc.impl is None:
                    raise ValueError("Run config is missing an implementation.", rc)
                a["impl"] = impl_names(rc.impl)
    # construct coordinates
    coords = {a: list(set(va[a] for va in version_attrs)) for a in va_keys}
    coords.update({a: list(set(ra[a] for ra in rc_attrs)) for a in rca_keys})

    dim_sizes = [len(coords[a]) for a in coords]

    # create dataset
    ds = xr.Dataset(coords=coords)
    for attr in util.get_class_attribute_names(Runtime):
        # initialize data
        data = np.full(dim_sizes, np.nan, dtype=float)

        array = xr.DataArray(data, coords, name=attr)
        has_entries = False  # indicates whether the current runtime attribute has entries
        for result, rca in zip(results, rc_attrs):
            for r, va in zip(result, version_attrs):
                entry = r.__dict__[attr]  # get the runtime entry
                if entry is not None:
                    has_entries = True
                array.loc[va | rca] = entry
        if has_entries:  # only add runtimes which were actually recorded
            ds[attr] = array
    ds.attrs["name"] = experiment_name
    return ds


def create_cores_dimension(
    results: xr.Dataset,
    contributors: list[str] = ["workers", "cluster_config_cores_per_worker"],
    cores_dim: str = "cores",
    reduction: Callable = np.min,
) -> xr.Dataset:
    """
    Merges the dimensions in the results array which contribute
    to the total amount of cores into a single 'cores' dimension.
    The number of cores are calculated by the product of the coordinates of the individual dimensions.
    The resulting dimension is sorted in increasing core order.

    Args:
        results (xr.Dataset): The results array
        contributors (list[str], optional): List of dimension names which contribute to the total core count
        cores_dim (str, optional): The dimension name of the new 'cores' dimension. Defaults to 'cores'.
        reduction (Callable, optional): How runtimes should be combined which have the same amount of cores.
            Defaults to ``np.min``

    Returns:
        The results dataset with merged cores dimensions.

    Group:
        Evaluation
    """
    out = results.stack({cores_dim: contributors})
    multis = out[cores_dim]  # this is now a multiindex
    coords_dup = [np.prod(x) for x in multis.data]  # calculate the number of cores
    out = out.drop_vars([cores_dim] + contributors)
    out[cores_dim] = coords_dup
    # reduce cores_dim to have unique values
    coords = np.unique(coords_dup)  # output is sorted
    out_cols = [out.loc[{cores_dim: c}] for c in coords]  # collect columns according to core size
    out_cols = [
        col if cores_dim in col.dims else col.expand_dims(cores_dim) for col in out_cols
    ]  # make sure there is cores_dim
    out_cols = [col.reduce(reduction, cores_dim) for col in out_cols]
    out = xr.concat(out_cols, cores_dim)
    out[cores_dim] = coords
    out.attrs = results.attrs
    return out


def rename_labels(
    results: xr.Dataset,
    renames: dict[str, dict[Any, Any] | list[Any]] | None = None,
    **kwargs: dict[Any, Any] | list[Any],
) -> xr.Dataset:
    """
    Rename labels for some dimensions. This changes the coordinates in the results dataset

    Args:
        results (xr.Dataset): The results dataset
        renames (dict[str, dict[Any, Any] | list[Any]] | None, optional):
            A dictionary mapping dimension names to rename instructions.
            Rename instructions can be either in the form of a dictionary,
            mapping old values to new values, or in the form of a list, completely replacing the old values directly.
            Defaults to None, which means that the rename instructions are passed in ``kwargs``.
        kwargs: The renames argument as kwargs. If neither `renames` or `kwargs` are given, nothing happens

    Returns:
        The results dataset with renamed labels

    Group:
        Evaluation
    """
    if renames is None:
        renames = kwargs
    out = results.copy()
    for k, v in renames.items():
        if isinstance(v, list):
            out[k] = v
        else:
            out[k] = xr.apply_ufunc(lambda x: v[x] if x in v else x, out[k], vectorize=True)
    return out


def remove_labels(results: xr.Dataset, labels: list[str], main_dim: str) -> xr.Dataset:
    """
    Removes the given labels in the given main dimension from the results array.

    Group:
        Evaluation
    """
    return results.drop_sel({main_dim: labels})


def simple_lin_reg(x: np.ndarray, y: np.ndarray, axis: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs simple linear regression along the given axis.

    Simple linear regression minimizes the mean squared error for a 2D linear model.

    .. math::

        y = \alpha + \betax


    Args:
        x (np.ndarray):
            The input values
        y (np.ndarray):
            The (measured) output values
        axis (int | None, optional):
            The axis along which a measurement series is stored.
            If None (default), x and y will be flattened.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            :math:`\alpha` and :math:`\beta` which minimize the mean squared error of the 2D linear model.

    Group:
        Evaluation
    """
    xm = x.mean(axis=axis)
    ym = y.mean(axis=axis)
    if axis is not None:
        xd = x - np.expand_dims(xm, axis)
        yd = y - np.expand_dims(ym, axis)
    else:
        xd = x - xm
        yd = y - ym
    beta = np.sum(xd * yd, axis=axis) / np.sum(xd * xd, axis=axis)
    alpha = ym - beta * xm
    return alpha, beta


def speedup(runtimes: np.ndarray, axis: int = -1, base: np.ndarray | None = None) -> np.ndarray:
    """
    Calculates the speedup for an array of runtimes.

    Args:
        runtimes (np.ndarray): The array of runtimes to convert to speedups
        axis (int, optional): The axis which defines a series of runtimes.
            Only relevant if `base` is not given.
            Defaults to the last dimension.
        base (np.ndarray | None, optional): The base runtimes from which to compute the speedup.
            Should have one dimension less than runtimes.
            By default (None), the base will be determined from the first element in the runtime series.

    Returns:
        An array of speedups with the same shape as ``runtimes``.

    Group:
        Evaluation
    """
    if base is None:
        first_index: list[slice | int] = [slice(x) for x in runtimes.shape]
        first_index[axis] = 0
        base = runtimes[tuple(first_index)]
    base = np.expand_dims(base, axis)
    out: np.ndarray = base / runtimes
    return out


@deprecated("Serial overhead analysis should be used instead.", version="0.0.1a1")
def find_scaling(scale: np.ndarray, speedup: np.ndarray, axis: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the scaling factor and scaling rate for a series of speedups.
    This is done via regression on functions of the type $y = \alpha * x^\beta$.
    $\alpha$ indicates the scaling factor and $\beta$ the scaling rate.
    This assumes that the speedup for scale = 1 is 1.

    Group:
        Evaluation
    """
    alpha, beta = simple_lin_reg(np.log(scale), np.log(speedup), axis=axis)
    alpha = np.exp(alpha)
    return alpha, beta


def amdahl_speedup(f: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Returns the speedups for a serial runtime fractions and a selection of core counts.

    Args:
        f (np.ndarray): A numpy array of serial runtime fractions. These must be between 0 and 1.
        c (np.ndarray): A numpy array of core counts. This must have the same shape as ``f`` or must be broadcastable.

    Returns:
        A numpy array of speedups for the given serial fractions. Has the same shape as ``f``.

    Group:
        Evaluation
    """
    out: np.ndarray = 1 / (f + (1 - f) / c)
    return out


def serial_overhead_analysis(
    t: np.ndarray,
    c: np.ndarray,
    t1: np.ndarray | None = None,
    c1: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimates the serial fraction of the total runtime.
    This is done via the closed-form solution of least squares regression with Amdahl's law.

    Args:
        t (np.ndarray): shape: ``(n_implementations, n_core_selections)``.
            The runtime measurements
        c (np.ndarray): shape: ``(n_implementations, n_core_selections)``.
            The core selections
        t1 (np.ndarray | None, optional): shape: ``n_implementations``.
            The runtime baseline.
            If None, the first column of `t` will be used.
        c1 (np.ndarray | None, optional): shape: ``n_implementations``.
            The core counts for the runtime baselines.
            If None, the first column of `c` will be used.

    Returns:
        A 1-D numpy array of length ``n_implementations`` of estimated serial fractions.

    Group:
        Evaluation
    """
    # prepare arguments
    if t1 is None:
        t1 = t[:, 0]
    if c1 is None:
        c1 = c[:, 0]
    t1 = t1[:, np.newaxis]
    c1 = c1[:, np.newaxis]
    c = c / c1

    cf = (c - 1) / c
    f1 = (t - t1 / c) * cf
    f1 = np.sum(f1, axis=1)
    f2 = cf * cf
    f2 = t1.reshape(-1) * np.sum(f2, axis=1)
    f: np.ndarray = f1 / f2
    f = f.flatten()

    return f


def get_plots_dir(results: xr.Dataset) -> util.PathLike:
    """
    Returns the path to the directory where plots should be stored for a specific results dataset.

    Group:
        Plot
    """
    base = cfg["evaluation"]["plot_dir"]
    args = [base, results.attrs["name"]]
    if base == cfg["evaluation"]["dir"]:
        args.append("plots")
    return util.get_path(*args)


plot_style = matplotx.styles.duftify(matplotx.styles.dracula)
"""
The plot style to use for creating plots.

Group:
    Plot
"""


def create_plots(
    results: xr.Dataset,
    reduction: Callable = np.nanmin,
    main_dim: str = "impl",
    relative_rt_dims: list[Hashable] | dict[Hashable, Any] = [],
    scaling_dims: list[str] = [],
    estimate_serial: bool = True,
    plot_scaling_fits: bool = False,
    plot_scaling_baseline: bool = True,
    runtime_selection: list[str] | None = None,
) -> None:
    """
    Creates a series of plots for the results array.
    The plot creation works as follows.
    Every plot has multiple different lines, which correspond to the different implementations.
    The y-axis indicates the value of the result, while the x-axis is a dimension of the result array.
    For every dimension of size greater than 1, except for the 'imp' dimension, a new plot will be created.
    The other dimensions will then be reduced by flattening and then reducing according to the given reduction function.

    If the coordinates of a dimension have type `str`,
    a bar plot will be generated. Otherwise a standard line plot will be saved.
    The plots will be stored in config's `plot_dir` in a directory according to the experiment name.

    Args:
        results (xr.DataArray): The result array
        reduction (Callable): The reduction function. (See `xarray.DataArray.reduce`)
        relative_rt_dims (list[Hashable] | dict[Hashable, Any]):
            Dimensions for which a relative runtime plot will be produced.
            If a dictionary is passed, the keys will be used to identify
            the dimensions for which to plot a relative runtime
            and the values will be used to identify which entry of the main
            dimension should be used as a reference (identified by label / coordinate value).
            If a list is passed, the first entry will be used.
            A relative runtime plot is often more practical for comparing results, instead of the raw data.
            If the main_dim has only one entry, no normalization will happen.
        scaling_dims (list[str]): Dimensions which are used for scalability plots.
            For those dimensions, a plot of the speedup will be created in addition to the usual runtime plot.
        estimate_serial (bool): Whether to estimate the serial overhead.
        plot_scaling_fits (bool):
            Whether to plot the functions which are being fitted for calculating the scaling factor and rate.
        plot_scaling_baseline (bool):
            Whether to plot a baseline for scaling dimensions.
        runtime_selection (list[str] | None, optional):
            The runtime types to plot.
            Defaults to all recorded runtimes.

    Group:
        Plot
    """

    # disable debug logs from matplotlib
    plt.set_loglevel("warning")  # type: ignore

    path = pathlib.Path(get_plots_dir(results))

    def save_plot(
        dim: str,
        runtime_type: str,
        extra: str | None = None,
        format: util.ImgSuffix = "png",
    ) -> None:
        name = dim + "_" + runtime_type
        if extra is not None:
            name += "_" + extra
        name += "." + format
        plt.savefig(path.joinpath(name), format=format, bbox_inches="tight")

    for d in results.dims:
        d = str(d)
        if d != main_dim and results.sizes[d] > 1:
            # reduce dimensions other than d and impl
            to_reduce = [dd for dd in results.dims if dd != main_dim and dd != d]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                to_plot = results.reduce(reduction, to_reduce)
            # sort x dimension
            to_plot = to_plot.sortby(d)
            # get index of reference entry for relative runtime plotting
            if d in relative_rt_dims:
                if isinstance(relative_rt_dims, list):
                    ref_idx = 0
                else:
                    ref_idx = to_plot.indexes[main_dim].get_loc(relative_rt_dims[d])  # type: ignore
            # reorder dimensions
            to_plot = to_plot.transpose(main_dim, d)
            for runtime_type, runtime_data in to_plot.data_vars.items():
                if (
                    runtime_selection is not None
                    and runtime_type not in runtime_selection
                    or d in scaling_dims
                    and runtime_type not in ["full", "compute"]
                ):
                    continue
                if np.isnan(runtime_data).any():
                    # only create plots where all runtime data is provided
                    continue
                # get plotting arguments
                labels = to_plot.coords[main_dim].data
                labels = [str(lb) for lb in labels]
                ticks = to_plot.coords[d].data
                # convert to numpy array
                runtime_array: np.ndarray = runtime_data.data
                # plot
                plt.clf()
                with plt.style.context(plot_style):  # type: ignore
                    if isinstance(ticks[0], str):
                        # bar plot
                        xpos, bar_width = get_pyplot_grouped_bar_pos(len(ticks), len(labels))
                        for lb, rt, xp in zip(labels, runtime_array, xpos):
                            plt.bar(xp, rt, width=bar_width, label=lb)  # type: ignore
                        plt.xticks(range(len(ticks)), ticks)
                        plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
                        plt.xlabel(d)
                        matplotx.ylabel_top("Runtime [s]")
                    else:
                        # line plot
                        if d in scaling_dims:
                            # create speedup plot for scaling dimension
                            # compute speedup
                            spd = speedup(runtime_array)
                            # calculate scaling rate and factor
                            spd_labels = labels
                            if estimate_serial:
                                cs = np.tile(ticks, (runtime_array.shape[0], 1))
                                fs = serial_overhead_analysis(runtime_array, cs)
                                spd_labels = [lb + r", $f=" + "%.2f" % (f * 100) + r"$%" for lb, f in zip(labels, fs)]
                            # plot fitted scaling functions
                            if plot_scaling_fits and estimate_serial:
                                cycler = plt.rcParams["axes.prop_cycle"]  # type: ignore
                                for f, c in zip(fs, cycler):
                                    x = np.linspace(ticks[0], ticks[-1], 100)
                                    xt = x / ticks[0]
                                    y = amdahl_speedup(f, xt)
                                    color = c["color"]
                                    plt.plot(x, y, linestyle=":", color=color)
                            # plot speedups plot
                            for lb, rt in zip(spd_labels, spd):
                                plt.plot(ticks, rt, label=lb)
                            # plot baseline
                            if plot_scaling_baseline:
                                base_label = "Perfect linear scaling"
                                if estimate_serial:
                                    base_label += r", $f=0\%$"
                                plt.plot(ticks, ticks / ticks[0], label=base_label, linestyle="--")
                            plt.xlabel(d)
                            matplotx.ylabel_top("Speedup")
                            matplotx.line_labels()
                            save_plot(d, runtime_type, "speedup")
                            plt.clf()
                        # plot runtime
                        ylabel = "Runtime [s]"
                        # normalize
                        if d in relative_rt_dims and runtime_array.shape[0] > 1:
                            ylabel = "Relative Runtime"
                            runtime_array /= runtime_array[ref_idx, :]
                        for lb, rt in zip(labels, runtime_array):
                            plt.plot(ticks, rt, label=lb)
                        plt.xlabel(d)
                        # plt.xticks(ticks)
                        matplotx.ylabel_top(ylabel)
                        matplotx.line_labels()
                        plt.margins(y=0.05)  # type: ignore
                    # save plot
                    save_plot(d, runtime_type)


def plot_runtime_parts(results: xr.Dataset, first_dims: list[str] = []) -> None:
    """
    Plots how the full runtimes are split up.

    Args:
        results: The results dataset
        first_dims: The dimensions to prioritize in the order in which they are plotted.
            The first entry will be interpreted as the main dimension.

    Group:
        Plot
    """
    # drop full runtimes
    results = results.drop_vars("full")
    # create array
    data_arr = results.to_array(dim="rt_types")
    rt_types = data_arr["rt_types"].data
    # reorder dimensions
    dim_order = ["rt_types"] + first_dims
    dim_order += [str(d) for d in data_arr.dims if d not in dim_order]
    data_arr = data_arr.transpose(*dim_order)
    # capture tick labels
    if first_dims:
        tick_labels = data_arr[first_dims[0]].data
    # convert to numpy and flatten
    array: np.ndarray = data_arr.data.reshape(len(rt_types), -1)
    # remove nan columns
    array = array[:, ~np.isnan(array).any(axis=0)]
    # normalize
    array /= np.sum(array, axis=0)[np.newaxis, :]

    plt.clf()
    with plt.style.context(plot_style):  # type: ignore
        # get ticks
        bars = array.shape[1]
        if first_dims:
            groups = len(tick_labels)
            xpos, bar_width = get_pyplot_grouped_bar_pos(groups, bars // groups)
            xpos = np.ravel(xpos, "F")
        else:
            xpos = np.arange(bars)
            bar_width = 1
        bottom: float | np.ndarray = 0.0
        for i, rt_type in enumerate(rt_types):
            row = array[i, :]
            plt.bar(xpos, row, bottom=bottom, label=rt_type, width=bar_width)  # type: ignore
            bottom += row
        path = pathlib.Path(get_plots_dir(results))
        plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1))
        if first_dims:
            plt.xticks(range(groups), tick_labels)
            plt.xlabel(first_dims[0])
        else:
            plt.xticks([])
            plt.xlabel("Measurements")
        matplotx.ylabel_top("Runtime %")
        plt.savefig(path.joinpath("runtime_parts.png"), format="png", bbox_inches="tight")


def store_config(results: xr.Dataset) -> None:
    """
    Stores the configuration of the runtime experiment as a yaml.
    The configuration are the coordinate values of the results array.

    Group:
        Evaluation
    """
    path = util.get_path(cfg["evaluation"]["config_dir"], results.attrs["name"], "config.yaml")
    # create config dict
    res_config = {c: a.data.tolist() for c, a in results.coords.items()}
    res_config = {k: a[0] if len(a) == 1 else a for k, a in res_config.items()}
    with open(path, mode="w") as f:
        yaml.dump(res_config, f)
