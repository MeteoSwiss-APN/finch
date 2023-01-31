import functools
import pathlib
import random
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from time import perf_counter, time
from typing import Any, TypeVar

import dask
import dask.array as da
import dask.distributed
import dask.graph_manipulation
import numpy as np
import xarray as xr
from dask.distributed import performance_report
from deprecated.sphinx import deprecated

from . import config, env, scheduler, util
from .data import Input
from .util import PbarArg

DefaultOperator = Callable[[xr.Dataset], xr.DataArray]
"""
Default interface of operators in finch.

Group:
    Finch
"""


@dataclass
class RunConfig(util.Config):
    """
    Abstract class for configuring and setting up the environment for experiments.

    Group:
        Experiments
    """

    impl: Callable = None
    """The operator implementation to run"""
    prep: Callable[[], dict] = None
    """
    A function with preparations to be made before running the implementation.
    The runtime of this function won't be measured.
    The output dictionary of this function will be used as arguments for the implementation runner.
    """

    def setup(self):
        """
        Sets up the environment for this config.
        """
        pass


@dataclass
class DaskRunConfig(RunConfig):
    """
    A run configuration class for running operators on a dask cluster.

    Group:
        Experiments
    """

    cluster_config: scheduler.ClusterConfig = scheduler.ClusterConfig()
    """The cluster configuration to use"""
    workers: int = 1
    """The number of dask workers to spawn"""

    def setup(self):
        """
        Start the scheduler and wait for the workers.
        """
        scheduler.start_scheduler(cfg=self.cluster_config)
        scheduler.scale_and_wait(self.workers)


@dataclass
class Runtime:
    """A class for capturing runtimes of different stages.
    The runtimes can be cathegorized into serial for serial overheads or parallel for runtimes in parallel regions.

    Group:
        Experiments
    """

    full: float = None
    """The full runtime of the experiment."""
    input_prep: float = None
    """Serial. The runtime used for preparing the input."""
    graph_construction: float = None
    """Serial. The runtime used for constructing the dask graph."""
    graph_opt: float = None
    """Serial. The runtime used for optimizing the dask graph."""
    graph_serial: float = None
    """Serial. The runtime used for serializing the dask graph."""
    compute: float = None
    """
    Parallel (and some unmeasurable serial). The runtime used for running the final computation.
    This includes the parallel computation as well as some serial overhead that cannot be measured separately.
    """


def _reduce_runtimes(rt: list[Runtime], reduction: Callable[[np.ndarray, int], np.ndarray]) -> Runtime:
    attr = util.get_class_attributes(Runtime)
    array = [[r.__dict__[a] for a in attr] for r in rt]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        out = reduction(np.array(array, dtype=float), axis=0)
    out = {k: o for k, o in zip(attr, out) if o is not float("NaN")}
    return Runtime(**out)


_RTList = list[list[Runtime]]


def measure_runtimes(
    run_config: list[RunConfig] | RunConfig,
    inputs: list[Callable[[], list]] | Callable[[], list] | list[list] | None = None,
    iterations: int = 1,
    impl_runner: Callable[..., Runtime | None] = None,
    reduction: Callable[[np.ndarray, int], np.ndarray] = np.nanmean,
    warmup: bool = False,
    pbar: PbarArg = True,
    dask_report: bool = False,
    **kwargs
) -> _RTList:
    """
    Measures the runtimes of multiple functions, each accepting the same inputs.

    Args:
        run_config (list[RunConfig] | RunConfig): The functions to be benchmarked
        inputs: The inputs to the functions to be benchmarked. These can be passed in different forms:

            - ``None``: Default. Can be passed if the functions do not accept any arguments.
            - ``list[list]``: A list of concrete arguments to the functions
            - ``list[Callable[[], list]]``: A list of argument generating functions.
                These will be run to collect the arguments for the functions to be benchmarked.
            - ``Callable[[], list]``: A single argument generating function if the same should be used for every function to be benchmarked.

            The preparation of the input is timed and included in the full runtime.
        iterations (int, optional): The number of times to repeat a run (including input preparation).
        impl_runner (Callable): The function responsible for running
            the impl argument of a run configuration for the given list of arguments.
            The first argument of impl_runner is the current run configuration while the remaining arguments are the arguments passed to `impl` of the run configuration.
            The execution of `impl_runner` will be timed and reported.
            A runtime can be returned if the implementation runner supports fine-grained runtime repoting.
            Defaults to directly running the passed function on the passed arguments.
        reduction (Callable):
            The function to be used to combine the results of the iterations.
            This is a reduction function which is able to reduce a specific dimenion (kwarg axis) of a numpy array.
        warmup (bool): If ``True``, runs the function once before measuring.
        pbar (PbarArg): Progressbar argument
        dask_report (bool): Whether or not to produce a dask report for the experiment.
            The location of the report is configured via finch's configuration.

    Returns:
        The runtimes as a list of lists, or a flat list, or a float, depending on whether a single function or a single version (None or Callable) were passed.

    Group:
        Experiments
    """
    # prepare run config
    singleton_rc = isinstance(run_config, RunConfig)
    if singleton_rc:
        run_config = [run_config]
    # ensure increasing worker sizes
    run_config = sorted(enumerate(run_config), key=lambda x: x[1].workers)
    rc_order, run_config = zip(*run_config)

    # prepare inputs to all have the same form
    singleton_input = False
    if inputs is None:
        inputs = [[]]
        singleton_input = True
    if isinstance(inputs, Callable):
        inputs = [inputs]
        singleton_input = True
    if isinstance(inputs[0], list):
        inputs = [lambda: i for i in inputs]

    # prepare impl_runner
    if impl_runner is None:
        impl_runner = lambda f, args: f(args)

    if warmup:
        iterations += 1

    pbar = util.get_pbar(pbar, len(run_config) * len(inputs) * iterations)

    reportfile = util.get_path(config["evaluation"]["perf_report_dir"], "dask-report.html")

    times = []
    for c in run_config:
        c.setup()
        with performance_report(filename=reportfile) if dask_report else nullcontext():
            f_times = []
            for in_prep in inputs:
                cur_times = []
                for _ in range(iterations):
                    start = perf_counter()
                    args = in_prep()
                    end = perf_counter()
                    in_prep_rt = end - start
                    ir_args = dict()
                    if c.prep is not None:
                        ir_args = c.prep()
                    start = perf_counter()
                    runtime = impl_runner(c.impl, *args, **ir_args)
                    end = perf_counter()
                    if runtime is None:
                        runtime = Runtime()
                    if runtime.full is None:
                        runtime.full = end - start + in_prep_rt
                    runtime.input_prep = in_prep_rt
                    cur_times.append(runtime)
                    pbar.update()
                if warmup:
                    cur_times = cur_times[1:]
                f_times.append(_reduce_runtimes(cur_times, reduction))
            times.append(f_times)
    # reorder according to original run config order
    out = [0] * len(times)
    for i, t in zip(rc_order, times):
        out[i] = t
    # adjust output form
    if singleton_input:
        out = [o[0] for o in out]
    if singleton_rc:
        out = out[0]
    return out


output_dir = util.get_path(config["global"]["tmp_dir"], "exp_out")
"""
The output directory of the experiments

Group:
    Experiments
"""


def xr_run_prep(remove_existing_output: bool = True, clear_scheduler: bool = False) -> dict[str, Any]:
    """
    A run preparation for standard xarray operators, which can be used in a run config.
    The parameters are provided for customizations via ``functools.partial``.

    Args:
        remove_existing_output (bool, optional): Whether to remove preexisting outputs. Defaults to True.
        clear_scheduler (bool, optional): Whether to clear the scheduler. Defaults to False.

    Returns:
        dict[str, Any]: The arguments for the xarray implementation runner.

    Group:
        Experiments
    """
    impl_runner_args = dict()
    if remove_existing_output:
        clear_output()
        impl_runner_args["output_exists"] = False
    if clear_scheduler:
        scheduler.clear_memory()
    return impl_runner_args


def xr_impl_runner(
    impl: Callable[[xr.Dataset], xr.DataArray], ds: xr.Dataset, output_exists: bool = True, **kwargs
) -> Runtime:
    """
    Implementation runner for standard xarray operators.

    Args:
        impl: The operator implementation to be run
        ds: The input for the implementation
        output_exists: Whether an output already exists at the output path and should be overwritten.
        kwargs: Additional arguments, which will be ignored

    Returns:
        The runtimes of the individual stages of the computation.

    Group:
        Experiments

    """
    runtime = Runtime()
    # construct the dask graph
    start = perf_counter()
    ds = ds + xr.full_like(ds, random.random())  # instead of clone. See https://github.com/dask/dask/issues/9621
    out = impl(ds)
    # clone the graph to ensure that the scheduler does not use results computed in an earlier round
    # cloned: da.Array = dask.graph_manipulation.clone(out.data)
    stored = out.data.to_zarr(str(output_dir), overwrite=output_exists, compute=False)
    end = perf_counter()
    runtime.graph_construction = end - start
    # optimize the graph
    start = perf_counter()
    optimized: da.Array = dask.optimize(stored)[0]
    end = perf_counter()
    runtime.graph_opt = end - start
    # compute
    client = scheduler.get_client()
    if client is not None:
        start = perf_counter()
        fut: da.Array = client.persist(optimized)
        end = perf_counter()
        runtime.graph_serial = end - start
        start = perf_counter()
        dask.distributed.wait(fut)
        end = perf_counter()
        runtime.compute = end - start
    else:
        start = perf_counter()
        optimized.compute()
        end = perf_counter()
        runtime.compute = end - start
    return runtime


def clear_output():
    """
    Clears the experiment output directory

    Group:
        Experiments
    """
    util.clear_dir(output_dir)


def xr_input_prep(input: Input, version: Input.Version) -> list[xr.Dataset]:
    """Input preparation for standard xarray operators.

    Args:
        input (Input): The input to be used for this operator
        version (Input.Version): The version which should be used for this experiment

    Returns:
        list[xr.Dataset]: The input of the operator

    Group:
        Experiments
    """
    out, _ = input.get_version(version)
    return [out]


def measure_operator_runtimes(
    run_config: list[RunConfig] | RunConfig, input: Input, versions: list[Input.Version] | Input.Version, **kwargs
) -> _RTList:
    """
    Measures the runtimes of different implementations of a standard xarray operator against different input versions.

    Args:
        run_config: The runtime configurations
        input: The input object for the operator
        versions: The different input versions to be benchmarked
        kwargs: Arguments for :func:`measure_runtimes`

    Returns:
        The runtimes as returned by :func:`measure_runtimes`

    See Also:
        :func:`measure_runtimes`

    Group:
        Experiments
    """
    single_version = False
    if isinstance(versions, Input.Version):
        single_version = True
        versions = [versions]
    preps = [functools.partial(xr_input_prep, input=input, version=v) for v in versions]
    if single_version:
        preps = preps[0]
    return measure_runtimes(run_config, preps, impl_runner=xr_impl_runner, **kwargs)


@deprecated("Replaced by the :class:`Runtime` class.", version="0.0.1a1")
def measure_loading_times(input: Input, versions: list[Input.Version], **kwargs) -> list[float]:
    """
    Measures the loading times of different versions of an input.

    This function will currently fail with a ``NotImplementedError``.
    Due to a bug in dask's graph cloning, it is currently not possible to properly
    measure the data loading time.

    Args:
        input: The input to be loaded
        versions: The different versions to be measured
        kwargs: Arguments for :func:`measure_runtimes`

    Returns:
        A list of times in seconds, indicating the loading times for the given versions of the input.

    Group:
        Experiments
    """
    raise NotImplementedError("Measuring pure load times is currently impossible.")
