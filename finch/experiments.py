import functools
import random
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import dask
import dask.array as da
import dask.distributed
import dask.graph_manipulation
import numpy as np
import xarray as xr
from dask.distributed import performance_report

from . import cfg, scheduler, util
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

    impl: Callable | None = None
    """The operator implementation to run"""
    prep: Callable[[], dict] | None = None
    """
    A function with preparations to be made before running the implementation.
    The runtime of this function won't be measured.
    The output dictionary of this function will be used as arguments for the implementation runner.
    """

    def setup(self) -> None:
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

    def setup(self) -> None:
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

    full: float | None = None
    """The full runtime of the experiment."""
    input_prep: float | None = None
    """Serial. The runtime used for preparing the input."""
    graph_construction: float | None = None
    """Serial. The runtime used for constructing the dask graph."""
    graph_opt: float | None = None
    """Serial. The runtime used for optimizing the dask graph."""
    graph_serial: float | None = None
    """Serial. The runtime used for serializing the dask graph."""
    compute: float | None = None
    """
    Parallel (and some unmeasurable serial). The runtime used for running the final computation.
    This includes the parallel computation as well as some serial overhead that cannot be measured separately.
    """


def _reduce_runtimes(rt: list[Runtime], reduction: Callable[[np.ndarray, int], np.ndarray]) -> Runtime:
    attr = util.get_class_attribute_names(Runtime)
    array = [[r.__dict__[a] for a in attr] for r in rt]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        out = reduction(np.array(array, dtype=float), 0)
    kwargs: dict[str, float] = {k: o for k, o in zip(attr, out) if o is not float("NaN")}
    return Runtime(**kwargs)


_RTList = list[list[Runtime]]


def measure_runtimes(
    run_config: list[RunConfig] | RunConfig,
    inputs: list[Callable[[], list]] | Callable[[], list] | list[list] | None = None,
    iterations: int = 1,
    impl_runner: Callable[..., Runtime | None] | None = None,
    reduction: Callable[[np.ndarray, int], np.ndarray] = np.nanmean,
    warmup: bool = False,
    pbar: PbarArg = True,
    dask_report: bool = False,
) -> _RTList:
    """
    Measures the runtimes of multiple functions, each accepting the same inputs.

    Args:
        run_config (list[RunConfig] | RunConfig): The functions to be benchmarked
        inputs: The inputs to the functions to be benchmarked. These can be passed in different forms:

            -   ``None``: Default. Can be passed if the functions do not accept any arguments.
            -   ``list[list]``: A list of concrete arguments to the functions
            -   ``list[Callable[[], list]]``: A list of argument generating functions.
                These will be run to collect the arguments for the functions to be benchmarked.
            -   ``Callable[[], list]``:
                A single argument generating function if the same should be used for every function to be benchmarked.

            The preparation of the input is timed and included in the full runtime.
        iterations (int, optional): The number of times to repeat a run (including input preparation).
        impl_runner (Callable[..., Runtime | None] | None): The function responsible for running
            the impl argument of a run configuration for the given list of arguments.
            The first argument of impl_runner is the current run configuration while
            the remaining arguments are the arguments passed to `impl` of the run configuration.
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
        The runtimes as a list of lists, or a flat list, or a float,
        depending on whether a single function or a single version (None or Callable) were passed.

    Group:
        Experiments
    """
    # prepare run config
    if isinstance(run_config, RunConfig):
        rc_list = [run_config]
    else:
        rc_list = run_config
    # ensure increasing worker sizes
    rc_order = range(len(rc_list))
    if not isinstance(run_config, RunConfig) and util.is_list_of(run_config, DaskRunConfig):
        rc_labeled = sorted(enumerate(run_config), key=lambda x: x[1].workers)
        unzipped: Any = zip(*rc_labeled)  # Currently not possible to properly type hint zip
        rc_order, rc_list = unzipped

    # prepare inputs to all have the same form
    if inputs is None:
        in_preps: list[Callable[[], list]] = []
    elif callable(inputs):
        in_preps = [inputs]
    elif util.is_list_of(inputs, list):
        in_preps = [lambda: i for i in inputs]
    else:
        assert util.is_callable_list(inputs)
        in_preps = inputs

    # prepare impl_runner
    if impl_runner is None:

        def simple_runner(f: Callable, *args: Any) -> None:
            f(args)

        impl_runner = simple_runner

    if warmup:
        iterations += 1

    progress = util.get_pbar(pbar, len(rc_list) * len(in_preps) * iterations)

    reportfile = util.get_path(cfg["evaluation"]["perf_report_dir"], "dask-report.html")

    times: list[list[Runtime]] = []
    for c in rc_list:
        c.setup()
        if dask_report:
            ctx: AbstractContextManager = performance_report(filename=reportfile)
        else:
            ctx = nullcontext()
        with ctx:
            f_times: list[Runtime] = []
            for in_prep in in_preps:
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
                    if progress:
                        progress.update()
                if warmup:
                    cur_times = cur_times[1:]
                f_times.append(_reduce_runtimes(cur_times, reduction))
            times.append(f_times)
    # reorder according to original run config order
    out: list[list[Runtime]] = [[]] * len(times)
    for i, t in zip(rc_order, times):
        out[i] = t
    return out


output_dir = util.get_path(cfg["global"]["tmp_dir"], "exp_out")
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
    impl: Callable[[xr.Dataset], xr.DataArray], ds: xr.Dataset, output_exists: bool = True, **kwargs: Any
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


def clear_output() -> None:
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
    run_config: list[RunConfig] | RunConfig, input: Input, versions: list[Input.Version] | Input.Version, **kwargs: Any
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
    if isinstance(versions, Input.Version):
        versions = [versions]
    preps = [functools.partial(xr_input_prep, input=input, version=v) for v in versions]
    assert util.is_callable_list(preps)
    return measure_runtimes(run_config, preps, impl_runner=xr_impl_runner, **kwargs)
