import abc
import gc
import random
import string
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar

import dask
import dask.array as da
import dask.distributed
import dask.graph_manipulation
import dask.typing
import numpy as np
import numpy.typing as npt
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
class Runtime:
    """A class for capturing runtimes of different stages.
    The runtimes can be cathegorized into serial for serial overheads or parallel for runtimes in parallel regions.

    Group:
        Experiments
    """

    full: float = np.nan
    """
    The full runtime of the experiment.
    """
    input_loading: float | None = None
    """Serial. The runtime used for loading the input."""
    compute: float | None = None
    """
    Parallel (and some unmeasurable serial). The runtime used for running the final computation.
    This includes the parallel computation as well as some serial overhead that cannot be measured separately.
    """


RT = TypeVar("RT", bound=Runtime)


def combine_runtimes(rt: list[RT], reduction: Callable[[npt.NDArray[np.float64], int], npt.NDArray[np.float64]]) -> RT:
    """
    Combines multiple runtime objects into one.

    Args:
        rt (list[RT]): The runtime objects to combine. Must all be members of the same class.
        reduction (Callable[[npt.NDArray[np.float64], int], npt.NDArray[np.float64]]):
            A function defining how the runtime parts are combined.
            It reduces a specific dimension of a numpy array.

    Returns:
        RT: The combined runtime object.
    """
    assert len(rt) > 0
    rt_type = rt[0].__class__
    attr = util.get_class_attribute_names(rt_type)
    array = [[r.__dict__[a] for a in attr] for r in rt]
    assert util.is_2d_list_of(array, float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        out = reduction(np.array(array, dtype=float), 0)
    kwargs: dict[str, float] = {k: o for k, o in zip(attr, out) if o is not float("NaN")}
    return rt_type(**kwargs)


@dataclass(kw_only=True)
class RunConfig(util.Config, abc.ABC):
    """
    Class for configuring and setting up the environment for experiments.

    Group:
        Experiments
    """

    impl: Callable
    """The operator implementation to run"""

    iterations: int = 5
    """
    The number of iterations to run.
    The runtimes will be combined according to `runtime_reduction`.
    """

    warmup: bool = True
    """
    If set to `True`, an additional warmup iteration will be added at the start of the measurement iterations,
    whose runtime will be discarded.
    """

    runtime_reduction: Callable[[npt.NDArray[np.float64], int], npt.NDArray[np.float64]] = np.nanmean
    """
    A function which reduces a specific dimension of a numpy array.
    This is used for comining multiple runtimes into one.
    """

    def setup(self) -> None:
        """
        Sets up the environment for this configuration.
        This will be called once before the measurement iterations start.
        """
        pass

    def cleanup(self) -> None:
        """
        Perform cleanup after the measurement iterations.
        """
        pass

    @abc.abstractmethod
    def load_input(self) -> list[Any]:
        """
        Loads the input for the implementation.
        """
        pass

    def measure(self) -> Runtime:
        """
        Measures the runtime of the implementation.
        """
        start = perf_counter()
        input = self.load_input()
        end = perf_counter()
        input_load_time = end - start
        start = perf_counter()
        self.impl(*input)
        end = perf_counter()
        compute_time = end - start
        return Runtime(full=compute_time + input_load_time, input_loading=input_load_time, compute=compute_time)


def measure_runtimes(
    run_config: list[RunConfig] | RunConfig,
    pbar: PbarArg = True,
) -> list[Runtime]:
    """
    Measures the runtimes of multiple run configurations.

    Args:
        run_config (list[RunConfig] | RunConfig): The run configurations to measure.
        pbar (PbarArg): Progressbar argument.

    Returns:
        The measured runtimes of the run configurations.

    Group:
        Experiments
    """
    # prepare run configs
    if isinstance(run_config, RunConfig):
        run_config = [run_config]
    else:
        run_config = run_config

    # prepare progress bar
    total_iterations = sum([c.iterations for c in run_config]) + sum([c.warmup for c in run_config])
    progress = util.get_pbar(pbar, total_iterations)

    # disable garbage collection
    gc_enabled: bool = gc.isenabled()
    gc.disable()

    times: list[Runtime] = []
    for c in run_config:
        c.setup()
        cur_times: list[Runtime] = []
        iterations = c.iterations
        if c.warmup:
            iterations += 1
        for _ in range(iterations):
            gc.collect()  # manually clear garbage
            runtime = c.measure()
            cur_times.append(runtime)
            if progress:
                progress.update()
        if c.warmup:
            cur_times = cur_times[1:]
        times.append(combine_runtimes(cur_times, c.runtime_reduction))

    # re-enable garbage collection
    if gc_enabled:
        gc.enable()

    return times


@dataclass
class DaskRuntime(Runtime):
    """
    A class for reporting runtimes of a dask operator.

    Group:
        Experiments
    """

    graph_construction: float | None = None
    """
    Serial. The runtime used for constructing the dask graph.
    """

    optimization: float | None = None
    """
    Serial. The runtime used for optimizing the dask graph.
    """

    serialization: float | None = None
    """
    Serial. The runtime used for serializing the dask graph.
    """


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
    create_report: bool = False
    """Whether to create a dask report."""

    def setup(self) -> None:
        # start the scheduler and wait for the workers
        scheduler.start_scheduler(cfg=self.cluster_config)
        scheduler.scale_and_wait(self.workers)
        # start the dask report session
        if self.create_report:
            reportfile = util.get_path(cfg["evaluation"]["perf_report_dir"], "dask-report.html")
            self.__dask_report_ctx = performance_report(reportfile)
            self.__dask_report_ctx.__enter__()

    def cleanup(self) -> None:
        # end the dask report session
        if self.create_report:
            self.__dask_report_ctx.__exit__(None, None, None)

    @abc.abstractmethod
    def construct_output(self, *args: Any) -> list[dask.typing.DaskCollection]:
        """
        Abstract class which constructs the output dask collections to be computed.

        Args:
            args: The output of `load_input`.
        """
        pass

    def measure(self) -> DaskRuntime:
        # measure the runtime of the graph construction
        full_start = perf_counter()
        start = perf_counter()
        op_input = self.load_input()
        end = perf_counter()
        input_loading = end - start
        start = perf_counter()
        output = self.construct_output(*op_input)
        end = perf_counter()
        graph_construction = end - start
        # optimize the graph
        start = perf_counter()
        optimized: list[dask.typing.DaskCollection] = dask.optimize(*output)
        end = perf_counter()
        optimization_time = end - start
        # compute
        client = scheduler.get_client()
        if client is not None:
            start = perf_counter()
            fut = client.persist(*optimized)
            end = perf_counter()
            serialization_time = end - start
            start = perf_counter()
            dask.distributed.wait(fut)
            end = perf_counter()
            compute_time = end - start
            del fut
        else:
            serialization_time = None
            start = perf_counter()
            dask.compute(*optimized)
            end = perf_counter()
            compute_time = end - start
        full_end = perf_counter()

        return DaskRuntime(
            full=full_start - full_end,
            input_loading=input_loading,
            compute=compute_time,
            graph_construction=graph_construction,
            optimization=optimization_time,
            serialization=serialization_time,
        )


@dataclass(kw_only=True)
class OperatorRunConfig(DaskRunConfig):
    """
    A run configuration class for running operators conforming to the standard operator signature.

    Group:
        Experiments
    """

    input_obj: Input
    """
    The input object to use.
    """

    input_version: Input.Version
    """
    The input version to use.
    """

    store_output: bool = True
    """
    Whether to store the output to zarr or not.
    """

    def load_input(self) -> list[Any]:
        inp, _ = self.input_obj.get_version(self.input_version)
        return [inp]

    def construct_output(self, *args: xr.Dataset) -> list[dask.typing.DaskCollection]:
        inp = args[0]
        ds = inp + xr.full_like(inp, random.random())  # instead of clone. See https://github.com/dask/dask/issues/9621
        out = self.impl(ds)
        # clone the graph to ensure that the scheduler does not use results computed in an earlier round
        # cloned: da.Array = dask.graph_manipulation.clone(out.data)
        if self.store_output:
            dir_name = "".join(random.choices(string.ascii_letters, k=16))
            self.output_dir = util.get_path(cfg["global"]["tmp_dir"], dir_name)
            out_array: da.Array = out.data.to_zarr(str(self.output_dir), overwrite=False, compute=False)
        else:
            out_array = out.data

        return [out_array]

    def measure(self) -> DaskRuntime:
        # run measurements
        out = super().measure()
        # clear output
        if self.store_output:
            util.remove_if_exists(self.output_dir)
        return out
