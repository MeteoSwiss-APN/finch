from copy import copy
from dataclasses import dataclass
import pathlib
from time import perf_counter, time
from collections.abc import Callable
from typing import Any, TypeVar
import numpy as np
import xarray as xr
from . import Input
from .util import PbarArg
from . import util
from . import config
from . import env
from . import scheduler
import dask.graph_manipulation
import dask.array as da
import dask
import warnings
import dask.distributed

@dataclass
class RunConfig(util.Config):
    impl: Callable = None
    cluster_config: scheduler.ClusterConfig = scheduler.ClusterConfig()
    workers: int = 1

    def setup(self):
        scheduler.start_scheduler(cfg=self.cluster_config)
        scheduler.scale_and_wait(self.workers)

    @classmethod
    def get_class_attr(cls) -> list[str]:
        return util.get_class_attributes(cls)

def list_run_configs(**kwargs) -> list[RunConfig]:
    """
    Returns a list of run configurations, 
    which is the euclidean product between the given lists of individual configurations.
    """
    configs: list[dict[str, Any]] = []
    for arg in kwargs:
        vals = kwargs[arg]
        if not isinstance(vals, list):
            vals = [vals]
        updates = [{arg : v} for v in vals]
        if len(configs) == 0:
            configs = updates
        else:
            configs = [c | u for c in configs for u in updates]
    return [RunConfig(**c) for c in configs]

@dataclass
class Runtime():
    full: float = None
    graph_construction: float = None
    graph_opt: float = None
    graph_serial: float = None
    compute: float = None

def _reduce_runtimes(rt: list[Runtime], reduction: Callable[[np.ndarray, int], np.ndarray]) -> Runtime:
    array = [[r.full, r.graph_construction, r.graph_opt, r.graph_serial, r.compute] for r in rt]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        out = reduction(np.array(array, dtype=float), axis=0)
    out = [o if o is not float("NaN") else None for o in out]
    return Runtime(*out)

def measure_runtimes(
    run_config: list[RunConfig] | RunConfig, 
    inputs: list[Callable[[], list]] | Callable[[], list] | list[list] | None = None, 
    iterations: int = 1,
    impl_runner: Callable[..., Runtime | None] = None,
    cache_inputs: bool = False,
    reduction: Callable[[np.ndarray, int], np.ndarray] = np.nanmean,
    warmup: bool = False,
    pbar: PbarArg = True,
    **kwargs
) -> list[list[Runtime]] | list[Runtime] | Runtime:
    """
    Measures the runtimes of multiple functions, each accepting the same inputs.
    Parameters
    ---
    - funcs: The functions to be benchmarked
    - inputs: The inputs to the functions to be benchmarked. These can be passed in different forms:
        - `None`: Default. Can be passed if the functions do not accept any arguments.
        - `list[list]`: A list of concrete arguments to the functions
        - `list[Callable[[], list]]`: A list of argument generating functions.
        These will be run to collect the arguments for the functions to be benchmarked.
        - `Callable[[], list]`: A single argumnent generating function 
        if the same should be used for every function to be benchmarked.
    - iterations: int, optional. The number of times to repeat a run (including input preparation).
    - impl_runner: Callable. The function responsible for running 
    the impl argument of a run configuration for the given list of arguments.
    The first argument of impl_runner is `impl` while the remaining arguments are the arguments passed to `impl`.
    The execution of `impl_runner` will be timed and reported.
    A runtime can be returned if the implementation runner supports fine-grained runtime repoting.
    Defaults to directly running the passed function on the passed arguments.
    - cache_inputs: bool, default: `True`. Whether to reuse the input for a function for its iterations.
    - reduction: Callable, default: `np.nanmean`. 
    The function to be used to combine the results of the iterations.
    This is a reduction function which is able to reduce a specific dimenion (kwarg axis) of a numpy array.
    - warmup: bool, default: `False`. If `True`, runs the function once before measuring.
    - pbar: PbarArg, default: `True`. Progressbar argument

    Returns
    ---
    The runtimes as a list of lists, or a flat list, or a float, 
    depending on whether a single function or a single version (None or Callable) were passed.
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
        inputs = [lambda : i for i in inputs]

    # prepare impl_runner
    if impl_runner is None:
        impl_runner = lambda f, args: f(args)

    if warmup:
        iterations += 1

    pbar = util.get_pbar(pbar, len(run_config) * len(inputs) * iterations)

    times = []
    for c in run_config:
        c.setup()
        f_times = []
        for prep in inputs:
            cur_times = []
            if cache_inputs:
                args = prep()
                prep = lambda a=args : a
            for _ in range(iterations):
                args = prep()
                start = perf_counter()
                runtime = impl_runner(c.impl, *args)
                end = perf_counter()
                if runtime is None:
                    runtime = Runtime()
                if runtime.full is None:
                    runtime.full = end-start
                cur_times.append(runtime)
                pbar.update()
            if warmup:
                cur_times = cur_times[1:]
            f_times.append(_reduce_runtimes(cur_times, reduction))
        times.append(f_times)
    # reorder according to original run config order
    out = [0]*len(times)
    for i, t in zip(rc_order, times):
        out[i] = t
    # adjust output form
    if singleton_input:
        out = [o[0] for o in out]
    if singleton_rc:
        out = out[0]
    return out

def xarray_impl_runner(impl: Callable[[xr.Dataset], xr.DataArray], ds: xr.Dataset) -> Runtime:
    runtime = Runtime()
    # construct the dask graph
    start = perf_counter()
    out = impl(ds)
    cloned: da.Array = dask.graph_manipulation.clone(out.data)
    end = perf_counter()
    runtime.graph_construction = end-start
    # optimize the graph
    start = perf_counter()
    optimized = dask.optimize(cloned)[0]
    end = perf_counter()
    runtime.graph_opt = end-start
    # compute
    client = scheduler.get_client()
    if client is not None:
        start = perf_counter()
        fut = client.persist(optimized)
        end = perf_counter()
        runtime.graph_serial = end-start
        start = perf_counter()
        dask.distributed.wait(fut)
        end = perf_counter()
        runtime.compute = end-start
    else:
        start = perf_counter()
        optimized.compute()
        end = perf_counter()
        runtime.compute = end-start
    return runtime


def measure_operator_runtimes(
    run_config: list[RunConfig] | RunConfig, 
    input: Input,
    versions: list[Input.Version] | Input.Version,
    **kwargs
) -> list[list[float]] | list[float] | float:
    """
    Measures the runtimes of different implementations of an operator against different input versions.

    Parameters
    ---
    - run_config: The runtime configurations
    - input: The input object for the operator
    - versions: The different input versions to be benchmarked
    - kwargs: Arguments for `measure_runtimes`
    """
    single_version = False
    if isinstance(versions, Input.Version):
        single_version = True
        versions = [versions]
    preps = [
        lambda v=v : [input.get_version(v)[0]]
        for v in versions
    ]
    if single_version:
        preps = preps[0]
    return measure_runtimes(run_config, preps, impl_runner=xarray_impl_runner, **kwargs)

def measure_loading_times(
    input: Input,
    versions: list[Input.Version],
    **kwargs
) -> list[float]:
    """
    Measures the loading times of different versions of an input

    Parameters
    ---
    - input: The input to be loaded
    - versions: The different versions to be measured
    - kwargs: Arguments for `measure_runtimes`
    """
    funcs = [lambda v=v : input.get_version(v) for v in versions]
    run_config = list_run_configs(impl=funcs)
    return measure_runtimes(run_config, None, **kwargs)