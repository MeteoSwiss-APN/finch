from time import time
from collections.abc import Callable
from typing import Any, TypeVar
import numpy as np
import xarray as xr
from . import Input
from .util import PbarArg
from . import util
import tqdm

def measure_runtimes(
    funcs: list[Callable[..., Any]] | Callable[..., Any], 
    inputs: list[Callable[[], list]] | Callable[[], list] | list[list] | None = None, 
    iterations: int = 1,
    cache_inputs: bool = True,
    reduction: Callable[[list[float]], float] = np.mean,
    warmup: bool = False,
    pbar: PbarArg = True,
    **kwargs
) -> list[list[float]] | list[float] | float:
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
    - cache_inputs: bool, default: `True`. Whether to reuse the input for a function for its iterations.
    - reduction: Callable[[list[float]], float], default: `np.mean`. The function to be used to combine the results of the iterations.
    - warmup: bool, default: `False`. If `True`, runs the function once before measuring.
    - pbar: PbarArg, default: `True`. Progressbar argument

    Returns
    ---
    The runtimes as a list of lists, or a flat list, or a float, 
    depending on whether a single function or a single version (None or Callable) were passed.
    """
    # prepare funcs
    singleton_funcs = isinstance(funcs, Callable)
    if singleton_funcs:
        funcs = [funcs]

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

    if warmup:
        iterations += 1

    pbar = util.get_pbar(pbar, len(funcs) * len(inputs) * iterations)

    out = []
    for f in funcs:
        f_out = []
        for prep in inputs:
            cur_times = []
            if cache_inputs:
                args = prep()
                prep = lambda : args
            for _ in range(iterations):
                args = prep()
                start = time()
                f(*args)
                end = time()
                cur_times.append(end - start)
                pbar.update()
            if warmup:
                cur_times = cur_times[1:]
            f_out.append(reduction(cur_times))
        out.append(f_out)
    if singleton_input:
        out = [o[0] for o in out]
    if singleton_funcs:
        out = out[0]
    return out

def measure_operator_runtimes(
    funcs: list[Callable[[xr.Dataset], Any]] | Callable[[xr.Dataset], Any], 
    input: Input,
    versions: list[Input.Version] | Input.Version,
    **kwargs
) -> list[list[float]] | list[float] | float:
    """
    Measures the runtimes of different implementations of an operator against different input versions.

    Parameters
    ---
    - funcs: The operator implementations
    - input: The input object for the operator
    - versions: The different input versions to be benchmarked
    - kwargs: Arguments for `measure_runtimes`
    """
    if isinstance(versions, list):
        preps = [
            lambda v=v : [input.get_version(v)[0]]
            for v in versions
        ]
    else:
        preps = lambda : input.get_version(versions)
    # make sure to run compute
    if isinstance(funcs, Callable):
        funcs = lambda x, funcs=funcs : funcs(x).compute()
    else:
        funcs = [lambda x, f=f : f(x).compute() for f in funcs]
    return measure_runtimes(funcs, preps, **kwargs)

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
    return measure_runtimes(funcs, None, **kwargs)