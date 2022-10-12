from time import time
from typing import Any, Callable, TypeVar
import numpy as np

def measure_runtimes(
    funcs: list[Callable[..., Any]], 
    inputs: list[Callable[[], list]] | list[list] | None = None, 
    iterations: int = 1,
    cache_inputs: bool = True,
    reduction: Callable[[list[float]], float] = np.mean,
    warmup: bool = False,
    **kwargs
) -> list[float]:
    """
    Measures the runtimes of multiple functions.

    Arguments:
    ---
    - funcs: The functions to be benchmarked
    - inputs: The inputs to the functions to be benchmarked. These can be passed in different forms:
        - `None`: Default. Can be passed if the functions do not accept any arguments.
        - `list[list]`: A list of concrete arguments to the functions
        - `list[Callable[[], list]]`: A list of argument generating functions.
        These will be run to collect the arguments for the functions to be benchmarked.
    - iterations: int, optional. The number of times to repeat a run (including input preparation).
    - cache_inputs: bool, default: `True`. Whether to reuse the input for a function for its iterations.
    - reduction: Callable[[list[float]], float], default: `np.mean`. The function to be used to combine the results of the iterations.
    - warmup: bool, default: `False`. If `True`, runs the function once before measuring.
    """
    # special case
    if len(funcs) == 0:
        return []
    # prepare inputs to all have the same form
    if inputs is None:
        inputs = [[]] * len(funcs)
    if isinstance(inputs[0], list):
        inputs = [lambda : i for i in inputs]

    if warmup:
        iterations += 1

    out = []
    for f, prep in zip(funcs, inputs):
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
        if warmup:
            cur_times = cur_times[1:]
        out.append(reduction(cur_times))
    return out

def measure_runtime(f: Callable[..., Any], input_prep: Callable[[], list] | list | None = None, **kwargs) -> float:
    """
    Convenience function to run `measure_runtimes` on just a single input
    """
    if input_prep is None:
        input_prep = []
    return measure_runtimes([f], [input_prep], **kwargs)[0]