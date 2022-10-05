from time import time
from typing import Any, Callable, TypeVar
import numpy as np

def measure_runtimes(funcs: list[Callable[..., Any]], input_preps: list[Callable[[], list]], iterations: int = 1) -> list[float]:
    """
    Measures the runtimes of multiple functions.

    Arguments:
    ---
    - funcs: The functions to be benchmarked
    - input_preps: A list of functions which prepare the inputs of `funcs`. The inputs must be returned as lists.
    - iterations: int, optional. The number of times to repeat a run (including input preparation). The final reported time will be an average.
    """
    out = []
    for f, prep in zip(funcs, input_preps):
        cur_times = []
        for _ in range(iterations):
            args = prep()
            start = time()
            f(*args)
            end = time()
            cur_times.append(end - start)
        out.append(np.array(cur_times).mean())
    return out