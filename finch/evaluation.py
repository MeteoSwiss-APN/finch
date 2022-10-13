from typing import Any
from collections.abc import Callable
from . import Input


def print_version_results(results: list[Any], versions: list[Input.Version]):
    """
    Prints the results of an experiment for different input versions.
    """
    for r, v in zip(results, versions):
        print(f"{r}\n    {v}")

def print_imp_results(results: list[list[Any]], imps: list[Callable], versions: list[Input.Version]):
    """
    Prints the results of an experiment for different implementations and input versions.
    """
    for imp, r in zip(imps, results):
        print(imp.__name__)
        print()
        print_version_results(r, versions)
        print()