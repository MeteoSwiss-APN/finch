# basic utility functions
# they can be used during module setup

from typing import TypeVar

T = TypeVar("T")
def arg2list(x: T | list[T]) -> list[T]:
    """Return a single-element list if x is not a list. Otherwise return x."""
    if not isinstance(x, list):
        x = [x]
    return x

def parse_bool(b: str) -> bool:
    b = b.lower()
    if b == "true":
        return True
    elif b == "false":
        return False
    else:
        raise ValueError(f"Could not parse {b} to boolean type.")

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
def map_keys(d1: dict[T, U], d2: dict[T, V]) -> dict[V, U]:
    """
    Maps the keys of `d1` according to `d2`.
    """
    return {d2[k] : v for k, v in d1.items()}