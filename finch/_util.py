# basic utility functions
# can be used during module setup, which is the only place where this module should be imported directly
# otherwise use the `util` module

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
    Returns a dictionary where the keys of `d1` were mapped according to `d2`.
    """
    return {d2[k] : v for k, v in d1.items()}

T = TypeVar("T")
U = TypeVar("U")
def inverse(d: dict[T, U]) -> dict[U, V]:
    """Returns the inverse dictionary of the input."""
    return {v : k for k, v in d.items()}