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