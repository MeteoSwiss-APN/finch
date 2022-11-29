import functools
from collections.abc import Callable
import xarray as xr
from . import impl
from .. import util
from .. import DefaultOperator as Operator

brn = impl.brn_blocked_cpp
"""
The default brn implementation

Group:
    BRN
"""
thetav = impl.thetav_blocked_cpp
"""
The default thetav implementation

Group:
    THETAV
"""

THETAV_REGEX = "thetav*"
"""Function name regex for discovering implementations of thetav"""
BRN_REGEX = "brn*"
"""Function name regex for discovering implementations of brn"""


def list_thetav_implementations() -> list[Operator]:
    """Returns all available implementations of the thetav operator.

    Returns:
        list[Operator]: A list of all available implementations of the thetav operator

    Group:
        THETAV
    """
    return util.list_funcs_matching(impl, THETAV_REGEX, Operator)


def list_brn_implementations() -> list[Operator]:
    """Returns all available implementations of the brn operator

    Returns:
        list[Operator]: A list of all abailable implementations of the brn operator

    Group:
        BRN
    """
    return util.list_funcs_matching(impl, BRN_REGEX, Operator)


def get_repeated_brn(n: int, base: Callable[[xr.Dataset, int], xr.DataArray]=impl.brn_blocked_cpp) -> Operator:
    """Returns a repeated version of a BRN implementation.
    A repeated version repeats the brn computation iteratively, while the output of a previous iteration is used as an input of the next iteration.

    Args:
        n (int): The number of iterations to perform.
        base (Callable[[xr.Dataset, int], xr.DataArray], optional): The BRN implementation to repeat.
            This implementation must support a `reps` argument.
            Defaults to impl.brn_blocked_cpp.

    Returns:
        Operator: The repeated BRN version as a finch operator.

    Group:
        BRN
    """
    return functools.partial(base, reps=n)


def get_repeated_brn_name(impl: functools.partial) -> str:
    """Returns a descriptive name of a repeated BRN operator.

    Args:
        impl (functools.partial): The repeated BRN operator
            This must be the output of :func:`finch.brn.get_repeated_brn`.

    Returns:
        str: A descriptive name of the repeated operator

    See Also:
        :func:`finch.brn.get_repeated_brn`

    Group:
        BRN
    """
    reps = str(impl.keywords["reps"])
    return f"repeated_{reps}"
