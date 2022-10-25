import functools
import inspect
from collections.abc import Callable
import xarray as xr
from . import impl
from .. import util

brn = impl.brn_blocked_cpp
"""The default brn implementation"""
thetav = impl.thetav_blocked_cpp
"""The default thetav implementation"""

THETAV_SIG = Callable[[xr.Dataset],xr.DataArray]
"""Function signature for implementations of thetav."""

BRN_SIG = Callable[[xr.Dataset],xr.DataArray]
"""Function signature for implementations of brn."""

THETAV_REGEX = "thetav*"
"""Function name regex for discovering implementations of thetav"""
BRN_REGEX = "brn*"
"""Function name regex for discovering implementations of brn"""

def list_thetav_implementations() -> list[THETAV_SIG]:
    return util.list_funcs_matching(impl, THETAV_REGEX, THETAV_SIG)

def list_brn_implementations() -> list[BRN_SIG]:
    return util.list_funcs_matching(impl, BRN_REGEX, BRN_SIG)

def get_repeated_implementation(n: int):
    return functools.partial(impl.brn_blocked_cpp, reps=n)
