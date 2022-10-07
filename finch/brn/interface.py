import inspect
from typing import Callable
import xarray as xr
from . import impl
from .. import util

brn = impl.brn_blocked_cpp
"""The default brn implementation"""
thetav = impl.thetav_blocked_cpp
"""The default thetav implementation"""

THETAV_SIG = inspect.signature(thetav)
"""Function signature for implementations of thetav."""

BRN_SIG = inspect.signature(brn)
"""Function signature for implementations of brn."""

THETAV_REGEX = "thetav*"
"""Function name regex for discovering implementations of thetav"""
BRN_REGEX = "brn*"
"""Function name regex for discovering implementations of brn"""

def list_thetav_implementations() -> list[Callable]:
    util.list_funcs_matching(impl, THETAV_REGEX, THETAV_SIG)

def list_brn_implementations() -> list[Callable]:
    util.list_funcs_matching(impl, BRN_REGEX, BRN_SIG)

def list_thetav_input_preps(thetav_imps: list[Callable], filetype: str = "zarr") -> list[Callable]:
    out = []
    for imp in thetav_imps:
        pass # TODO: Define an elegant way for specifying input preparations for functions
