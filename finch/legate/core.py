from dataclasses import dataclass
import xarray as xr
from collections.abc import Callable
import subprocess
import os

from .. import RunConfig
from .. import env

@dataclass
class LegateRunConfig(RunConfig):
    """
    A run configuration for running legate implementations.

    Group:
        Legate
    """

    nodes: int = None
    """
    The number of nodes to run computation on
    """
    cores: int = None
    """
    The number of cores available per node
    """
    gpus: int = None
    """
    The number of GPU units available per node
    """
    mem: int = None
    """
    The amount of memory available per node (in MB)
    """
    fbmem: int = None
    """
    The amount of framebuffer memory available per GPU (in MB)
    """

def legate_impl_runner(cfg: LegateRunConfig, ds: xr.Dataset):
    """
    An implementation runner for legate implementations

    Args:
        cfg (LegateRunConfig): The run configuration to run
        ds (xr.Dataset): The imput for the implementation
    """
    cmd = ["legate"]
    if cfg.nodes is not None:
        cmd += ["--nodes", str(cfg.nodes)]
    if cfg.cores is not None:
        cmd += ["--cpus", str(cfg.cores)]
    if cfg.gpus is not None:
        cmd += ["--gpus", str(cfg.gpus)]
    if cfg.mem is not None:
        cmd += ["--sysmem", str(cfg.mem)]
    if cfg.fbmem is not None:
        cmd += ["--fbmem", str(cfg.fbmem)]

    module = cfg.impl.__module__
    file = os.path.join(env.package_root, "legate", module)
    fname = cfg.impl.__name__
    cmd += [file, fname] # TODO: Provide input