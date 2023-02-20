from . import _util
from . import environment as env

__version__: env.Version = env.get_version()
"""
The current version of this package.
"""

import os

from .config import config as cfg
from .config import debug, set_debug_mode, set_log_level

os.environ["GRIB_DEFINITION_PATH"] = cfg["data"]["grib_definition_path"]

if 1:  # this is a hack to avoid flake8 to complain about E402
    from . import constants as const
    from . import data
    from . import evaluation as eval
    from . import operators as ops
    from . import scheduler, util
    from .experiments import (
        DaskRunConfig,
        DaskRuntime,
        DefaultOperator,
        OperatorRunConfig,
        RunConfig,
        Runtime,
        measure_runtimes,
    )

__all__ = [
    "_util",
    "cfg",
    "debug",
    "log_level",
    "logging_format",
    "set_debug_mode",
    "set_log_level",
    "const",
    "data",
    "eval",
    "ops",
    "scheduler",
    "util",
    "DaskRunConfig",
    "DefaultOperator",
    "RunConfig",
    "DaskRunConfig",
    "OperatorRunConfig",
    "Runtime",
    "DaskRuntime",
    "measure_runtimes",
    "__version__",
]
