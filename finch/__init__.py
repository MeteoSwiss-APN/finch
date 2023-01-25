from . import _util
from . import environment as env

__version__ = env.get_version()
"""
The current version of this package.
"""

import os

from .config import config as cfg
from .config import debug, log_level, logging_format, set_debug_mode, set_log_level

os.environ["GRIB_DEFINITION_PATH"] = config["data"]["grib_definition_path"]

from . import brn
from . import constants as const
from . import data
from . import evaluation as eval
from . import scheduler, util
from .experiments import (
    DaskRunConfig,
    DefaultOperator,
    RunConfig,
    measure_loading_times,
    measure_operator_runtimes,
    measure_runtimes,
    xr_impl_runner,
    xr_input_prep,
    xr_run_prep,
)
