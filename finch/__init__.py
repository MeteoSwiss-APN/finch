from packaging.version import Version
import importlib.metadata
version = Version("0.0.1a1")
# __version__ = Version(importlib.metadata.version("finch_mch")) # use this for the future
"""
The current version of this package.
"""

from . import _util
from . import environment as env
from .config import config, set_debug_mode, set_log_level, debug, logging_format, log_level
import os
os.environ["GRIB_DEFINITION_PATH"] = config["data"]["grib_definition_path"]

from . import util
from . import constants as const
from . import data
from . import scheduler
from .experiments import measure_runtimes, measure_operator_runtimes, measure_loading_times
from .experiments import RunConfig, DaskRunConfig
from .experiments import xr_impl_runner, get_xr_run_prep, xr_input_prep
from .experiments import DefaultOperator
from . import evaluation as eval

from . import brn