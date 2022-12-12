from . import _util
from . import environment as env

# retrieve package version from pyproject.toml
from packaging.version import Version
import importlib.util
if importlib.util.find_spec('finch') is not None:
    import importlib.metadata
    __version__ = Version(importlib.metadata.version('finch'))
else:
    # read the toml if finch is not installed
    import toml
    __toml_data = toml.load(env.__proj_toml)
    __version__ = Version(__toml_data["project"]["version"])
"""
The current version of this package.
"""

from .config import config, set_debug_mode, set_log_level, debug, logging_format, log_level
import os
os.environ["GRIB_DEFINITION_PATH"] = config["data"]["grib_definition_path"]

from . import util
from . import constants as const
from . import data
from . import scheduler
from .experiments import measure_runtimes, measure_operator_runtimes, measure_loading_times
from .experiments import RunConfig, DaskRunConfig
from .experiments import xr_impl_runner, xr_run_prep, xr_input_prep
from .experiments import DefaultOperator
from . import evaluation as eval

from . import brn