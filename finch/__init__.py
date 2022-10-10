from .config import config
import os
os.environ["GRIB_DEFINITION_PATH"] = config["data"]["grib_definition_path"]

from .scheduler import start_scheduler, start_slurm
from . import environment as env
from .experiments import measure_runtimes

from . import brn as brn_
from .brn.impl import brn_blocked_cpp as brn
from .brn.impl import thetav_blocked_cpp as thetav