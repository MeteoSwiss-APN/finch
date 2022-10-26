from . import environment as env
from .config import config
import os
os.environ["GRIB_DEFINITION_PATH"] = config["data"]["grib_definition_path"]

from . import util
from . import constants as const
from .data import Input
from . import data
from .scheduler import start_scheduler, start_slurm_cluster
from .experiments import measure_runtimes, measure_operator_runtimes, measure_loading_times
from .evaluation import print_version_results, print_results
from . import evaluation as eval

from . import brn