import logging
import os
import pathlib
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

debug = False
"""Debug mode toggle"""

log_level = logging.INFO
"""The current log level"""

def set_debug_mode(dbg: bool):
    global debug, log_level
    debug = dbg
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level)

set_debug_mode(debug)

proj_root = str(pathlib.Path(__file__).parent.parent.absolute())
"""The root directory of the project"""

proj_config = os.path.join(proj_root, "config", "finch.ini")
"""The location of the project configuration file"""

default_custom_config = os.path.join(proj_root, "config", "custom.ini")
"""The default location for a custom configuration file"""

custom_config_env_var = "CONFIG"
"""The name of the environment variable specifying the location of a custom configuration file."""