from dataclasses import dataclass
import logging
import os
import pathlib
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from . import util

debug = False
"""Debug mode toggle"""

log_level = logging.INFO
"""The current log level"""

logging_format = '[%(levelname)s]: %(message)s'
"""The format used for logging outputs"""

def set_log_level(level):
    global log_level
    log_level = level
    logging.basicConfig(format=logging_format, level=log_level)

def set_debug_mode(dbg: bool):
    global debug, log_level
    debug = dbg
    set_log_level(logging.DEBUG if debug else logging.INFO)

set_debug_mode(debug)

proj_root = str(pathlib.Path(__file__).parent.parent.absolute())
"""The root directory of the project"""

proj_config = os.path.join(proj_root, "config", "finch.ini")
"""The location of the project configuration file"""

default_custom_config = os.path.join(proj_root, "config", "custom.ini")
"""The default location for a custom configuration file"""

custom_config_env_var = "CONFIG"
"""The name of the environment variable specifying the location of a custom configuration file."""

class WorkerEnvironment():
    """
    This class manages environments for dask workers.
    """

    omp_threads: int = 1
    """The number of threads available to openmp"""

    env_var_map = {
        "omp_threads": ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT"]
    }
    """Maps attributes of this class to environment variables"""

    @classmethod
    def load(cls):
        """Returns a new worker environment whose attributes are initialized according to the current environment."""
        out = cls()
        for k, v in out.env_var_map.items():
            if isinstance(v, list):
                v = v[0]
            out.__dict__[k] = out.__dict__[k].__class__(os.environ[v])
        return out

    @property
    def _env_vars(self) -> dict[str, str]:
        """This worker environment as a dictionary mapping environment variable names to values"""
        return {e : str(self.__dict__[v]) for v, es in self.env_var_map for e in util.arg2list(es)}

    def set(self):
        """Sets environment variables according to this worker environment"""
        os.environ.update(self._env_vars)

    def get_job_script_prologue(self) -> list[str]:
        """Returns a list of bash commands setting up the environment of a worker."""
        return [f"export {e}={v}" for e, v in self._env_vars.items()]