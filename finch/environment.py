import os
import pathlib
from . import _util

proj_root = str(pathlib.Path(__file__).parent.parent.absolute())
"""The root directory of the project"""

proj_config = os.path.join(proj_root, "config", "finch.ini")
"""The location of the project configuration file"""

default_custom_config = os.path.join(proj_root, "config", "custom.ini")
"""The default location for a custom configuration file"""

custom_config_env_var = "CONFIG"
"""The name of the environment variable specifying the location of a custom configuration file."""

node_name_env_var = "SLURMD_NODENAME"

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
        return {e : str(self.__dict__[v]) for v, es in self.env_var_map.items() for e in _util.arg2list(es)}

    def set(self):
        """Sets environment variables according to this worker environment"""
        os.environ.update(self._env_vars)

    def get_job_script_prologue(self) -> list[str]:
        """Returns a list of bash commands setting up the environment of a worker."""
        return [f"export {e}={v}" for e, v in self._env_vars.items()]