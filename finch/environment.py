import os
import pathlib

proj_root = str(pathlib.Path(__file__).parent.parent.absolute())
"""The root directory of the project"""

proj_config = os.path.join(proj_root, "config", "finch.ini")
"""The location of the project configuration file"""

default_custom_config = os.path.join(proj_root, "config", "custom.ini")
"""The default location for a custom configuration file"""

custom_config_env_var = "CONFIG"
"""The name of the environment variable specifying the location of a custom configuration file."""

cluster = None
"""The currently active dask.distributed cluster"""

hostname = os.environ.get("SLURMD_NODENAME", default="localhost")
"""
The network hostname where finch is running.
This is either the SLURM node name if finch is running on a compute node or `localhost`.
"""