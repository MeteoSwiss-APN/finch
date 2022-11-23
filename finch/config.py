from configparser import ConfigParser, ExtendedInterpolation
from . import env
from . import _util
import os
import logging
import pathlib
from io import StringIO
from expandvars import expand

config = ConfigParser(interpolation=ExtendedInterpolation())

def read_config(cfg_path: str, optional: bool = False):
    """
    Reads a config file.

    Arguments:
    ---
    - cfg_path: The path to the config file
    - optional: If True, nothing happens if the file does not exist.
    If False, an error will be raised.
    """
    if optional and not pathlib.Path(cfg_path).exists():
        return
    with open(cfg_path) as f:
        cfg_txt = expand(f.read(), var_symbol="%")
        config.readfp(StringIO(cfg_txt))

# built-in config (defaults)
read_config(env.proj_config)

# custom config from default location
read_config(env.default_custom_config, optional=True)

# custom config from custom location
if env.custom_config_env_var in os.environ:
    read_config(os.environ[env.custom_config_env_var])

# logging

log_level = logging.INFO
"""The current log level"""

logging_format = '[%(levelname)s]: %(message)s'
"""The format used for logging outputs"""

def set_log_level(level):
    global log_level
    log_level = level
    logging.basicConfig(format=logging_format, level=log_level)

# debugging

debug = False
"""Debug mode toggle"""

if "debug_mode" in config["global"] and config["global"]["debug_mode"] != "":
    debug = _util.parse_bool(config["global"]["debug_mode"])

def set_debug_mode(dbg: bool):
    global debug, log_level
    debug = dbg
    set_log_level(logging.DEBUG if debug else logging.INFO)

set_debug_mode(debug)

def get_debug_mode() -> bool:
    return debug