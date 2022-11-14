from configparser import ConfigParser, ExtendedInterpolation
from . import env
from . import _util
import os
import pathlib
import logging

config = ConfigParser(os.environ, interpolation=ExtendedInterpolation())

# built-in config (defaults)
with open(env.proj_config) as f:
    config.read_file(f)

# custom config from default location
if pathlib.Path(env.default_custom_config).exists():
    with open(env.default_custom_config) as f:
        config.read_file(f)

# custom config from custom location
if env.custom_config_env_var in os.environ:
    with open(os.environ[env.custom_config_env_var]) as f:
        config.read_file(f)

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