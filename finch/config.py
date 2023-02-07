import logging
import os
import pathlib
from configparser import ConfigParser, ExtendedInterpolation
from io import StringIO

from expandvars import expand  # type: ignore

from . import env

config: ConfigParser = ConfigParser(interpolation=ExtendedInterpolation())
"""
This variable contains the configuration of the finch core library.
It is initialized from finch's config files when importing finch.

Group:
    Config
"""


def read_config(cfg_path: str, optional: bool = False) -> None:
    """
    Reads a config file.

    Args:
        cfg_path (str): The path to the config file
        optional (str): If True, nothing happens if the file does not exist.
            If False, an error will be raised.

    Group:
        Config
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

__log_level: str | int | None = None
if "log_level" in config["global"]:
    __log_level = config["global"]["log_level"]


def set_log_level(level: str | int) -> None:
    """Overwrite the current logging level.

    Group:
        Config
    """
    global __log_level
    __log_level = level
    config["global"]["log_level"] = logging.getLevelName(level)
    logging.basicConfig(format=config["global"]["log_format"], level=level)


# debugging

debug = False
"""Debug mode toggle"""

if "debug_mode" in config["global"] and config["global"]["debug_mode"] != "":
    debug = config.getboolean("global", "debug_mode")


def set_debug_mode(dbg: bool) -> None:
    """
    Toggles the debug mode.
    If True, debug mode is enabled.
    If False, it is disabled.
    If the log level was not set explicitely,
    it will be set to INFO if debug is disabled and to DEBUG if it is enabled.

    Group:
        Config
    """
    global debug, __log_level
    debug = dbg
    config["global"]["debug_mode"] = str(debug)
    if __log_level is None:
        set_log_level(logging.DEBUG if debug else logging.INFO)


set_debug_mode(debug)
