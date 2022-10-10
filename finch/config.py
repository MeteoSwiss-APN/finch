from configparser import ConfigParser, ExtendedInterpolation
from . import environment as env
import os
import pathlib

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