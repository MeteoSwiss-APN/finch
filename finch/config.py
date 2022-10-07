import configparser
from . import environment as env

config = configparser.ConfigParser()
with open(env.proj_root + "/config/finch.ini") as f:
    config.read_file(f)