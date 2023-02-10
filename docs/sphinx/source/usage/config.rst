.. _config:

Configuration
==================

Finch
------

Finch uses Python's `configparser <https://docs.python.org/3/library/configparser.html>`_ library for configuration.
The default configuration can be found at ``config/finch.ini``, which lists all available configuration options.
You can provide a custom configuration file at the location ``config/custom.ini``.
Alternatively, you can specify the location of a custom configuration file via the ``CONFIG`` environment variable.

A finch config file is a standard ``.ini`` file.
You can cross-reference variables with ``${section:variable}``.
You can reference environment variables with the ``%`` prefix, which allows for standard bash substitutions.

Run and Debug Configuration
---------------------------

The ``run.py`` script has its own configuration system via Python source files.
The purpose of this configuration is to control how the built-in experiments provided by finch are being run.
The default run configuration is provided in ``run_config.py`` and the default debug configuration is provided in ``debug_config.py``.
If you want to overwrite those settings, you can duplicate ``custom_config_template.py`` and rename it to ``custom_config.py``.
There you can import the default debug and run configuration and overwrite their values.
