.. _config:

Configuration
=============

Specifying Config Files
-----------------------

Finch uses Python's `configparser <https://docs.python.org/3/library/configparser.html>`_ library for configuration.
The default configuration can be found at ``src/finch/data/config/default.ini``.
You can provide a custom configuration file named ``finch.ini`` in the current working directory.
Alternatively, you can specify the location of a custom configuration file via the ``CONFIG`` environment variable.
A custom configuration file will overwrite the individual configuration options from the default configuration.

Config File Content
-------------------

A finch config file is a standard ``.ini`` file.
You can cross-reference variables with ``${section:variable}``.
You can reference environment variables with the ``%`` prefix, which allows for standard bash substitutions.
