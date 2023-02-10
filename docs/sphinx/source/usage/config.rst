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
