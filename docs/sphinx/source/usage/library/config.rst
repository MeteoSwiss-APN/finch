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

Configurable Values
^^^^^^^^^^^^^^^^^^^

``[global]``
""""""""""""

.. confval:: scratch_dir

    :type: string
    :default: ``%SCRATCH``

    The location of a fast file storage system.

.. confval:: tmp_dir

    :type: string
    :default: ``${scratch_dir}/tmp``

    A path to a directory for temporary files

.. confval:: log_dir

    :type: string
    :default: ``${scratch_dir}/logs``

    A path to a directory for log files

.. confval:: log_level

    :type: string, optional
    :default: None

    Sets the log level for finch.
    This can be one of the levels from python's `logging <https://docs.python.org/3/library/logging.html#logging-levels>`_ library.
    This defaults to ``INFO`` if :confval:`debug_mode` is disabled and ``DEBUG`` otherwise.

.. confval:: log_format

    :type: string
    :default: ``"[%(levelname)s]: %(message)s"``

    The format to use for logging, as documented in python's `logging <https://docs.python.org/3/howto/logging.html#changing-the-format-of-displayed-messages>`_ library.

.. confval:: debug_mode

    :type: bool
    :default: ``%DEBUG``

    Toggles debug mode for finch.
    This sets the default :confval:`log_level` to ``DEBUG`` and runs finch on a local dask cluster instead of a SLURM cluster.

``[data]``
""""""""""

.. confval:: grib_definition_path

    :type: string
    :default: ``%GRIB_DEFINITION_PATH``

    The path to a grib definition. Multiple paths can be passed, separated by a colon (``:``).

.. confval:: input_store

    :type: string
    :default: ``${global:scratch_dir}/finch_store``

    The path to a directory which holds the input store for finch.

``[experiments]``
"""""""""""""""""

.. confval:: results_dir

    :type: string
    :default: ``${global:tmp_dir}/results``

    .. warning:: Deprecated. Will be moved to run configurations.

    Path to a directory where experiment results are stored.

.. confval:: scaling_timeout

    :type: int
    :default: 60

    The timeout for waiting for worker startup in seconds, when scaling the dask cluster.

``[evaluation]``
""""""""""""""""

.. confval:: dir

    :type: string
    :default: ``${global:scratch_dir}/finch_eval``

    .. warning:: Deprecated. Will be moved to run configurations.

    Path to a directory where evaluation results are stored.

.. confval:: pref_report_dir

    :type: string
    :default: ``${dir}``

    .. warning:: Deprecated. Will be moved to run configurations.

    Path to a directory where performance reports are stored.

.. confval:: plot_dir

    :type: string
    :default: ``${dir}``

    .. warning:: Deprecated. Will be moved to run configurations.

    Path to a directory where plots are stored.
    If this is the same as :confval:`dir`, the plots will be stored in a separate experiment-specific directory inside the :confval:`dir` directory.

.. confval:: config_dir

    :type: string
    :default: ``${dir}``

    .. warning:: Deprecated. Will be moved to run configurations.

    Path to a directory where experiment configurations are stored.

.. confval:: config_dir

    :type: string
    :default: ``${dir}``

    .. warning:: Deprecated. Will be moved to run configurations.

    Path to a directory where experiment results are stored.

``[brn]``
"""""""""

.. confval:: grib_index_dir

    :type: string
    :default: ``${global:tmp_dir}``

    The path where grib index files for BRN experiments are stored and loaded.

``[run]``
"""""""""

.. confval:: config_path

    :type: string
    :default: ``finch_run_config.yaml``

    The path to a custom run configuration file.

.. confval:: debug_config_path

    :type: string
    :default: ``finch_debug_config.yaml``

    The path to a custom debug configuration file.
