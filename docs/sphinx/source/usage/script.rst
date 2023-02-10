Running Finch
=============

If you've installed finch regularly, the run script should be available on your path.
Make sure that you run finch on a system with `SLURM <https://slurm.schedmd.com/>`_ support.

.. code-block:: bash

    finch

Configuration
-------------

This runs finch with the default configuration, which can be found in ``src/finch/data/config/run.yaml``.
The default run configuration file contains all configurable options along with documentation.
If you want to override the default configuration, duplicate the default configuration file into the current directory and rename it to ``finch_run_config.yaml``.
This file will be picked up by finch and will override the default configuration with the values which are present in ``finch_run_config.yaml``.

.. note:: You can override the default path of the custom run (and debug) configuration, as documented in :ref:`config`.

You can also use the command line options ``-c`` or ``--config`` to specify the location of a run config file.

.. code-block:: bash

    finch -c custom_run_config.yaml


Debugging
---------

You can launch ``finch`` in debug mode with the flag ``-d`` or ``--debug``.

.. code-block:: bash

    finch -d

In debug mode, finch will print debug information and will run the experiments on a local dask cluster instead of the SLURM cluster.

You can provide a separate config file for debugging, which overrides the run configuration, wherever specified.
The default debug config file is located at ``src/finch/data/config/debug.yaml``.
You can provide a custom debug config in a file called ``finch_debug_config.yaml``
Alternatively, you can specify the location of your debug config with the ``-p`` or ``--debug-config`` flag.

.. code-block:: bash

    finch -p custom_debug_config.yaml
