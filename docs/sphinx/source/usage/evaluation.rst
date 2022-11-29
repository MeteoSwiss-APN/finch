Evaluation
==========

The Results Dataset
-------------------

The runtimes from our experiments are returned in a rather raw format with lists.
While Python lists are easy to understand, they get rather impractical when trying to work with the data.
For this reason, the evaluation module uses ``xarray.Dataset`` to work with experiment results.
We can easily create such a results dataset from the output of a runtime experiment with :func:`finch.evaluation.create_results_dataset`.
::
    from finch import eval, RunConfig
    from finch.brn import list_brn_implementations, get_brn_input

    input = get_brn_input()
    versions = input.list_versions()
    rc = RunConfig.list_configs(impl=list_brn_implementations())
    times = measure_operator_runtimes(rc, input, versions),

    results = eval.create_results_dataset(times, rc, versions, input, name="brn")

The results daataset holds all the information necessary for reproducing our original experiment.
Every configurable attribute in the run configurations and the versions will receive a separate dimension in the dataset.
The values of the configuration options will be stored as coordinates for the dimensions.
These coordinates are of numeric form, if the values were initially numeric (not including booleans).
If they are not of numeric form, finch will try to create an expressive string representation of them.
Every runtime that was recorded will receive a dedicated array in the dataset.

We can use xarray's built-in functions for loading and storing experiment results.
::
    import xarray as xr
    
    results.to_netcdf("results.nc")
    results = xr.open_dataset("results.nc")

NetCDF files are not human-readable, which might be impractical when we want to see the configuration of a previous experiment.
For this reason, finch also supports storing the configuration as a yaml file with :func:`finch.evaluation.store_config`
::
    eval.store_config(results)


Plotting Runtime Results
------------------------

Finch's evaluation module provides the function :func:`finch.evaluation.create_plots` for plotting the results of a runtime experiment.
Per default, it creates a plot for every runtime type and every configuration option, which had multiple different values.
The plot is a line plot, if the configuration option is of a numeric type (excluding boolean), and a bar plot otherwise.

The function uses `matplotlib <https://matplotlib.org/stable/index.html>`_ along with `matplotx <https://github.com/nschloe/matplotx>`_ for plotting.
::
    eval.create_plots(results)

The plots will be stored in a directory named after the name of the experiment inside the plot directory.
The plot directory can be configured via finch's configuration.
If the plot directory is the same as the evaluation directory, the plots will be put inside a separate directory of the experiment directory.
This separates the plots properly from other experiment-specific data.
The location of the plots for an experiment can be retrieved with the :func:`finch.evaluation.get_plots_dir` function.
::
    print(eval.get_plots_dir(results))

A dimension will be selected as a main dimension, which can be used for direct comparison between the configured values within a plot.
In a line plot, there will be individual lines for the different values of the main dimension.
In a bar plot, there will be individual, grouped bars for the different values.
By default, the ``"impl"`` dimension will be used, which corresponds to the different implementations of the operator.
We can change the main dimension via the ``main_dim`` parameter.
The name must match a dimension in the results dataset.
These are the names of the configuration options for the experiment.
Nested configurations are flattened and their names are combined with an underscore (_).
For example, the number of cores per worker can be configured via the cluster configuration in the run configuration.
Its name will be ``"cluster_config_cores_per_worker"``.
::
    eval.create_plots(results, main_dim="format")