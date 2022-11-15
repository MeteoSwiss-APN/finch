from contextlib import nullcontext
import functools
import logging
from dask.distributed import performance_report
import pathlib
import sys
import argparse
import os

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true")
cmd_args = parser.parse_args(sys.argv[1:])

######################################################
# configuration
######################################################

# pre-import configurations

debug = cmd_args.debug
"""Debug mode"""
debug_finch = debug
"""Whether to use finch in debug mode."""

# apply pre-import configurations
os.environ["DEBUG"] = str(debug_finch)
import finch
import xarray as xr


# general configurations
######################################################

iterations = 1 if debug else 5
"""The number of iterations when measuring runtime"""
warmup = not debug
"""Whether to perform a warmup before measuring the runtimes"""
cache_input = True
"""Whether to cache the input between multiple iterations"""
pbar = True
"""Whether to print a progress bar"""

# BRN experiment settings
######################################################

run_brn = True
"""Whether to run brn experiments"""
brn_results_file = pathlib.Path(finch.config["global"]["tmp_dir"], "brn_results.nc")

# input management

brn_add_input_version = False
"""Whether to alter the brn input versions"""
brn_add_input_version = finch.Input.Version(
    format=finch.data.Format.NETCDF,
    dim_order="xyz",
    chunks={"x": 10},
    coords=False
)
"""New brn input version to add"""

# input loading

brn_load_experiment = False
"""Whether to measure the different input loading times"""

# single run

brn_single_run = False
"""Whether to perform a single run experiment with brn"""
brn_imp_to_inspect = finch.brn.impl.brn_blocked_cpp
"""The brn implementation to inspect during the single run experiment"""
brn_single_versions = [finch.Input.Version(
    format=finch.data.Format.ZARR,
    dim_order="xyz",
    chunks={"x": 30},
    coords=False
)]
"""The input version for the brn single run experiment"""
brn_single_workers = 3
"""The number of jobs to spawn for the brn single run"""
brn_single_perf_report = True
"""Whether to record a performance report for the brn single run"""
brn_single_iterations = iterations
"""The number of iterations for the brn single run experiment"""
brn_single_name = "single"
"""The name for the brn single run experiment"""

# runtime measurements

brn_measure_runtimes = False
"""Wether to measure brn runtimes"""
brn_input_versions = finch.Input.Version.list_configs(
    format=finch.data.Format.ZARR,
    dim_order="xyz",
    coords=False,
    chunks={"x" : 30}
) + finch.Input.Version.list_configs(
    format=finch.data.Format.NETCDF,
    dim_order="xyz",
    coords=False,
    chunks={"x" : 30}
) + finch.Input.Version.list_configs(
    format=finch.data.Format.GRIB,
    dim_order="zyx",
    coords=False,
    chunks={"z" : 2}
)
"""The input versions for the runtime experiment"""
brn_imps = finch.brn.impl.brn_xr #[finch.brn.interface.get_repeated_implementation(n, base=finch.brn.impl.brn_blocked_cpp) for n in [10, 20, 30, 40, 50]]
"""The brn implementations used"""
brn_exp_name = "input_loading"
"""The name of the runtime experiment"""
brn_workers = [1, 5, 10, 15, 20, 25, 30, 35, 40]
"""A list of the number of workers to spawn for the runtime experiment"""
brn_cores_per_worker = 1
"""The number of cores dedicated to each worker"""
brn_omp = False
"""Whether to delegate parallelism to OpenMP for a worker"""
brn_cluster_configs = finch.scheduler.ClusterConfig.list_configs(
    cores_per_worker=brn_cores_per_worker,
    omp_parallelism=brn_omp,
    exclusive_jobs=False
)
"""The cluster configurations"""
run_configs = finch.experiments.RunConfig.list_configs(
    workers=brn_workers,
    impl=brn_imps,
    cluster_config=brn_cluster_configs
)
"""The run configurations"""
brn_perf_report = len(run_configs) == 1
"""Whether to write a dask performance report"""

# evaluation

brn_evaluation = True
"""Whether or not to run evaluation"""
brn_eval_runtimes_plot = ["full"]
"""The runtimes to plot"""
brn_eval_main_dim = "format"


######################################################
# script
######################################################

if __name__ == "__main__":

    # configure logging
    logging.basicConfig(format=finch.logging_format, level=logging.INFO)

    # configure debug setup
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        finch.set_log_level(logging.DEBUG)

    # brn experiments

    config = {
        "iterations": iterations,
        "warmup": warmup,
        "cache_inputs": cache_input,
        "pbar": pbar
    }

    brn_input = finch.brn.brn_input

    if run_brn:

        if brn_add_input_version:
            logging.info("Adjusting input versions")
            if brn_add_input_version:
                brn_input.add_version(brn_add_input_version)

        if brn_load_experiment:
            logging.info("Measuring brn input load times")
            times = finch.measure_loading_times(brn_input, brn_input.versions, **config)
            finch.print_version_results(times, brn_input.versions)
            print()
        
        if brn_measure_runtimes:
            logging.info(f"Measuring runtimes of brn implementations")
            reportfile = finch.util.get_path(finch.config["evaluation"]["perf_report_dir"], "dask-report.html")
            with performance_report(filename=reportfile) if brn_perf_report else nullcontext():
                times = finch.measure_operator_runtimes(run_configs, brn_input, brn_input_versions, **config)
            results = finch.eval.create_result_dataset(times, run_configs, brn_input_versions, finch.brn.brn_input, "brn_"+brn_exp_name)
            results.to_netcdf(brn_results_file)

        if brn_evaluation:
            logging.info(f"Evaluating experiment results")
            results = xr.open_dataset(brn_results_file)
            results = finch.eval.create_cores_dimension(results)
            plot_fits = results["full"].sizes[brn_eval_main_dim] < 10
            finch.eval.create_plots(results, main_dim=brn_eval_main_dim, runtime_selection=brn_eval_runtimes_plot, plot_scaling_fits=plot_fits)
            if len(results.data_vars) > 1:
                finch.eval.plot_runtime_parts(results)
            finch.eval.store_config(results)
            results.to_netcdf(finch.util.get_path(finch.config["evaluation"]["results_dir"], results.attrs["name"], "results.nc"))
