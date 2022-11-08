from contextlib import nullcontext
import functools
import logging
from dask.distributed import performance_report
import pathlib
import sys
import argparse
import xarray as xr
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

brn_modify_input_versions = False
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

# multi run

brn_multi_run = True
"""Wether to perform a run experiment with different run configurations"""
brn_multi_versions = finch.Input.Version.list_configs(
    format=finch.data.Format.FAKE,
    dim_order="xyz",
    coords=False,
    chunks=[{"x" : x} for x in [10, 20, 30, 50]]
)
"""The input versions for the brn multi run experiment"""
brn_multi_imps = finch.brn.list_brn_implementations()
brn_multi_imps += [finch.brn.interface.get_repeated_implementation(10)]
"""The brn implementations used"""
brn_multi_name = "rand_multi"
"""The name of the brn multi run experiment"""
brn_multi_workers = [1, 2, 4, 8, 16]
"""A list of the number of workers to spawn for the brn multi run"""
brn_multi_cores_per_worker = 4
"""The number of cores dedicated to each worker"""
brn_multi_omp = False
"""Whether to delegate parallelism to OpenMP for a worker"""

# repeated experiment

brn_repeated_run = False
"""Whether to perform a run experiment for the brn repeated function(s)"""
brn_repeated_n = range(10, 50, 10)
"""A list with the number of times to repeat the computation"""
brn_repeated_workers = [1, 2, 4, 8, 16]
"""A list of the number of workers to spawn"""
brn_repeated_cores_per_worker = 4
"""The number of cores available per worker"""
brn_repeated_omp = False
"""Whether to reserve parallelism to OpenMP"""
brn_repeated_input_version = finch.Input.Version(
    format=finch.data.Format.FAKE,
    dim_order="xyz",
    chunks={"x": 10},
    coords=False
)
brn_repeated_name = "repeated"
"""The name of the repeated experiment"""

# multicore experiment

brn_multicore_run = True
"""Whether to run the brn multicore experiment"""

# evaluation

brn_evaluation = True
"""Whether or not to run evaluation"""
brn_eval_runtimes_plot = ["full"]
"""The runtimes to plot"""


######################################################
# script
######################################################

if __name__ == "__main__":

    # configure logging
    logging.basicConfig(format=finch.env.logging_format, level=logging.INFO)

    # configure debug setup
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        finch.env.set_log_level(logging.DEBUG)

    # brn experiments

    config = {
        "iterations": iterations,
        "warmup": warmup,
        "cache_inputs": cache_input,
        "pbar": pbar
    }

    brn_input = finch.brn.brn_input

    if run_brn:

        if brn_modify_input_versions:
            logging.info("Adjusting input versions")
            if brn_add_input_version:
                brn_input.add_version(brn_add_input_version)

        if brn_load_experiment:
            logging.info("Measuring brn input load times")
            times = finch.measure_loading_times(brn_input, brn_input.versions, **config)
            finch.print_version_results(times, brn_input.versions)
            print()

        if brn_single_run:
            logging.info(f"Measuring runtime of function {brn_imp_to_inspect.__name__}")
            run_configs = finch.experiments.RunConfig(impl=brn_imp_to_inspect, workers=brn_single_workers)
            with performance_report(filename=pathlib.Path(finch.config["evaluation"]["perf_report_dir"], "dask-report.html")) \
                if brn_single_perf_report else nullcontext():
                times = finch.measure_operator_runtimes(run_configs, brn_input, brn_single_versions, **config)
            results = finch.eval.create_result_array(times, run_configs, brn_single_versions, "brn_"+brn_single_name)
            results.to_netcdf(brn_results_file)
        
        if brn_multi_run:
            logging.info(f"Measuring runtimes of brn implementations")
            cluster_configs = finch.scheduler.ClusterConfig.list_configs(
                cores_per_worker=brn_multi_cores_per_worker,
                omp_parallelism=brn_multi_omp,
                exclusive_jobs=False
            )
            run_configs = finch.experiments.RunConfig.list_configs(
                workers=brn_multi_workers,
                impl=brn_multi_imps,
                cluster_config=cluster_configs
            )
            times = finch.measure_operator_runtimes(run_configs, brn_input, brn_multi_versions, **config)
            results = finch.eval.create_result_dataset(times, run_configs, brn_multi_versions, "brn_"+brn_multi_name)
            results.to_netcdf(brn_results_file)

        if brn_repeated_run:
            logging.info(f"Measuring runtimes of repeated brn and thetav runs")
            cluster_configs = finch.scheduler.ClusterConfig.list_configs(
                cores_per_worker=brn_repeated_cores_per_worker,
                omp_parallelism=brn_repeated_omp,
                exclusive_jobs=False
            )
            run_configs = finch.experiments.RunConfig.list_configs(
                workers=brn_repeated_workers,
                cluster_config=cluster_configs,
                impl=[finch.brn.get_repeated_implementation(n) for n in brn_repeated_n]
            )
            times = finch.measure_operator_runtimes(run_configs, finch.brn.brn_input, brn_repeated_input_version, **config)
            results = finch.eval.create_result_dataset(
                times, 
                run_configs, 
                brn_repeated_input_version, 
                brn_repeated_name, 
                impl_names=finch.brn.get_repeated_impl_name
            )
            results.to_netcdf(brn_results_file)

        if brn_evaluation:
            logging.info(f"Evaluating experiment results")
            results = xr.open_dataset(brn_results_file)
            results = finch.eval.create_cores_dimension(results)
            finch.eval.create_plots(results, runtime_selection=brn_eval_runtimes_plot)
            if len(results.data_vars) > 1:
                finch.eval.plot_runtime_parts(results)
