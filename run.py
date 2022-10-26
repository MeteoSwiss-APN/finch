from locale import normalize
import logging
from fileinput import filename
from dask.distributed import performance_report
import pathlib
import sys
import finch
import argparse
import xarray as xr

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true")
cmd_args = parser.parse_args(sys.argv[1:])

######################################################
# configuration
######################################################


# general configurations
######################################################

debug = cmd_args.debug
"""Debug mode"""
debug_scheduler = False
"""Whether to launch a debugable scheduler or the normal distributed one."""
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
brn_single_jobs = 3
"""The number of jobs to spawn for the brn single run"""
brn_single_perf_report = True
"""Whether to record a performance report for the brn single run"""
brn_single_iterations = iterations
"""The number of iterations for the brn single run experiment"""

# multi run

brn_multi_run = False
"""Wether to perform a run experiment with every available implementation"""
brn_multi_versions = finch.Input.Version.list_configs(
    format=finch.data.Format.NETCDF,
    dim_order="xyz",
    coords=False,
    chunks=[{"x" : x, "y" : -1, "z" : -1} for x in [10, 20, 30, 50, 100]]
)
"""The input versions for the brn multi run experiment"""
brn_multi_name = "netcdf_scaling"
"""The name of the brn multi run experiment"""
brn_multi_jobs = 1 if debug else [1,2,3,5,10,20]
"""A list of the number of jobs to spawn for the brn multi run"""

# repeated experiment

brn_repeated_run = True
"""Whether to performa a run experiment for the brn repeated function(s)"""
brn_repeated_n = range(1, 10, 1)
"""A list with the number of times to repeat the computation"""
brn_repeated_jobs = [1, 2, 3, 5, 10, 20]
"""A list of the number of jobs to spawn"""
brn_repeated_input_version = finch.Input.Version(
    format=finch.data.Format.ZARR,
    dim_order="xyz",
    chunks={"x": 30},
    coords=False
)
brn_repeated_name = "repeated"
"""The name of the repeated experiment"""

# multicore experiment

brn_multicore_run = True
"""Whether to run the brn multicore experiment"""

# evaluation

brn_evaluation = brn_multi_run or brn_repeated_run
"""Whether or not to run evaluation"""
brn_eval_normalize = brn_repeated_run
"""Whether to normalize line plots"""


######################################################
# script
######################################################

# adjust logging
logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)
if debug:
    logging.basicConfig(level=logging.DEBUG)

# start scheduler
client = finch.start_scheduler(debug=debug_scheduler)

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
        run_config = finch.experiments.RunConfig(brn_imp_to_inspect, brn_single_jobs)
        if brn_single_perf_report:
            with performance_report(filename=pathlib.Path(finch.config["evaluation"]["perf_report_dir"], "dask-report.html")):
                times = finch.measure_operator_runtimes(run_config, brn_input, brn_single_versions, **config)
        else:
            times = finch.measure_operator_runtimes(run_config, brn_input, brn_single_versions, **config)
        finch.print_version_results(times, brn_single_versions)
        print()
    
    if brn_multi_run:
        logging.info(f"Measuring runtimes of brn implementations")
        run_configs = finch.experiments.RunConfig.list_configs(
            jobs=brn_multi_jobs,
            impl=finch.brn.list_brn_implementations()
        )
        times = finch.measure_operator_runtimes(run_configs, brn_input, brn_multi_versions, **config)
        results = finch.eval.create_result_array(times, run_configs, brn_multi_versions, "brn_"+brn_multi_name)
        results.to_netcdf(brn_results_file)

    if brn_repeated_run:
        logging.info(f"Measuring runtimes of repeated brn and thetav runs")
        run_configs = finch.experiments.RunConfig.list_configs(
            jobs=brn_repeated_jobs,
            impl=[finch.brn.get_repeated_implementation(n) for n in brn_repeated_n]
        )
        impl_names = [f"{n} repeats" for n in brn_repeated_n] * len(brn_repeated_jobs)
        times = finch.measure_operator_runtimes(run_configs, finch.brn.brn_input, brn_repeated_input_version, **config)
        results = finch.eval.create_result_array(
            times, 
            run_configs, 
            brn_repeated_input_version, 
            brn_repeated_name, 
            impl_names=impl_names
        )
        results.to_netcdf(brn_results_file)

    if brn_evaluation:
        logging.info(f"Evaluating experiment results")
        results = xr.open_dataarray(brn_results_file)
        finch.eval.create_plots(results, normalize_lines=brn_eval_normalize)
