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
    format=finch.data.Format.ZARR,
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
brn_imp_to_inspect = finch.brn.impl.brn_xr
"""The brn implementation to inspect during the single run experiment"""
brn_single_versions = [finch.Input.Version(
    format=finch.data.Format.GRIB,
    dim_order="zyx",
    chunks={"z": 1},
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

brn_multi_run = True
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

# evaluation

brn_evaluation = True
"""Whether or not to run evaluation"""


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
        logging.log(f"Measuring runtime of function {brn_imp_to_inspect.__name__}")
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
        finch.eval.create_plots(results)

    if brn_evaluation:
        logging.info(f"Evaluating experiment results")
        results = xr.open_dataarray(brn_results_file)
        finch.eval.create_plots(results)
