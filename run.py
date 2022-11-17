from contextlib import nullcontext
import functools
import logging
from dask.distributed import performance_report
import pathlib
import sys
import argparse
import os
import matplotx

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

iterations = 1 if debug else 10
"""The number of iterations when measuring runtime"""
warmup = not debug
"""Whether to perform a warmup before measuring the runtimes"""
cache_input = True
"""Whether to cache the input between multiple iterations"""
pbar = True
"""Whether to print a progress bar"""
results_file = pathlib.Path(finch.config["global"]["tmp_dir"], "results.nc")
"""The file where runtime results are temporarily stored."""

# BRN experiment settings
######################################################

run_brn = True
"""Whether to run brn experiments"""
brn_manage_input = False
"""Whether to alter the brn input versions"""
brn_load_experiment = False
"""Whether to measure the different input loading times"""
brn_measure_runtimes = False
"""Wether to measure brn runtimes"""
brn_evaluation = True
"""Whether or not to run evaluation"""

# input management

brn_add_input_version = finch.Input.Version(
    format=finch.data.Format.ZARR,
    dim_order="zyx",
    chunks={"x": 10, "y": -1, "z": -1},
    coords=False
)
"""New brn input version to add"""
brn_input_management_cluster = finch.scheduler.ClusterConfig(
    workers_per_job=1,
    cores_per_worker=5,
    omp_parallelism=False,
    exclusive_jobs=False
)
"""The cluster configuration used for input management"""
brn_input_management_workers = 8
"""The number of workers used for input management"""

# input loading

# runtime measurements

brn_exp_name = "chunk_size"
"""The name of the runtime experiment"""
brn_input_versions = finch.Input.Version.list_configs(
    format=finch.data.Format.ZARR,
    dim_order="xyz",
    coords=False,
    chunks=[{"x" : n, "y": -1, "z": -1} for n in [10, 20, 30, 40, 50]]
# ) + finch.Input.Version.list_configs(
#     format=finch.data.Format.ZARR,
#     dim_order="zyx",
#     coords=False,
#     chunks={"z" : 2, "y": -1, "x": -1}
# ) + finch.Input.Version.list_configs(
#     format=finch.data.Format.GRIB,
#     dim_order="zyx",
#     coords=False,
#     chunks={"z" : 2}
)
"""The input versions for the runtime experiment"""
brn_imps = finch.brn.impl.brn_xr #[finch.brn.interface.get_repeated_implementation(n, base=finch.brn.impl.brn_blocked_cpp) for n in [10, 20, 30, 40, 50]]
"""The brn implementations used"""
brn_workers = [1] + list(range(5, 41, 5))
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

brn_eval_exp_name = "brn_blockwise"
"""If not None, then this variable is used to locate the results file. Otherwise the (temporary) results file will be used."""
brn_eval_runtimes_plot = ["full"]
"""The runtimes to plot"""
brn_eval_main_dim = "impl"
"""The dimension in the results dataset to choose as the main dimension for comparison"""
brn_eval_plot_fits = False
"""Whether to plot fitted scaling model"""
brn_eval_estimate_serial = False
"""Whether to estimate the serial overhead for the runtime results"""
brn_eval_plot_dark_mode = False
"""Whether to use dark or light mode for plotting"""
brn_eval_rename_labels = {"brn_xr": "xarray", "brn_blocked_cpp" : "C++", "brn_blocked_np": "NumPy"}
"""If not None, then this dictionary will be used to rename the labels of the main dimensions."""
brn_eval_reference_labels = {"cores": "NumPy"}
"""The keys of this dictionary indicate the plots for which to plot a relative runtime. 
The values indicate the label in the main dimension of the reference."""


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

        if brn_manage_input:
            logging.info("Adding new input version")
            if brn_add_input_version:
                finch.scheduler.start_scheduler(debug, brn_input_management_cluster)
                finch.scheduler.scale_and_wait(brn_input_management_workers)
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
            results.to_netcdf(results_file)

        if brn_evaluation:
            logging.info(f"Evaluating experiment results")
            if not brn_eval_plot_dark_mode:
                finch.eval.plot_style = matplotx.styles.dufte
            if brn_eval_exp_name is None:
                results = xr.open_dataset(results_file)
                results.to_netcdf(finch.util.get_path(finch.config["evaluation"]["results_dir"], results.attrs["name"], "results.nc"))
                results = finch.eval.create_cores_dimension(results)
            else:
                results = xr.open_dataset(finch.util.get_path(finch.config["evaluation"]["results_dir"], brn_eval_exp_name, "results.nc"))
            if brn_eval_rename_labels:
                brn_eval_rename_labels = {brn_eval_main_dim : brn_eval_rename_labels}
                results = finch.eval.rename_labels(results, brn_eval_rename_labels)
            finch.eval.create_plots(results, main_dim=brn_eval_main_dim, relative_rt_dims=brn_eval_reference_labels, runtime_selection=brn_eval_runtimes_plot, estimate_serial=brn_eval_estimate_serial, plot_scaling_fits=brn_eval_plot_fits)
            if len(results.data_vars) > 1:
                finch.eval.plot_runtime_parts(results)
            finch.eval.store_config(results)
