import finch
import pathlib

######################################################
# general configurations
######################################################

iterations = 10
"""The number of iterations when measuring runtime"""
warmup = True
"""Whether to perform a warmup before measuring the runtimes"""
cache_input = True
"""Whether to cache the input between multiple iterations"""
pbar = True
"""Whether to print a progress bar"""
results_file = pathlib.Path(finch.config["global"]["tmp_dir"], "results.nc")
"""The file where runtime results are temporarily stored."""

######################################################
# BRN experiment settings
######################################################

run_brn = True
"""Whether to run brn experiments"""
brn_manage_input = False
"""Whether to alter the brn input versions"""
brn_load_experiment = False
"""Whether to measure the different input loading times"""
brn_measure_runtimes = True
"""Wether to measure brn runtimes"""
brn_evaluation = True
"""Whether or not to run evaluation"""


# input management
######################################################

brn_add_input_version: finch.Input.Version = None
"""New brn input version to add"""
brn_input_management_cluster = finch.scheduler.ClusterConfig(
    workers_per_job=1,
    cores_per_worker=1,
    omp_parallelism=False,
    exclusive_jobs=False
)
"""The cluster configuration used for input management"""
brn_input_management_workers = 1
"""The number of workers used for input management"""


# input loading
######################################################


# runtime measurements
######################################################

brn_exp_name = "all"
"""The name of the runtime experiment"""
brn_input_versions = finch.brn.brn_input.list_versions()
"""The input versions for the runtime experiment"""
run_configs = finch.experiments.RunConfig.list_configs(
    workers=1,
    impl=finch.brn.list_brn_implementations(),
    cluster_config=finch.scheduler.ClusterConfig.list_configs(
        cores_per_worker=1,
        omp_parallelism=False,
        exclusive_jobs=False,
    ),
    prep=finch.experiments.get_xr_run_prep()
)
"""The run configurations"""
brn_dask_report = False
"""Whether to write a dask performance report"""

# evaluation
######################################################

brn_eval_exp_name = None
"""If not None, then this variable is used to locate the results file. Otherwise the (temporary) results file will be used."""
brn_eval_runtimes_plot = ["full"]
"""The runtimes to plot"""
brn_eval_main_dim = "output_overwrite"
"""The dimension in the results dataset to choose as the main dimension for comparison"""
brn_eval_speedup_dims = ["cores"]
"""The dimensions for which to plot the speedups"""
brn_eval_estimate_serial = True
"""Whether to estimate the serial overhead for the runtime results"""
brn_eval_plot_fits = True
"""Whether to plot fitted scaling model"""
brn_eval_plot_dark_mode = False
"""Whether to use dark or light mode for plotting"""
brn_eval_rename_labels = {"True": "Overwrites", "False": "No Overwrites"}
"""If not None, then this dictionary will be used to rename the labels of the main dimensions."""
brn_eval_reference_labels = {"cores": "Overwrites"}
"""The keys of this dictionary indicate the plots for which to plot a relative runtime. 
The values indicate the label in the main dimension of the reference."""
brn_eval_rt_parts_order = [brn_eval_main_dim, "cores"]
"""The order of dimension (must not be complete) for plotting the runtime parts"""