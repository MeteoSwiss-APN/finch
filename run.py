import logging
import sys
import argparse
import os
import matplotx

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true")
cmd_args = parser.parse_args(sys.argv[1:])

# debug arguments must be set before importing finch
debug = cmd_args.debug
"""Debug mode"""
os.environ["DEBUG"] = str(debug)

import finch
import xarray as xr

# handle configuration
import run_config as rc
import debug_config as dc
try:
    import custom_config
except ImportError:
    pass


# script

if __name__ == "__main__":

    # configure logging
    logging.basicConfig(format=finch.logging_format, level=logging.INFO)

    # configure debug setup
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        finch.set_log_level(logging.DEBUG)

    # brn experiments

    brn_input = finch.brn.brn_input

    if rc.run_brn:

        measure_cfg = dict(
            iterations = dc.iterations if debug else rc.iterations,
            warmup = dc.warmup if debug else rc.warmup,
            pbar = rc.pbar
        )

        if rc.brn_manage_input:
            logging.info("Adding new input version")
            if rc.brn_add_input_version:
                finch.scheduler.start_scheduler(debug, rc.brn_input_management_cluster)
                finch.scheduler.scale_and_wait(rc.brn_input_management_workers)
                brn_input.add_version(rc.brn_add_input_version)

        if rc.brn_load_experiment:
            logging.info("Measuring brn input load times")
            times = finch.measure_loading_times(
                brn_input, 
                brn_input.versions,
                **measure_cfg
            )
            finch.print_version_results(times, brn_input.versions)
            print()
        
        if rc.brn_measure_runtimes:
            logging.info(f"Measuring runtimes of brn implementations")
            times = finch.measure_operator_runtimes(
                rc.run_configs, 
                brn_input, 
                rc.brn_input_versions, 
                dask_report = rc.brn_dask_report, 
                **measure_cfg
            )
            results = finch.eval.create_result_dataset(times, rc.run_configs, rc.brn_input_versions, finch.brn.brn_input, "brn_"+rc.brn_exp_name)
            results.to_netcdf(rc.results_file)

        if rc.brn_evaluation:
            logging.info(f"Evaluating experiment results")
            if not rc.brn_eval_plot_dark_mode:
                finch.eval.plot_style = matplotx.styles.dufte
            if rc.brn_eval_exp_name is None:
                results = xr.open_dataset(rc.results_file)
                results.to_netcdf(finch.util.get_path(finch.config["evaluation"]["results_dir"], results.attrs["name"], "results.nc"))
            else:
                results = xr.open_dataset(finch.util.get_path(finch.config["evaluation"]["results_dir"], rc.brn_eval_exp_name, "results.nc"))
            results = finch.eval.create_cores_dimension(results)
            if rc.brn_eval_rename_labels:
                brn_eval_rename_labels = {rc.brn_eval_main_dim :rc. brn_eval_rename_labels}
                results = finch.eval.rename_labels(results, brn_eval_rename_labels)
            finch.eval.create_plots(results, 
                main_dim=rc.brn_eval_main_dim, 
                scaling_dims=rc.brn_eval_speedup_dims, 
                relative_rt_dims=rc.brn_eval_reference_labels, 
                runtime_selection=rc.brn_eval_runtimes_plot, 
                estimate_serial=rc.brn_eval_estimate_serial, 
                plot_scaling_fits=rc.brn_eval_plot_fits
            )
            if len(results.data_vars) > 1:
                finch.eval.plot_runtime_parts(results, first_dims=rc.brn_eval_rt_parts_order)
            finch.eval.store_config(results)
