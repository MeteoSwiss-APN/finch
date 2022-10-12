import functools
import sys
import finch
import argparse

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
debug_scheduler = debug
"""Whether to launch a debugable scheduler or the normal distributed one."""
iterations = 10
"""The number of iterations when measuring runtime"""
warmup = True
"""Whether to perform a warmup before measuring the runtimes"""

# BRN experiment settings
######################################################

run_brn = True
"""Whether to run brn experiments"""

# input loading

brn_load_experiment = True
"""Whether to measure the different input loading times"""

# single run

brn_single_run = True
"""Whether to perform a single run experiment with brn"""
brn_imp_to_inspect = finch.brn.impl.brn_blocked_np
"""The brn implementation to inspect during the single run experiment"""
brn_single_format = finch.data.Format.GRIB
"""The file type for the input data for the brn single run experiment"""
brn_single_reps = 1
"""The number of repetitions to do the brn single run experiment"""

# multi run

brn_multi_run = True
"""Wether to perform a run experiment with every available implementation"""
brn_multi_format = finch.data.Format.ZARR
"""The input format for the brn multi run experiment"""
brn_multi_dim_order = "xyz"
"""The input dimension order for the brn multi run experiment"""


######################################################
# script
######################################################

# start scheduler
client = finch.start_scheduler(debug=debug_scheduler)

# brn experiments

config = {
    "iterations": iterations,
    "warmup": warmup
}

if run_brn:
    if brn_load_experiment:
        print("Measuring brn input load times")
        load_funcs = finch.util.funcs_from_args(finch.brn.load_input, [{"format" : f} for f in finch.data.Format])
        runtimes = finch.measure_runtimes(load_funcs, **config)
        for f, r in zip(finch.data.Format, runtimes):
            print(f"{f.value}: {r}")
    if brn_single_run:
        print(f"Measuring runtime of function {brn_imp_to_inspect.__name__}")
        arrays = finch.brn.load_input(format=brn_single_format)
        runtime = finch.measure_runtime(
            lambda *x: brn_imp_to_inspect(*x).compute(),
            arrays,
            **config
        )
        print(runtime)
    
    if brn_multi_run:
        print(f"Measuring runtimes of brn implementations")
        imps = finch.brn.list_brn_implementations()
        input = functools.partial(finch.brn.load_input, format=brn_multi_format, dim_order=brn_multi_dim_order)
        runtimes = finch.measure_runtimes(imps, input, **config)
        for f, r, in zip(imps, runtimes):
            print(f"{f.__name__}: {r}")
