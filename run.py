import finch

######################################################
# configuration
######################################################


# general configurations
######################################################

debug = False

# BRN experiment settings
######################################################

run_brn = True
"""Whether to run brn experiments"""

# single run

brn_single_run = True
"""Whether to perform a single run experiment with brn"""
brn_imp_to_inspect = finch.brn.impl.brn_blocked_np
"""The brn implementation to inspect during the single run experiment"""
brn_single_format = finch.data.Formats
"""The file type for the input data for the brn single run experiment"""
brn_single_reps = 1
"""The number of repetitions to do the brn single run experiment"""


######################################################
# script
######################################################

# start scheduler
client = finch.start_scheduler(debug=debug)

# brn experiments
if run_brn:
    if brn_single_run:
        arrays = finch.brn.load_input(format=brn_single_format)
        runtime = finch.measure_runtimes(
            [lambda *x: brn_imp_to_inspect(*x).compute()], 
            [lambda : arrays], iterations=brn_single_reps
            )[0]
        print(runtime)
