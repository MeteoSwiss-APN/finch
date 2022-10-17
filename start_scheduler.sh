#!/bin/bash -l

# script to run a distributed scheduler from finch on the cluster
# running this script starts a SLURMCluster instance and opens an interactive python shell for it.

PREFIX=

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--slurm)
            PREFIX="srun --nodes=1 --partition=postproc --pty"
            shift
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done    

$PREFIX python -i -c "
import finch;
import finch.environment as env;
finch.start_scheduler(verbose=True);
print(\"You can scale the cluster with \`env.cluster.scale(...)\` within this console.\");
print(\"Exit with Ctrl-D\")
"