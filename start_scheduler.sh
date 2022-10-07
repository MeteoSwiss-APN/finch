#!/bin/bash -l

# script to run a distributed scheduler from finch on the cluster
# running this script starts a SLURMCluster instance and opens an interactive python shell for it.

srun --nodes=1 --partition=postproc --pty python -i -c "
import finch;
finch.start_scheduler(on_slurm=False, verbose=True);
print(\"You can scale the cluster with \`cluster.scale(...)\` within this console.\");
print(\"Exit with Ctrl-D\")
"