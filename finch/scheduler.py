from dask_jobqueue import SLURMCluster
import os
import dask
from dask.distributed import Client
import argparse
from . import util
from . import environment as env

def start_slurm(scheduler_port: int = 8785, dashboard_port: int = 8877, cores_per_node: int = 20, memory_per_node: str = "24GiB") -> Client:
    """
    Tries to start a new SLURM cluster scheduler at port `scheduler_port` and exposes a dashboard at port `dashboard_port`.
    A client for the scheduler is registered and returned.
    If `scheduler_port` is already open, it is assumed that a scheduler is already running there and 
    a client will be returned for the running scheduler.

    Arguments:
    ---
    - cores_per_node: int. The number of available cores per node on the cluster.
    - memory_per_node: int. The amount of available memory per node on the cluster.
    """
    dashboard_address = f":{dashboard_port}"
    if not util.check_socket_open(port=scheduler_port):
        cluster = SLURMCluster(
                queue="postproc",
                cores=cores_per_node,
                memory=memory_per_node,
                job_extra=["--exclusive"],
                n_workers=cores_per_node,
                processes=cores_per_node,
                log_directory=env.scratch_dir + "/out",
                scheduler_options={"port": scheduler_port, "dashboard_address": dashboard_address},
                local_directory=env.scratch_dir
            )
        print("SLURM cluster started at address: %s" % cluster.scheduler_address)
        print("Dashboard available at address: http://localhost%s/status" % dashboard_address)
        print("You can scale the cluster with `cluster.scale(...)` within this console.")
    else:
        print(f"Did not start new cluster. Port {scheduler_port} is already in use.")
    return Client(f"127.0.0.1:{scheduler_port}")

def start_scheduler(debug: bool = False) -> Client | None:
    """
    Starts a new default schedule or connects to an existing one.
    If `debug` is `False`, a client connected to a SLURM cluster scheduler will be returned.
    If no scheduler is available at the default port, a new one will be started.
    If `debug` is `True`, `None` is returned and dask is configured to run a synchronous scheduler.
    """
    if debug:
        dask.config.set(scheduler="synchronous")
        return None
    else:
        client = start_slurm()
        client.restart()
        return client
