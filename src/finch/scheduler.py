import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import dask
import dask.config
import dask.utils
from dask.distributed import Client
from dask_jobqueue import SLURMCluster  # type: ignore
from distributed.deploy.cluster import Cluster

from . import cfg as finch_cfg
from . import debug, env, util


def dask_config_get_not_none(key: str, default: Any) -> Any:
    """
    Returns the value of `dask.config.get(key, default)` and returns the default if `None` would be returned.

    Group:
        Util
    """
    out = dask.config.get(key, default)
    if out is None:
        return default
    else:
        return out


def parse_slurm_time(t: str) -> timedelta:
    """
    Returns a timedelta from the given duration as is being passed to SLURM

    Args:
        t: The time in SLURM format

    Returns:
        A timedelta object representing the passed SLURM time.

    Group:
        Util
    """
    has_days = "-" in t
    d = "0"
    if has_days:
        d, t = t.split("-")
        tl = t.split(":")
        h, m, s = tl + ["0"] * (3 - len(tl))
    else:
        tl = t.split(":")
        if len(tl) == 1:
            tl = ["0", *t, "0"]
        elif len(tl) == 2:
            tl = ["0", *tl]
        h, m, s = tl
    return timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))


@dataclass
class ClusterConfig(util.Config):
    """
    A configuration class for configuring a dask SLURM cluster.

    Group:
        Dask
    """

    workers_per_job: int = 1
    """The number of workers to spawn per SLURM job"""
    cores_per_worker: int = dask_config_get_not_none("jobqueue.slurm.cores", 1)
    """The number of cores available per worker"""
    omp_parallelism: bool = False
    """
    Toggle whether the cores of the worker should be reserved to the implementation of the task.
    If true, a worker thinks it has only one one thread available and won't run tasks in parallel.
    Instead, zebra is configured with the given number of threads.
    """
    exclusive_jobs: bool = False
    """Toggle whether to use a full node exclusively for one job."""
    queuing: bool = False
    """If True, queuing will be used by dask. If False, it will be disabled."""


client: Client | None = None
"""
The currently active dask client.

Group:
    Dask
"""
_active_config: ClusterConfig | None = None


def start_slurm_cluster(cfg: ClusterConfig = ClusterConfig()) -> Client:
    """
    Starts a new SLURM cluster with the given config and returns a client for it.
    If a cluster is already running with a different config, it is shut down.

    Args:
        cfg: The configuration of the cluster to start

    Returns:
        A client connected to the newly started SLURM cluster.

    Group:
        Dask
    """
    global client, _active_config

    if cfg == _active_config:
        assert client is not None
        return client

    if client is not None:
        cluster = client.cluster
        client.close()
        assert isinstance(cluster, SLURMCluster)
        cluster.close()
        logging.info("Closed SLURM cluster")

    worker_env = env.WorkerEnvironment()

    walltime = dask_config_get_not_none("jobqueue.slurm.walltime", "01:00:00")
    if walltime is None:
        walltime = "01:00:00"
    node_cores = dask_config_get_not_none("jobqueue.slurm.cores", 1)
    if node_cores is None:
        node_cores = 1
    node_memory: str = dask_config_get_not_none("jobqueue.slurm.memory", "1GB")
    if node_memory is None:
        node_memory = "1GB"
    node_memory_bytes = dask.utils.parse_bytes(node_memory)

    job_cpu = cfg.cores_per_worker * cfg.workers_per_job
    jobs_per_node = node_cores // job_cpu
    job_mem = dask.utils.format_bytes(node_memory_bytes // jobs_per_node)

    cores = (
        job_cpu if not cfg.omp_parallelism else cfg.workers_per_job
    )  # the number of cores dask believes it has available per job
    worker_env.omp_threads = 1 if not cfg.omp_parallelism else cfg.cores_per_worker

    walltime_delta = parse_slurm_time(walltime)
    worker_lifetime_td = walltime_delta - timedelta(minutes=3)
    worker_lifetime = int(worker_lifetime_td.total_seconds())

    dashboard_address = ":8877"

    if cfg.queuing:
        dask.config.set({"distributed.scheduler.worker-saturation": 1.0})
    else:
        dask.config.set({"distributed.scheduler.worker-saturation": "inf"})

    cluster = SLURMCluster(
        # resources
        walltime=walltime,
        cores=cores,
        memory=job_mem,
        processes=cfg.workers_per_job,
        job_cpu=job_cpu,
        job_extra_directives=["--exclusive"] if cfg.exclusive_jobs else [],
        # scheduler / worker options
        scheduler_options={
            "dashboard_address": dashboard_address,
        },
        worker_extra_args=["--lifetime", f"{worker_lifetime}s", "--lifetime-stagger", "2m", "--lifetime-restart"],
        # filesystem config
        local_directory=finch_cfg["global"]["scratch_dir"],
        shared_temp_directory=finch_cfg["global"]["tmp_dir"],
        log_directory=finch_cfg["global"]["log_dir"],
        # other
        job_script_prologue=worker_env.get_job_script_prologue(),
        nanny=True,
    )

    client = Client(cluster)
    _active_config = cfg
    logging.info(f"Started new SLURM cluster. Dashboard available at {cluster.dashboard_link}")
    if env.node_name_env_var in os.environ:
        nodename = os.environ[env.node_name_env_var]
    else:
        nodename = "local"
    logging.info(f"Current node name: {nodename}")
    logging.debug(cluster.job_script())
    return client


def start_scheduler(debug: bool = debug, *cluster_args: Any, **cluster_kwargs: Any) -> Client | None:
    """
    Starts a new scheduler either in debug or run mode.

    Args:
        debug (bool):
            If `False`, a new SLURM cluster will be started and a client connected to the new cluster is returned.
            If `True`, `None` is returned and dask is configured to run a synchronous scheduler.
        cluster_args: The positional arguments passed to :py:func:`finch.start_slurm_cluster`
        cluster_kwargs: The keyword arguments passed to :py:func:`finch.start_slurm_cluster`

    Returns:
        A client connected to the new cluster / scheduler or `None`, depending on `debug`.

    Group:
        Dask
    """
    if debug:
        dask.config.set(scheduler="synchronous")
        return None
    else:
        return start_slurm_cluster(*cluster_args, **cluster_kwargs)


def clear_memory() -> None:
    """
    Clears the memory of the current scheduler and workers.
    **Attention**: This function currently raises a `NotImplementedError`,
    because dask currently provides no efficient way of clearning the memory of the scheduler.

    Group:
        Dask
    """
    # Currently the only possible way to completely reset memory is via client.restart(),
    # which won't work many times in a row on a SLURM Cluster.
    raise NotImplementedError()


def get_client() -> Client | None:
    """
    Returns the currently registered client.

    Group:
        Dask
    """
    return client


def scale_and_wait(n: int) -> None:
    """
    Scales the current registered cluster to `n` workers and waits for them to start up.

    Group:
        Dask
    """
    if client:
        assert isinstance(client.cluster, Cluster)
        client.cluster.scale(n)
        client.wait_for_workers(n, timeout=finch_cfg.getfloat("experiments", "scaling_timeout"))
