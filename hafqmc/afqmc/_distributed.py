"""Early JAX distributed initialization helpers for AFQMC."""

from __future__ import annotations

import logging
import os

import jax


def maybe_initialize_distributed(logger: logging.Logger | None = None) -> None:
    if hasattr(jax.distributed, "is_initialized"):
        try:
            if jax.distributed.is_initialized():
                return
        except Exception:
            pass

    coord_addr = os.environ.get("JAX_COORDINATOR_ADDRESS")
    num_processes = os.environ.get("JAX_NUM_PROCESSES")
    process_id = os.environ.get("JAX_PROCESS_ID")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    local_ids = None
    if cuda_visible and cuda_visible != "-1":
        visible_list = [x.strip() for x in cuda_visible.split(",") if x.strip()]
        if visible_list:
            local_ids = list(range(len(visible_list)))

    if coord_addr and num_processes and process_id:
        if local_ids is None:
            jax.distributed.initialize()
        else:
            jax.distributed.initialize(local_device_ids=local_ids)
        if logger is not None and int(process_id) == 0:
            logger.info(
                "Initialized JAX distributed: coordinator=%s process_id=%s num_processes=%s",
                coord_addr,
                process_id,
                num_processes,
            )
        return

    slurm_n_tasks = os.environ.get("SLURM_NTASKS")
    slurm_proc_id = os.environ.get("SLURM_PROCID")
    if slurm_n_tasks and slurm_proc_id and int(slurm_n_tasks) > 1:
        if local_ids is None:
            jax.distributed.initialize()
        else:
            jax.distributed.initialize(local_device_ids=local_ids)
        if logger is not None and int(slurm_proc_id) == 0:
            logger.info(
                "Initialized JAX distributed from SLURM: proc_id=%s ntasks=%s",
                slurm_proc_id,
                slurm_n_tasks,
            )


__all__ = ["maybe_initialize_distributed"]
