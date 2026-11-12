"""Multi-rank plumbing for the lean trainer (SPMD: every rank runs the same loop).

``init_dist`` reads the torchrun-style environment (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR,
MASTER_PORT), set either by torchrun or by our srun wrapper. LOCAL_RANK is the *physical* GPU
index this rank uses (all node GPUs stay visible; node 0 skips the vLLM GPUs), so the trainer
device is ``cuda:{LOCAL_RANK}``.

Tensor collectives use the default group (NCCL on CUDA, else gloo); pickled-object
collectives (``broadcast_object``, ``gather_object``, ``barrier``) use a separate gloo group so
CPU objects never go through NCCL.
"""

from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as tdist

from local_backend.distributed import shard_by_tokens

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 14400.0


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    is_main: bool


_INFO: DistInfo | None = None
_OBJ_GROUP: Any = None


def init_dist() -> DistInfo | None:
    """Initialise torch.distributed from the environment. Idempotent.

    Returns None (and does nothing) when WORLD_SIZE is unset or 1. The collective timeout is
    ``$LEAN_DIST_TIMEOUT_S`` (default 14400 s); it is also the backstop if a peer dies.
    """
    global _INFO, _OBJ_GROUP
    if _INFO is not None and tdist.is_initialized():
        return _INFO
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size <= 1:
        return None
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    timeout = datetime.timedelta(
        seconds=float(os.environ.get("LEAN_DIST_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
    )
    if not tdist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            # device_id binds NCCL to the physical GPU (barrier etc. would otherwise guess
            # cuda:{rank}, which is wrong when node 0 skips the vLLM GPUs).
            tdist.init_process_group(
                backend="nccl", rank=rank, world_size=world_size, timeout=timeout,
                device_id=torch.device(f"cuda:{local_rank}"),
            )
        else:
            tdist.init_process_group(
                backend="gloo", rank=rank, world_size=world_size, timeout=timeout
            )
    _OBJ_GROUP = tdist.new_group(backend="gloo", timeout=timeout)
    _INFO = DistInfo(
        rank=tdist.get_rank(),
        world_size=tdist.get_world_size(),
        local_rank=local_rank,
        is_main=tdist.get_rank() == 0,
    )
    logger.info(
        "init_dist: rank %d/%d local_rank %d backend %s",
        _INFO.rank, _INFO.world_size, local_rank, tdist.get_backend(),
    )
    return _INFO


def device_for(info: DistInfo | None) -> str:
    """``cuda:{LOCAL_RANK}`` (multi-rank) / ``cuda`` (single) when CUDA is available, else cpu."""
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{info.local_rank}" if info is not None else "cuda"


def _multi() -> bool:
    return tdist.is_available() and tdist.is_initialized() and tdist.get_world_size() > 1


def obj_group() -> Any:
    return _OBJ_GROUP


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Pickled broadcast over the gloo group; returns ``obj`` unchanged when single-process."""
    if not _multi():
        return obj
    buf = [obj]
    tdist.broadcast_object_list(buf, src=src, group=_OBJ_GROUP)
    return buf[0]


def gather_object(obj: Any, dst: int = 0) -> list[Any] | None:
    """Gather one object per rank to ``dst`` (list in rank order there, None elsewhere)."""
    if not _multi():
        return [obj]
    out = [None] * tdist.get_world_size() if tdist.get_rank() == dst else None
    tdist.gather_object(obj, out, dst=dst, group=_OBJ_GROUP)
    return out


def barrier() -> None:
    if _multi():
        tdist.barrier(group=_OBJ_GROUP)


def any_true(local: bool) -> bool:
    """True on every rank if ``local`` is True on any rank (one-int all-reduce MAX on the gloo
    group). A collective: every rank must make the same sequence of calls."""
    if not _multi():
        return bool(local)
    t = torch.tensor([1 if local else 0], dtype=torch.int32)
    tdist.all_reduce(t, op=tdist.ReduceOp.MAX, group=_OBJ_GROUP)
    return bool(t.item())


def shard_indices(lengths: Sequence[int], dist: DistInfo | None) -> list[int]:
    """Indices (ascending) of the items this rank works on; all items when single-rank."""
    if dist is None or dist.world_size <= 1:
        return list(range(len(lengths)))
    return shard_by_tokens(list(lengths), dist.world_size)[dist.rank]


def shard(data: list, lengths: list[int], dist: DistInfo | None) -> list:
    """This rank's token-balanced share of ``data`` (may be empty with few items)."""
    return [data[i] for i in shard_indices(lengths, dist)]
