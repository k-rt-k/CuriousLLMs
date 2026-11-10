"""
Driver/worker data parallelism for the local LoRA trainer.

Launch with `torchrun` (or anything that sets WORLD_SIZE/RANK/LOCAL_RANK and
MASTER_ADDR/MASTER_PORT). Rank 0 (the *driver*) runs the whole RL loop exactly
as in the single-process backend. Ranks 1..W-1 (*workers*) build the same
`LocalTrainingClient` and then sit in `serve_worker`, executing the commands
the driver broadcasts:

    driver                                   worker
    ------                                   ------
    send_command(FWD_BWD, shards)  ───────▶  recv_command
    _do_fwd_bwd(own shard)                   _do_fwd_bwd(its shard)
        local fwd/bwd (token-SUM loss)           local fwd/bwd
        all_reduce(grads, SUM)  ◀──────────▶     all_reduce(grads, SUM)
        gather_object(outputs)  ◀───────────     gather_object
    send_command(OPTIM, adam)      ───────▶  _do_optim (identical step)
    ...
    shutdown_workers               ───────▶  SHUTDOWN → serve_worker returns

Gradients are reduced with SUM, never mean: every rank backprops the token-sum
of its shard, so the sum over ranks equals the single-process token-sum, and
the optimizer update is identical to the 1-GPU / Tinker update.

Commands travel on a dedicated gloo group (`obj_group`) via pickled object
collectives; tensors (the gradient all-reduce, the optional sync check) use
the default group (NCCL when CUDA is available, else gloo).
"""

from __future__ import annotations

import datetime
import enum
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from local_backend.training_client import LocalTrainingClient

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 14400.0


@dataclass
class DistContext:
    rank: int
    world_size: int
    local_rank: int
    obj_group: Any
    # Driver-side protocol state: True while a command is in flight (workers
    # are inside a collective), so shutdown must not scatter another command.
    busy: bool = field(default=False, repr=False)
    shut_down: bool = field(default=False, repr=False)

    @property
    def is_driver(self) -> bool:
        return self.rank == 0


_CONTEXT: Optional[DistContext] = None


def get_context() -> Optional[DistContext]:
    """The process-global context set by `init_distributed()` (None if single-process)."""
    return _CONTEXT


def init_distributed(timeout_s: Optional[float] = None) -> Optional[DistContext]:
    """Initialise torch.distributed from torchrun-style env vars. Idempotent.

    Returns None (and does nothing) when WORLD_SIZE is unset or 1.
    `timeout_s` defaults to $LOCAL_DIST_TIMEOUT_S or 14400 s; it bounds every
    collective, so it is also the backstop if a peer dies mid-protocol.
    """
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size <= 1:
        return None
    if timeout_s is None:
        timeout_s = float(os.environ.get("LOCAL_DIST_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
    timeout = datetime.timedelta(seconds=timeout_s)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(
            backend=backend, rank=rank, world_size=world_size, timeout=timeout
        )
    obj_group = dist.new_group(backend="gloo", timeout=timeout)
    _CONTEXT = DistContext(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        obj_group=obj_group,
    )
    logger.info(
        "init_distributed: rank %d/%d local_rank %d backend %s",
        _CONTEXT.rank, _CONTEXT.world_size, local_rank, dist.get_backend(),
    )
    return _CONTEXT


def shard_by_tokens(lengths: list[int], world_size: int) -> list[list[int]]:
    """Greedy longest-first onto the least-loaded rank.

    Returns one sorted index list per rank; every index appears exactly once.
    Ranks may get an empty list when there are fewer items than ranks.
    """
    shards: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for i in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        r = min(range(world_size), key=lambda r: (loads[r], r))
        shards[r].append(i)
        loads[r] += lengths[i]
    return [sorted(s) for s in shards]


class Cmd(enum.Enum):
    FWD_BWD = "fwd_bwd"
    OPTIM = "optim"
    LOAD_STATE = "load_state"
    LOAD_ADAPTER = "load_adapter"
    SHUTDOWN = "shutdown"


def send_command(ctx: DistContext, cmd: Cmd, per_rank_payloads: list[Any]) -> Any:
    """Driver: scatter `(cmd, payload_r)` to every rank; returns the driver's own payload."""
    assert ctx.is_driver, "send_command is driver-only"
    if ctx.shut_down:
        raise RuntimeError(f"send_command({cmd}) after workers were shut down")
    if len(per_rank_payloads) != ctx.world_size:
        raise ValueError(
            f"need {ctx.world_size} payloads, got {len(per_rank_payloads)}"
        )
    out: list[Any] = [None]
    dist.scatter_object_list(
        out, [(cmd, p) for p in per_rank_payloads], src=0, group=ctx.obj_group
    )
    return out[0][1]


def recv_command(ctx: DistContext) -> tuple[Cmd, Any]:
    """Worker: block until the driver scatters the next command."""
    out: list[Any] = [None]
    dist.scatter_object_list(out, None, src=0, group=ctx.obj_group)
    cmd, payload = out[0]
    return cmd, payload


def shutdown_workers(ctx: Optional[DistContext]) -> None:
    """Driver: release the workers. Safe to call more than once, and from `finally`.

    If a command is still in flight (the driver failed mid-protocol), the
    workers are blocked in a collective and cannot receive SHUTDOWN; we skip
    the send and rely on process teardown / the collective timeout instead.
    """
    if ctx is None or not ctx.is_driver or ctx.shut_down:
        return
    if ctx.busy:
        logger.error(
            "shutdown_workers: driver failed mid-command; workers will exit on "
            "process teardown or collective timeout"
        )
        return
    try:
        send_command(ctx, Cmd.SHUTDOWN, [None] * ctx.world_size)
    finally:
        ctx.shut_down = True


def serve_worker(ctx: DistContext, client: "LocalTrainingClient") -> None:
    """Worker loop: execute driver commands until SHUTDOWN."""
    assert not ctx.is_driver, "serve_worker runs on ranks 1..W-1"
    while True:
        cmd, payload = recv_command(ctx)
        if cmd is Cmd.SHUTDOWN:
            logger.info("rank %d: SHUTDOWN", ctx.rank)
            return
        if cmd is Cmd.FWD_BWD:
            shard, loss_fn = payload
            client._do_fwd_bwd(shard, loss_fn)
        elif cmd is Cmd.OPTIM:
            client._do_optim(payload)
        elif cmd is Cmd.LOAD_STATE:
            client._do_load_state(payload)
        elif cmd is Cmd.LOAD_ADAPTER:
            client._do_load_adapter(payload)
        else:  # pragma: no cover
            raise RuntimeError(f"unknown command {cmd!r}")
