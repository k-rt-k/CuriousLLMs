"""Multi-rank plumbing for the lean trainer (SPMD: every rank runs the same loop).

``init_dist`` reads the torchrun-style environment (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR,
MASTER_PORT), set either by torchrun or by our srun wrapper. LOCAL_RANK is the *physical* GPU
index this rank uses (all node GPUs stay visible; node 0 skips the vLLM GPUs), so the trainer
device is ``cuda:{LOCAL_RANK}``.

Tensor collectives use the default group (NCCL on CUDA, else gloo); pickled-object
collectives (``broadcast_object``, ``gather_object``, ``barrier``) use a separate gloo group so
CPU objects never go through NCCL.

Fail-fast abort (``start_watchdog`` / ``abort`` / ``shutdown``): a rank that hits an unhandled
exception logs it, sets an abort key in rank 0's c10d TCPStore and hard-exits non-zero; every
rank's watchdog thread polls that key (and the store itself: it lives in rank 0's process; three
consecutive failed checks count as rank 0 gone) about once a second and hard-exits non-zero too, even while its main thread is blocked in a
collective. A clean exit stops the watchdogs before rank 0 (and its store) can go away.
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Any, NoReturn, Sequence

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

_ABORT_KEY = "lean_rl_abort"
_WATCHDOG_INTERVAL_S = 1.0
_STORE_FAILURES_TO_ABORT = 3  # consecutive failed checks (~3 s) before rank 0's store counts as gone
_FLUSH_TIMEOUT_S = 2.0  # _hard_exit never waits longer than this for a (possibly stuck) flush
_ABORT_STORE: Any = None  # a TCPStore client of its own (never blocked behind the main thread)
_WATCHDOG: tuple[threading.Thread, threading.Event] | None = None


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
            # Backstop for the abort watchdog: NCCL errors/timeouts tear the process down.
            os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            # The NCCL heartbeat monitor (default 480 s) aborts ranks whose watchdog thread
            # stalls, and it has fired while ranks 1..N waited for rank 0's >15 min cold FSDP
            # load (job 11196098), even with that wait on the gloo group. Hangs are already
            # bounded by the collective timeout and the abort watchdog: align the two.
            os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", str(int(timeout.total_seconds())))
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


# ---------------------------------------------------------------------------- fail-fast abort
def _rank_tag() -> str:
    return f"[rank {_INFO.rank if _INFO is not None else '?'}]"


def _hard_exit(msg: str, code: int = 1) -> NoReturn:
    """Write ``msg`` to stderr, flush logs and ``os._exit`` (no atexit, no collectives).

    The write/flush runs in a daemon thread given ``_FLUSH_TIMEOUT_S``: a blocked handler
    (stalled NFS log file, full pipe) must not keep the process from exiting."""
    tag = _rank_tag()

    def flush() -> None:
        try:
            sys.stderr.write("".join(f"{tag} {line}\n" for line in msg.rstrip("\n").split("\n")))
            for h in logging.getLogger().handlers:
                h.flush()
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 -- exiting regardless
            pass

    t = threading.Thread(target=flush, name="lean-abort-flush", daemon=True)
    t.start()
    t.join(_FLUSH_TIMEOUT_S)
    os._exit(code)


def _abort_store() -> Any:
    """This process's own client of rank 0's TCPStore (MASTER_ADDR:MASTER_PORT, reachable from
    every node). Separate from the default store so a main thread blocked in a store op
    cannot block the watchdog."""
    global _ABORT_STORE
    if _ABORT_STORE is None:
        _ABORT_STORE = tdist.TCPStore(
            os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]),
            is_master=False, timeout=datetime.timedelta(seconds=30), wait_for_workers=False,
        )
    return _ABORT_STORE


def _watch(store: Any, stop: threading.Event, interval_s: float, exit_fn=None) -> None:
    """Watchdog body. A present abort key exits at once; a store error exits only after
    ``_STORE_FAILURES_TO_ABORT`` consecutive failures (a transient TCP hiccup to rank 0's node
    must not kill the job). ``exit_fn`` (default ``_hard_exit``) is a test hook."""
    exit_fn = exit_fn or _hard_exit
    failures = 0
    while not stop.wait(interval_s):
        try:
            hit = store.check([_ABORT_KEY])
        except Exception as e:  # noqa: BLE001
            failures += 1
            if failures >= _STORE_FAILURES_TO_ABORT and not stop.is_set():
                exit_fn(
                    f"aborting: lost rank 0's c10d store ({failures} failed checks, last {e!r});"
                    " rank 0 died"
                )
                return
            continue
        failures = 0
        if hit and not stop.is_set():
            try:
                who = store.get(_ABORT_KEY).decode()
            except Exception:  # noqa: BLE001 -- rank 0 (the store) may already be gone
                who = "?"
            exit_fn(f"aborting: rank {who} failed")
            return


def start_watchdog(interval_s: float = _WATCHDOG_INTERVAL_S) -> None:
    """Start this rank's daemon abort watchdog (multi-rank only; idempotent). Hard-exits 1 when
    any rank set the abort key, or when rank 0's store is gone while the watchdog still runs
    (rank 0 died without reaching ``abort``)."""
    global _WATCHDOG
    if not _multi() or _WATCHDOG is not None:
        return
    store = _abort_store()
    stop = threading.Event()
    t = threading.Thread(
        target=_watch, args=(store, stop, interval_s), name="lean-abort-watchdog", daemon=True
    )
    t.start()
    _WATCHDOG = (t, stop)


def stop_watchdog() -> None:
    global _WATCHDOG
    if _WATCHDOG is not None:
        t, stop = _WATCHDOG
        stop.set()
        t.join(timeout=30)
        _WATCHDOG = None


def abort(exc: BaseException | None = None) -> NoReturn:
    """Unhandled error on this rank: log it (rank-tagged), tell every rank via the store's
    abort key, flush and hard-exit 1. Never returns and never enters a collective."""
    rank = _INFO.rank if _INFO is not None else -1
    msg = "unhandled exception, aborting all ranks:\n"
    if exc is not None:
        msg += "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:  # first: the peers learn of it even if our own log writes stall
        _abort_store().set(_ABORT_KEY, str(rank))
    except Exception as e:  # noqa: BLE001 -- best effort: peers still see the store vanish
        msg += f"could not set the abort key: {e!r}\n"
    _hard_exit(msg + f"rank {rank} failed; exiting 1")


def shutdown() -> None:
    """Clean multi-rank exit: stop the watchdog, then a barrier (so no rank's watchdog can
    still be polling when rank 0 and its store exit), then destroy the process group."""
    stop_watchdog()
    if _multi():
        barrier()
    if tdist.is_initialized():
        tdist.destroy_process_group()
