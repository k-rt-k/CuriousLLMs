"""Lean preemption-safe checkpoints: atomic write, retention, SIGTERM/SIGUSR1 flag.

Layout: ``<log_path>/ckpt/step_{step:06d}/`` containing ``adapter/`` (PEFT dir, for serving /
warm start), ``optimizer.pt`` (``Trainer.state_dict_for_ckpt()``: exact LoRA weights, AdamW
state, init snapshot, step), ``rng.pt``, ``loop_state.json`` and optionally ``rnd.pt``.
A checkpoint is written to ``step_N.tmp`` and renamed into place; older ones are deleted
only after the rename, and ``.tmp`` dirs are never loaded (``remove_stale_tmp`` clears them at
startup). ``load_latest`` falls back to an older checkpoint if the newest does not load.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import signal
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from lean_rl.trainer import Trainer

logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"^step_(\d+)$")


@dataclass
class LoopState:
    step: int
    data_position: int
    wandb_run_id: str | None = None


def _ckpt_root(log_path: str) -> str:
    return os.path.join(log_path, "ckpt")


def _list_steps(log_path: str) -> list[tuple[int, str]]:
    root = _ckpt_root(log_path)
    if not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        m = _STEP_RE.match(name)  # excludes "step_000002.tmp"
        if m and os.path.isfile(os.path.join(root, name, "loop_state.json")):
            out.append((int(m.group(1)), os.path.join(root, name)))
    return sorted(out)


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(s: dict[str, Any]) -> None:
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def save_checkpoint(
    log_path: str,
    step: int,
    trainer: "Trainer",
    loop_state: LoopState,
    rnd_state: dict | None,
    keep: int = 2,
    *,
    is_main: bool = True,
) -> str:
    """Atomically write the step checkpoint and prune to the newest ``keep``.

    Multi-rank: call on ALL ranks. Every rank takes part in ``trainer.save_adapter`` and
    ``trainer.state_dict_for_ckpt`` (collectives under FSDP); only ``is_main`` writes files,
    renames and prunes. A final barrier keeps the other ranks from racing ahead (e.g. into
    ``load_latest``) before the checkpoint is in place. Returns the final path on every rank.
    """
    from lean_rl.dist import barrier

    final = os.path.join(_ckpt_root(log_path), f"step_{step:06d}")
    tmp = final + ".tmp"
    if is_main:
        os.makedirs(_ckpt_root(log_path), exist_ok=True)
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
    trainer.save_adapter(os.path.join(tmp, "adapter"))  # writes on the trainer's main rank only
    trainer_state = trainer.state_dict_for_ckpt()
    if not is_main:
        del trainer_state
        barrier()
        return final
    torch.save(trainer_state, os.path.join(tmp, "optimizer.pt"))
    del trainer_state
    torch.save(_rng_state(), os.path.join(tmp, "rng.pt"))
    if rnd_state is not None:
        torch.save(rnd_state, os.path.join(tmp, "rnd.pt"))
    # loop_state.json last: its presence marks the directory complete.
    with open(os.path.join(tmp, "loop_state.json"), "w") as f:
        json.dump(asdict(loop_state), f)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.replace(tmp, final)
    for _, path in _list_steps(log_path)[: -max(keep, 1)]:
        shutil.rmtree(path, ignore_errors=True)
    barrier()
    return final


def remove_stale_tmp(log_path: str) -> list[str]:
    """Delete ``ckpt/step_*.tmp`` dirs left by a save interrupted by SIGKILL. Call on the
    writing rank (rank 0) only, before resuming. Returns the removed paths."""
    root = _ckpt_root(log_path)
    if not os.path.isdir(root):
        return []
    removed = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if name.startswith("step_") and name.endswith(".tmp") and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
            logger.warning(f"Removed stale partial checkpoint {path}")
    return removed


def _check_loadable(path: str) -> None:
    """Light restore check: unpickle every file ``restore_checkpoint`` reads (no trainer)."""
    for name in ("optimizer.pt", "rng.pt", "rnd.pt"):
        f = os.path.join(path, name)
        if name == "rnd.pt" and not os.path.exists(f):
            continue
        torch.load(f, map_location="cpu", weights_only=False)


def load_latest(log_path: str, *, validate: bool = True) -> tuple[str, LoopState] | None:
    """Newest complete checkpoint that loads: ``(path, LoopState)``, or None if there is none.

    With ``validate`` (default) each candidate's files are unpickled first; an unreadable one
    is skipped with a warning and the next older one tried. Raises only if checkpoints exist
    but none loads. Multi-rank: call on rank 0 and broadcast the result, so every rank
    restores the same path."""
    steps = _list_steps(log_path)
    if not steps:
        return None
    errors = []
    for _, path in reversed(steps):
        try:
            with open(os.path.join(path, "loop_state.json")) as f:
                state = LoopState(**json.load(f))
            if validate:
                _check_loadable(path)
            return path, state
        except Exception as e:  # noqa: BLE001 -- any unreadable file disqualifies the checkpoint
            logger.warning(f"Checkpoint {path} does not load ({e!r}); trying an older one")
            errors.append(f"{path}: {e!r}")
    raise RuntimeError("No loadable checkpoint: " + "; ".join(errors))


def restore_checkpoint(path: str, trainer: "Trainer", *, restore_rng: bool = True) -> dict | None:
    """Load trainer state (and this process's RNG) from ``path``; returns rnd_state or None.

    Multi-rank: call on ALL ranks (each reads the same files; ``Trainer.load_ckpt_state``)."""
    trainer.load_ckpt_state(torch.load(os.path.join(path, "optimizer.pt"), weights_only=False))
    if restore_rng:
        _set_rng_state(torch.load(os.path.join(path, "rng.pt"), weights_only=False))
    rnd_path = os.path.join(path, "rnd.pt")
    if os.path.exists(rnd_path):
        return torch.load(rnd_path, weights_only=False)
    return None


class PreemptionFlag:
    """SIGTERM/SIGUSR1 set ``requested``; the loop saves at the next step boundary and exits 0."""

    SIGNALS = (signal.SIGTERM, signal.SIGUSR1)

    def __init__(self) -> None:
        self.requested = False

    def _handler(self, signum: int, frame: Any) -> None:
        self.requested = True

    def install(self) -> "PreemptionFlag":
        for sig in self.SIGNALS:
            signal.signal(sig, self._handler)  # ValueError if not the main thread
        return self
