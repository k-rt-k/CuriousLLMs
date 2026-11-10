"""Lean preemption-safe checkpoints: atomic write, retention, SIGTERM/SIGUSR1 flag.

Layout: ``<log_path>/ckpt/step_{step:06d}/`` containing ``adapter/`` (PEFT dir, for serving /
warm start), ``optimizer.pt`` (``Trainer.state_dict_for_ckpt()``: exact LoRA weights, AdamW
state, init snapshot, step), ``rng.pt``, ``loop_state.json`` and optionally ``rnd.pt``.
A checkpoint is written to ``step_N.tmp`` and renamed into place; older ones are deleted
only after the rename, and ``.tmp`` dirs are never loaded.
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

    Non-main ranks return the would-be path without writing (safe to call on all ranks).
    """
    final = os.path.join(_ckpt_root(log_path), f"step_{step:06d}")
    if not is_main:
        return final
    tmp = final + ".tmp"
    os.makedirs(_ckpt_root(log_path), exist_ok=True)
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    trainer.save_adapter(os.path.join(tmp, "adapter"))
    torch.save(trainer.state_dict_for_ckpt(), os.path.join(tmp, "optimizer.pt"))
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
    return final


def load_latest(log_path: str) -> tuple[str, LoopState] | None:
    steps = _list_steps(log_path)
    if not steps:
        return None
    path = steps[-1][1]
    with open(os.path.join(path, "loop_state.json")) as f:
        return path, LoopState(**json.load(f))


def restore_checkpoint(path: str, trainer: "Trainer", *, restore_rng: bool = True) -> dict | None:
    """Load trainer state (and this process's RNG) from ``path``; returns rnd_state or None."""
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
