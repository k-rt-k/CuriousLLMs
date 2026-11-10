"""Entrypoint: ``python -m lean_rl.train key=value ...`` (see ``lean_rl.config.LeanRLConfig``).

Under a multi-rank launch (``WORLD_SIZE > 1``; RANK/LOCAL_RANK/MASTER_ADDR/MASTER_PORT set by
``slurm/sbatch_lean.sh``) every rank runs this same command. A preempted run exits 0 after
checkpointing; rerunning the same command with the same ``log_path`` resumes.
"""

from __future__ import annotations

import asyncio

import chz

from lean_rl.config import LeanRLConfig
from lean_rl.loop import run


def main() -> None:
    cfg = chz.entrypoint(LeanRLConfig)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
