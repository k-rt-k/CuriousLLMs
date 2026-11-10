"""Toy preemptible loop used by test_checkpoint.py (``python -m tests.lean_rl.sigterm_loop``)."""

import argparse
import sys
import time

import tinker
import torch

from lean_rl.checkpoint import (
    LoopState, PreemptionFlag, load_latest, restore_checkpoint, save_checkpoint,
)
from lean_rl.trainer import Trainer
from tests.lean_rl.test_trainer import _make_data

TOTAL = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--marker")  # touched after the rollout of --wait-step, then wait for a signal
    ap.add_argument("--wait-step", type=int, default=-1)
    a = ap.parse_args()

    flag = PreemptionFlag().install()
    torch.manual_seed(0)  # identical PEFT lora_A init across processes
    trainer = Trainer(a.model, rank=4, device="cpu", dtype=torch.float32)
    torch.manual_seed(123)
    pos = 0
    found = load_latest(a.log)
    if found:
        restore_checkpoint(found[0], trainer)
        pos = found[1].data_position
    adam = tinker.AdamParams(learning_rate=1e-2, beta1=0.9, beta2=0.95, eps=1e-8)
    while trainer.step < TOTAL:
        data = _make_data(seed=int(torch.randint(0, 10**6, (1,))))  # "rollout" uses global RNG
        pos += 1
        if trainer.step == a.wait_step and a.marker:
            open(a.marker, "w").close()
            t0 = time.time()
            while not flag.requested and time.time() - t0 < 120:
                time.sleep(0.05)
        trainer.train_step(data, "importance_sampling", adam)  # step completes before exiting
        if trainer.step < TOTAL and flag.requested:
            save_checkpoint(a.log, trainer.step, trainer, LoopState(trainer.step, pos, None), None)
            return 0
    torch.save(
        {"step": trainer.step, "pos": pos, "lora": trainer.state_dict_for_ckpt()["lora"]}, a.out
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
