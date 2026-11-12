"""One rank of a 2-rank lean RL loop against the fake vLLM, used by test_abort.py as a real
subprocess (``python -m tests.lean_rl.abort_loop``) so exit codes and hangs are observable.

Scenarios: ``ok`` (2 steps), ``preempt`` (rank 1 SIGUSR1 mid-train_step), ``rollout_error``
(rank 0's vLLM request raises during the first rollouts while rank 1 waits in a collective),
``comm_error`` (rank 1 fails building ``_Comm`` in run's setup), ``train_error`` (rank 1 raises inside train_step: before the grad all-reduce in replicated
mode, in its first micro-batch under FSDP), ``watchdog_rank{0,1}_fails`` (no loop: the given
rank aborts while the other blocks its main thread in a 600 s sleep, standing in for a hung
NCCL collective that never notices the dead peer, so only the watchdog can end it).
A failing rank writes ``<log>/fail_time`` first.
"""

import argparse
import asyncio
import os
import signal
import sys
import time


class _Patch:
    def setattr(self, obj, name, value):
        setattr(obj, name, value)


def _fail(log: str, msg: str):
    with open(os.path.join(log, "fail_time"), "w") as f:
        f.write(str(time.time()))
    raise RuntimeError(msg)


def _watchdog_only(a, failing: int) -> int:
    from lean_rl import dist as ldist

    os.makedirs(a.log, exist_ok=True)
    ldist.init_dist()
    ldist.start_watchdog()
    ldist.barrier()  # both watchdogs are up
    if a.rank == failing:
        try:
            _fail(a.log, "boom outside any collective")
        except RuntimeError as e:
            ldist.abort(e)
    time.sleep(600)
    print(f"rank {a.rank} slept through the abort", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--fsdp", type=int, default=0)
    a = ap.parse_args()
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(a.port), WORLD_SIZE="2", RANK=str(a.rank),
        LOCAL_RANK=str(a.rank),
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    import torch

    torch.set_num_threads(1)
    if a.scenario.startswith("watchdog_"):
        return _watchdog_only(a, failing=int(a.scenario[len("watchdog_rank")]))
    import lean_rl.loop as loop_mod
    from lean_rl.trainer import Trainer
    from tests.lean_rl.test_loop_e2e import FakeVLLM, _cfg

    os.makedirs(a.log, exist_ok=True)
    kw = dict(num_batches=2, fsdp=bool(a.fsdp), keep_checkpoints=2)
    on_completion = None
    if a.scenario == "rollout_error" and a.rank == 0:
        loop_mod._PREEMPT_GRACE_S = 0.5

        def on_completion(n):
            _fail(a.log, "boom in rollout")
    elif a.scenario == "comm_error" and a.rank == 1:  # fails in run() before any collective
        loop_mod._Comm = lambda *x, **k: _fail(a.log, "boom in setup")
    elif a.scenario == "train_error" and a.rank == 1:
        if a.fsdp:
            Trainer._fwd_bwd_group = lambda self, *x, **k: _fail(a.log, "boom in train_step")
        else:
            Trainer._all_reduce_lora_grads = lambda self: _fail(a.log, "boom in train_step")
    elif a.scenario == "preempt":
        kw.update(num_batches=3, save_every=100, micro_batch_tokens=16)
        if a.rank == 1:
            real_fb = Trainer._fwd_bwd_group

            def fwd_bwd(self, *x, **k):
                out = real_fb(self, *x, **k)
                if self.step == 1:
                    os.kill(os.getpid(), signal.SIGUSR1)
                return out

            Trainer._fwd_bwd_group = fwd_bwd
    fake = FakeVLLM(on_completion=on_completion)
    transport = fake.install(_Patch())
    asyncio.run(loop_mod.run(_cfg(a.model, a.log, **kw), transport=transport))
    print(f"rank {a.rank} finished cleanly", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
