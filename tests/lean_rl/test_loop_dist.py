"""2-rank smoke of the lean RL loop (replicated and FSDP) against the fake vLLM. Every rank runs
``lean_rl.loop.run`` (SPMD); the run must not hang and rank 0 alone writes checkpoints / metrics
/ adapters.

- Two steps with save_every=1, then a resume for one more step; rank 1's ``load_latest`` is
  patched to pick an older checkpoint, which must not matter (rank 0's choice is broadcast).
- Preemption mid-``train_step`` on rank 1 only: both ranks abandon the step without an
  optimizer step, save the last completed step and exit cleanly.
- Preemption that kills vLLM during rank 0's adapter sync: both ranks save and exit cleanly.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

pytest.importorskip("tinker")
pytest.importorskip("chz")

ROOT = Path(__file__).resolve().parents[2]
DEVICE = os.environ.get("DIST_TEST_DEVICE", "cpu")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Patch:
    def setattr(self, obj, name, value):
        setattr(obj, name, value)


def _record_restores(rec: dict) -> None:
    """Record, on this rank, the checkpoint path restored and the LoRA right after it."""
    import lean_rl.loop as loop_mod

    real = loop_mod.restore_checkpoint

    def restore(path, trainer, **kw):
        out = real(path, trainer, **kw)
        rec["restored"] = path
        rec["lora_after_restore"] = trainer._lora_full()  # collective under FSDP: all ranks
        return out

    loop_mod.restore_checkpoint = restore


def _preempt_mid_train_step(rank: int, rec: dict) -> None:
    """Rank 1 raises SIGUSR1 after its first micro-batch of the second train_step; every rank
    records its LoRA before / after each train_step call."""
    from lean_rl.trainer import Trainer

    real_step, real_fb = Trainer.train_step, Trainer._fwd_bwd_group
    rec["steps"] = []

    def fwd_bwd(self, *a, **k):
        out = real_fb(self, *a, **k)
        if rank == 1 and self.step == 1 and not rec.get("signalled"):
            rec["signalled"] = True
            os.kill(os.getpid(), signal.SIGUSR1)
        return out

    def train_step(self, *a, **k):
        before = self._lora_full()  # collective under FSDP: every rank wraps train_step
        out = real_step(self, *a, **k)
        rec["steps"].append({
            "stopped": out is None, "before": before, "after": self._lora_full(), "step": self.step,
        })
        return out

    Trainer.train_step, Trainer._fwd_bwd_group = train_step, fwd_bwd


def _proc(rank, world_size, port, model_dir, log, num_batches, fsdp, out_dir, scenario):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tinker-cookbook"))
    import lean_rl.loop as loop_mod
    from lean_rl.checkpoint import LoopState
    from lean_rl.loop import run
    from tests.lean_rl.test_loop_e2e import FakeVLLM, _cfg

    rec: dict = {}
    kw = dict(num_batches=num_batches, fsdp=fsdp, compute_post_kl=True, keep_checkpoints=2)
    on_lora_load = None
    if scenario == "resume":
        _record_restores(rec)
        if rank == 1:  # a different (older) checkpoint than rank 0 would pick
            older = os.path.join(log, "ckpt", "step_000001")
            loop_mod.load_latest = lambda *a, **k: (older, LoopState(1, 1, None))
    elif scenario == "train_preempt":
        _preempt_mid_train_step(rank, rec)
        kw.update(save_every=100, micro_batch_tokens=16)  # one datum per micro-batch
    elif scenario == "sync_preempt":
        kw.update(save_every=100)

        def on_lora_load(n):  # rank 0 only (it alone talks to vLLM): dies at the 2nd sync
            if n == 2:
                fake.dead = True
                os.kill(os.getpid(), signal.SIGUSR1)

    fake = FakeVLLM(on_lora_load=on_lora_load)
    transport = fake.install(_Patch())
    asyncio.run(run(_cfg(model_dir, log, **kw), transport=transport))
    rec.update(lora_loads=len(fake.lora_loads), completions=fake.completions)
    torch.save(rec, os.path.join(out_dir, f"r{rank}_{scenario}_n{num_batches}.pt"))


def _spawn(model_dir, log, num_batches, fsdp, out_dir, scenario="plain"):
    mp.spawn(_proc, args=(2, _free_port(), model_dir, log, num_batches, fsdp, out_dir, scenario),
             nprocs=2, join=True)  # raises if any rank exits non-zero
    return [
        torch.load(os.path.join(out_dir, f"r{r}_{scenario}_n{num_batches}.pt"), weights_only=False)
        for r in range(2)
    ]


def _rows(log):
    with open(os.path.join(log, "metrics.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def _ckpts(log):
    return sorted(os.listdir(os.path.join(log, "ckpt")))


def _same(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def _skip_without_gpus():
    if DEVICE == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")


FSDP = pytest.mark.parametrize("fsdp", [False, True], ids=["replicated", "fsdp"])


@FSDP
def test_two_rank_loop_saves_and_resumes(tiny_model_dir, tmp_path, fsdp):
    _skip_without_gpus()
    log = str(tmp_path / "run")
    r0, r1 = _spawn(tiny_model_dir, log, 2, fsdp, str(tmp_path))
    assert [r["step"] for r in _rows(log)] == [0, 1]
    assert _ckpts(log) == ["step_000001", "step_000002"]
    # only rank 0 samples and hot-swaps adapters
    assert r0["lora_loads"] == 2 and r1["lora_loads"] == 0 and r1["completions"] == 0
    assert os.listdir(os.path.join(log, "serving_adapter")) == ["step_000002"]

    # Rank 1's load_latest would pick step_000001: every rank must restore rank 0's choice.
    r0, r1 = _spawn(tiny_model_dir, log, 3, fsdp, str(tmp_path), "resume")
    newest = os.path.join(log, "ckpt", "step_000002")
    assert r0["restored"] == r1["restored"] == newest
    _same(r0["lora_after_restore"], r1["lora_after_restore"])
    rows = _rows(log)
    assert [r["step"] for r in rows] == [0, 1, 2]  # the resumed run did exactly one step
    assert all(r["grad_norm"] > 0 for r in rows)
    assert _ckpts(log) == ["step_000002", "step_000003"]
    with open(os.path.join(log, "ckpt", "step_000003", "loop_state.json")) as f:
        assert json.load(f)["step"] == 3
    assert os.path.isfile(os.path.join(log, "ckpt", "step_000003", "adapter", "adapter_model.safetensors"))
    assert r0["lora_loads"] == 2 and r1["lora_loads"] == 0  # resume sync + one step


@FSDP
def test_two_rank_preemption_mid_train_step(tiny_model_dir, tmp_path, fsdp):
    _skip_without_gpus()
    log = str(tmp_path / "run")
    r0, r1 = _spawn(tiny_model_dir, log, 3, fsdp, str(tmp_path), "train_preempt")
    assert r1["signalled"] and not r0.get("signalled")
    for r in (r0, r1):
        assert [s["stopped"] for s in r["steps"]] == [False, True]  # both ranks stopped
        last = r["steps"][-1]
        assert last["step"] == 1  # no optimizer step
        _same(last["after"], last["before"])  # weights are the pre-step weights
    _same(r0["steps"][-1]["before"], r1["steps"][-1]["before"])
    assert [r["step"] for r in _rows(log)] == [0]
    assert _ckpts(log) == ["step_000001"]  # the last completed step
    state = torch.load(os.path.join(log, "ckpt", "step_000001", "optimizer.pt"), weights_only=False)
    assert state["step"] == 1
    _same(state["lora"], r0["steps"][-1]["before"])


def test_two_rank_vllm_dies_during_adapter_sync(tiny_model_dir, tmp_path):
    _skip_without_gpus()
    log = str(tmp_path / "run")
    r0, r1 = _spawn(tiny_model_dir, log, 3, True, str(tmp_path), "sync_preempt")
    assert r0["lora_loads"] == 2 and r1["lora_loads"] == 0
    assert [r["step"] for r in _rows(log)] == [0, 1]
    assert _ckpts(log) == ["step_000002"]  # step 2 completed; its sync failed


@pytest.mark.parametrize(
    "local_rank, expected", [(None, "cuda"), ("1", "cuda:1"), ("3", "cuda:3")]
)
def test_single_rank_device_honours_local_rank(monkeypatch, local_rank, expected):
    """One trainer GPU (sbatch_lean.sh --gres=gpu:2): vLLM owns GPU 0 and the trainer gets
    LOCAL_RANK=1. Plain "cuda" would put the trainer on vLLM's GPU (smoke job 11150506 OOM)."""
    from lean_rl import loop

    picked: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: picked.append(d))
    monkeypatch.setenv("WORLD_SIZE", "1")
    if local_rank is None:
        monkeypatch.delenv("LOCAL_RANK", raising=False)
    else:
        monkeypatch.setenv("LOCAL_RANK", local_rank)
    r, device = loop._init_rank()
    assert device == expected
    assert r.info is None
    assert picked == ([] if local_rank is None else [int(local_rank)])
