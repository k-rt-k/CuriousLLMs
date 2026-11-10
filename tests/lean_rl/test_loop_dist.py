"""2-rank smoke of the lean RL loop (replicated and FSDP) against the fake vLLM: two steps with
save_every=1, then a resume for one more step. Every rank runs ``lean_rl.loop.run`` (SPMD);
the run must not hang, rank 0 alone writes checkpoints / metrics / adapters, and the resumed
run continues the step count."""

from __future__ import annotations

import asyncio
import json
import os
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


def _proc(rank, world_size, port, model_dir, log, num_batches, fsdp, out_dir):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tinker-cookbook"))
    from lean_rl.loop import run
    from tests.lean_rl.test_loop_e2e import FakeVLLM, _cfg

    fake = FakeVLLM()
    transport = fake.install(_Patch())
    cfg = _cfg(model_dir, log, num_batches=num_batches, fsdp=fsdp, compute_post_kl=True,
               keep_checkpoints=2)
    asyncio.run(run(cfg, transport=transport))
    with open(os.path.join(out_dir, f"r{rank}_n{num_batches}.json"), "w") as f:
        json.dump({"lora_loads": len(fake.lora_loads), "completions": fake.completions}, f)


def _spawn(model_dir, log, num_batches, fsdp, out_dir):
    mp.spawn(_proc, args=(2, _free_port(), model_dir, log, num_batches, fsdp, out_dir),
             nprocs=2, join=True)
    return [json.load(open(os.path.join(out_dir, f"r{r}_n{num_batches}.json"))) for r in range(2)]


def _rows(log):
    with open(os.path.join(log, "metrics.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("fsdp", [False, True], ids=["replicated", "fsdp"])
def test_two_rank_loop_saves_and_resumes(tiny_model_dir, tmp_path, fsdp):
    if DEVICE == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    log = str(tmp_path / "run")
    r0, r1 = _spawn(tiny_model_dir, log, 2, fsdp, str(tmp_path))
    assert [r["step"] for r in _rows(log)] == [0, 1]
    assert sorted(os.listdir(os.path.join(log, "ckpt"))) == ["step_000001", "step_000002"]
    # only rank 0 samples and hot-swaps adapters
    assert r0["lora_loads"] == 2 and r1["lora_loads"] == 0 and r1["completions"] == 0
    assert os.listdir(os.path.join(log, "serving_adapter")) == ["step_000002"]

    r0, r1 = _spawn(tiny_model_dir, log, 3, fsdp, str(tmp_path))
    rows = _rows(log)
    assert [r["step"] for r in rows] == [0, 1, 2]  # the resumed run did exactly one step
    assert all(r["grad_norm"] > 0 for r in rows)
    assert sorted(os.listdir(os.path.join(log, "ckpt"))) == ["step_000002", "step_000003"]
    with open(os.path.join(log, "ckpt", "step_000003", "loop_state.json")) as f:
        assert json.load(f)["step"] == 3
    assert os.path.isfile(os.path.join(log, "ckpt", "step_000003", "adapter", "adapter_model.safetensors"))
    assert r0["lora_loads"] == 2 and r1["lora_loads"] == 0  # resume sync + one step
