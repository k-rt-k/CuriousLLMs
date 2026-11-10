"""math_train's torchrun entry path, on CPU (gloo) with the tiny model.

Launches `torchrun --standalone --nproc_per_node=2` on a stub driver that uses the
same wiring as `math_train._entry`: `run_worker_if_needed` on every rank (rank 1
enters `serve_worker`), rank 0 builds its training client through
`LocalServiceClient` (picking up the process-global context), runs one
forward_backward + optim_step, then `shutdown_workers` in a `finally`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

STUB = textwrap.dedent(
    """
    import asyncio, os, sys
    from pathlib import Path

    sys.path.insert(0, os.environ["ROOT"])
    sys.path.insert(0, os.path.join(os.environ["ROOT"], "tinker-cookbook"))
    import tinker
    import math_train
    from local_backend import LocalServiceClient
    from local_backend.distributed import get_context, shutdown_workers
    from tests.test_distributed_trainer import _make_data

    out = Path(os.environ["OUT_DIR"])
    rank = int(os.environ["RANK"])
    cfg = math_train.CLIConfig(
        backend="local", model_name=os.environ["MODEL_DIR"], lora_rank=4,
        log_path=str(out / f"logs_r{rank}"),
    )

    async def main():
        if await math_train.run_worker_if_needed(cfg):
            (out / f"worker_done_r{rank}").touch()
            return
        ctx = get_context()
        assert ctx is not None and ctx.is_driver and ctx.world_size == 2
        try:
            svc = LocalServiceClient(vllm_urls="http://unused:1", log_path=cfg.log_path)
            client = await svc.create_lora_training_client_async(
                cfg.model_name, rank=cfg.lora_rank, device="cpu"
            )
            assert client.dist is ctx
            fb = await (await client.forward_backward_async(_make_data(4), "ppo")).result_async()
            assert len(fb.loss_fn_outputs) == 4
            adam = tinker.AdamParams(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8)
            await (await client.optim_step_async(adam)).result_async()
            (out / "driver_stepped").touch()
        finally:
            shutdown_workers(get_context())
        (out / "driver_done").touch()

    asyncio.run(main())
    """
)


@pytest.mark.skipif(shutil.which("torchrun") is None, reason="torchrun not on PATH")
def test_torchrun_worker_enters_serve_worker_and_exits_on_shutdown(tmp_path, tiny_model_dir):
    stub = tmp_path / "stub_driver.py"
    stub.write_text(STUB)
    env = dict(
        os.environ,
        ROOT=str(ROOT),
        OUT_DIR=str(tmp_path),
        MODEL_DIR=tiny_model_dir,
        PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "tinker-cookbook")]),
        GLOO_SOCKET_IFNAME="lo",
        OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
    )
    proc = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=2", str(stub)],
        env=env, cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    assert (tmp_path / "driver_stepped").exists()
    assert (tmp_path / "driver_done").exists()
    assert (tmp_path / "worker_done_r1").exists()


def test_tinker_backend_rejects_multi_process(monkeypatch):
    import asyncio

    import math_train

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="backend=local"):
        asyncio.run(math_train.run_worker_if_needed(math_train.CLIConfig(backend="tinker")))


def test_tinker_backend_single_process_is_not_a_worker(monkeypatch):
    import asyncio

    import math_train

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert asyncio.run(math_train.run_worker_if_needed(math_train.CLIConfig(backend="tinker"))) is False
