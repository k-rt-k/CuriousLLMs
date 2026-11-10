"""Data-parallel LocalTrainingClient: W ranks must reproduce the 1-rank update.

Each test spawns W CPU processes (gloo). Rank 0 drives forward_backward /
optim_step; ranks 1..W-1 run `serve_worker`. Rank 0 dumps the LoRA state (and
grads / logprobs / loss) to `tmp_path/ws{W}.pt`, then shuts the workers down.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tinker-cookbook"))

from local_backend.distributed import shard_by_tokens  # noqa: E402

LENGTHS = [3, 11, 5, 8, 4, 7]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_data(n: int):
    import tinker
    from tinker import TensorData

    g = torch.Generator().manual_seed(1234)
    data = []
    for T in LENGTHS[:n]:
        tokens = torch.randint(2, 64, (T,), generator=g)
        targets = torch.randint(2, 64, (T,), generator=g)
        n_prompt = 1 + (T // 3)
        adv = torch.zeros(T)
        adv[n_prompt:] = torch.randn(T - n_prompt, generator=g)
        # Near the random-init policy's logprob (~log 1/64) so PPO ratios ~1.
        logp = -4.16 + 0.1 * torch.randn(T, generator=g)
        data.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(tokens.tolist()),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(targets),
                    "logprobs": TensorData.from_torch(logp),
                    "advantages": TensorData.from_torch(adv),
                },
            )
        )
    return data


def _lora_state(client) -> dict:
    return {
        n: p.detach().cpu().clone()
        for n, p in client.model.named_parameters()
        if p.requires_grad
    }


def _lora_grads(client) -> dict:
    return {
        n: (p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p).cpu())
        for n, p in client.model.named_parameters()
        if p.requires_grad
    }


def _proc(rank, world_size, port, model_dir, out_dir, scenario, n_datums):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tinker-cookbook"))

    import tinker

    from local_backend.distributed import init_distributed, serve_worker, shutdown_workers
    from local_backend.training_client import LocalTrainingClient

    ctx = init_distributed()
    assert (ctx is None) == (world_size == 1)

    torch.manual_seed(0)
    client = LocalTrainingClient(
        base_model=model_dir,
        lora_rank=4,
        vllm_url="http://placeholder",
        log_path=os.path.join(out_dir, f"logs_r{rank}"),
        device="cpu",
        dtype=torch.float32,
        dist=ctx,
    )
    asyncio.run(client._init_async())

    if ctx is not None and not ctx.is_driver:
        serve_worker(ctx, client)
        Path(out_dir, f"worker_exit_ws{world_size}_r{rank}").touch()
        return

    async def drive():
        data = _make_data(n_datums)
        # "accum": two forward_backward calls accumulate before one optim step.
        batches = [data[:3], data[3:]] if scenario == "accum" else [data]
        out = {"logprobs": [], "loss": [], "response_tokens": 0.0}
        for batch in batches:
            fb = await (await client.forward_backward_async(batch, "ppo")).result_async()
            out["logprobs"] += [o["logprobs"].to_torch() for o in fb.loss_fn_outputs]
            out["loss"].append(fb.metrics["loss"])
            out["response_tokens"] += fb.metrics["response_tokens"]
        out["grads"] = _lora_grads(client)
        adam = tinker.AdamParams(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8)
        if scenario in ("step", "raise", "accum", "load"):
            await (await client.optim_step_async(adam)).result_async()
        if scenario == "load":
            # Workers must follow the driver through LOAD_STATE / LOAD_ADAPTER,
            # else the next all-reduced step mixes grads from different weights.
            saved = (await (await client.save_state_async("s1")).result_async()).path

            async def step():
                await (await client.forward_backward_async(data, "ppo")).result_async()
                await (await client.optim_step_async(adam)).result_async()

            await step()
            await (await client.load_state_async(saved)).result_async()
            await step()
            await (await client.load_adapter_weights_async(saved)).result_async()
            assert client.step_counter == 0
            await step()
        out["state"] = _lora_state(client)
        torch.save(out, os.path.join(out_dir, f"ws{world_size}.pt"))
        if scenario == "raise":
            raise RuntimeError("driver failure after one step")

    try:
        asyncio.run(drive())
    finally:
        if ctx is not None:
            shutdown_workers(ctx)
            if scenario == "raise":
                # Give workers a moment to record a clean exit before this
                # process dies and spawn() terminates the rest.
                deadline = time.monotonic() + 20
                markers = [Path(out_dir, f"worker_exit_ws{world_size}_r{r}") for r in range(1, world_size)]
                while time.monotonic() < deadline and not all(m.exists() for m in markers):
                    time.sleep(0.1)


def _run(world_size, model_dir, out_dir, scenario="step", n_datums=6):
    os.makedirs(out_dir, exist_ok=True)
    port = _free_port()
    mp.spawn(
        _proc,
        args=(world_size, port, model_dir, str(out_dir), scenario, n_datums),
        nprocs=world_size,
        join=True,
    )
    return torch.load(os.path.join(out_dir, f"ws{world_size}.pt"), weights_only=False)


def _assert_same_dicts(a: dict, b: dict, atol: float):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], atol=atol, rtol=0), (
            f"{k}: max abs diff {(a[k] - b[k]).abs().max().item():.3e}"
        )


def test_shard_by_tokens_partitions_and_balances():
    lengths = [10, 1, 1, 1, 9, 2]
    shards = shard_by_tokens(lengths, 2)
    assert len(shards) == 2
    flat = [i for s in shards for i in s]
    assert sorted(flat) == list(range(6))
    assert len(flat) == len(set(flat))
    assert sorted(sum(lengths[i] for i in s) for s in shards) == [12, 12]


def test_shard_by_tokens_more_ranks_than_items():
    shards = shard_by_tokens([5, 3], 3)
    assert len(shards) == 3
    assert sorted(i for s in shards for i in s) == [0, 1]
    assert sum(1 for s in shards if not s) == 1


def test_two_ranks_match_one_rank(tiny_model_dir, tmp_path):
    one = _run(1, tiny_model_dir, tmp_path)
    two = _run(2, tiny_model_dir, tmp_path)
    _assert_same_dicts(one["state"], two["state"], atol=1e-6)
    assert len(one["logprobs"]) == len(two["logprobs"]) == 6
    for a, b in zip(one["logprobs"], two["logprobs"]):
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-6, rtol=0)
    assert abs(one["loss"][0] - two["loss"][0]) < 1e-6
    assert one["response_tokens"] == two["response_tokens"]
    # The step must actually have moved the weights.
    init = _run(1, tiny_model_dir, tmp_path / "init", scenario="fwd")
    assert any(not torch.equal(init["state"][k], one["state"][k]) for k in one["state"])


def test_grad_is_sum_not_mean(tiny_model_dir, tmp_path):
    one = _run(1, tiny_model_dir, tmp_path, scenario="fwd")
    two = _run(2, tiny_model_dir, tmp_path, scenario="fwd")
    assert any(g.abs().max() > 1e-4 for g in one["grads"].values()), "need non-trivial grads"
    _assert_same_dicts(one["grads"], two["grads"], atol=1e-6)


def test_empty_shard_rank(tiny_model_dir, tmp_path):
    one = _run(1, tiny_model_dir, tmp_path, n_datums=2)
    three = _run(3, tiny_model_dir, tmp_path, n_datums=2)
    _assert_same_dicts(one["state"], three["state"], atol=1e-6)
    for a, b in zip(one["logprobs"], three["logprobs"]):
        assert torch.allclose(a, b, atol=1e-6, rtol=0)
    assert abs(one["loss"][0] - three["loss"][0]) < 1e-6


def test_grad_accumulation_across_calls(tiny_model_dir, tmp_path):
    """Two forward_backward calls before one optim_step: earlier (already reduced)
    grads must not be all-reduced again."""
    one = _run(1, tiny_model_dir, tmp_path, scenario="accum")
    two = _run(2, tiny_model_dir, tmp_path, scenario="accum")
    single_call = _run(1, tiny_model_dir, tmp_path / "single", scenario="step")
    _assert_same_dicts(one["grads"], two["grads"], atol=1e-6)
    _assert_same_dicts(one["state"], two["state"], atol=1e-6)
    _assert_same_dicts(single_call["state"], two["state"], atol=1e-6)
    for a, b in zip(one["loss"], two["loss"]):
        assert abs(a - b) < 1e-6


def test_load_state_and_adapter_reach_workers(tiny_model_dir, tmp_path):
    one = _run(1, tiny_model_dir, tmp_path, scenario="load")
    two = _run(2, tiny_model_dir, tmp_path, scenario="load")
    _assert_same_dicts(one["state"], two["state"], atol=1e-6)


def test_driver_exception_terminates_workers(tiny_model_dir, tmp_path, monkeypatch):
    # Backstop only: a hung collective would fail after this, not hang the suite.
    monkeypatch.setenv("LOCAL_DIST_TIMEOUT_S", "30")
    t0 = time.monotonic()
    with pytest.raises(mp.ProcessRaisedException, match="driver failure"):
        _run(2, tiny_model_dir, tmp_path, scenario="raise")
    assert time.monotonic() - t0 < 60
    # The worker left serve_worker via SHUTDOWN (not via spawn's terminate).
    assert Path(tmp_path, "worker_exit_ws2_r1").exists()
