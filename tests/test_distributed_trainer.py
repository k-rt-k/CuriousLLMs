"""Data-parallel LocalTrainingClient: W ranks must reproduce the 1-rank update.

Each test spawns W CPU processes (gloo). Rank 0 drives forward_backward /
optim_step; ranks 1..W-1 run `serve_worker`. Rank 0 dumps the LoRA state (and
grads / logprobs / loss) to `tmp_path/ws{W}.pt`, then shuts the workers down;
each worker dumps its own LoRA state to `tmp_path/state_ws{W}_r{rank}.pt` on exit.
Every rank is seeded differently, so equal results also prove the rank-0 LoRA
init broadcast.
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

from local_backend.distributed import (  # noqa: E402
    Cmd,
    DistContext,
    send_command,
    shard_by_tokens,
)

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


def _snapshot(client) -> dict:
    return {
        "state": _lora_state(client),
        "opt_state_len": len(client.optimizer.state),
        "step_counter": client.step_counter,
    }


def _proc(rank, world_size, port, model_dir, out_dir, scenario, n_datums, load_path):
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

    # Different RNG per rank: replicas agree only via the rank-0 init broadcast.
    torch.manual_seed(rank)
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

    if scenario == "load":
        # Record every rank's params / optimizer right after the warm start.
        orig_load_adapter = client._do_load_adapter

        def _load_adapter_and_snapshot(path):
            orig_load_adapter(path)
            torch.save(_snapshot(client), os.path.join(out_dir, f"adapter_snap_ws{world_size}_r{rank}.pt"))

        client._do_load_adapter = _load_adapter_and_snapshot
    if scenario == "load_fail" and rank == world_size - 1:

        def _broken_load_state(path):
            raise RuntimeError(f"injected load failure on rank {rank}")

        client._do_load_state = _broken_load_state

    if ctx is not None and not ctx.is_driver:
        serve_worker(ctx, client)
        torch.save(_lora_state(client), os.path.join(out_dir, f"state_ws{world_size}_r{rank}.pt"))
        Path(out_dir, f"worker_exit_ws{world_size}_r{rank}").touch()
        return

    async def drive():
        data = _make_data(n_datums)
        # "accum": two forward_backward calls accumulate before one optim step.
        batches = [data[:3], data[3:]] if scenario == "accum" else [data]
        out = {"logprobs": [], "loss": [], "response_tokens": 0.0}
        if scenario == "resume_b2":
            batches = []  # resume straight from the checkpoint, no prior fwd/bwd
        for batch in batches:
            fb = await (await client.forward_backward_async(batch, "ppo")).result_async()
            out["logprobs"] += [o["logprobs"].to_torch() for o in fb.loss_fn_outputs]
            out["loss"].append(fb.metrics["loss"])
            out["response_tokens"] += fb.metrics["response_tokens"]
        out["grads"] = _lora_grads(client)
        adam = tinker.AdamParams(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8)

        async def step():
            await (await client.forward_backward_async(data, "ppo")).result_async()
            await (await client.optim_step_async(adam)).result_async()

        if scenario in ("step", "raise", "accum", "load", "resume_a", "resume_b1", "load_fail", "resume_moves"):
            await (await client.optim_step_async(adam)).result_async()
        if scenario == "resume_a":
            await step()
            await step()
        if scenario == "resume_b1":
            await step()
            out["saved"] = (await (await client.save_state_async("s2")).result_async()).path
        if scenario == "resume_b2":
            # Fresh process group: resume from run B1's checkpoint, take one step.
            await (await client.load_state_async(load_path)).result_async()
            await step()
        if scenario == "resume_moves":
            saved = (await (await client.save_state_async("s1")).result_async()).path
            await (await client.load_state_async(saved)).result_async()
            live = client._trainable_params()
            bound = [p for g in client.optimizer.param_groups for p in g["params"]]
            out["opt_bound_live"] = len(bound) == len(live) and all(a is b for a, b in zip(bound, live))
            out["opt_state_len"] = len(client.optimizer.state)
            out["before"] = _lora_state(client)
            await step()
        if scenario == "load_fail":
            saved = (await (await client.save_state_async("s1")).result_async()).path
            try:
                await (await client.load_state_async(saved)).result_async()
            except RuntimeError as e:
                out["load_error"] = str(e)
            # The protocol must still be in sync: another step works on every rank.
            await step()
        if scenario == "load":
            # Workers must follow the driver through LOAD_STATE / LOAD_ADAPTER,
            # else the next all-reduced step mixes grads from different weights.
            saved = (await (await client.save_state_async("s1")).result_async()).path
            out["saved"] = saved

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


def _run(world_size, model_dir, out_dir, scenario="step", n_datums=6, load_path=None):
    os.makedirs(out_dir, exist_ok=True)
    port = _free_port()
    mp.spawn(
        _proc,
        args=(world_size, port, model_dir, str(out_dir), scenario, n_datums, load_path),
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


def _worker_state(out_dir, world_size, rank) -> dict:
    return torch.load(os.path.join(out_dir, f"state_ws{world_size}_r{rank}.pt"), weights_only=False)


def test_load_state_and_adapter_reach_workers(tiny_model_dir, tmp_path):
    from safetensors.torch import load_file

    one = _run(1, tiny_model_dir, tmp_path, scenario="load")
    two = _run(2, tiny_model_dir, tmp_path, scenario="load")
    _assert_same_dicts(one["state"], two["state"], atol=1e-6)
    _assert_same_dicts(two["state"], _worker_state(tmp_path, 2, 1), atol=1e-6)

    # Warm start: right after LOAD_ADAPTER every rank holds exactly the adapter
    # file's tensors, with a fresh optimizer and step 0.
    adapter = load_file(os.path.join(two["saved"], "adapter_model.safetensors"))
    for rank in range(2):
        snap = torch.load(tmp_path / f"adapter_snap_ws2_r{rank}.pt", weights_only=False)
        assert snap["opt_state_len"] == 0 and snap["step_counter"] == 0
        live = {k.replace(".default.", "."): v for k, v in snap["state"].items()}
        _assert_same_dicts(adapter, live, atol=0.0)


def test_load_state_rebinds_optimizer_single_rank(tiny_model_dir, tmp_path):
    """After load_state the optimizer must drive the live LoRA params (with the
    restored Adam moments), not the tensors PEFT deleted during the reload."""
    out = _run(1, tiny_model_dir, tmp_path, scenario="resume_moves")
    assert out["opt_bound_live"], "optimizer still bound to pre-reload param tensors"
    assert out["opt_state_len"] > 0, "Adam moments were not restored"
    assert any(not torch.equal(out["before"][k], out["state"][k]) for k in out["state"]), (
        "optim_step after load_state did not change the live LoRA params"
    )


def test_resume_matches_uninterrupted(tiny_model_dir, tmp_path):
    """W=2: 3 straight steps == 2 steps, save, new process group, load, 1 step."""
    a = _run(2, tiny_model_dir, tmp_path / "a", scenario="resume_a")
    b1 = _run(2, tiny_model_dir, tmp_path / "b1", scenario="resume_b1")
    b2 = _run(2, tiny_model_dir, tmp_path / "b2", scenario="resume_b2", load_path=b1["saved"])
    # Each rank's final params, both runs.
    _assert_same_dicts(a["state"], b2["state"], atol=1e-6)
    _assert_same_dicts(_worker_state(tmp_path / "a", 2, 1), _worker_state(tmp_path / "b2", 2, 1), atol=1e-6)
    _assert_same_dicts(b2["state"], _worker_state(tmp_path / "b2", 2, 1), atol=1e-6)
    # And resuming did something: the result is not just the 2-step checkpoint.
    assert any(not torch.equal(b1["state"][k], b2["state"][k]) for k in b2["state"])


def test_load_failure_on_worker_is_reported(tiny_model_dir, tmp_path, monkeypatch):
    """A worker whose load raises reports it via the closing gather and keeps
    serving; the driver raises naming that rank, and the protocol stays in sync."""
    monkeypatch.setenv("LOCAL_DIST_TIMEOUT_S", "30")
    out = _run(2, tiny_model_dir, tmp_path, scenario="load_fail")
    err = out.get("load_error")
    assert err is not None, "driver did not raise on a worker load failure"
    assert "rank(s) [1]" in err and "injected load failure on rank 1" in err
    assert Path(tmp_path, "worker_exit_ws2_r1").exists()


def test_send_command_refuses_while_busy():
    ctx = DistContext(rank=0, world_size=2, local_rank=0, obj_group=None, busy=True)
    with pytest.raises(RuntimeError, match="protocol desynced: previous command did not complete"):
        send_command(ctx, Cmd.OPTIM, [None, None])


def test_driver_exception_terminates_workers(tiny_model_dir, tmp_path, monkeypatch):
    # Backstop only: a hung collective would fail after this, not hang the suite.
    monkeypatch.setenv("LOCAL_DIST_TIMEOUT_S", "30")
    t0 = time.monotonic()
    with pytest.raises(mp.ProcessRaisedException, match="driver failure"):
        _run(2, tiny_model_dir, tmp_path, scenario="raise")
    assert time.monotonic() - t0 < 60
    # The worker left serve_worker via SHUTDOWN (not via spawn's terminate).
    assert Path(tmp_path, "worker_exit_ws2_r1").exists()
