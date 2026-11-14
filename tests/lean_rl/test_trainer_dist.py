"""Multi-rank lean_rl Trainer (replicated DP and FSDP2): W ranks must reproduce 1 rank.

Each configuration spawns W processes (gloo on CPU) that all run the same SPMD script
(`_proc`); rank 0 dumps its observations to ``out_dir/ws{W}.pt``. Every rank is seeded
differently, so agreement with the 1-rank run also proves the rank-0 LoRA init broadcast.
A small ``micro_batch_tokens`` gives ranks different micro-batch counts, which FSDP must
pad with dummy forwards/backwards.

DIST_TEST_DEVICE=cuda runs the same tests on one GPU per rank (NCCL); configurations
needing more ranks than GPUs are skipped.
"""

from __future__ import annotations

import math
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

tinker = pytest.importorskip("tinker")

ROOT = Path(__file__).resolve().parents[2]
DEVICE = os.environ.get("DIST_TEST_DEVICE", "cpu")
ATOL = float(os.environ.get("DIST_TEST_ATOL", "1e-5"))
BF16_ATOL = 2e-2
LENGTHS = [3, 11, 5, 8, 4, 7]
MICRO_BATCH_TOKENS = 24
ADAM = dict(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_data(lengths=LENGTHS, seed=1234, vocab=64):
    from tinker import TensorData

    g = torch.Generator().manual_seed(seed)
    data = []
    for T in lengths:
        seq = torch.randint(2, vocab, (T + 1,), generator=g)
        n_prompt = 1 + (T // 3)
        mask = torch.zeros(T)
        mask[n_prompt:] = 1.0
        adv = torch.zeros(T)
        adv[n_prompt:] = torch.randn(T - n_prompt, generator=g)
        logp = -4.16 + 0.1 * torch.randn(T, generator=g)
        data.append(tinker.Datum(
            model_input=tinker.ModelInput.from_ints(seq[:-1].tolist()),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(seq[1:].clone()),
                "logprobs": TensorData.from_torch(logp),
                "advantages": TensorData.from_torch(adv),
                "mask": TensorData.from_torch(mask),
            },
        ))
    return data


def _proc(rank, world_size, port, model_dir, out_dir, fsdp):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tinker-cookbook"))

    from lean_rl.dist import device_for, init_dist
    from lean_rl.trainer import Trainer, _full

    info = init_dist()
    assert (info is None) == (world_size == 1)
    if fsdp and rank > 0:
        _forbid_weight_loading()
    torch.manual_seed(100 + rank)  # replicas agree only via the rank-0 init broadcast
    tr = Trainer(
        model_dir, rank=4, device=device_for(info) if DEVICE == "cuda" else "cpu",
        dtype=torch.float32, dist=info, fsdp=fsdp, micro_batch_tokens=MICRO_BATCH_TOKENS,
    )
    assert tr.fsdp == (fsdp and world_size > 1)
    if tr.fsdp:
        from torch.distributed.tensor import DTensor

        assert all(isinstance(p, DTensor) for p in tr.model.parameters())
        assert all(p.to_local().device.type != "meta" for p in tr.model.parameters())
        assert all(b.device.type != "meta" for b in tr.model.buffers())
    # Same nonzero lora_B on every rank so lora_A gets gradient too.
    full = tr._lora_full()
    g = torch.Generator().manual_seed(7)
    for n in sorted(full):
        if "lora_B" in n:
            full[n] = 0.05 * torch.randn(full[n].shape, generator=g)
    tr._lora_assign(full)
    tr.snapshot_init_adapter()

    data = _make_data()
    adam = tinker.AdamParams(**ADAM)
    out: dict = {"state0": tr._lora_full()}
    out["lp5"] = tr.logprobs(data[:5], "policy")
    out["fb"] = tr._forward_backward(data, "ppo")
    out["grads"] = {n: _full(p.grad).detach().cpu().clone() for n, p in tr._named_lora()}
    tr._ensure_optimizer().zero_grad(set_to_none=True)
    out["m1"] = tr.train_step(data, "ppo", adam)
    out["state1"] = tr._lora_full()
    # One datum: with 2 ranks one shard is empty (train and logprobs).
    out["m2"] = tr.train_step(data[:1], "ppo", adam)
    out["state2"] = tr._lora_full()
    out["lp1"] = tr.logprobs(data[:1], "policy")
    out["lp_after"] = tr.logprobs(data, "policy")
    out["lp_init"] = tr.logprobs(data, "init")
    out["lp_base"] = tr.logprobs(data, "base")
    adapter = os.path.join(out_dir, "adapter")
    # Every rank calls save_adapter; only rank 0 writes (rank r>0 gets its own path here so
    # a stray write would be visible).
    tr.save_adapter(adapter if rank == 0 else f"{adapter}_r{rank}")
    if rank > 0:
        assert not os.path.exists(f"{adapter}_r{rank}")
    # Checkpoint round trip, with gradient clipping active.
    ck = tr.state_dict_for_ckpt()
    clip = tinker.AdamParams(**ADAM, grad_clip_norm=1e-2)
    out["m3"] = tr.train_step(data, "importance_sampling", clip)
    out["state3"] = tr._lora_full()
    tr.load_ckpt_state(ck)
    out["m3b"] = tr.train_step(data, "importance_sampling", clip)
    out["state3b"] = tr._lora_full()
    tr.load_adapter(adapter)
    out["lp_loaded"] = tr.logprobs(data, "policy")
    if rank == 0:
        torch.save(out, os.path.join(out_dir, f"ws{world_size}.pt"))
    else:
        assert out["lp5"] is None
        # Replicas / shards agree with rank 0 (rank 0's file is compared in the tests).
        torch.save({"state3b": out["state3b"]}, os.path.join(out_dir, f"ws{world_size}_r{rank}.pt"))


def _forbid_weight_loading():
    """FSDP ranks > 0 must never materialise the full checkpoint (they build on meta)."""
    from transformers import AutoModelForCausalLM

    def boom(*a, **k):
        raise AssertionError("rank > 0 called from_pretrained under FSDP")

    AutoModelForCausalLM.from_pretrained = boom


def _proc_bf16(rank, world_size, port, model_dir, out_dir, fsdp):
    """Production dtype path: bf16 frozen base, fp32 LoRA (FSDP: param_dtype=bf16)."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
        RANK=str(rank), LOCAL_RANK=str(rank),
    )
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tinker-cookbook"))
    from lean_rl.dist import device_for, init_dist
    from lean_rl.trainer import Trainer

    info = init_dist()
    if fsdp and rank > 0:
        _forbid_weight_loading()
    torch.manual_seed(100 + rank)
    tr = Trainer(
        model_dir, rank=4, device=device_for(info) if DEVICE == "cuda" else "cpu",
        dtype=torch.bfloat16, dist=info, fsdp=fsdp, micro_batch_tokens=MICRO_BATCH_TOKENS,
    )
    assert tr.fsdp == (fsdp and world_size > 1)
    assert all(p.dtype == torch.float32 for _, p in tr._named_lora())
    full = tr._lora_full()
    g = torch.Generator().manual_seed(7)
    for n in sorted(full):
        if "lora_B" in n:
            full[n] = 0.05 * torch.randn(full[n].shape, generator=g)
    tr._lora_assign(full)
    data = _make_data()
    # The loop's first forward is the KL "base" pass (PEFT disable_adapter flips LoRA
    # requires_grad off). FSDP2 must still reduce LoRA grads in fp32 and cast them back to the
    # fp32 shards; torch >= 2.11 raises on a bf16 grad, older versions silently keep it.
    tr.logprobs(data, "base")
    checked = 0
    for m in tr._fsdp_modules:
        group = m._get_fsdp_state()._fsdp_param_group
        if group is not None and any(p.sharded_param.requires_grad for p in group.fsdp_params):
            assert group._orig_dtype == torch.float32, group._orig_dtype
            assert group._reduce_dtype == torch.float32, group._reduce_dtype
            checked += 1
    assert checked > 0 or not tr.fsdp, "no FSDP param group with LoRA params found (torch internals changed?)"
    out = {"lp0": tr.logprobs(data, "policy")}
    out["m"] = tr.train_step(data, "ppo", tinker.AdamParams(**ADAM, grad_clip_norm=1.0))
    out["lp1"] = tr.logprobs(data, "policy")
    assert all(p.dtype == torch.float32 for _, p in tr._named_lora())
    if rank == 0:
        torch.save(out, os.path.join(out_dir, f"bf16_ws{world_size}.pt"))


_CACHE: dict = {}


def _run(world_size: int, fsdp: bool, tmp_factory) -> dict:
    key = (world_size, fsdp)
    if key in _CACHE:
        return _CACHE[key]
    if DEVICE == "cuda" and world_size > torch.cuda.device_count():
        pytest.skip(f"needs {world_size} GPUs")
    out_dir = tmp_factory.mktemp(f"ws{world_size}_fsdp{int(fsdp)}")
    mp.spawn(_proc, args=(world_size, _free_port(), MODEL[0], str(out_dir), fsdp),
             nprocs=world_size, join=True)
    res = torch.load(os.path.join(out_dir, f"ws{world_size}.pt"), weights_only=False)
    res["out_dir"] = str(out_dir)
    res["others"] = [
        torch.load(os.path.join(out_dir, f"ws{world_size}_r{r}.pt"), weights_only=False)
        for r in range(1, world_size)
    ]
    _CACHE[key] = res
    return res


MODEL: list[str] = []


@pytest.fixture(scope="module")
def runs(tiny_model_dir, tmp_path_factory):
    MODEL[:] = [tiny_model_dir]
    return lambda ws, fsdp: _run(ws, fsdp, tmp_path_factory)


def _same(a: dict, b: dict, atol=ATOL):
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], atol=atol, rtol=0, msg=lambda m, k=k: f"{k}: {m}")


def _same_list(a, b, atol=ATOL):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.shape == y.shape
        torch.testing.assert_close(x, y, atol=atol, rtol=0)


def _same_metrics(a: dict, b: dict, atol=ATOL):
    assert a.keys() == b.keys()
    for k in a:
        assert abs(a[k] - b[k]) <= atol * max(1.0, abs(b[k])), (k, a[k], b[k])


FSDP = pytest.mark.parametrize("fsdp", [False, True], ids=["replicated", "fsdp"])


def test_shard_and_init_dist_single(monkeypatch):
    from lean_rl.dist import DistInfo, broadcast_object, init_dist, shard

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert init_dist() is None
    assert broadcast_object({"a": 1}) == {"a": 1}
    data = list("abcdef")
    assert shard(data, LENGTHS, None) == data
    parts = [shard(data, LENGTHS, DistInfo(r, 2, r, r == 0)) for r in range(2)]
    assert sorted(x for p in parts for x in p) == data


@FSDP
def test_two_ranks_match_one_rank(runs, fsdp):
    one, two = runs(1, False), runs(2, fsdp)
    _same(one["state0"], two["state0"])
    _same(one["state1"], two["state1"])
    _same_metrics(one["m1"], two["m1"])
    assert any(not torch.equal(one["state0"][k], one["state1"][k]) for k in one["state0"])
    _same_list(one["lp_after"], two["lp_after"])
    _same_list(one["lp_init"], two["lp_init"])
    _same_list(one["lp_base"], two["lp_base"])
    # Checkpoint round trip (clipped step) is exact and matches 1 rank on every rank.
    _same(two["state3"], two["state3b"], atol=1e-6)
    _same(one["state3"], two["state3"])
    _same_metrics(one["m3"], two["m3"])
    for other in two["others"]:
        _same(two["state3b"], other["state3b"], atol=1e-6)


@FSDP
def test_grad_is_sum_not_mean(runs, fsdp):
    one, two = runs(1, False), runs(2, fsdp)
    assert any(g.abs().max() > 1e-3 for g in one["grads"].values()), "need non-trivial grads"
    _same(one["grads"], two["grads"])
    _same_metrics(one["fb"], two["fb"])


@FSDP
def test_fewer_datums_than_ranks(runs, fsdp):
    one, two = runs(1, False), runs(2, fsdp)
    _same(one["state2"], two["state2"])
    _same_metrics(one["m2"], two["m2"])
    _same_list(one["lp1"], two["lp1"])


@FSDP
def test_logprobs_gathered_in_order(runs, fsdp):
    one, two = runs(1, False), runs(2, fsdp)
    assert len(two["lp5"]) == 5
    assert [len(x) for x in two["lp5"]] == LENGTHS[:5]
    _same_list(one["lp5"], two["lp5"])


@FSDP
def test_save_adapter_under_fsdp_loads_in_single_rank(runs, fsdp, tiny_model_dir):
    from lean_rl.trainer import Trainer

    two = runs(2, fsdp)
    tr = Trainer(tiny_model_dir, rank=4, device="cpu", dtype=torch.float32)
    tr.load_adapter(os.path.join(two["out_dir"], "adapter"))
    _same_list(tr.logprobs(_make_data(), "policy"), two["lp_after"])
    # ... and the multi-rank trainer reloads its own adapter exactly.
    _same_list(two["lp_loaded"], two["lp_after"], atol=1e-6)


@pytest.mark.parametrize("fsdp", [False, True], ids=["replicated", "fsdp"])
def test_bf16_base_fp32_lora(tiny_model_dir, tmp_path, fsdp):
    """bf16 base / fp32 LoRA runs; 2-rank logprobs and metrics track 1 rank within bf16 noise."""
    res = {}
    for ws in (1, 2):
        if DEVICE == "cuda" and ws > torch.cuda.device_count():
            pytest.skip(f"needs {ws} GPUs")
        mp.spawn(_proc_bf16, args=(ws, _free_port(), tiny_model_dir, str(tmp_path), fsdp and ws > 1),
                 nprocs=ws, join=True)
        res[ws] = torch.load(tmp_path / f"bf16_ws{ws}.pt", weights_only=False)
    one, two = res[1], res[2]
    # Before the step only the LoRA compute dtype differs (FSDP param_dtype=bf16 vs PEFT's
    # fp32 LoRA): ~1e-2. After one AdamW step the per-element normalisation turns bf16 noise
    # in tiny grads into lr-sized weight differences, so allow more (observed 0.077 on CPU).
    # Replicated is not bit-exact either: the per-rank micro-batches have other shapes than the
    # 1-rank run, so bf16 GEMMs pick other kernels / reduction orders (observed 4.6e-3 on L40,
    # 7.5e-4 on some Babel CPU nodes, 0 on others), so it gets the bf16 tolerance too.
    for key, atol in (("lp0", BF16_ATOL), ("lp1", 0.15)):
        for a, b in zip(one[key], two[key]):
            assert torch.isfinite(b).all()
            torch.testing.assert_close(b, a, atol=atol, rtol=0)
    for k in ("loss", "grad_norm"):
        assert math.isfinite(two["m"][k])
        assert abs(two["m"][k] - one["m"][k]) <= 0.05 * max(1.0, abs(one["m"][k])), k
    assert two["m"]["response_tokens"] == one["m"]["response_tokens"]
