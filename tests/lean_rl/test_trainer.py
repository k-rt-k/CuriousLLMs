import asyncio
import math
import os

import pytest
import torch

tinker = pytest.importorskip("tinker")
from tinker import TensorData  # noqa: E402

from lean_rl.trainer import Trainer  # noqa: E402

ADAM = dict(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8)


def _datum(tokens, targets, logp, adv, mask):
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([int(t) for t in tokens]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.as_tensor(targets, dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.as_tensor(logp, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.as_tensor(adv, dtype=torch.float32)),
            "mask": TensorData.from_torch(torch.as_tensor(mask, dtype=torch.float32)),
        },
    )


def _make_data(lengths=(5, 9, 12), seed=1234, vocab=64):
    """Like tests/test_distributed_trainer.py::_make_data, plus the cookbook `mask` key.

    Targets are the input shifted by one (as assemble_training_data produces) so the
    full sequence is model_input + last target.
    """
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
        data.append(_datum(seq[:-1], seq[1:], logp, adv, mask))
    return data


def _trainer(path, **kw):
    kw.setdefault("rank", 4)
    kw.setdefault("dtype", torch.float32)
    torch.manual_seed(0)  # identical PEFT lora_A init across trainers
    return Trainer(path, device="cpu", **kw)


def _perturb_lora_b(model, seed=7, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and "lora_B" in n:
                p.copy_(scale * torch.randn(p.shape, generator=g))


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def _lora(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _with_datum_field(data, key, fn):
    out = []
    for d in data:
        inputs = dict(d.loss_fn_inputs)
        inputs[key] = TensorData.from_torch(fn(d, inputs[key].to_torch()).to(torch.float32))
        out.append(tinker.Datum(model_input=d.model_input, loss_fn_inputs=inputs))
    return out


def test_single_rank_dist_and_fsdp_fallback(tiny_model_dir):
    from lean_rl.dist import DistInfo

    # fsdp needs >1 rank: ignored single-process; a 1-rank DistInfo is treated as no dist.
    tr = _trainer(tiny_model_dir, fsdp=True, dist=DistInfo(0, 1, 0, True))
    assert tr.fsdp is False and tr.dist is None
    with pytest.raises(RuntimeError, match="init_dist"):
        _trainer(tiny_model_dir, dist=DistInfo(0, 2, 0, True))


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo"])
def test_token_sum_matches_local_backend(tiny_model_dir, loss_fn):
    dtype = torch.float32
    from local_backend.training_client import LocalTrainingClient

    tr = _trainer(tiny_model_dir, dtype=dtype)
    _perturb_lora_b(tr.model)
    client = LocalTrainingClient(
        base_model=tiny_model_dir, lora_rank=4, vllm_urls=[], device="cpu",
        dtype=dtype, lora_target_modules=list(tr.placement.target_modules),
    )
    asyncio.run(client._init_async())
    ref_params = dict(client.model.named_parameters())
    with torch.no_grad():
        for n, p in _lora(tr.model).items():
            ref_params[n].copy_(p)

    data = _make_data()
    m = tr._forward_backward(data, loss_fn)

    async def run():
        return await (await client.forward_backward_async(data, loss_fn)).result_async()
    fb = asyncio.run(run())

    g_new, g_ref = _grads(tr.model), _grads(client.model)
    assert set(g_new) == set(g_ref) and g_new
    # atol 1e-6 relative to the gradient scale: grads are token sums of magnitude ~10, and
    # the reference computes per-datum mean * n_resp, so fp32 reduction order differs.
    scale = max(1.0, max(float(g.abs().max()) for g in g_ref.values()))
    for n in g_new:
        torch.testing.assert_close(g_new[n], g_ref[n], atol=1e-6 * scale, rtol=0, msg=lambda m, n=n: f"{n}: {m}")
    # local backend reports the per-token mean; ours is the token sum.
    assert m["response_tokens"] == fb.metrics["response_tokens"]
    assert math.isclose(m["loss"], fb.metrics["loss"] * fb.metrics["response_tokens"],
                        rel_tol=1e-5, abs_tol=1e-6)


def test_padded_batch_matches_per_datum(tiny_model_dir):
    data = _make_data(lengths=(3, 40, 7))
    out = []
    for mbt in (10_000, 1):
        tr = _trainer(tiny_model_dir, micro_batch_tokens=mbt)
        _perturb_lora_b(tr.model)
        torch.manual_seed(0)
        m = tr._forward_backward(data, "ppo")
        out.append((m, _grads(tr.model)))
    (m1, g1), (m2, g2) = out
    assert math.isclose(m1["loss"], m2["loss"], abs_tol=1e-6, rel_tol=1e-6)
    assert m1["response_tokens"] == m2["response_tokens"]
    for n in g1:
        torch.testing.assert_close(g1[n], g2[n], atol=1e-6, rtol=1e-5, msg=lambda m, n=n: f"{n}: {m}")


def test_logprobs_base_vs_policy(tiny_model_dir):
    tr = _trainer(tiny_model_dir)
    data = _make_data()
    pol0, base0 = tr.logprobs(data, "policy"), tr.logprobs(data, "base")
    for a, b in zip(pol0, base0):
        assert a.dtype == torch.float32 and a.device.type == "cpu" and not a.requires_grad
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    tr.train_step(data, "importance_sampling", tinker.AdamParams(**ADAM))
    pol1, base1 = tr.logprobs(data, "policy"), tr.logprobs(data, "base")
    assert any(not torch.allclose(a, b) for a, b in zip(pol1, base1))
    tr.train_step(data, "importance_sampling", tinker.AdamParams(**ADAM))
    for a, b in zip(base0, tr.logprobs(data, "base")):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
    for d, lp in zip(data, pol1):
        assert lp.shape == (len(d.model_input.to_ints()),)


def test_init_reference_frozen(tiny_model_dir, tmp_path):
    src = _trainer(tiny_model_dir)
    _perturb_lora_b(src.model)
    p = str(tmp_path / "adapter")
    src.save_adapter(p)

    data = _make_data()
    tr = _trainer(tiny_model_dir)
    tr.load_adapter(p)
    tr.snapshot_init_adapter()
    adam = tinker.AdamParams(**ADAM)
    tr.train_step(data, "ppo", adam)
    tr.train_step(data, "ppo", adam)
    init_lp = tr.logprobs(data, "init")

    fresh = _trainer(tiny_model_dir)
    fresh.load_adapter(p)
    expected = fresh.logprobs(data, "policy")
    for a, b in zip(init_lp, expected):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
    # the policy really moved, and the init swap restored it
    assert any(not torch.allclose(a, b) for a, b in zip(tr.logprobs(data, "policy"), expected))


def test_ref_client_contract(tiny_model_dir):
    from tinker_cookbook.rl.metrics import incorporate_kl_penalty

    tr = _trainer(tiny_model_dir)
    data = _make_data()
    client = tr.as_ref_client("base")
    full = data[0].model_input.append_int(int(data[0].loss_fn_inputs["target_tokens"].data[-1]))
    lps = asyncio.run(client.compute_logprobs_async(full))
    assert len(lps) == len(full.to_ints()) and lps[0] is None
    base = tr.logprobs(data, "base")
    torch.testing.assert_close(torch.tensor(lps[1:]), base[0], atol=1e-5, rtol=0)

    diffs, n = 0.0, 0.0
    for d, b in zip(data, base):
        mask = d.loss_fn_inputs["mask"].to_torch()
        diffs += float(((d.loss_fn_inputs["logprobs"].to_torch() - b) * mask).sum())
        n += float(mask.sum())
    out = asyncio.run(incorporate_kl_penalty(data, client, 0.1, 0.0))
    assert math.isclose(out["kl_policy_base"], diffs / n, rel_tol=1e-4, abs_tol=1e-5)


def test_adam_hparams_from_tinker(tiny_model_dir):
    data = _make_data()
    adam = tinker.AdamParams(learning_rate=3e-3, beta1=0.8, beta2=0.99, eps=1e-6,
                             weight_decay=0.05, grad_clip_norm=1e-3)
    tr = _trainer(tiny_model_dir)
    _perturb_lora_b(tr.model)
    ref = _trainer(tiny_model_dir)
    with torch.no_grad():
        rp = dict(ref.model.named_parameters())
        for n, p in _lora(tr.model).items():
            rp[n].copy_(p)

    metrics = tr.train_step(data, "ppo", adam)
    (g,) = tr.optimizer.param_groups
    assert g["lr"] == adam.learning_rate and g["betas"] == (adam.beta1, adam.beta2)
    assert g["eps"] == adam.eps and g["weight_decay"] == adam.weight_decay

    # manual: grads -> clip to 1e-3 -> torch AdamW
    ref._forward_backward(data, "ppo")
    names = [n for n, p in ref.model.named_parameters() if p.requires_grad and p.grad is not None]
    params = [rp[n] for n in names]
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params]))
    assert math.isclose(metrics["grad_norm"], float(norm), rel_tol=1e-5)
    assert float(norm) > 1e-3  # clipping is active
    for p in params:
        p.grad.mul_(1e-3 / (float(norm) + 1e-6))
    opt = torch.optim.AdamW(params, lr=adam.learning_rate, betas=(adam.beta1, adam.beta2),
                            eps=adam.eps, weight_decay=adam.weight_decay)
    opt.step()
    new = dict(tr.model.named_parameters())
    for n in names:
        torch.testing.assert_close(new[n].detach(), rp[n].detach(), atol=1e-6, rtol=1e-5, msg=lambda m, n=n: f"{n}: {m}")


def test_logprob_gap_metric(tiny_model_dir):
    tr = _trainer(tiny_model_dir)
    data = _make_data()
    pol = tr.logprobs(data, "policy")
    lp_iter = iter(pol)
    matched = _with_datum_field(data, "logprobs", lambda d, _: next(lp_iter))
    adam = tinker.AdamParams(**ADAM)
    m = tr.train_step(matched, "importance_sampling", adam)
    assert m["logprob_gap/sampler_vs_trainer_mean_abs"] < 1e-6

    tr2 = _trainer(tiny_model_dir)
    pol2 = tr2.logprobs(data, "policy")
    lp_iter = iter(pol2)
    shifted = _with_datum_field(data, "logprobs", lambda d, _: next(lp_iter) + 0.5)
    m2 = tr2.train_step(shifted, "importance_sampling", adam)
    assert math.isclose(m2["logprob_gap/sampler_vs_trainer_mean_abs"], 0.5, abs_tol=1e-5)


def test_ckpt_state_roundtrip(tiny_model_dir):
    data = _make_data()
    adam = tinker.AdamParams(**ADAM)
    a = _trainer(tiny_model_dir)
    a.train_step(data, "ppo", adam)
    a.snapshot_init_adapter()
    state = a.state_dict_for_ckpt()
    b = _trainer(tiny_model_dir)
    b.load_ckpt_state(state)
    a.train_step(data, "ppo", adam)
    b.train_step(data, "ppo", adam)
    for (n, x), y in zip(_lora(a.model).items(), _lora(b.model).values()):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=0, msg=lambda m, n=n: f"{n}: {m}")


def test_gpt_oss_expert_lora_gets_grads(tmp_path):
    from transformers import GptOssConfig, GptOssForCausalLM

    torch.manual_seed(0)
    model = GptOssForCausalLM(GptOssConfig(
        hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, vocab_size=64, num_local_experts=4,
        num_experts_per_tok=2, layer_types=["full_attention", "sliding_attention"]))
    d = str(tmp_path / "gpt_oss")
    model.save_pretrained(d)

    tr = _trainer(d)
    # nonzero lora_B so lora_A (whose grad is proportional to B) also receives gradient
    _perturb_lora_b(tr.model)
    tr._forward_backward(_make_data(), "importance_sampling")
    expert = {n: p for n, p in tr.model.named_parameters() if p.requires_grad and "experts" in n}
    assert expert
    for n, p in expert.items():
        assert p.grad is not None and p.grad.abs().sum() > 0, n


def test_save_adapter_has_only_lora_keys(tiny_model_dir, tmp_path):
    from safetensors import safe_open

    tr = _trainer(tiny_model_dir)
    p = tmp_path / "adapter"
    tr.save_adapter(str(p))
    with safe_open(str(p / "adapter_model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert not [k for k in keys if "base_layer" in k]
    assert any("lm_head" in k and "lora_A" in k for k in keys)
    assert any("lm_head" in k and "lora_B" in k for k in keys)


def test_chunked_logprobs_and_checkpointing_exact(tiny_model_dir):
    """Small logprob chunks / no gradient checkpointing give the same loss and grads."""
    data = _make_data(lengths=(3, 40, 7))
    out = []
    for kw in ({}, {"logprob_chunk_tokens": 5}, {"gradient_checkpointing": False}):
        tr = _trainer(tiny_model_dir, **kw)
        _perturb_lora_b(tr.model)
        m = tr._forward_backward(data, "ppo")
        out.append((m, _grads(tr.model), tr.logprobs(data, "policy")))
        assert not tr.model.training
    m0, g0, lp0 = out[0]
    assert any("lm_head" in n for n in g0)
    # Chunking changes fp32 accumulation order in the lm_head matmul/backward; grads are token
    # sums of magnitude ~10, so compare at 1e-6 relative to the gradient scale.
    scale = max(1.0, max(float(x.abs().max()) for x in g0.values()))
    for m, g, lp in out[1:]:
        assert math.isclose(m["loss"], m0["loss"], abs_tol=1e-6, rel_tol=1e-6)
        for n in g0:
            torch.testing.assert_close(g[n], g0[n], atol=1e-6 * scale, rtol=0,
                                       msg=lambda s, n=n: f"{n}: {s}")
        for a, b in zip(lp, lp0):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def test_attn_implementation_default(tiny_model_dir, tmp_path):
    from transformers import GptOssConfig, GptOssForCausalLM

    assert _trainer(tiny_model_dir).model.config._attn_implementation == "sdpa"
    d = str(tmp_path / "gpt_oss")
    GptOssForCausalLM(GptOssConfig(
        hidden_size=32, intermediate_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64, num_local_experts=2,
        num_experts_per_tok=1, layer_types=["full_attention"])).save_pretrained(d)
    assert _trainer(d).model.config._attn_implementation == "eager"
    assert _trainer(tiny_model_dir, attn_implementation="eager").model.config._attn_implementation == "eager"


def test_gpt_oss_loads_mxfp4_dequantized(tmp_path, monkeypatch):
    """A pre-quantized (MXFP4) gpt-oss checkpoint gets an explicit Mxfp4Config(dequantize=True)
    from_pretrained kwarg; a plain gpt-oss checkpoint (real load) does not."""
    import json

    from transformers import GptOssConfig, GptOssForCausalLM, Mxfp4Config

    import lean_rl.trainer as trainer_mod

    d = str(tmp_path / "gpt_oss")
    GptOssForCausalLM(GptOssConfig(
        hidden_size=32, intermediate_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=64, num_local_experts=2,
        num_experts_per_tok=1, layer_types=["full_attention"])).save_pretrained(d)
    calls = []
    real = trainer_mod.AutoModelForCausalLM.from_pretrained

    def spy(*a, **k):
        calls.append(k)
        return real(*a, **k)

    monkeypatch.setattr(trainer_mod.AutoModelForCausalLM, "from_pretrained", spy)
    tr = _trainer(d)
    assert "quantization_config" not in calls[-1]
    assert any("experts" in n for n, _ in tr._named_lora())

    # Mark the checkpoint MXFP4 like openai/gpt-oss-*; stop at from_pretrained (the tiny
    # checkpoint has no *_blocks tensors to dequantize).
    cfg_path = os.path.join(d, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {"quant_method": "mxfp4", "modules_to_not_convert": ["lm_head"]}
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    class _Stop(Exception):
        pass

    def capture(*a, **k):
        calls.append(k)
        raise _Stop

    monkeypatch.setattr(trainer_mod.AutoModelForCausalLM, "from_pretrained", capture)
    with pytest.raises(_Stop):
        _trainer(d)
    q = calls[-1].get("quantization_config")
    assert isinstance(q, Mxfp4Config) and q.dequantize


def test_train_step_should_stop_abandons_step(tiny_model_dir):
    """should_stop -> None after the first micro-batch: no update, no grads, step unchanged."""
    adam = tinker.AdamParams(**ADAM)
    data = _make_data(lengths=(5, 9, 12, 7))
    tr = _trainer(tiny_model_dir, micro_batch_tokens=16)  # several micro-batches
    _perturb_lora_b(tr.model)
    tr.train_step(data, "ppo", adam)
    before = _lora(tr.model)
    opt_before = {k: v.clone() for k, v in tr.optimizer.state[tr._trainable()[0]].items()}
    polls = []

    def stop():
        polls.append(1)
        return len(polls) >= 2

    assert tr.train_step(data, "ppo", adam, should_stop=stop) is None
    assert len(polls) == 2  # stopped between micro-batches, not at the end
    assert tr.step == 1
    assert all(p.grad is None for p in tr._trainable())
    for n, p in _lora(tr.model).items():
        assert torch.equal(p, before[n]), n
    for k, v in tr.optimizer.state[tr._trainable()[0]].items():
        assert torch.equal(v, opt_before[k]), k
    # A step that is never told to stop matches a plain train_step.
    m = tr.train_step(data, "ppo", adam, should_stop=lambda: False)
    assert m is not None and tr.step == 2
