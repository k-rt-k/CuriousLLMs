"""GPU spike: does vLLM serve our LoRA placement (experts, lm_head) with HF-matching logprobs?

Builds a PEFT model with the Task-1 placement, randomises every ``lora_B`` (seed 0) so the
adapter is non-trivial, saves it, loads it into a running vLLM server, and compares
teacher-forced logprobs from vLLM against HF on four fixed prompts.

Two measurements:
- prompt path (diagnostic): vLLM prompt_logprobs vs HF on the fixed prompts. vLLM 0.20 applies
  lm_head LoRA with a per-request sampler mapping, so this path under-applies an lm_head adapter;
  training never reads it (KL/reference logprobs come from the trainer).
- sampled path (decides): vLLM samples SAMPLES_PER_PROMPT x SAMPLE_TOKENS tokens with and without
  the adapter, and its returned logprobs are compared with HF teacher-forced logprobs of the same
  tokens. These are the logprobs training uses (importance-ratio denominator).

PASS iff the adapter has an effect (HF adapted-vs-base mean |d| >= 0.10 on both paths, and vLLM's
sampled logprobs differ from HF base by >= 0.10) AND on sampled tokens mean |vLLM - HF(adapted)|
<= 0.05 or <= (the same gap for the base model) + 0.02, which factors out the MXFP4-vs-bf16 /
kernel numerics of the base. Every outcome, including a server rejecting the adapter, emits a JSON
verdict with a "diagnosis" field.

    python -m lean_rl.tools.spike_lora_serving --model Qwen/Qwen3-8B \
        --vllm-url http://localhost:8000 --out /tmp/spike --with-lm-head

Prints a one-line JSON verdict as the last stdout line; exit code 0 iff PASS.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
from pathlib import Path

MEAN_ABS_TOL = 0.05
MIN_ADAPTER_EFFECT = 0.10
SAMPLED_EXCESS_TOL = 0.02  # allowed vLLM-vs-HF gap beyond the base-model gap on sampled tokens
SAMPLE_TOKENS = 48
SAMPLES_PER_PROMPT = 4
LORA_RANK = 8
ADAPTER_NAME = "spike"
PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?",
    "Let x be a real number such that x^2 - 5x + 6 = 0. Then the sum of all possible values of x is",
    "The derivative of f(x) = x^3 * sin(x) is f'(x) = 3x^2 sin(x) + x^3 cos(x). Evaluating at x = 0 gives",
    "A rectangle has perimeter 36 and area 80. Its side lengths are 8 and 10, so its diagonal has length",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--vllm-url", required=True)
    p.add_argument("--out", required=True, help="directory for the saved adapter")
    p.add_argument("--with-lm-head", action="store_true", help="also put a LoRA adapter on lm_head")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rank", type=int, default=LORA_RANK)
    p.add_argument("--b-std", type=float, default=0.02, help="std of the random lora_B init")
    p.add_argument("--zero-b-regex", default=None,
                   help="keep lora_B at zero for params whose name matches (ablation: e.g. 'experts')")
    p.add_argument("--adapter-name", default=ADAPTER_NAME, help="vLLM adapter name (distinct per run on one server)")
    return p.parse_args(argv)


def _hf_logprobs(model, ids):
    import torch

    x = torch.tensor([ids], device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(input_ids=x).logits.float()[0, :-1]
    lp = torch.log_softmax(logits, dim=-1)
    return lp.gather(-1, x[0, 1:, None])[:, 0].cpu()


def run(args) -> dict:
    import torch
    import tinker
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from lean_rl.lora import log_parity_warnings, lora_placement, make_lora_config
    from lean_rl.trainer import _gpt_oss_quant_kw, _is_mxfp4
    from local_backend.sampling_client import LocalSamplingClient
    from local_backend.weight_sync import load_lora

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    hf_config = AutoConfig.from_pretrained(args.model)
    is_gpt_oss = getattr(hf_config, "model_type", "") == "gpt_oss"
    load_kw = dict(
        torch_dtype=torch.bfloat16,
        # Straight onto --device: a CPU load followed by .to() dequantizes gpt-oss's MXFP4
        # experts on the default GPU, which is the one vLLM occupies.
        device_map={"": args.device},
        # Same as the trainer: eager attention (HF gpt-oss needs it for attention sinks) and an
        # explicit MXFP4 dequantize.
        attn_implementation="eager" if is_gpt_oss else "sdpa",
    )
    if is_gpt_oss and _is_mxfp4(hf_config):
        load_kw.update(_gpt_oss_quant_kw())
    if torch.device(args.device).type == "cuda":
        # transformers' MXFP4 dequantize launches on the current device; loading onto another
        # one hits an illegal memory access (gpt-oss, job 11146348). The trainer sets it too.
        torch.cuda.set_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    model.eval()

    model_type = model.config.model_type
    placement = lora_placement(
        model, train_unembed=args.with_lm_head, gpt_oss_lm_head=args.with_lm_head
    )
    log_parity_warnings(placement, model_type)
    print(
        f"placement: modules={placement.target_modules} params={placement.target_parameters} "
        f"lm_head={placement.includes_lm_head}",
        file=sys.stderr,
    )
    peft_model = get_peft_model(model, make_lora_config(placement, args.rank))

    torch.manual_seed(0)
    zero_re = re.compile(args.zero_b_regex) if args.zero_b_regex else None
    n_b = 0
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                # Draw for every lora_B so the ablations keep the other modules' values.
                val = torch.randn(p.shape, dtype=torch.float32) * args.b_std
                if zero_re is not None and zero_re.search(name):
                    continue
                p.copy_(val.to(p.dtype))
                n_b += 1
    print(f"randomised {n_b} lora_B tensors", file=sys.stderr)
    for name, p in peft_model.named_parameters():  # layer-0 LoRA names, for --zero-b-regex
        if "lora_" in name and ".layers.0." in name:
            print(f"  {name} {tuple(p.shape)}", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Same as Trainer.save_adapter: with lm_head targeted PEFT would otherwise also save the
    # full base lm_head weight (lm_head.base_layer.weight) into the adapter.
    peft_model.save_pretrained(str(out), save_embedding_layers=False)
    verdict = {"model": args.model, "with_lm_head": bool(args.with_lm_head),
               "b_std": args.b_std, "zero_b_regex": args.zero_b_regex}
    try:
        load_lora([args.vllm_url], args.adapter_name, str(out.resolve()))
    except Exception as e:  # server rejected the adapter: still emit a verdict
        verdict.update(pass_=False, error=f"{type(e).__name__}: {e}", diagnosis="adapter rejected by vLLM")
        return _finalize(verdict)

    adapted = LocalSamplingClient(
        vllm_urls=[args.vllm_url], base_model_name=args.model, adapter_name=args.adapter_name
    )
    # No adapter_name -> requests use the base model name (sampling_client._model_field).
    base = LocalSamplingClient(vllm_urls=[args.vllm_url], base_model_name=args.model)

    def as_tensor(lp):
        return torch.tensor([float("nan") if v is None else v for v in lp[1:]])

    async def query_all(prompt_ids):
        # One event loop for every request: the clients keep an httpx.AsyncClient bound to the
        # loop they first ran on, so per-call asyncio.run() fails from the second call on.
        out = []
        for ids in prompt_ids:
            mi = tinker.ModelInput.from_ints(ids)
            out.append((as_tensor(await adapted.compute_logprobs_async(mi)),
                        as_tensor(await base.compute_logprobs_async(mi))))
        return out

    async def sample_all(prompt_ids):
        params = tinker.SamplingParams(max_tokens=SAMPLE_TOKENS, temperature=1.0, seed=0)
        out = []
        for ids in prompt_ids:
            mi = tinker.ModelInput.from_ints(ids)
            out.append((await adapted.sample_async(mi, SAMPLES_PER_PROMPT, params),
                        await base.sample_async(mi, SAMPLES_PER_PROMPT, params)))
        return out

    all_ids = [tok(text, add_special_tokens=True)["input_ids"] for text in PROMPTS]

    async def all_requests(prompt_ids):
        # A single asyncio.run for everything (see query_all).
        return await query_all(prompt_ids), await sample_all(prompt_ids)

    vllm_out, sampled = asyncio.run(all_requests(all_ids))

    cols = {k: [] for k in ("vs_hf", "base_gap", "delta_gap", "vllm_eff", "hf_eff")}
    for ids, (v_ad, v_base) in zip(all_ids, vllm_out):
        h_ad = _hf_logprobs(peft_model, ids)
        with peft_model.disable_adapter():
            h_base = _hf_logprobs(peft_model, ids)
        cols["vs_hf"].append((v_ad - h_ad).abs())
        cols["base_gap"].append((v_base - h_base).abs())
        cols["delta_gap"].append(((v_ad - v_base) - (h_ad - h_base)).abs())
        cols["vllm_eff"].append((v_ad - v_base).abs())
        cols["hf_eff"].append((h_ad - h_base).abs())
    # Sampled path (what training consumes: the importance-ratio denominator). vLLM applies
    # lm_head LoRA with a per-request (sampler) mapping, so prompt_logprobs over every prompt
    # position can miss it; sampled-token logprobs must not.
    for ids, (resp_ad, resp_base) in zip(all_ids, sampled):
        for seq in resp_ad.sequences:
            full = list(ids) + list(seq.tokens)
            h_ad = _hf_logprobs(peft_model, full)[len(ids) - 1:]
            with peft_model.disable_adapter():
                h_base = _hf_logprobs(peft_model, full)[len(ids) - 1:]
            v = torch.tensor(seq.logprobs, dtype=torch.float32)
            cols.setdefault("s_vs_hf", []).append((v - h_ad).abs())
            cols.setdefault("s_vllm_eff", []).append((v - h_base).abs())
            cols.setdefault("s_hf_eff", []).append((h_ad - h_base).abs())
        for seq in resp_base.sequences:
            full = list(ids) + list(seq.tokens)
            with peft_model.disable_adapter():
                h_base = _hf_logprobs(peft_model, full)[len(ids) - 1:]
            v = torch.tensor(seq.logprobs, dtype=torch.float32)
            cols.setdefault("s_base_gap", []).append((v - h_base).abs())
    m = {k: float(torch.cat(v).mean()) for k, v in cols.items()}  # NaN propagates -> fails
    verdict.update(
        mean_abs_vs_hf=m["vs_hf"],
        adapter_effect=m["hf_eff"],
        vllm_effect=m["vllm_eff"],
        base_gap=m["base_gap"],
        lora_delta_gap=m["delta_gap"],
    )
    verdict.update(
        sampled_vs_hf=m["s_vs_hf"],
        sampled_base_gap=m["s_base_gap"],
        sampled_adapter_effect=m["s_hf_eff"],
        sampled_vllm_vs_hf_base=m["s_vllm_eff"],
    )
    # Prompt path: diagnostic only (training never reads vLLM prompt_logprobs).
    close = m["vs_hf"] <= MEAN_ABS_TOL
    delta_ok = m["delta_gap"] <= MEAN_ABS_TOL
    verdict["prompt_path_ok"] = bool(close or delta_ok)
    # Sampled path decides: the adapter visibly moves the logprobs of vLLM's own samples, and
    # vLLM-vs-HF(adapted) is within tolerance or adds at most SAMPLED_EXCESS_TOL on top of the
    # base-model numerics gap measured the same way.
    hf_ok = m["hf_eff"] >= MIN_ADAPTER_EFFECT and m["s_hf_eff"] >= MIN_ADAPTER_EFFECT
    vllm_ok = m["s_vllm_eff"] >= MIN_ADAPTER_EFFECT
    s_close = m["s_vs_hf"] <= MEAN_ABS_TOL
    s_base_only = m["s_vs_hf"] <= m["s_base_gap"] + SAMPLED_EXCESS_TOL
    verdict["pass_"] = bool(hf_ok and vllm_ok and (s_close or s_base_only))
    if not hf_ok:
        diag = "HF adapter has no effect (spike invalid)"
    elif not vllm_ok:
        diag = "adapter not applied in vLLM sampling"
    elif s_close:
        diag = "ok"
    elif s_base_only:
        diag = "base numerics gap only"
    else:
        diag = "lora mismatch on sampled tokens"
    verdict["diagnosis"] = diag
    return _finalize(verdict)


def _finalize(verdict: dict) -> dict:
    """Rename pass_ -> pass and map NaN/inf to null."""
    verdict["pass"] = verdict.pop("pass_")
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in verdict.items()
    }


def main(argv=None) -> int:
    verdict = run(parse_args(argv))
    print(json.dumps(verdict))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
