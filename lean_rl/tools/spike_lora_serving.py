"""GPU spike: does vLLM serve our LoRA placement (experts, lm_head) with HF-matching logprobs?

Builds a PEFT model with the Task-1 placement, randomises every ``lora_B`` (seed 0) so the
adapter is non-trivial, saves it, loads it into a running vLLM server, and compares
teacher-forced logprobs from vLLM against HF on four fixed prompts.

PASS iff the adapter has an effect on both sides (HF and vLLM adapted-vs-base mean |d| >= 0.10)
AND (mean |vLLM - HF(adapted)| <= 0.05 OR the LoRA delta agrees: mean |(vllm_ad - vllm_base) -
(hf_ad - hf_base)| <= 0.05, which factors out the MXFP4-vs-bf16 base gap). Every outcome,
including a server rejecting the adapter, emits a JSON verdict with a "diagnosis" field.

    python -m lean_rl.tools.spike_lora_serving --model Qwen/Qwen3-8B \
        --vllm-url http://localhost:8000 --out /tmp/spike --with-lm-head

Prints a one-line JSON verdict as the last stdout line; exit code 0 iff PASS.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

MEAN_ABS_TOL = 0.05
MIN_ADAPTER_EFFECT = 0.10
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lean_rl.lora import log_parity_warnings, lora_placement, make_lora_config
    from local_backend.sampling_client import LocalSamplingClient
    from local_backend.weight_sync import load_lora

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
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
    n_b = 0
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_B" in name:
                p.copy_((torch.randn(p.shape, dtype=torch.float32) * 0.02).to(p.dtype))
                n_b += 1
    print(f"randomised {n_b} lora_B tensors", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out))
    verdict = {"model": args.model, "with_lm_head": bool(args.with_lm_head)}
    try:
        load_lora([args.vllm_url], ADAPTER_NAME, str(out.resolve()))
    except Exception as e:  # server rejected the adapter: still emit a verdict
        verdict.update(pass_=False, error=f"{type(e).__name__}: {e}", diagnosis="adapter rejected by vLLM")
        return _finalize(verdict)

    adapted = LocalSamplingClient(
        vllm_urls=[args.vllm_url], base_model_name=args.model, adapter_name=ADAPTER_NAME
    )
    # No adapter_name -> requests use the base model name (sampling_client._model_field).
    base = LocalSamplingClient(vllm_urls=[args.vllm_url], base_model_name=args.model)

    def vllm_lp(client, ids):
        lp = client.compute_logprobs(tinker.ModelInput.from_ints(ids))[1:]
        return torch.tensor([float("nan") if v is None else v for v in lp])

    cols = {k: [] for k in ("vs_hf", "base_gap", "delta_gap", "vllm_eff", "hf_eff")}
    for text in PROMPTS:
        ids = tok(text, add_special_tokens=True)["input_ids"]
        v_ad, v_base = vllm_lp(adapted, ids), vllm_lp(base, ids)
        h_ad = _hf_logprobs(peft_model, ids)
        with peft_model.disable_adapter():
            h_base = _hf_logprobs(peft_model, ids)
        cols["vs_hf"].append((v_ad - h_ad).abs())
        cols["base_gap"].append((v_base - h_base).abs())
        cols["delta_gap"].append(((v_ad - v_base) - (h_ad - h_base)).abs())
        cols["vllm_eff"].append((v_ad - v_base).abs())
        cols["hf_eff"].append((h_ad - h_base).abs())
    m = {k: float(torch.cat(v).mean()) for k, v in cols.items()}  # NaN propagates -> fails
    verdict.update(
        mean_abs_vs_hf=m["vs_hf"],
        adapter_effect=m["hf_eff"],
        vllm_effect=m["vllm_eff"],
        base_gap=m["base_gap"],
        lora_delta_gap=m["delta_gap"],
    )
    close = m["vs_hf"] <= MEAN_ABS_TOL
    delta_ok = m["delta_gap"] <= MEAN_ABS_TOL
    hf_ok = m["hf_eff"] >= MIN_ADAPTER_EFFECT
    vllm_ok = m["vllm_eff"] >= MIN_ADAPTER_EFFECT
    verdict["pass_"] = bool(hf_ok and vllm_ok and (close or delta_ok))
    if not hf_ok:
        diag = "HF adapter has no effect (spike invalid)"
    elif not vllm_ok:
        diag = "adapter not applied in vLLM"
    elif close:
        diag = "ok"
    elif delta_ok:
        diag = "base quantization gap only"
    else:
        diag = "lora mismatch"
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
