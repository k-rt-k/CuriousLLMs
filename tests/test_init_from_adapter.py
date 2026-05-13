"""Smoke test for warm-start (init_from_adapter) path.

Verifies that `load_adapter_weights_async`:
1. Loads adapter weights bit-for-bit from a previously-saved state dir.
2. Leaves `step_counter == 0` (fresh, not restored from meta.json).
3. Rebuilds the optimizer (state dict is empty -> AdamW has not seen a step).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
import torch

# Repo root + tinker-cookbook submodule on the path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tinker-cookbook"))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="warm-start test needs CUDA (the trainer is GPU-only).",
)
def test_load_adapter_weights_fresh_optimizer(tmp_path):
    """save_state -> snapshot -> fresh client -> load_adapter_weights -> same weights, fresh optim."""
    from local_backend.training_client import LocalTrainingClient

    model_name = os.environ.get("FORMAT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    log_path = str(tmp_path)

    async def go():
        # Seed run: build a client, take a real optimizer step so LoRA weights
        # diverge from random init, then snapshot the *post-update* weights.
        seed = LocalTrainingClient(
            base_model=model_name,
            lora_rank=8,
            vllm_url="http://placeholder",
            log_path=log_path,
            gpu_memory_fraction=0.6,
        )
        await seed._init_async()
        # Real (synthetic) gradients so AdamW step actually mutates LoRA weights.
        for p in seed.model.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p) * 0.01
        seed.optimizer.step()
        seed.step_counter = 42
        # Snapshot AFTER the step — verifies load brings back trained weights,
        # not just random init.
        seed_lora_state = {
            n: p.detach().cpu().clone()
            for n, p in seed.model.named_parameters()
            if p.requires_grad
        }
        fut = await seed.save_state_async("seed")
        seed_path = (await fut.result_async()).path

        # Free GPU memory before constructing the second client.
        del seed
        torch.cuda.empty_cache()

        # Fresh client: should random-init to weights that DIFFER from seed.
        fresh = LocalTrainingClient(
            base_model=model_name,
            lora_rank=8,
            vllm_url="http://placeholder",
            log_path=log_path,
            gpu_memory_fraction=0.6,
        )
        await fresh._init_async()
        fresh_pre = {
            n: p.detach().cpu().clone()
            for n, p in fresh.model.named_parameters()
            if p.requires_grad
        }
        n_differ_pre = sum(
            1 for n in seed_lora_state
            if not torch.allclose(seed_lora_state[n], fresh_pre[n].to(seed_lora_state[n].dtype))
        )
        assert n_differ_pre > 0, (
            "fresh client's LoRA weights should differ from seed pre-warm-start; "
            "if they match, the test cannot distinguish 'load worked' from 'no-op'."
        )

        # Warm-start.
        await (await fresh.load_adapter_weights_async(seed_path)).result_async()

        # Weights must now match seed exactly.
        fresh_post = {
            n: p.detach().cpu()
            for n, p in fresh.model.named_parameters()
            if p.requires_grad
        }
        assert set(seed_lora_state.keys()) == set(fresh_post.keys()), (
            "LoRA parameter names should match"
        )
        for name, seed_t in seed_lora_state.items():
            fresh_t = fresh_post[name]
            assert torch.allclose(seed_t, fresh_t.to(seed_t.dtype), atol=0, rtol=0), (
                f"LoRA weight {name} differs after warm-start"
            )

        # Fresh optimizer: no state entries (AdamW state is keyed per-param after a step).
        assert fresh.step_counter == 0, f"step_counter should be 0, got {fresh.step_counter}"
        assert len(fresh.optimizer.state) == 0, (
            f"AdamW state should be empty on warm-start, got {len(fresh.optimizer.state)} entries"
        )
        assert fresh._last_loss_value is None, "_last_loss_value should reset on warm-start"

    asyncio.run(go())
