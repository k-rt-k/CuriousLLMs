"""
Offline smoke test for the local_backend adapter classes.

Doesn't need a running vLLM. Verifies:
  - sampling_client._normalize_stop handles the various tinker stop shapes.
  - sampling_client._to_sample_response parses an OpenAI-shaped completion
    response into a tinker.SampleResponse.
  - losses.compute_loss produces finite gradients on a tiny synthetic batch.

Run with:
  cd /home/ksnair/worktrees/slurm
  python -m tests.test_local_backend_smoke
"""

from __future__ import annotations

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tinker
import torch

from local_backend.sampling_client import _normalize_stop, _to_sample_response
from local_backend import losses


def test_normalize_stop():
    assert _normalize_stop(None) == (None, None)
    assert _normalize_stop("</s>") == (["</s>"], None)
    assert _normalize_stop(["</s>", "<|endoftext|>"]) == (["</s>", "<|endoftext|>"], None)
    assert _normalize_stop([2, 3, 4]) == (None, [2, 3, 4])
    print("test_normalize_stop: OK")


def test_to_sample_response():
    fake = {
        "choices": [
            {
                "text": "abc",
                "finish_reason": "stop",
                "logprobs": {
                    "token_ids": [11, 22, 33],
                    "token_logprobs": [-0.1, -0.2, -0.3],
                },
            },
            {
                "text": "xyz",
                "finish_reason": "length",
                "logprobs": {
                    "token_ids": [44, 55],
                    "token_logprobs": [-0.4, -0.5],
                },
            },
        ]
    }
    resp = _to_sample_response(fake, prompt_ids=[1, 2, 3])
    assert isinstance(resp, tinker.SampleResponse)
    assert len(resp.sequences) == 2
    assert resp.sequences[0].tokens == [11, 22, 33]
    assert resp.sequences[0].logprobs == [-0.1, -0.2, -0.3]
    assert resp.sequences[0].stop_reason == "stop"
    assert resp.sequences[1].stop_reason == "length"
    print("test_to_sample_response: OK")


def test_losses():
    torch.manual_seed(0)
    T = 8
    new_logp = torch.randn(T, requires_grad=True)
    sample_logp = torch.randn(T)
    advantages = torch.tensor([0, 0, 0, 1.0, 1.0, 1.0, -0.5, -0.5])
    mask = (advantages != 0).float()

    is_loss = losses.importance_sampling_loss(new_logp, sample_logp, advantages, mask)
    ppo = losses.ppo_loss(new_logp, sample_logp, advantages, mask, clip_eps=0.2)
    assert torch.isfinite(is_loss).item()
    assert torch.isfinite(ppo).item()
    # Both should produce non-zero gradients into the masked positions
    is_loss.backward(retain_graph=True)
    grad_pre = new_logp.grad.clone()
    assert (grad_pre[:3] == 0).all(), "Prompt positions should have zero IS gradient"
    assert (grad_pre[3:].abs().sum() > 0), "Response positions should have non-zero gradient"
    print("test_losses: OK")


def test_local_imports():
    import local_backend
    from local_backend import LocalServiceClient, LocalTrainingClient, LocalSamplingClient
    from local_backend.futures import LocalFuture, SavedPath
    from local_backend.weight_sync import sync_lora_to_vllm  # noqa: F401
    from local_backend.vllm_proc import start_vllm_server  # noqa: F401
    print("test_local_imports: OK")


if __name__ == "__main__":
    test_local_imports()
    test_normalize_stop()
    test_to_sample_response()
    test_losses()
    print("\nAll local_backend smoke tests passed.")
