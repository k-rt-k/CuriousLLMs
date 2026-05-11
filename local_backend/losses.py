"""
PPO and importance-sampling losses on top of per-token logprobs.

These match the contract that tinker uses on the remote side:

- Input: `new_logp_btT` and `sample_logp_btT` are per-token logprobs of the
  TARGET tokens under the current policy and the sampler-time policy.
- `advantages_BT` is per-token advantage (0 on prompt tokens, A on response tokens).
- `mask_BT` is 1 on response tokens, 0 on prompt tokens and padding.

Reduction is per-token mean over response tokens (mask.sum() denominator).
"""

from __future__ import annotations

from typing import Literal

import torch


def _ratio(new_logp: torch.Tensor, sample_logp: torch.Tensor) -> torch.Tensor:
    return torch.exp(new_logp - sample_logp)


def importance_sampling_loss(
    new_logp: torch.Tensor,
    sample_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    r = _ratio(new_logp, sample_logp)
    contrib = r * advantages * mask
    denom = mask.sum().clamp_min(1.0)
    return -(contrib.sum() / denom)


def ppo_loss(
    new_logp: torch.Tensor,
    sample_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    r = _ratio(new_logp, sample_logp)
    clipped = torch.clamp(r, 1.0 - clip_eps, 1.0 + clip_eps)
    unclipped_obj = r * advantages
    clipped_obj = clipped * advantages
    obj = torch.minimum(unclipped_obj, clipped_obj) * mask
    denom = mask.sum().clamp_min(1.0)
    return -(obj.sum() / denom)


def compute_loss(
    loss_fn: Literal["ppo", "importance_sampling"],
    new_logp: torch.Tensor,
    sample_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    ppo_clip_eps: float = 0.2,
) -> torch.Tensor:
    if loss_fn == "ppo":
        return ppo_loss(new_logp, sample_logp, advantages, mask, clip_eps=ppo_clip_eps)
    if loss_fn == "importance_sampling":
        return importance_sampling_loss(new_logp, sample_logp, advantages, mask)
    raise ValueError(f"Unknown loss_fn: {loss_fn}")
