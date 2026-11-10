"""Tinker's RL policy losses, reduced as a token SUM (never a mean).

Tinker's ``importance_sampling`` and ``ppo`` losses sum the per-token objective
over response tokens of the whole batch. ``ratio = exp(logp_new - logp_sampler)``
where ``logp_sampler`` is the sampler (vLLM) logprob carried in the datum.
"""

from __future__ import annotations

from typing import Literal

import torch

PPO_CLIP_LOW = 0.8
PPO_CLIP_HIGH = 1.2


def token_sum_loss(
    loss_fn: Literal["ppo", "importance_sampling"],
    new_logp: torch.Tensor,
    sampler_logp: torch.Tensor,
    adv: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Scalar loss summed over tokens where ``mask`` is nonzero. All inputs ``[B, T]``."""
    ratio = torch.exp(new_logp - sampler_logp)
    if loss_fn == "importance_sampling":
        obj = ratio * adv
    elif loss_fn == "ppo":
        obj = torch.minimum(ratio * adv, torch.clamp(ratio, PPO_CLIP_LOW, PPO_CLIP_HIGH) * adv)
    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn!r}")
    # where() rather than multiply so inf/nan on padded positions cannot leak in.
    return -torch.where(mask > 0, obj * mask, torch.zeros_like(obj)).sum()
