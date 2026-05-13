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


def nll_loss(
    new_logp: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted negative log-likelihood for supervised fine-tuning.

    `new_logp` is the per-token logprob of the target token under the current
    policy; `weights` is 1.0 on response tokens and 0.0 on prompt/padding
    tokens (the response-mask convention used by
    ``tinker_cookbook.supervised.common.datum_from_tokens_weights``).
    """
    denom = weights.sum().clamp_min(1.0)
    return -(new_logp * weights).sum() / denom


def compute_loss(
    loss_fn: Literal["ppo", "importance_sampling", "nll"],
    new_logp: torch.Tensor,
    sample_logp: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    *,
    ppo_clip_eps: float = 0.2,
) -> torch.Tensor:
    if loss_fn == "ppo":
        return ppo_loss(new_logp, sample_logp, advantages, mask, clip_eps=ppo_clip_eps)
    if loss_fn == "importance_sampling":
        return importance_sampling_loss(new_logp, sample_logp, advantages, mask)
    if loss_fn == "nll":
        return nll_loss(new_logp, weights)
    raise ValueError(f"Unknown loss_fn: {loss_fn}")
