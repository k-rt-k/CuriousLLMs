"""Unit tests for NLL loss + the SFT branch of forward_backward_async."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tinker-cookbook"))


def test_nll_loss_response_mask():
    """Weighted NLL = -(new_logp * weights).sum() / weights.sum()."""
    from local_backend.losses import compute_loss, nll_loss

    new_logp = torch.tensor([-0.1, -0.2, -0.5, -1.0])
    weights = torch.tensor([0.0, 0.0, 1.0, 1.0])  # mask: prompt-prompt-response-response
    expected = -((-0.5) + (-1.0)) / 2.0  # 0.75
    out = nll_loss(new_logp, weights)
    assert torch.allclose(out, torch.tensor(expected), atol=1e-6), out

    # compute_loss dispatch agrees.
    out2 = compute_loss("nll", new_logp=new_logp, weights=weights)
    assert torch.allclose(out, out2)


def test_nll_loss_zero_weights_no_nan():
    """All-zero weights: clamp denominator to 1 -> loss is 0, no NaN."""
    from local_backend.losses import nll_loss

    new_logp = torch.tensor([-0.5, -0.5])
    weights = torch.zeros(2)
    out = nll_loss(new_logp, weights)
    assert torch.isfinite(out)
    assert torch.allclose(out, torch.tensor(0.0))


def test_compute_loss_unknown_raises():
    from local_backend.losses import compute_loss

    with pytest.raises(ValueError):
        compute_loss(
            "not_a_real_loss",  # type: ignore[arg-type]
            new_logp=torch.zeros(3),
            weights=torch.ones(3),
        )
