"""Unit tests for compute_entropy_bonus / apply_entropy_bonus."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tinker-cookbook"))


def _group(logprobs_per_traj, final_rewards):
    """logprobs_per_traj: list (trajectories) of list (transitions) of logprob lists or None."""
    import tinker
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

    ob = tinker.ModelInput.empty()
    trajs = []
    for transitions in logprobs_per_traj:
        ts = [
            Transition(
                ob=ob,
                ac=TokensWithLogprobs(
                    tokens=list(range(len(lp))) if lp is not None else [0], maybe_logprobs=lp
                ),
                reward=0.0,
                episode_done=True,
            )
            for lp in transitions
        ]
        trajs.append(Trajectory(transitions=ts, final_ob=ob))
    return TrajectoryGroup(
        trajectories_G=trajs,
        final_rewards_G=list(final_rewards),
        metrics_G=[{} for _ in trajs],
    )


def test_entropy_bonus_is_mean_negative_logprob():
    from curiosity_train import compute_entropy_bonus

    g = _group([[[-1.0, -3.0]], [[-1.0], [-2.0, -3.0]]], [0.0, 0.0])
    assert compute_entropy_bonus([g]) == [[pytest.approx(2.0), pytest.approx(2.0)]]


def test_entropy_bonus_no_logprobs_is_zero():
    from curiosity_train import compute_entropy_bonus

    g = _group([[None]], [0.0])
    assert compute_entropy_bonus([g]) == [[0.0]]


def test_apply_entropy_bonus_adds_to_final_rewards():
    from curiosity_train import apply_entropy_bonus

    g = _group([[[-1.0, -3.0]]], [1.0])
    metrics = {}
    apply_entropy_bonus([g], 0.1, metrics)
    assert g.final_rewards_G == [pytest.approx(1.2)]
    assert metrics["entropy/bonus_mean"] == pytest.approx(2.0)
    assert metrics["entropy/bonus_std"] == pytest.approx(0.0)
    assert metrics["entropy/bonus_coef"] == 0.1


@pytest.mark.parametrize("coef", [None, 0.0])
def test_apply_entropy_bonus_skipped_when_none_or_zero(coef):
    from curiosity_train import apply_entropy_bonus

    g = _group([[[-1.0, -3.0]]], [1.0])
    metrics = {}
    apply_entropy_bonus([g], coef, metrics)
    assert g.final_rewards_G == [1.0]
    assert not any(k.startswith("entropy/") for k in metrics)
