"""lean_rl.shaping: reward shaping extracted from curiosity_train.prepare_minibatch."""

from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tinker-cookbook"))

# Frozen from the PRE-refactor curiosity_train.prepare_minibatch on the inputs below
# (judge and RND disabled, entropy_bonus_coef=0.1, kl_penalty_coef=0). Timing keys ("time/*")
# are excluded from the value comparison; only their presence is asserted.
GOLDEN_REWARDS = [
    [1.2, 0.05, 1.2333333333333334, 0.020000000000000004],
    [0.2, 1.1, 0.30000000000000004, 0.51],
]
GOLDEN_SHAPING_METRICS = {
    "entropy/bonus_coef": 0.1,
    "entropy/bonus_mean": 1.3916666666666666,
    "entropy/bonus_std": 1.0155389920847178,
    "env/all/ac_tokens_per_turn": 1.4444444444444444,
    "env/all/by_group/frac_all_bad": 0.0,
    "env/all/by_group/frac_all_good": 0.0,
    "env/all/by_group/frac_mixed": 1.0,
    "env/all/ob_tokens_per_turn": 2.0,
    "env/all/reward/total": 0.4375,
    "env/all/total_ac_tokens": 13,
    "env/all/total_episodes": 8,
    "env/all/total_ob_tokens": 18,
    "env/all/total_turns": 9,
    "env/all/turns_per_episode": 1.125,
}


class _Tok:
    def decode(self, ids, **kwargs):
        return " ".join(f"t{i}" for i in ids)


class _Builder:
    def logging_tags(self):
        return ["math"]

    async def make_envs(self):
        return [object() for _ in range(4)]


def _group(logprobs_per_traj, final_rewards):
    """logprobs_per_traj: list (trajectories) of list (transitions) of logprob lists."""
    import tinker
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

    ob = tinker.ModelInput.from_ints([5, 6])
    trajs = []
    for transitions in logprobs_per_traj:
        ts = [
            Transition(
                ob=ob,
                ac=TokensWithLogprobs(tokens=[7 + j for j in range(len(lp))], maybe_logprobs=lp),
                reward=0.0,
                episode_done=True,
            )
            for lp in transitions
        ]
        trajs.append(Trajectory(transitions=ts, final_ob=ob))
    return TrajectoryGroup(
        trajectories_G=trajs, final_rewards_G=list(final_rewards), metrics_G=[{} for _ in trajs]
    )


def _groups():
    return [
        _group([[[-1.0, -3.0]], [[-0.5]], [[-2.0], [-4.0, -1.0]], [[-0.2, -0.2]]], [1.0, 0.0, 1.0, 0.0]),
        _group([[[-2.0]], [[-1.0, -1.0]], [[-3.0]], [[-0.1]]], [0.0, 1.0, 0.0, 0.5]),
    ]


def _shape_kwargs(**overrides):
    kw = dict(
        i_batch=0,
        gemini_judge=None,
        reasoning_reward_coef=0.5,
        novelty_model=None,
        rnd_buffer=None,
        intrinsic_reward_coef=0.1,
        rnd_update_steps=50,
        rnd_minibatch_size=1024,
        penalize_incorrect_novelty=True,
        correctness_threshold=0.5,
        curiosity_warmup_batches=0,
        entropy_bonus_coef=0.1,
    )
    kw.update(overrides)
    return kw


def _non_time(metrics):
    return {k: v for k, v in metrics.items() if not k.startswith("time/")}


def test_shape_rewards_matches_prepare_minibatch_pre_refactor():
    from lean_rl.shaping import shape_rewards

    groups = _groups()
    metrics = asyncio.run(
        shape_rewards([_Builder(), _Builder()], groups, _Tok(), **_shape_kwargs())
    )
    assert [g.final_rewards_G for g in groups] == GOLDEN_REWARDS
    assert _non_time(metrics) == GOLDEN_SHAPING_METRICS
    assert "time/compute_entropy_bonus" in metrics


def test_prepare_minibatch_reproduces_golden():
    import curiosity_train as ct

    groups = _groups()
    _data, metrics = asyncio.run(
        ct.prepare_minibatch(
            [_Builder(), _Builder()],
            groups,
            tokenizer=_Tok(),
            service_client=None,
            model_name="unused",
            kl_penalty_coef=0.0,
            kl_discount_factor=0.0,
            entropy_bonus_coef=0.1,
        )
    )
    assert [g.final_rewards_G for g in groups] == GOLDEN_REWARDS
    assert _non_time(metrics) == GOLDEN_SHAPING_METRICS
    assert {"time/compute_entropy_bonus", "time/assemble_training_data"} <= set(metrics)


def test_rnd_path_called_with_pre_bonus_rewards():
    """RND must see extrinsic rewards from BEFORE the entropy bonus."""
    import torch

    from lean_rl.shaping import shape_rewards

    seen = {}

    class StubNovelty:
        concat_before_emb = False
        device = "cpu"
        _optimizer = None

        def encode_problems_responses_separately(self, problems, responses):
            return torch.zeros(len(problems), 2), torch.zeros(len(responses), 2)

        def compute_rewards_stratified_normalized(self, **kw):
            seen["extrinsic"] = kw["extrinsic_rewards"].tolist()
            return torch.zeros(len(kw["extrinsic_rewards"]))

    class StubBuffer:
        def add_batch(self, *a, **k):
            pass

        def get_current_size(self):
            return {"num_responses": 0, "memory_mb": 0.0}

    groups = _groups()
    metrics = asyncio.run(
        shape_rewards(
            [_Builder(), _Builder()],
            groups,
            _Tok(),
            **_shape_kwargs(novelty_model=StubNovelty(), rnd_buffer=StubBuffer()),
        )
    )
    assert seen["extrinsic"] == pytest.approx([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5])
    # the bonus was applied to the stored rewards, but RND did not see it
    assert [g.final_rewards_G for g in groups] == GOLDEN_REWARDS
    assert metrics["rnd/num_correct"] == 4
