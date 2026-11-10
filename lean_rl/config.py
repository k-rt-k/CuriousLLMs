"""CLI configuration for the lean local RL loop (``python -m lean_rl.train key=value ...``).

Field names and defaults mirror ``math_train.CLIConfig`` where the two overlap, so a Tinker run
config maps onto a lean run one-to-one. Options the lean loop does not support
(``dataset_schedule``, ``remove_constant_reward_groups``, ``num_substeps``, ``backend``,
``max_steps_off_policy``) are deliberately not fields, so chz rejects them.
"""

from __future__ import annotations

from typing import Literal, Tuple

import chz


@chz.chz
class LeanRLConfig:
    # Model
    model_name: str = "meta-llama/Llama-3.2-3B"
    lora_rank: int = 32
    renderer_name: str | None = None
    init_from_adapter: str | None = None  # PEFT adapter dir: warm start, fresh optimizer

    # Environment / data
    env: str = "mixed"  # arithmetic, math, polaris, deepmath, gsm8k, mixed
    seed: int = 0
    math_train_size: int | None = 8000
    deepmath_train_size: int = 8000
    deepmath_test_size: int = 500
    deepmath_seed: int = 42

    # RL hyperparameters (Tinker parity)
    group_size: int = 16
    groups_per_batch: int = 128
    learning_rate: float = 7e-5
    max_tokens: int = 2048
    loss_fn: Literal["importance_sampling", "ppo"] = "ppo"
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0
    kl_reference: Literal["base", "init"] = "base"
    compute_post_kl: bool = False
    entropy_bonus_coef: float = 0.0

    # Reasoning rewards (Gemini judge, rank 0 only)
    use_reasoning_rewards: bool = True
    reasoning_reward_coef: float = 0.5

    # RND curiosity (SemanticRND on rank 0's GPU); names/defaults as math_train.CLIConfig
    use_rnd_curiosity: bool = True
    rnd_buffer_size_multiplier: int = 5
    rnd_update_steps: int = 50
    rnd_minibatch_size: int = 1024
    curiosity_reward_coef: float | Tuple[float, float] = 0.2
    rnd_learning_rate: float = 1e-3
    penalize_incorrect_novelty: bool = True
    correctness_threshold: float = 0.6
    curiosity_warmup_batches: int = 20
    target_layers: Tuple[int, ...] = (128, 8)
    predictor_layers: Tuple[int, ...] = (256, 8)

    # Loop
    eval_every: int = 20
    save_every: int = 10
    keep_checkpoints: int = 2
    num_batches: int | None = None  # stop after this many total batches (smoke runs)

    # Trainer
    fsdp: bool = False
    micro_batch_tokens: int = 16384
    gpt_oss_lm_head: bool = False
    attn_implementation: str | None = None  # None: eager for gpt-oss, sdpa otherwise

    # Sampling servers: comma-separated vLLM base URLs (rank 0 only talks to them)
    vllm_urls: str | None = None

    # Logging
    log_path: str
    wandb_project: str | None = None
    wandb_name: str | None = None
