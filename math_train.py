import asyncio
import logging
from datetime import datetime
from typing import Literal

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl import (
    arithmetic_env,
    math_env,
)
from curiosity_train import AsyncConfig, Config, main
from tinker_cookbook.rl.types import RLDatasetBuilder
import math_grade

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-3B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    env: str = "mixed"  # Options: arithmetic, math, polaris, deepmath, gsm8k, mixed
    seed: int = 0  # Random seed for data shuffling

    # Training hyperparameters
    group_size: int = 16
    groups_per_batch: int = 128
    learning_rate: float = 7e-5
    max_tokens: int = 2048
    kl_penalty_coef: float = 0.0

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = "rnd_train"
    wandb_name: str | None = "Llama-3B"
    compute_post_kl: bool = False

    # Evals
    eval_every: int = 20

    # Checkpointing
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None
    loss_fn: Literal["importance_sampling", "ppo"] = "ppo"

    # BEGIN REASONING CODE
    # Reasoning reward configuration
    use_reasoning_rewards: bool = True
    reasoning_reward_coef: float = 0.5
    # END REASONING CODE

    # BEGIN SEMANTIC_RND CODE
    # RND Buffer-based training configuration
    use_rnd_curiosity: bool = True  # Whether to use RND-based curiosity rewards
    rnd_buffer_size_multiplier: int = 3  # S = buffer holds S batches of rollouts
    rnd_update_steps: int = 50  # K = number of RND gradient updates per LLM batch
    rnd_minibatch_size: int = min(1024, group_size * groups_per_batch)  # N = samples per RND update (default = B*G)
    curiosity_reward_coef: float = 0.2  # Coefficient for curiosity rewards
    rnd_learning_rate: float = 1e-3  # Learning rate for RND predictor
    penalize_incorrect_novelty: bool = True  # Whether to apply negative reward for incorrect novel responses
    correctness_threshold: float = 0.6  # Threshold for determining correctness (reward >= threshold)
    curiosity_warmup_batches: int = 20  # Number of batches before curiosity rewards are added (RND still trains during warmup)
    # END SEMANTIC_RND CODE

    # Mixed dataset configuration (only used when env="mixed")
    math_train_size: int | None = 6000  # None = use all ~12000 Math samples
    deepmath_train_size: int = 6000
    deepmath_test_size: int = 500
    deepmath_seed: int = 42


def get_dataset_builder(
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
    # Mixed dataset parameters
    math_train_size: int | None = None,
    deepmath_train_size: int = 8000,
    deepmath_test_size: int = 500,
    deepmath_seed: int = 42,
) -> RLDatasetBuilder:
    if env == "arithmetic":
        return arithmetic_env.ArithmeticDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            n_batches=100,
            include_fewshot=True,
            group_size=group_size,
        )
    elif env == "math":
        return math_grade.get_math_dataset_builder(
            dataset_name="math",
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    elif env == "mixed":
        return math_grade.get_math_dataset_builder(
            dataset_name="mixed",
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
            math_train_size=math_train_size,
            deepmath_train_size=deepmath_train_size,
            deepmath_test_size=deepmath_test_size,
            deepmath_seed=deepmath_seed,
        )
    elif env in ["polaris", "deepmath", "gsm8k"]:
        return math_env.get_math_dataset_builder(
            dataset_name=env,
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown environment: {env}")


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get tokenizer for stop sequences
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_name = cli_config.model_name.replace("/", "-")
    run_name = f"{cli_config.env}-{model_name}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.group_size}group-{cli_config.groups_per_batch}batch-{cli_config.loss_fn}-seed{cli_config.seed}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    # create log path if it doesn't exist
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/math_rl/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name
    # Create full config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
            # Mixed dataset parameters
            math_train_size=cli_config.math_train_size,
            deepmath_train_size=cli_config.deepmath_train_size,
            deepmath_test_size=cli_config.deepmath_test_size,
            deepmath_seed=cli_config.deepmath_seed,
        ),
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        loss_fn=cli_config.loss_fn,
        # BEGIN REASONING CODE
        use_reasoning_rewards=cli_config.use_reasoning_rewards,
        reasoning_reward_coef=cli_config.reasoning_reward_coef,
        # END REASONING CODE
        # BEGIN SEMANTIC_RND CODE
        use_rnd_curiosity=cli_config.use_rnd_curiosity,
        rnd_buffer_size_multiplier=cli_config.rnd_buffer_size_multiplier,
        rnd_update_steps=cli_config.rnd_update_steps,
        rnd_minibatch_size=cli_config.rnd_minibatch_size,
        curiosity_reward_coef=cli_config.curiosity_reward_coef,
        rnd_learning_rate=cli_config.rnd_learning_rate,
        group_size=cli_config.group_size,
        penalize_incorrect_novelty=cli_config.penalize_incorrect_novelty,
        correctness_threshold=cli_config.correctness_threshold,
        curiosity_warmup_batches=cli_config.curiosity_warmup_batches,
        # END SEMANTIC_RND CODE
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
