"""
Implements RL on general MDPs
"""

import asyncio
import io
import logging
import os
import time
from typing import Any, Callable, List, Literal, Sequence, Iterator, Tuple

import chz
import numpy as np
import tinker
import torch
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import (
    compute_kl_sample_train,
    compute_post_kl,
    compute_sampling_client_metrics,
    incorporate_kl_penalty,
)
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import logtree, ml_log
from tinker_cookbook.utils.misc_utils import safezip, split_list, timed
from tinker_cookbook.utils.trace import scope, trace_init, get_scope_context
from contextlib import contextmanager
from encoder import Encoder
# BEGIN SEMANTIC_RND CODE
from rnd import SemanticRND
# END SEMANTIC_RND CODE
from llm_as_a_judge import GeminiJudge, JudgeResult

logger = logging.getLogger(__name__)


# BEGIN ADDED CODE
# BEGIN SEMANTIC_RND CODE
@scope
async def extract_responses_from_trajectories(
    trajectory_groups_P: list[TrajectoryGroup],
    env_group_builders_P: Sequence[EnvGroupBuilder],
    tokenizer: Tokenizer,
) -> List[List[Tuple[str, str]]]:
    """
    Helper function to extract (problem, response) pairs from trajectory groups.
    
    Args:
        trajectory_groups_P: List of trajectory groups
        env_group_builders_P: List of environment group builders
        tokenizer: Tokenizer for decoding tokens
        
    Returns:
        List[List[Tuple[str, str]]] - Nested list structure for SemanticRND compatibility
    """
# END SEMANTIC_RND CODE
    from tinker_cookbook.rl.problem_env import ProblemEnv
    
    problem_response_pairs = []
    
    for env_group_builder, traj_group in safezip(env_group_builders_P, trajectory_groups_P):
        # Create environments to access get_question()
        envs = await env_group_builder.make_envs()
        
        for env, trajectory in safezip(envs, traj_group.trajectories_G):
            # Cast to ProblemEnv to access get_question()
            if isinstance(env, ProblemEnv):
                # Extract the problem/question
                problem = env.get_question()
                
                # Extract the response from the trajectory
                # The response is the final action taken (the model's generated text)
                if len(trajectory.transitions) > 0:
                    # Get the last transition's action tokens
                    last_action_tokens = trajectory.transitions[-1].ac.tokens
                    # Decode to get response string
                    response = tokenizer.decode(last_action_tokens)
                else:
                    # Empty trajectory - use empty response
                    response = ""
                
                problem_response_pairs.append((problem, response))
    
    # logger.warning(f"Extracted {len(problem_response_pairs)} problem-response pairs.")
    # logger.warning(f"Sample pair: {problem_response_pairs[0] if problem_response_pairs else 'N/A'}")
    # BEGIN SEMANTIC_RND CODE
    # Wrap in a list to match SemanticRND's expected format: List[List[Tuple[str, str]]]
    return [problem_response_pairs] # length = 1 batch with (group_size * batch_size) pairs
    # END SEMANTIC_RND CODE
# END ADDED CODE


# BEGIN SEMANTIC_RND CODE
@scope
async def extract_questions_references_answers_from_trajectories(
    trajectory_groups_P: list[TrajectoryGroup],
    env_group_builders_P: Sequence[EnvGroupBuilder],
    tokenizer: Tokenizer,
) -> tuple[list[str], list[str], list[str]]:
    """
    Helper function to extract (question, reference, answer) tuples from trajectory groups.
    
    Args:
        trajectory_groups_P: List of trajectory groups
        env_group_builders_P: List of environment group builders
        tokenizer: Tokenizer for decoding tokens
        
    Returns:
        Tuple of (questions, references, answers) lists, one entry per trajectory
    """
    from tinker_cookbook.rl.problem_env import ProblemEnv
    
    questions = []
    references = []
    answers = []
    
    for env_group_builder, traj_group in safezip(env_group_builders_P, trajectory_groups_P):
        # Create environments to access get_question() and get_reference_answer()
        envs = await env_group_builder.make_envs()
        
        for env, trajectory in safezip(envs, traj_group.trajectories_G):
            # Cast to ProblemEnv to access methods
            if isinstance(env, ProblemEnv):
                # Extract the problem/question
                question = env.get_question()
                
                # Extract the reference answer #FIXME: USE SOLUTIONS IN PLACE OF ANSWER
                reference = env.get_reference_answer()
                
                # Extract the response from the trajectory
                # The response is the final action taken (the model's generated text)
                if len(trajectory.transitions) > 0:
                    # Get the last transition's action tokens
                    last_action_tokens = trajectory.transitions[-1].ac.tokens
                    # Decode to get response string
                    answer = tokenizer.decode(last_action_tokens)
                else:
                    # Empty trajectory - use empty response
                    answer = ""
                
                questions.append(question)
                references.append(reference)
                answers.append(answer)
    
    return questions, references, answers
# END SEMANTIC_RND CODE


def _get_evaluator_name(evaluator: SamplingClientEvaluator) -> str:
    return (
        evaluator.name
        if isinstance(evaluator, RLTestSetEvaluator) and evaluator.name is not None
        else ""
    )


@contextmanager
def _get_logtree_scope(
    log_path: str | None, num_groups_to_log: int, f_name: str, scope_name: str
) -> Iterator[None]:
    """
    Creates a context manager; all log inside this context will be logged under the section `scope_name`.
    It will create a file with the path of log_path/f_name.html
    If num_groups_to_log is 0, it will disable logging (but note that this function does not actually implement the logic for logging itself!)
    """
    if log_path is not None and num_groups_to_log > 0:
        logtree_path = os.path.join(log_path, f"{f_name}.html")
        with logtree.init_trace(scope_name, path=logtree_path):
            yield
    else:
        yield


@scope
def _select_representative_inds(scores: list[float], num_inds: int) -> list[int]:
    assert num_inds <= len(scores)
    sorted_inds = np.argsort(scores)
    uniform_inds = np.linspace(0, len(sorted_inds) - 1, num_inds).astype(int)
    return [int(sorted_inds[i]) for i in uniform_inds]


@scope
def print_group(traj_group: TrajectoryGroup, tokenizer: Tokenizer):
    """
    Print a subset of the trajectory group to the console.
    """
    # Cut down the number of trajectories to print
    max_trajs_to_print = 4
    if len(traj_group.trajectories_G) > max_trajs_to_print:
        inds = _select_representative_inds(traj_group.get_total_rewards(), max_trajs_to_print)
        traj_group = TrajectoryGroup(
            trajectories_G=[traj_group.trajectories_G[i] for i in inds],
            final_rewards_G=[traj_group.final_rewards_G[i] for i in inds],
            metrics_G=[traj_group.metrics_G[i] for i in inds],
        )

    rewards = traj_group.get_total_rewards()
    advantages_G = compute_advantages([traj_group])
    data_D, metadata_D = assemble_training_data([traj_group], advantages_G)

    buf = io.StringIO()

    @scope
    def bprint(s: str):
        print(s, file=buf)

    bprint("\n====== Trajectory Group ======")
    last_metadata = None
    for datum, metadata in safezip(data_D, metadata_D):
        idx = metadata["traj_idx"]
        if metadata != last_metadata:
            bprint(f"****** trajectory idx={idx}, reward={rewards[idx]:.3g} ******")
            # Print trajectory-level metrics
            if traj_group.metrics_G[idx]:
                bprint("Trajectory metrics:")
                for key, value in traj_group.metrics_G[idx].items():
                    bprint(f"  {key}: {value}")
            # Print per-transition metrics
            transition_metrics = [
                transition.metrics
                for transition in traj_group.trajectories_G[idx].transitions
                if transition.metrics
            ]
            if transition_metrics:
                bprint("Per-step metrics:")
                for i, metrics in enumerate(transition_metrics):
                    bprint(f"  Step {i}:")
                    for key, value in metrics.items():
                        bprint(f"    {key}: {value}")
        bprint("---- datum ----")
        bprint(colorize_example(datum, tokenizer, key="advantages"))
        last_metadata = metadata
    bprint("====== End Trajectory Group ======")
    logger.info(buf.getvalue().rstrip())


@scope
async def optim_step(
    training_client: tinker.TrainingClient,
    learning_rate: float,
) -> None:
    """Apply the accumulated gradients to update the model weights"""
    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    optim_step_future = await training_client.optim_step_async(adam_params)
    await optim_step_future.result_async()


@scope
def remove_mask(datum: tinker.Datum) -> tinker.Datum:
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


@scope
async def forward_backward(
    training_client: tinker.TrainingClient,
    batch_d: List[tinker.Datum],
    loss_fn: Literal["importance_sampling", "ppo"],
) -> List[torch.Tensor]:
    """Accumulate gradients on a minibatch of data"""
    fwd_bwd_future = await training_client.forward_backward_async(
        list(map(remove_mask, batch_d)), loss_fn=loss_fn
    )
    fwd_bwd_result = await fwd_bwd_future.result_async()

    # Extract training logprobs from loss_fn_outputs
    training_logprobs_D: list[torch.Tensor] = []
    for output in fwd_bwd_result.loss_fn_outputs:
        training_logprobs = output["logprobs"].to_torch()
        training_logprobs_D.append(training_logprobs)

    # We dont display fwd_bwd_result.metrics to avoid spam
    return training_logprobs_D


@scope
async def train_step(
    data_D: List[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: Literal["importance_sampling", "ppo"],
) -> List[torch.Tensor]:
    """Train the model on collected trajectories."""
    batches_md = split_list(data_D, min(num_substeps, len(data_D)))
    training_logprobs_D: list[torch.Tensor] = []
    for batch_d in batches_md:
        training_logprobs = await forward_backward(training_client, batch_d, loss_fn)
        training_logprobs_D.extend(training_logprobs)
        await optim_step(training_client, learning_rate)
    return training_logprobs_D


@chz.chz
class StreamMinibatchConfig:
    """
    Configuration for training with minibatch streaming.
    Once we have accumulated enough trajectories for a minibatch, we will
    immediately train on them, instead of waiting for the full batch of
    trajectories to be ready.
    """

    # Total number of trajectory groups across all minibatches and substeps
    groups_per_batch: int
    # For each substep, we will divide up the number of trajectory groups
    # into this many minibatches.
    # We will do num_minibatches forward_backward() passes and one optim_step()
    # per substep.
    num_minibatches: int


@chz.chz
class AsyncConfig:
    """Configuration for async RL training"""

    # If samples are generated from a sample more than this many steps ago,
    # we will skip training on them.
    max_steps_off_policy: int
    # We will ensure all batches have at least this many groups, even
    # as we discard stale samples
    groups_per_batch: int


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: RLDatasetBuilder  # also determines batch size
    model_name: str
    max_tokens: int
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0

    # Loss function to use for training: "importance_sampling" or "ppo"
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None
    enable_trace: bool = False

    remove_constant_reward_groups: bool = False
    eval_every: int = 20
    save_every: int = 20
    load_checkpoint_path: str | None = None

    async_config: AsyncConfig | None = None
    stream_minibatch_config: StreamMinibatchConfig | None = None

    # Logtree configuration
    num_groups_to_log: int = 4  # Number of groups to log per iteration (0 = disable logging)

    # BEGIN SEMANTIC_RND CODE
    # Reasoning reward configuration
    use_reasoning_rewards: bool = False  # Whether to use LLM-as-judge for reasoning rewards
    reasoning_reward_coef: float = 0.5  # Weight for reasoning rewards vs correctness (0.5 = equal weight)
    # END SEMANTIC_RND CODE


@scope
async def run_single_evaluation(evaluator, cfg, i_batch, sampling_client):
    ev_name = _get_evaluator_name(evaluator)
    with _get_logtree_scope(
        log_path=cfg.log_path,
        num_groups_to_log=cfg.num_groups_to_log,
        f_name=f"eval_{ev_name}_iteration_{i_batch:06d}",
        scope_name=f"Running evaluation {ev_name} {i_batch}",
    ):
        eval_metrics = await evaluator(sampling_client)
        return {f"test/{k}": v for k, v in eval_metrics.items()}


@scope
async def run_evaluations_parallel(
    evaluators: list[SamplingClientEvaluator],
    sampling_client: tinker.SamplingClient,
    cfg: Config,
    i_batch: int,
) -> dict[str, Any]:
    """Run all evaluators in parallel and return aggregated metrics."""

    # Create tasks for all evaluators with names for better traceability
    tasks = []
    for i, evaluator in enumerate(evaluators):
        ev_name = _get_evaluator_name(evaluator)
        task = asyncio.create_task(
            run_single_evaluation(evaluator, cfg, i_batch, sampling_client),
            name=f"eval_{ev_name or i}_iteration_{i_batch:06d}",
        )
        tasks.append(task)

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Merge all metrics
    metrics = {}
    for result in results:
        metrics.update(result)

    return metrics


@scope
async def do_sync_training_with_stream_minibatch(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """
    Implements fully synchronous on-policy training with minibatch streaming.
    Once we have accumulated enough trajectories for a minibatch, we will
    immediately train on them, instead of waiting for the full batch of
    trajectories to be ready. This allows us to overlap sampling and training.
    """
    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every, start_batch
    )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if (cfg.eval_every > 0 and i_batch % cfg.eval_every == 0) or i_batch == end_batch - 1:
            with timed("run_evals", metrics):
                eval_metrics = await run_evaluations_parallel(
                    evaluators, sampling_client, cfg, i_batch
                )
                metrics.update(eval_metrics)

        with _get_logtree_scope(
            cfg.log_path,
            cfg.num_groups_to_log,
            f"train_iteration_{i_batch:06d}",
            f"RL Iteration {i_batch}",
        ):
            # Samplers will produce trajectory groups asynchronously,
            # and the trainer will consume them as soon as they are ready
            trajectory_groups_queue = asyncio.Queue[WrappedTrajectoryGroup | None]()
            env_group_builders_P = dataset.get_batch(i_batch)

            @scope
            async def trajectory_group_worker_task(
                builder: EnvGroupBuilder, enable_logging: bool
            ) -> None:
                metrics = {}
                t_start = time.time()
                trajectory_group = await do_group_rollout_and_filter_constant_reward(
                    sampling_client,
                    builder,
                    max_tokens=cfg.max_tokens,
                    do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                    enable_logging=enable_logging,
                )
                metrics["time/trajectory_group_worker_loop/total"] = time.time() - t_start
                if trajectory_group is not None:
                    trajectory_groups_queue.put_nowait(
                        WrappedTrajectoryGroup(
                            trajectory_group=trajectory_group,
                            env_group_builder=builder,
                            sampling_client_step=i_batch,
                            metrics=metrics,
                        )
                    )
                else:
                    trajectory_groups_queue.put_nowait(None)

            # Sample all trajectories asynchronously. If we have multiple minibatches,
            # then sampling can overlap with training.
            for i, builder in enumerate(env_group_builders_P):
                asyncio.create_task(
                    trajectory_group_worker_task(builder, enable_logging=i < cfg.num_groups_to_log),
                    name=f"trajectory_group_worker_task_{i}",
                )

            # Run multiple optimizer substeps per training iteration
            (
                sampling_client,
                full_batch_metrics,
            ) = await do_train_step_streaming_and_get_sampling_client(
                cfg,
                i_batch,
                trajectory_groups_queue,
                training_client,
                service_client,
                tokenizer,
            )

        # Log metrics
        metrics.update(full_batch_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


@chz.chz
class WrappedTrajectoryGroup:
    """
    A wrapper around a trajectory group that includes metadata about how it was generated.
    Used when we need to overlap sampling and training.
    """

    trajectory_group: TrajectoryGroup
    # The env group builder that produced the trajectory group.
    # Pass this along in case the sampler is too stale, and we need to
    # requeue this group.
    env_group_builder: EnvGroupBuilder
    # The step that produced this trajectory group.
    sampling_client_step: int
    metrics: dict[str, Any] = chz.field(default_factory=dict)


# BEGIN ADDED CODE
# BEGIN SEMANTIC_RND CODE
@scope
async def do_async_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    # BEGIN SEMANTIC_RND CODE
    novelty_model: SemanticRND | None = None,
    # END SEMANTIC_RND CODE
    novelty_model_optimizer: torch.optim.Optimizer | None = None,
    intrinsic_reward_coef: float = 0.1,
    gemini_judge: GeminiJudge | None = None,
    reasoning_reward_coef: float = 0.5,
):
    """
    Implements async off-policy training, capped at K steps off policy.
    
    Args:
        start_batch: Starting batch index
        end_batch: Ending batch index (exclusive)
        num_batches: Total number of batches
        cfg: Training configuration
        training_client: Tinker training client
        service_client: Tinker service client
        evaluators: List of evaluators
        dataset: RL dataset
        ml_logger: ML logger for metrics
        tokenizer: Tokenizer
        novelty_model: Optional SemanticRND module for intrinsic rewards
        novelty_model_optimizer: Optimizer for the novelty model
        intrinsic_reward_coef: Coefficient for intrinsic rewards
        gemini_judge: Optional GeminiJudge for reasoning-level rewards
        reasoning_reward_coef: Coefficient for weighting reasoning rewards vs correctness rewards
    """
    assert cfg.async_config is not None
# END SEMANTIC_RND CODE
# END ADDED CODE

    shutdown_event = asyncio.Event()
    # We will have groups_per_batch worker generating rollouts, so cap the
    # queue size to be groups_per_batch.
    env_group_builders_queue = asyncio.Queue[EnvGroupBuilder | None](
        maxsize=cfg.async_config.groups_per_batch
    )
    trajectory_groups_queue = asyncio.Queue[WrappedTrajectoryGroup | None]()

    # Initial sampling client to use
    path_dict = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name=f"{start_batch:06d}",
        log_path=cfg.log_path,
        loop_state={"batch": start_batch},
        kind="both",
    )

    # This will be updated by the training loop
    sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
    sampling_client_step = start_batch
    sampling_client_updated_event = asyncio.Event()
    sampling_client_updated_event.set()

    @scope
    def shutdown_loops():
        """Trigger all loops to shutdown"""
        shutdown_event.set()
        assert cfg.async_config is not None
        for _ in range(cfg.async_config.groups_per_batch):
            env_group_builders_queue.put_nowait(None)
        sampling_client_updated_event.set()

    @scope
    async def dataloader_loop():
        """Gets the next set of env builders to run"""
        i_batch = start_batch
        while not shutdown_event.is_set() and i_batch < end_batch:
            env_group_builders_P = dataset.get_batch(i_batch)
            for env_group_builder in env_group_builders_P:
                await env_group_builders_queue.put(env_group_builder)
            i_batch += 1

    @scope
    async def trajectory_group_worker_loop():
        """Generates trajectories for a single env builder"""
        while not shutdown_event.is_set():
            env_group_builder = await env_group_builders_queue.get()
            if env_group_builder is None:
                break

            metrics = {}
            t_start = time.time()
            # Save a reference to the sampling client step in case it changes
            # while we're running the rollout
            sampling_client_step_copy = sampling_client_step
            trajectory_group = await do_group_rollout_and_filter_constant_reward(
                sampling_client,
                env_group_builder,
                max_tokens=cfg.max_tokens,
                do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
            )
            if trajectory_group is None:
                trajectory_groups_queue.put_nowait(None)
            else:
                metrics["time/trajectory_group_worker_loop/total"] = time.time() - t_start
                trajectory_groups_queue.put_nowait(
                    WrappedTrajectoryGroup(
                        trajectory_group=trajectory_group,
                        env_group_builder=env_group_builder,
                        sampling_client_step=sampling_client_step_copy,
                        metrics=metrics,
                    )
                )

    @scope
    async def training_loop():
        """
        Waits for a sufficient number of valid trajectories to be accumulated and trains on them.
        Will discard trajectories that are too stale.
        """
        assert cfg.async_config is not None

        i_batch = start_batch
        wrapped_trajectory_groups = []
        while i_batch < end_batch:
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if wrapped_trajectory_group is None:
                continue

            @scope
            def filter_stale_trajectory_group(
                wrapped_trajectory_group: WrappedTrajectoryGroup | None,
            ) -> bool:
                """Returns False if the trajectory group is too stale or not valid"""
                if wrapped_trajectory_group is None:
                    return False

                # If the samples are too stale, requeue the data so that it will be used eventually.
                # Requeue on a separate coroutine to avoid blocking the training loop
                assert cfg.async_config is not None
                if (
                    i_batch - wrapped_trajectory_group.sampling_client_step
                    > cfg.async_config.max_steps_off_policy
                ):
                    logger.info(f"[training_loop] Step {i_batch}: Samples are too stale, skipping")
                    asyncio.create_task(
                        env_group_builders_queue.put(wrapped_trajectory_group.env_group_builder),
                        name="requeue_stale_sample_task",
                    )
                    return False
                return True

            metrics = {
                "training_client/step": i_batch,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
            }
            t_start = time.time()

            nonlocal sampling_client
            nonlocal sampling_client_step
            if cfg.stream_minibatch_config is not None:
                (
                    sampling_client,
                    train_step_metrics,
                ) = await do_train_step_streaming_and_get_sampling_client(
                    cfg,
                    i_batch,
                    trajectory_groups_queue,
                    training_client,
                    service_client,
                    tokenizer,
                    filter_stale_trajectory_group,
                )
            else:
                if not filter_stale_trajectory_group(wrapped_trajectory_group):
                    continue

                # Dynamic sampling: Wait for enough trajectories to accumulate to
                # ensure all batch sizes are the same size. This avoids needing to adjust
                # the learning rate for different batch sizes.
                wrapped_trajectory_groups.append(wrapped_trajectory_group)
                if len(wrapped_trajectory_groups) < cfg.async_config.groups_per_batch:
                    continue
                logger.info(
                    f"[training_loop] Step {i_batch}: Will train on batch, num groups: {len(wrapped_trajectory_groups)}"
                )

                # Compute sampling client metrics, as samples may have been generated with
                # different sampler versions
                metrics.update(compute_sampling_client_metrics(wrapped_trajectory_groups))

                # TODO: For proper checkpointing, we also need to save dataloader state and
                # all queued trajectory groups that haven't been trained on yet
                # BEGIN ADDED CODE
                # BEGIN SEMANTIC_RND CODE
                sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
                    cfg,
                    i_batch,
                    training_client,
                    service_client,
                    tokenizer,
                    [g.env_group_builder for g in wrapped_trajectory_groups],
                    [g.trajectory_group for g in wrapped_trajectory_groups],
                    novelty_model=novelty_model,
                    novelty_model_optimizer=novelty_model_optimizer,
                    intrinsic_reward_coef=intrinsic_reward_coef,
                    gemini_judge=gemini_judge,
                    reasoning_reward_coef=reasoning_reward_coef,
                )
                # END SEMANTIC_RND CODE
                # END ADDED CODE
            sampling_client_step = i_batch + 1
            sampling_client_updated_event.set()

            # Log metrics
            metrics.update(train_step_metrics)
            metrics["time/training_loop/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=i_batch)
            i_batch += 1
            wrapped_trajectory_groups = []

        shutdown_loops()

    @scope
    async def evaluation_loop():
        """Runs evals periodically"""
        if len(evaluators) == 0 or cfg.eval_every == 0:
            return

        while not shutdown_event.is_set():
            await sampling_client_updated_event.wait()
            sampling_client_updated_event.clear()

            metrics = {}
            t_start = time.time()
            # Save a reference to the original values in case it changes
            # while we're running the evals
            sampling_client_eval_step = sampling_client_step
            sampling_client_eval = sampling_client
            if cfg.eval_every > 0 and sampling_client_eval_step % cfg.eval_every == 0:
                with timed("run_evals", metrics):
                    for evaluator in evaluators:
                        eval_metrics = await evaluator(sampling_client_eval)
                        metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})
                metrics["time/evaluation_loop/total"] = time.time() - t_start
                ml_logger.log_metrics(metrics, step=sampling_client_eval_step)

    await asyncio.gather(
        asyncio.create_task(dataloader_loop(), name="dataloader_loop"),
        *[
            asyncio.create_task(
                trajectory_group_worker_loop(), name=f"trajectory_group_worker_loop_{i}"
            )
            for i in range(cfg.async_config.groups_per_batch)
        ],
        asyncio.create_task(training_loop(), name="training_loop"),
        asyncio.create_task(evaluation_loop(), name="evaluation_loop"),
    )


@scope
async def do_group_rollout_and_filter_constant_reward(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    do_remove_constant_reward_groups: bool,
    enable_logging: bool = True,
) -> TrajectoryGroup | None:
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)

    with logtree.optional_enable_logging(enable_logging):
        trajectory_group = await do_group_rollout(env_group_builder, policy)

    # Remove if all trajectories have the same reward
    trajectory_groups = [trajectory_group]
    if do_remove_constant_reward_groups:
        trajectory_groups = remove_constant_reward_groups(trajectory_groups)
    if len(trajectory_groups) == 0:
        return None
    return trajectory_groups[0]


@scope
async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    metrics = {}
    with timed("save_checkpoint", metrics):
        path_dict = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"{i_batch:06d}",
            log_path=log_path,
            loop_state={"batch": i_batch},
            kind="both" if (i_batch > start_batch and i_batch % save_every == 0) else "sampler",
        )
        return training_client.create_sampling_client(path_dict["sampler_path"]), metrics


# BEGIN ADDED CODE
# BEGIN SEMANTIC_RND CODE
@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    service_client: tinker.ServiceClient,
    model_name: str,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    novelty_model: SemanticRND | None = None,
    intrinsic_reward_coef: float = 0.1,
    gemini_judge: GeminiJudge | None = None,
    reasoning_reward_coef: float = 0.5,
    i_batch: int = 0,
) -> tuple[list[tinker.Datum], dict[str, Any], torch.Tensor | None]:
    """
    Converts the trajectories into a minibatch, and provides metrics about the minibatch.
    
    Args:
        env_group_builders_P: Sequence of environment group builders
        trajectory_groups_P: List of trajectory groups
        tokenizer: Tokenizer for decoding
        service_client: Service client for KL computation
        model_name: Name of the base model
        kl_penalty_coef: Coefficient for KL penalty
        kl_discount_factor: Discount factor for KL penalty
        novelty_model: Optional SemanticRND module for intrinsic rewards
        intrinsic_reward_coef: Coefficient for combining intrinsic rewards with extrinsic
        gemini_judge: Optional GeminiJudge for reasoning-level rewards
        reasoning_reward_coef: Coefficient for weighting reasoning rewards vs correctness rewards
        i_batch: Current batch index for logging purposes
        
    Returns:
        Tuple of (training data, metrics dict, rnd_loss_tensor or None)
    """
# END SEMANTIC_RND CODE

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Print up to two trajectory groups
    for traj_group in trajectory_groups_P[:2]:
        print_group(traj_group, tokenizer)

    # BEGIN SEMANTIC_RND CODE
    # Compute reasoning-level rewards using GeminiJudge if provided
    if gemini_judge is not None:
        with timed("compute_reasoning_rewards", metrics):
            # Extract questions, references, and answers from trajectories
            questions, references, answers = await extract_questions_references_answers_from_trajectories(
                trajectory_groups_P, env_group_builders_P, tokenizer
            )
            
            logger.info(f"Calling GeminiJudge on {len(questions)} trajectories...")
            
            # Call judge_batch to evaluate reasoning
            judge_results: list[JudgeResult] = gemini_judge.judge_batch(
                questions=questions,
                answers=answers,
                references=references,
                filename_fingerprint=f"batch_{i_batch:06d}_{len(questions)}",
                retry_on_error=False,
            )
            
            # Convert verdicts to float rewards
            verdict_to_reward = {
                "correct": 1.0,
                "incorrect": 0.0,
                "uncertain": 0.2,
            }
            
            reasoning_rewards = [
                verdict_to_reward.get(result.verdict.value.lower(), 0.0) #FIXME: DEFAULT TO 0.0 IF UNKNOWN VERDICT
                for result in judge_results
            ]
            
            # Combine reasoning rewards with existing correctness rewards
            # Store original correctness rewards for metrics
            original_correctness_rewards = []
            idx = 0
            for traj_group in trajectory_groups_P:
                for i in range(len(traj_group.final_rewards_G)):
                    if idx < len(reasoning_rewards):
                        original_reward = traj_group.final_rewards_G[i]
                        original_correctness_rewards.append(original_reward)
                        
                        # Combine: reasoning_coef * judge_reward + (1 - reasoning_coef) * correctness_reward
                        combined_reward = (
                            reasoning_reward_coef * reasoning_rewards[idx] +
                            (1 - reasoning_reward_coef) * original_reward
                        )
                        traj_group.final_rewards_G[i] = combined_reward
                        idx += 1
            
            # Compute reasoning metrics
            num_correct_reasoning = sum(1 for r in reasoning_rewards if r == 1.0)
            num_incorrect_reasoning = sum(1 for r in reasoning_rewards if r == 0.0)
            num_uncertain_reasoning = sum(1 for r in reasoning_rewards if r == 0.2)
            
            # Count how many correct answers have correct reasoning
            num_correct_answer_correct_reasoning = sum(
                1 for orig_r, judge_r in zip(original_correctness_rewards, reasoning_rewards)
                if orig_r > 0.5 and judge_r == 1.0
            )
            num_correct_answer_incorrect_reasoning = sum(
                1 for orig_r, judge_r in zip(original_correctness_rewards, reasoning_rewards)
                if orig_r > 0.5 and judge_r == 0.0
            )
            num_correct_answer_uncertain_reasoning = sum(
                1 for orig_r, judge_r in zip(original_correctness_rewards, reasoning_rewards)
                if orig_r > 0.5 and judge_r == 0.2
            )
            
            num_correct_answers = sum(1 for r in original_correctness_rewards if r > 0.5)
            
            # Log reasoning metrics
            metrics["reasoning/reward_mean"] = np.mean(reasoning_rewards)
            metrics["reasoning/reward_std"] = np.std(reasoning_rewards)
            metrics["reasoning/correct_frac"] = num_correct_reasoning / len(reasoning_rewards) if len(reasoning_rewards) > 0 else 0.0
            metrics["reasoning/incorrect_frac"] = num_incorrect_reasoning / len(reasoning_rewards) if len(reasoning_rewards) > 0 else 0.0
            metrics["reasoning/uncertain_frac"] = num_uncertain_reasoning / len(reasoning_rewards) if len(reasoning_rewards) > 0 else 0.0
            
            # Metrics for correct answers
            if num_correct_answers > 0:
                metrics["reasoning/correct_answer_correct_reasoning_frac"] = num_correct_answer_correct_reasoning / num_correct_answers
                metrics["reasoning/correct_answer_incorrect_reasoning_frac"] = num_correct_answer_incorrect_reasoning / num_correct_answers
                metrics["reasoning/correct_answer_uncertain_reasoning_frac"] = num_correct_answer_uncertain_reasoning / num_correct_answers
            else:
                metrics["reasoning/correct_answer_correct_reasoning_frac"] = 0.0
                metrics["reasoning/correct_answer_incorrect_reasoning_frac"] = 0.0
                metrics["reasoning/correct_answer_uncertain_reasoning_frac"] = 0.0
            
            # Overall combined metrics
            metrics["reasoning/combined_reward_mean"] = np.mean([tg.final_rewards_G for tg in trajectory_groups_P])
            
            logger.info(f"Reasoning evaluation complete: {num_correct_reasoning}/{len(reasoning_rewards)} correct, "
                       f"{num_uncertain_reasoning}/{len(reasoning_rewards)} uncertain, "
                       f"{num_incorrect_reasoning}/{len(reasoning_rewards)} incorrect")
    # END SEMANTIC_RND CODE

    # BEGIN SEMANTIC_RND CODE
    # Compute RND intrinsic rewards if novelty model is provided
    rnd_loss_tensor = None
    if novelty_model is not None:
        with timed("compute_rnd_rewards", metrics):
            # Extract (problem, response) pairs from trajectories
            # Returns List[List[Tuple[str, str]]] for SemanticRND compatibility
            problem_response_pairs = await extract_responses_from_trajectories(
                trajectory_groups_P, env_group_builders_P, tokenizer
            )
            
            # Get intrinsic rewards and RND loss from SemanticRND using forward method
            intrinsic_rewards_tensor, rnd_loss_tensor = novelty_model(
                problem_response_pairs,
                update_stats=True
            )
            
            # Convert to list for adding to rewards
            intrinsic_rewards = intrinsic_rewards_tensor.detach().cpu().flatten().tolist()
            
            # Add intrinsic rewards to the trajectories' final rewards
            idx = 0
            for traj_group in trajectory_groups_P:
                for i in range(len(traj_group.final_rewards_G)):
                    if idx < len(intrinsic_rewards):
                        # Combine extrinsic and intrinsic rewards
                        traj_group.final_rewards_G[i] += intrinsic_reward_coef * intrinsic_rewards[idx]
                        idx += 1
            
            # Log RND metrics (detach for logging only)
            metrics["rnd/loss"] = rnd_loss_tensor.detach().cpu().item()
            metrics["rnd/intrinsic_reward_mean"] = np.mean(intrinsic_rewards)
            metrics["rnd/intrinsic_reward_std"] = np.std(intrinsic_rewards)
            metrics["rnd/intrinsic_reward_min"] = np.min(intrinsic_rewards)
            metrics["rnd/intrinsic_reward_max"] = np.max(intrinsic_rewards)
    # END SEMANTIC_RND CODE

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0:
        with timed("kl_vs_base", metrics):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                service_client.create_sampling_client(base_model=model_name),
                # ^^^ TODO: replace with the model we load, if relevant
                kl_penalty_coef,
                kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics, rnd_loss_tensor
# END ADDED CODE


@scope
async def compute_full_batch_metrics_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    data_D: list[tinker.Datum],
    training_logprobs_D: list[torch.Tensor],
    log_path: str,
    save_every: int,
    do_compute_post_kl: bool,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    At the end of the iteration, this will compute metrics for the full batch
    and return the latest sampling client.

    The reason we return a sampling client is that if do_compute_post_kl is True,
    we need to create a sampling client from the post-update policy.
    """
    metrics = {}

    # Compute KL metrics
    with timed("compute_kl_sample_train", metrics):
        kl_sample_train_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
        metrics.update(kl_sample_train_metrics)

    # Get a sampling client using the new weights
    sampling_client, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
        training_client, i_batch, log_path, save_every
    )
    metrics.update(checkpoint_metrics)

    # Compute post-KL metrics if configured
    if do_compute_post_kl:
        with timed("compute_post_kl", metrics):
            post_kl_metrics = await compute_post_kl(data_D, sampling_client)
            metrics.update(post_kl_metrics)

    return sampling_client, metrics


@scope
async def do_train_step_streaming_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    trajectory_groups_queue: asyncio.Queue[WrappedTrajectoryGroup | None],
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    trajectory_group_filter: Callable[[WrappedTrajectoryGroup | None], bool] = lambda _: True,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    As soon as we have enough trajectories for a minibatch, we will train on them.
    This allows us to overlap sampling and training.
    """
    assert cfg.stream_minibatch_config is not None
    assert cfg.stream_minibatch_config.groups_per_batch % cfg.num_substeps == 0, (
        f"{cfg.stream_minibatch_config.groups_per_batch=} must be divisible by {cfg.num_substeps=}"
    )
    # Number of groups across all minibatches in each optimizer substep
    groups_per_substep = cfg.stream_minibatch_config.groups_per_batch // cfg.num_substeps
    assert groups_per_substep % cfg.stream_minibatch_config.num_minibatches == 0, (
        f"{groups_per_substep} must be divisible by {cfg.stream_minibatch_config.num_minibatches=}"
    )
    # Number of groups per minibatch in each optimizer substep
    groups_per_minibatch = groups_per_substep // cfg.stream_minibatch_config.num_minibatches

    context = get_scope_context()
    context.attributes["step"] = i_batch

    metrics = {}

    # Run multiple optimizer substeps per training iteration
    all_data_D = []
    all_training_logprobs_D = []
    all_wrapped_trajectory_groups = []
    for i_substep in range(cfg.num_substeps):
        # Run multiple minibatches per substep
        # Once we have enough trajectories for a minibatch, train on them
        wrapped_trajectory_groups = []
        i_minibatch = 0
        while i_minibatch < cfg.stream_minibatch_config.num_minibatches:
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if not trajectory_group_filter(wrapped_trajectory_group):
                continue
            wrapped_trajectory_groups.append(wrapped_trajectory_group)

            if len(wrapped_trajectory_groups) < groups_per_minibatch:
                continue
            logger.info(
                f"[stream_minibatch] Step {i_batch}, Substep {i_substep}/{cfg.num_substeps}, Minibatch {i_minibatch}/{cfg.stream_minibatch_config.num_minibatches}: Will train on minibatch, num groups: {len(wrapped_trajectory_groups)}"
            )

            # Note: we may have removed trajectory groups that have the same reward.
            # To have the same results as the sync implementation, we will
            # remove these and train on a smaller batch.
            wrapped_trajectory_groups = [g for g in wrapped_trajectory_groups if g is not None]
            data_D, prepare_minibatch_metrics = await prepare_minibatch(
                [g.env_group_builder for g in wrapped_trajectory_groups],
                [g.trajectory_group for g in wrapped_trajectory_groups],
                tokenizer,
                service_client,
                model_name=cfg.model_name,
                kl_penalty_coef=cfg.kl_penalty_coef,
                kl_discount_factor=cfg.kl_discount_factor,
            )
            metrics.update(prepare_minibatch_metrics)

            # Accumulate gradients across multiple minibatches
            with timed(
                f"train/forward_backward_substep_{i_substep}_minibatch_{i_minibatch}", metrics
            ):
                training_logprobs_D = await forward_backward(
                    training_client,
                    data_D,
                    cfg.loss_fn,
                )
            all_data_D.extend(data_D)
            all_training_logprobs_D.extend(training_logprobs_D)
            all_wrapped_trajectory_groups.extend(wrapped_trajectory_groups)
            i_minibatch += 1
            wrapped_trajectory_groups = []

        # Run optimizer step only once after all minibatches
        with timed(f"train/optim_substep_{i_substep}", metrics):
            await optim_step(training_client, cfg.learning_rate)

    # Aggregate metrics across the entire batch
    metrics.update(compute_sampling_client_metrics(all_wrapped_trajectory_groups))
    metrics.update(
        compute_trajectory_metrics(
            [g.trajectory_group for g in all_wrapped_trajectory_groups],
            [g.env_group_builder.logging_tags() for g in all_wrapped_trajectory_groups],
        )
    )
    (
        sampling_client,
        full_batch_metrics,
    ) = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        all_data_D,
        all_training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
    )
    metrics.update(full_batch_metrics)
    return sampling_client, metrics


# BEGIN ADDED CODE
# BEGIN SEMANTIC_RND CODE
@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    novelty_model: SemanticRND | None = None,
    novelty_model_optimizer: torch.optim.Optimizer | None = None,
    intrinsic_reward_coef: float = 0.1,
    gemini_judge: GeminiJudge | None = None,
    reasoning_reward_coef: float = 0.5,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    Perform a single training step and return updated sampling client.
    
    Args:
        cfg: Training configuration
        i_batch: Current batch index
        training_client: Tinker training client
        service_client: Tinker service client
        tokenizer: Tokenizer for decoding
        env_group_builders_P: Environment group builders
        trajectory_groups_P: Trajectory groups to train on
        novelty_model: Optional SemanticRND module for intrinsic rewards
        novelty_model_optimizer: Optimizer for the novelty model
        intrinsic_reward_coef: Coefficient for intrinsic rewards
        gemini_judge: Optional GeminiJudge for reasoning-level rewards
        reasoning_reward_coef: Coefficient for weighting reasoning rewards vs correctness rewards
        
    Returns:
        Tuple of (sampling client, metrics dict)
    """
# END SEMANTIC_RND CODE
    context = get_scope_context()
    context.attributes["step"] = i_batch

    metrics = {}
    # BEGIN SEMANTIC_RND CODE
    data_D, prepare_minibatch_metrics, rnd_loss_tensor = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        service_client,
        model_name=cfg.model_name,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        novelty_model=novelty_model,
        intrinsic_reward_coef=intrinsic_reward_coef,
        gemini_judge=gemini_judge,
        reasoning_reward_coef=reasoning_reward_coef,
        i_batch=i_batch,
    )
    # END SEMANTIC_RND CODE
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
        )

    # BEGIN SEMANTIC_RND CODE
    # Update RND model if provided
    # Single forward pass approach: reuse the tensor computed in prepare_minibatch
    if novelty_model is not None and novelty_model_optimizer is not None and rnd_loss_tensor is not None:
        with timed("train_rnd", metrics):
            novelty_model_optimizer.zero_grad()
            rnd_loss_tensor.backward()
            novelty_model_optimizer.step()
    # END SEMANTIC_RND CODE

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics
# END ADDED CODE


# BEGIN ADDED CODE
# BEGIN SEMANTIC_RND CODE
@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    novelty_model: SemanticRND | None = None,
    novelty_model_optimizer: torch.optim.Optimizer | None = None,
    intrinsic_reward_coef: float = 0.1,
    gemini_judge: GeminiJudge | None = None,
    reasoning_reward_coef: float = 0.5,
):
    """
    Implements fully synchronous on-policy training.
    
    Args:
        start_batch: Starting batch index
        end_batch: Ending batch index (exclusive)
        num_batches: Total number of batches
        cfg: Training configuration
        training_client: Tinker training client
        service_client: Tinker service client
        evaluators: List of evaluators
        dataset: RL dataset
        ml_logger: ML logger for metrics
        tokenizer: Tokenizer
        novelty_model: Optional SemanticRND module for intrinsic rewards
        novelty_model_optimizer: Optimizer for the novelty model
        intrinsic_reward_coef: Coefficient for intrinsic rewards
        gemini_judge: Optional GeminiJudge for reasoning-level rewards
        reasoning_reward_coef: Coefficient for weighting reasoning rewards vs correctness rewards
    """
# END SEMANTIC_RND CODE
    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every, start_batch
    )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                eval_metrics = await run_evaluations_parallel(
                    evaluators, sampling_client, cfg, i_batch
                )
                metrics.update(eval_metrics)

        # Get batch and sample trajectories
        env_group_builders_P = dataset.get_batch(i_batch)

        # Initialize logtree trace for this iteration if logging is enabled
        with _get_logtree_scope(
            log_path=cfg.log_path,
            num_groups_to_log=cfg.num_groups_to_log,
            f_name=f"train_iteration_{i_batch:06d}",
            scope_name=f"RL Iteration {i_batch}",
        ):
            trajectory_groups_P = await asyncio.gather(
                *[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            max_tokens=cfg.max_tokens,
                            do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                            enable_logging=i < cfg.num_groups_to_log,
                        ),
                        name=f"sample_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ],
            )
        trajectory_groups_P = [
            trajectory_group
            for trajectory_group in trajectory_groups_P
            if trajectory_group is not None
        ]

        # Train step
        # BEGIN SEMANTIC_RND CODE
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
            novelty_model=novelty_model,
            novelty_model_optimizer=novelty_model_optimizer,
            intrinsic_reward_coef=intrinsic_reward_coef,
            gemini_judge=gemini_judge,
            reasoning_reward_coef=reasoning_reward_coef,
        )
        # END SEMANTIC_RND CODE

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)
# END ADDED CODE


@scope
async def main(
    cfg: Config,
):
    """Main training loop for MDP RL."""
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    if cfg.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0
    
    # BEGIN SEMANTIC_RND CODE
    # --- RND Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize SemanticRND with encoder and RND networks
    novelty_model = SemanticRND(
        encoder_model_name="Alibaba-NLP/gte-modernbert-base",
        rnd_output_dim=512,
        rnd_hidden_dim=512,
        device=str(device),
        max_length=8192,
        concat_problem_answer=True,
        separator=" [SEP] "
    )
    # Initialize optimizer for RND predictor network
    novelty_model_optimizer = torch.optim.Adam(novelty_model.predictor_parameters(), lr=1e-4)
    logger.info(f"Initialized SemanticRND module on {device}")
    # --- End RND Initialization ---
    # END SEMANTIC_RND CODE

    # BEGIN SEMANTIC_RND CODE
    # --- GeminiJudge Initialization ---
    gemini_judge = None
    if cfg.use_reasoning_rewards:
        from llm_as_a_judge import GEMINI_MATH_JUDGE_SYSTEM_PROMPT
        gemini_judge = GeminiJudge(
            model_name="gemini-2.0-flash-lite",
            system_prompt=GEMINI_MATH_JUDGE_SYSTEM_PROMPT
        )
        logger.info(f"Initialized GeminiJudge for reasoning-level rewards (coef={cfg.reasoning_reward_coef})")
    # --- End GeminiJudge Initialization ---
    # END SEMANTIC_RND CODE

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    )
    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create dataset from thunk
    dataset, maybe_test_dataset = await cfg.dataset_builder()
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))

    num_batches = len(dataset)
    logger.info(f"Will train on {num_batches} batches")

    # BEGIN ADDED CODE
    # BEGIN SEMANTIC_RND CODE
    # RND configuration
    intrinsic_reward_coef = 0.1  # Can be made configurable via Config class
    
    # Training loop
    if cfg.async_config is not None:
        training_func = do_async_training
    elif cfg.stream_minibatch_config is not None:
        training_func = do_sync_training_with_stream_minibatch
    else:
        training_func = do_sync_training
    await training_func(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
        novelty_model=novelty_model,
        novelty_model_optimizer=novelty_model_optimizer,
        intrinsic_reward_coef=intrinsic_reward_coef,
        gemini_judge=gemini_judge,
        reasoning_reward_coef=cfg.reasoning_reward_coef,
    )
    # END SEMANTIC_RND CODE
    # END ADDED CODE

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
