"""Reward shaping shared by the Tinker loop (curiosity_train) and the lean_rl loop.

Extracted verbatim from ``curiosity_train.prepare_minibatch``: trajectory metrics,
Gemini-judge reasoning rewards, entropy bonus and the SemanticRND block. Rewards are
mutated in place on ``trajectory_groups_P[*].final_rewards_G``.
"""

from __future__ import annotations

import io
import logging
from typing import Any, List, Sequence, Tuple

import numpy as np
import psutil
import torch
from tinker_cookbook.display import colorize_example
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.problem_env import ProblemEnv
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils.misc_utils import safezip, timed
from tinker_cookbook.utils.trace import scope

from llm_as_a_judge import GeminiJudge, JudgeResult
from rnd import RNDBuffer, SemanticRND

logger = logging.getLogger(__name__)


def get_memory_usage() -> dict:
    """Get current CPU and GPU memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    
    result = {
        "cpu_rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size
        "cpu_vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size
    }
    
    if torch.cuda.is_available():
        result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        result["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return result


def log_memory_usage(label: str, include_gpu: bool = True):
    """Log current memory usage with a label."""
    mem = get_memory_usage()
    gpu_str = ""
    if include_gpu and "gpu_allocated_mb" in mem:
        gpu_str = f", GPU: {mem['gpu_allocated_mb']:.1f}MB alloc / {mem['gpu_reserved_mb']:.1f}MB reserved"
    logger.info(f"[Memory] {label}: CPU RSS={mem['cpu_rss_mb']:.1f}MB, VMS={mem['cpu_vms_mb']:.1f}MB{gpu_str}")


@scope
async def extract_problems_and_responses_separately(
    trajectory_groups_P: list[TrajectoryGroup],
    env_group_builders_P: Sequence[EnvGroupBuilder],
    tokenizer: Tokenizer,
) -> Tuple[List[str], List[str]]:
    """
    Extract problems and responses as separate flat lists for buffer-based RND training.
    
    Args:
        trajectory_groups_P: List of trajectory groups (B groups, each with G trajectories)
        env_group_builders_P: List of environment group builders (B builders)
        tokenizer: Tokenizer for decoding tokens
        
    Returns:
        Tuple[List[str], List[str]]:
            - problems_B: List of B unique problems (one per group)
            - responses_BG: List of B*G responses (flattened, G responses per problem)
    """
    from tinker_cookbook.rl.problem_env import ProblemEnv
    
    problems_B = []
    responses_BG = []
    
    for env_group_builder, traj_group in safezip(env_group_builders_P, trajectory_groups_P):
        # Create environments to access get_question()
        envs = await env_group_builder.make_envs()
        
        # Get problem from first env (all envs in group have same problem)
        first_env = envs[0]
        if isinstance(first_env, ProblemEnv):
            problem = first_env.get_question()
        else:
            problem = ""
        problems_B.append(problem)
        
        # Extract all G responses for this problem
        for env, trajectory in safezip(envs, traj_group.trajectories_G):
            if len(trajectory.transitions) > 0:
                last_action_tokens = trajectory.transitions[-1].ac.tokens
                response = tokenizer.decode(last_action_tokens)
            else:
                response = ""
            responses_BG.append(response)
    
    return problems_B, responses_BG


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


def compute_entropy_bonus(trajectory_groups_P: list[TrajectoryGroup]) -> list[list[float]]:
    """
    Compute per-token entropy/perplexity bonus for each trajectory in each group.
    
    The entropy bonus is: -sum(log_probs) / num_tokens
    This encourages the model to generate diverse, uncertain responses.
    
    Args:
        trajectory_groups_P: List of trajectory groups
        
    Returns:
        List of lists, where each inner list contains entropy bonuses for trajectories in that group
    """
    entropy_bonuses_P = []
    
    for traj_group in trajectory_groups_P:
        entropy_bonuses_G = []
        
        for trajectory in traj_group.trajectories_G:
            # Sum log probabilities across all tokens in all transitions
            total_log_prob = 0.0
            num_tokens = 0
            
            for transition in trajectory.transitions:
                # transition.ac is TokensWithLogprobs
                if transition.ac.maybe_logprobs is not None:
                    total_log_prob += sum(transition.ac.maybe_logprobs)
                    num_tokens += len(transition.ac.maybe_logprobs)
            
            # Entropy bonus is negative average log probability
            # (higher entropy = lower log prob = more uncertain responses)
            if num_tokens > 0:
                entropy_bonus = -total_log_prob / num_tokens
            else:
                entropy_bonus = 0.0
            
            entropy_bonuses_G.append(entropy_bonus)
        
        entropy_bonuses_P.append(entropy_bonuses_G)
    
    return entropy_bonuses_P


def apply_entropy_bonus(
    trajectory_groups_P: list[TrajectoryGroup],
    coef: float | None,
    metrics: dict[str, Any],
) -> list[float]:
    """Add coef * entropy bonus to each trajectory's final_rewards_G in place; no-op if coef is None or <= 0.

    Returns the flat (B*G) per-trajectory total rewards from BEFORE the bonus, so
    callers that need task correctness (the RND block) are not fooled by the bonus.
    """
    pre_bonus_total_rewards = [
        r for traj_group in trajectory_groups_P for r in traj_group.get_total_rewards()
    ]
    if coef is None or coef <= 0.0:
        return pre_bonus_total_rewards
    # The bonus uses sampler logprobs and assumes sampling temperature 1.0.
    with timed("compute_entropy_bonus", metrics):
        entropy_bonuses_P = compute_entropy_bonus(trajectory_groups_P)
        for traj_group, bonuses_G in zip(trajectory_groups_P, entropy_bonuses_P):
            for i, bonus in enumerate(bonuses_G):
                traj_group.final_rewards_G[i] += coef * bonus

        all_bonuses = [b for bonuses_G in entropy_bonuses_P for b in bonuses_G]
        mean = float(np.mean(all_bonuses)) if all_bonuses else 0.0
        metrics["entropy/bonus_mean"] = mean
        metrics["entropy/bonus_std"] = float(np.std(all_bonuses)) if all_bonuses else 0.0
        metrics["entropy/bonus_coef"] = coef
        logger.info(f"Applied entropy bonus: mean={mean:.4f}, coef={coef}")
    return pre_bonus_total_rewards


@scope
async def shape_rewards(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    *,
    i_batch: int,
    gemini_judge: GeminiJudge | None,
    reasoning_reward_coef: float,
    novelty_model: SemanticRND | None,
    rnd_buffer: RNDBuffer | None,
    intrinsic_reward_coef: float,
    rnd_update_steps: int,
    rnd_minibatch_size: int,
    penalize_incorrect_novelty: bool,
    correctness_threshold: float,
    curiosity_warmup_batches: int,
    entropy_bonus_coef: float | None,
) -> dict[str, Any]:
    """Compute trajectory metrics and apply all reward shaping in place.

    Returns the metrics dict (trajectory metrics, reasoning/*, entropy/*, rnd/*, timings).
    """
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Print up to two trajectory groups
    for traj_group in trajectory_groups_P[:2]:
        print_group(traj_group, tokenizer)

    # BEGIN SEMANTIC_RND CODE
    # Compute reasoning-level rewards using GeminiJudge if provided
    # Only judge trajectories with correct answers (reward >= 0.9) to save API costs
    # Skip reasoning rewards during warmup period (same as curiosity rewards)
    apply_reasoning_rewards = i_batch >= curiosity_warmup_batches
    if gemini_judge is not None and apply_reasoning_rewards:
        with timed("compute_reasoning_rewards", metrics):
            
            # Step 1: Identify correct trajectories and extract their data inline
            # This avoids calling make_envs() for trajectories we won't judge
            correct_indices: list[tuple[int, int]] = []  # (traj_group_idx, traj_idx)
            filtered_questions: list[str] = []
            filtered_references: list[str] = []
            filtered_answers: list[str] = []
            total_trajectories = 0
            
            for tg_idx, (env_group_builder, traj_group) in enumerate(safezip(env_group_builders_P, trajectory_groups_P)):
                # Use get_total_rewards() to get actual rewards (per-step + final group rewards)
                # final_rewards_G is just the group reward (often 0.0), not the per-step rewards
                total_rewards = traj_group.get_total_rewards()
                total_trajectories += len(total_rewards)
                
                # Check if any trajectories in this group are correct
                has_correct = any(reward >= 0.9 for reward in total_rewards)
                if not has_correct:
                    continue
                
                # Only create envs if we have correct trajectories in this group
                envs = await env_group_builder.make_envs()
                
                for t_idx, (env, trajectory, reward) in enumerate(safezip(
                    envs, traj_group.trajectories_G, total_rewards
                )):
                    if reward >= 0.9 and isinstance(env, ProblemEnv):
                        correct_indices.append((tg_idx, t_idx))
                        filtered_questions.append(env.get_question())
                        # Use get_reference_solution if available, else get_reference_answer
                        ref_fn = getattr(env, 'get_reference_solution', None) or env.get_reference_answer #TODO: VERIFY IF CORRECT -- SEEMS OKAY
                        filtered_references.append(ref_fn())
                        # Extract model's response
                        if len(trajectory.transitions) > 0:
                            filtered_answers.append(tokenizer.decode(trajectory.transitions[-1].ac.tokens))
                        else:
                            filtered_answers.append("")
            
            num_correct_answers = len(correct_indices)
            num_skipped = total_trajectories - num_correct_answers
            
            # Step 2: Only call judge if we have correct trajectories
            if num_correct_answers > 0:
                logger.info(f"Calling GeminiJudge on {num_correct_answers}/{total_trajectories} correct trajectories (skipping {num_skipped} incorrect)...")
                
                judge_results: list[JudgeResult] = gemini_judge.judge_synchronous(
                    questions=filtered_questions,
                    answers=filtered_answers,
                    references=filtered_references,
                    # filename_fingerprint=f"batch_{i_batch:06d}_{num_correct_answers}",
                    retry_on_error=False,
                )
                
                # Convert verdicts to float rewards
                verdict_to_reward = {
                    "correct": 1.0,
                    "incorrect": 0.0,
                    "uncertain": 0.2,
                }
                
                reasoning_rewards = [
                    verdict_to_reward.get(result.verdict.value.lower(), 0.0)
                    for result in judge_results
                ]
                
                # Step 3: Update rewards only for judged (correct) trajectories
                # We need to get the original total reward (per-step + group rewards)
                # and modify the final_rewards_G to incorporate the judge reward
                for i, (tg_idx, t_idx) in enumerate(correct_indices):
                    # Get the total reward for this trajectory (per-step + group)
                    traj = trajectory_groups_P[tg_idx].trajectories_G[t_idx]
                    per_step_reward = sum(t.reward for t in traj.transitions)
                    group_reward = trajectory_groups_P[tg_idx].final_rewards_G[t_idx]
                    original_total_reward = per_step_reward + group_reward
                    
                    judge_reward = reasoning_rewards[i]
                    # Combine: reasoning_coef * judge_reward + (1 - reasoning_coef) * correctness_reward
                    combined_reward = (
                        reasoning_reward_coef * judge_reward +
                        (1 - reasoning_reward_coef) * original_total_reward
                    )
                    # Update final_rewards_G so that new_total = per_step + new_final_reward = combined_reward
                    # new_final_reward = combined_reward - per_step_reward
                    trajectory_groups_P[tg_idx].final_rewards_G[t_idx] = combined_reward - per_step_reward
                
                # Step 4: Compute reasoning metrics (only for judged trajectories)
                num_correct_reasoning = sum(1 for r in reasoning_rewards if r == 1.0)
                num_incorrect_reasoning = sum(1 for r in reasoning_rewards if r == 0.0)
                num_uncertain_reasoning = sum(1 for r in reasoning_rewards if r == 0.2)
                
                # Log reasoning metrics
                metrics["reasoning/num_judged"] = num_correct_answers
                metrics["reasoning/num_skipped"] = num_skipped
                metrics["reasoning/reward_mean"] = np.mean(reasoning_rewards)
                metrics["reasoning/reward_std"] = np.std(reasoning_rewards)
                metrics["reasoning/correct_frac"] = num_correct_reasoning / num_correct_answers
                metrics["reasoning/incorrect_frac"] = num_incorrect_reasoning / num_correct_answers
                metrics["reasoning/uncertain_frac"] = num_uncertain_reasoning / num_correct_answers
                
                logger.info(
                    f"Reasoning evaluation complete: {num_correct_reasoning}/{num_correct_answers} correct reasoning, "
                    f"{num_uncertain_reasoning}/{num_correct_answers} uncertain, "
                    f"{num_incorrect_reasoning}/{num_correct_answers} incorrect reasoning"
                )
            else:
                # No correct trajectories to judge
                logger.info(f"Skipping GeminiJudge: no correct trajectories in batch (0/{total_trajectories})")
                metrics["reasoning/num_judged"] = 0
                metrics["reasoning/num_skipped"] = total_trajectories
                metrics["reasoning/reward_mean"] = 0.0
                metrics["reasoning/reward_std"] = 0.0
                metrics["reasoning/correct_frac"] = 0.0
                metrics["reasoning/incorrect_frac"] = 0.0
                metrics["reasoning/uncertain_frac"] = 0.0
    elif gemini_judge is not None and not apply_reasoning_rewards:
        # Warmup period - skip reasoning rewards but log the skip
        logger.info(f"Warmup batch {i_batch}/{curiosity_warmup_batches}: skipping reasoning rewards")
        metrics["reasoning/warmup_active"] = 1
    # END SEMANTIC_RND CODE

    # BEGIN ENTROPY_BONUS CODE
    # Applied to final_rewards_G (independent of RND), before the RND block and compute_advantages.
    # The returned pre-bonus totals are what the RND block uses for correctness.
    extrinsic_rewards_pre_bonus_BG = apply_entropy_bonus(trajectory_groups_P, entropy_bonus_coef, metrics)
    # END ENTROPY_BONUS CODE

    # BEGIN SEMANTIC_RND CODE
    # Buffer-based RND training with stratified normalization
    # 1. Extract problems (B) and responses (B*G) separately
    # 2. Encode embeddings for current batch
    # 3. Compute intrinsic rewards BEFORE training (using current RND state)
    # 4. Apply stratified normalization within each group's correct/incorrect subsets
    # 5. Add embeddings to buffer
    # 6. Train RND for K steps sampling from buffer
    if novelty_model is not None and rnd_buffer is not None:
        with timed("compute_rnd_rewards", metrics):
            log_memory_usage(f"  RND: before extract_problems_and_responses")
            
            # Validate concat_before_emb mode
            if novelty_model.concat_before_emb:
                raise ValueError(
                    "Buffer-based RND training requires concat_before_emb=False. "
                    "Please initialize SemanticRND with concat_before_emb=False."
                )
            
            # Extract problems and responses separately
            problems_B, responses_BG = await extract_problems_and_responses_separately(
                trajectory_groups_P, env_group_builders_P, tokenizer
            )
            log_memory_usage(f"  RND: after extract (B={len(problems_B)}, BG={len(responses_BG)})")
            
            B = len(problems_B)
            G = len(responses_BG) // B  # Number of rollouts per problem
            
            # Extrinsic rewards for determining correctness: get_total_rewards()
            # (per-step + final group rewards) captured BEFORE the entropy bonus,
            # so a wrong answer with a large bonus is never counted as correct.
            extrinsic_rewards_BG = torch.tensor(
                extrinsic_rewards_pre_bonus_BG, device=novelty_model.device
            )

            # Encode problems and responses separately
            problem_embeddings_B, response_embeddings_BG = novelty_model.encode_problems_responses_separately(
                problems_B, responses_BG
            )
            log_memory_usage(f"  RND: after encoding embeddings")
            
            # Compute stratified normalized rewards using current RND state (BEFORE training)
            # This gives us rewards normalized within each group's correct/incorrect subsets
            signed_intrinsic_rewards = novelty_model.compute_rewards_stratified_normalized(
                problem_embeddings=problem_embeddings_B,
                response_embeddings=response_embeddings_BG,
                extrinsic_rewards=extrinsic_rewards_BG,
                group_size=G,
                curiosity_coef=intrinsic_reward_coef,
                correctness_threshold=correctness_threshold,
                penalize_incorrect=penalize_incorrect_novelty,
            )
            log_memory_usage(f"  RND: after compute_rewards")
            
            # Determine correctness for buffer storage
            is_correct_BG = extrinsic_rewards_BG >= correctness_threshold
            
            # Add embeddings to buffer (for later RND training)
            rnd_buffer.add_batch(problem_embeddings_B, response_embeddings_BG, is_correct_BG)
            log_memory_usage(f"  RND: after add_batch to buffer")
            
            # Add signed intrinsic rewards to the trajectories (only after warmup period)
            # During warmup, RND still trains but rewards are not added to RL training
            apply_curiosity_rewards = i_batch >= curiosity_warmup_batches
            
            if apply_curiosity_rewards:
                intrinsic_rewards_list = signed_intrinsic_rewards.detach().cpu().tolist()
                idx = 0
                for traj_group in trajectory_groups_P:
                    for i in range(len(traj_group.final_rewards_G)):
                        if idx < len(intrinsic_rewards_list):
                            # intrinsic_rewards_list already has sign and coefficient applied
                            traj_group.final_rewards_G[i] += intrinsic_rewards_list[idx]
                            idx += 1
            else:
                logger.info(f"Warmup batch {i_batch}/{curiosity_warmup_batches}: RND training only, no curiosity rewards added")
            
            metrics["rnd/warmup_active"] = int(not apply_curiosity_rewards)
            metrics["rnd/rewards_applied"] = int(apply_curiosity_rewards)
            
            # Log reward metrics (split by correct/incorrect)
            correct_mask = is_correct_BG.cpu().numpy()
            raw_intrinsic = signed_intrinsic_rewards.detach().cpu().numpy()
            
            metrics["rnd/intrinsic_reward_mean"] = float(np.mean(raw_intrinsic))
            metrics["rnd/intrinsic_reward_std"] = float(np.std(raw_intrinsic))
            metrics["rnd/intrinsic_reward_min"] = float(np.min(raw_intrinsic))
            metrics["rnd/intrinsic_reward_max"] = float(np.max(raw_intrinsic))
            metrics["rnd/intrinsic_reward_p75"] = float(np.percentile(raw_intrinsic, 75))
            metrics["rnd/intrinsic_reward_p25"] = float(np.percentile(raw_intrinsic, 25))
            
            if correct_mask.sum() > 0:
                metrics["rnd/correct_reward_mean"] = float(np.mean(raw_intrinsic[correct_mask]))
            else:
                metrics["rnd/correct_reward_mean"] = 0.0
                
            if (~correct_mask).sum() > 0:
                metrics["rnd/incorrect_reward_mean"] = float(np.mean(raw_intrinsic[~correct_mask]))
            else:
                metrics["rnd/incorrect_reward_mean"] = 0.0
            
            metrics["rnd/num_correct"] = int(correct_mask.sum())
            metrics["rnd/num_incorrect"] = int((~correct_mask).sum())
            buffer_stats = rnd_buffer.get_current_size()
            metrics["rnd/buffer_num_responses"] = buffer_stats['num_responses']
            metrics["rnd/buffer_memory_mb"] = buffer_stats['memory_mb']
            
        # Train RND from buffer (K update steps)
        # Use the rnd_minibatch_size parameter (passed from config)
        effective_minibatch_size = min(rnd_minibatch_size, B * G)
        with timed("train_rnd_from_buffer", metrics):
            rnd_losses = []
            buffer_stats = rnd_buffer.get_current_size()
            total_responses = buffer_stats['num_responses']
            
            if total_responses >= effective_minibatch_size:
                for k in range(rnd_update_steps):
                    # Sample from buffer and train in one step
                    loss = novelty_model.train_step_from_buffer(
                        rnd_buffer, effective_minibatch_size, novelty_model._optimizer
                    )
                    rnd_losses.append(loss)
                
                metrics["rnd/loss_mean"] = float(np.mean(rnd_losses))
                metrics["rnd/loss_std"] = float(np.std(rnd_losses))
                metrics["rnd/num_updates"] = rnd_update_steps
                metrics["rnd/loss_max"] = float(np.max(rnd_losses))
                metrics["rnd/loss_min"] = float(np.min(rnd_losses))
                metrics["rnd/loss_p75"] = float(np.percentile(rnd_losses, 75))
                metrics["rnd/loss_p25"] = float(np.percentile(rnd_losses, 25))
                
                logger.info(
                    f"RND buffer training: {rnd_update_steps} updates, "
                    f"loss={np.mean(rnd_losses):.4f}±{np.std(rnd_losses):.4f}, "
                    f"buffer_size={total_responses}, memory={buffer_stats['memory_mb']:.1f}MB"
                )
            else:
                metrics["rnd/loss_mean"] = 0.0
                metrics["rnd/loss_std"] = 0.0
                metrics["rnd/num_updates"] = 0
                logger.info(
                    f"RND buffer too small for training: {total_responses}/{effective_minibatch_size} samples"
                )
    # END SEMANTIC_RND CODE
    return metrics
