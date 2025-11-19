"""
AIME 2024 Evaluation Script

This script provides comprehensive evaluation functionality for the AIME 2024 dataset.
It implements a custom SamplingClientEvaluator following Tinker's evaluation patterns,
similar to RLTestSetEvaluator.

Usage:
    python math_evaluation.py --model_path <tinker_model_path> --log_path <output_path>
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import chz
import numpy as np
import tinker
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.misc_utils import all_same

logger = logging.getLogger(__name__)
console = Console()


# ============================================================================
# AIME Environment Definition
# ============================================================================


class AIMEEnv(ProblemEnv):
    """Environment for AIME 2024 problems."""

    def __init__(
        self,
        problem: str,
        answer: str,
        solution: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
    ):
        super().__init__(renderer, convo_prefix, format_coef)
        self.problem = problem
        self.answer = answer
        self.solution = solution

    @classmethod
    def question_suffix(cls) -> str:
        """Suffix to add to the problem statement."""
        return "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        """Check if the response has the correct format (contains \\boxed{})."""
        try:
            extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        """Check if the extracted answer is correct."""
        try:
            extracted = extract_boxed(sample_str)
        except ValueError:
            return False
        return grade_answer(extracted, self.answer)

    def get_reference_answer(self) -> str:
        return self.answer

    def get_reference_solution(self) -> str:
        return self.solution

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        """Standard few-shot examples for AIME problems."""
        return [ #FIXME: MAYBE A SHORTER PROMPT
            {
                "role": "user",
                "content": (
                    "Find the number of ordered pairs $(a,b)$ of integers such that "
                    "$\\frac{a + 2}{a + 5} = \\frac{b}{4}$.\n\n"
                    "Please reason step by step, and put your final answer within \\boxed{}."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "We have $\\frac{a + 2}{a + 5} = \\frac{b}{4}$, so "
                    "$4(a + 2) = b(a + 5)$. Expanding gives $4a + 8 = ab + 5b$, so "
                    "$4a + 8 = b(a + 5)$. Rearranging gives $b = \\frac{4a + 8}{a + 5}$. "
                    "For $b$ to be an integer, we need $(a + 5) | (4a + 8)$. "
                    "We can write $4a + 8 = 4(a + 5) - 12$, so $(a + 5) | 12$. "
                    "The divisors of 12 are $\\pm 1, \\pm 2, \\pm 3, \\pm 4, \\pm 6, \\pm 12$. "
                    "This gives us 12 possible values of $a + 5$, and thus 12 ordered pairs. "
                    "Therefore, the answer is $\\boxed{12}$."
                ),
            },
        ]


# ============================================================================
# AIME Dataset Loader
# ============================================================================


def load_aime_dataset() -> list[dict]:
    """Load the AIME 2024 dataset from HuggingFace."""
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    return [
        {
            "id": item["id"],
            "problem": item["problem"],
            "answer": item["answer"],
            "solution": item["solution"],
        }
        for item in dataset
    ]


def create_aime_env_builder(
    problem_data: dict,
    renderer: renderers.Renderer,
    convo_prefix: list[renderers.Message] | None = None,
    group_size: int = 1,
) -> ProblemGroupBuilder:
    """Create an environment group builder for a single AIME problem."""
    return ProblemGroupBuilder(
        env_thunk=partial(
            AIMEEnv,
            problem_data["problem"],
            problem_data["answer"],
            problem_data["solution"],
            renderer,
            convo_prefix=convo_prefix,
        ),
        num_envs=group_size,
        dataset_name="aime_2024",
    )


# ============================================================================
# Metrics Computation
# ============================================================================


def _compute_by_group_metrics(
    trajectory_groups: list[TrajectoryGroup], good_thresh: float = 0.5
) -> dict[str, float]:
    """
    Compute metrics grouped by trajectory groups.
    Categorizes groups as all-good, all-bad, or mixed based on total rewards.
    """
    n_groups = len(trajectory_groups)
    if n_groups == 0:
        return {
            "by_group/frac_mixed": 0.0,
            "by_group/frac_all_good": 0.0,
            "by_group/frac_all_bad": 0.0,
        }

    n_mixed = n_good = n_bad = 0
    for tg in trajectory_groups:
        grp_rewards = tg.get_total_rewards()
        if all_same(grp_rewards):
            if grp_rewards[0] >= good_thresh:
                n_good += 1
            else:
                n_bad += 1
        else:
            n_mixed += 1

    return {
        "by_group/frac_mixed": n_mixed / n_groups,
        "by_group/frac_all_good": n_good / n_groups,
        "by_group/frac_all_bad": n_bad / n_groups,
    }


def _compute_trajectory_metrics(trajectory_groups: list[TrajectoryGroup]) -> dict[str, float]:
    """
    Compute comprehensive metrics from trajectory groups.
    Similar to metric_util.py's _compute_trajectory_metrics.
    """
    if not trajectory_groups:
        return {}

    flat_trajs = [traj for tg in trajectory_groups for traj in tg.trajectories_G]

    # Token and turn statistics
    ac_tokens_by_turn = [
        len(transition.ac.tokens) for traj in flat_trajs for transition in traj.transitions
    ]
    ob_tokens_by_turn = [
        transition.ob.length for traj in flat_trajs for transition in traj.transitions
    ]
    turns_by_trajectory = [len(traj.transitions) for traj in flat_trajs]

    # Basic metrics
    metrics = {
        "ac_tokens_per_turn": (
            sum(ac_tokens_by_turn) / sum(turns_by_trajectory) if turns_by_trajectory else 0.0
        ),
        "ob_tokens_per_turn": (
            sum(ob_tokens_by_turn) / sum(turns_by_trajectory) if turns_by_trajectory else 0.0
        ),
        "turns_per_episode": (
            sum(turns_by_trajectory) / len(flat_trajs) if flat_trajs else 0.0
        ),
        "total_episodes": len(flat_trajs),
        "total_turns": sum(turns_by_trajectory),
        "total_ac_tokens": sum(ac_tokens_by_turn),
        "total_ob_tokens": sum(ob_tokens_by_turn),
    }

    # Reward metrics
    total_rewards = [reward for tg in trajectory_groups for reward in tg.get_total_rewards()]
    metrics["reward/total"] = np.mean(total_rewards).item() if total_rewards else 0.0

    # Per-transition metrics (format, correct, etc.)
    transition_metrics = [ #FIXME: IS THIS CORRECT? SEEMS WEIRD. NEED TO COMPARE TO TRAINING LOOP
        transition.metrics
        for tg in trajectory_groups
        for traj in tg.trajectories_G
        for transition in traj.transitions
    ]
    traj_metrics = [metrics for tg in trajectory_groups for metrics in tg.metrics_G]

    # Aggregate metrics
    all_metrics = transition_metrics + traj_metrics
    if all_metrics:
        # Compute mean for each metric key
        metric_keys = set()
        for m in all_metrics:
            metric_keys.update(m.keys())

        for key in metric_keys:
            values = [m.get(key, 0.0) for m in all_metrics if key in m]
            if values:
                metrics[key] = np.mean(values).item()

    # By-group metrics
    metrics.update(_compute_by_group_metrics(trajectory_groups))

    return metrics


# ============================================================================
# AIME Evaluator
# ============================================================================


class AIMEEvaluator(SamplingClientEvaluator):
    """
    Custom evaluator for AIME 2024 dataset.
    Follows the pattern of RLTestSetEvaluator from metric_util.py.
    """

    def __init__(
        self,
        model_name: str,
        renderer_name: str = "llama3",
        max_tokens: int = 2048,
        group_size: int = 1,
        use_fewshot: bool = True,
        num_problems: int | None = None,
    ):
        """
        Initialize the AIME evaluator.

        Args:
            model_name: Name of the model for tokenizer
            renderer_name: Name of the renderer to use
            max_tokens: Maximum tokens to generate
            group_size: Number of samples per problem (for computing variance)
            use_fewshot: Whether to use few-shot examples
            num_problems: Number of problems to evaluate (None = all)
        """
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_tokens = max_tokens
        self.group_size = group_size
        self.use_fewshot = use_fewshot

        # Load dataset
        logger.info("Loading AIME 2024 dataset...")
        self.problems = load_aime_dataset()
        if num_problems is not None:
            self.problems = self.problems[:num_problems]
        logger.info(f"Loaded {len(self.problems)} AIME 2024 problems")

        # Setup tokenizer and renderer
        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

        # Setup conversation prefix
        self.convo_prefix = AIMEEnv.standard_fewshot_prefix() if use_fewshot else None

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run evaluation on the AIME dataset.

        Args:
            sampling_client: The sampling client to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Starting evaluation on {len(self.problems)} AIME 2024 problems...")

        # Create policy
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)

        # Run rollouts for each problem
        trajectory_groups = []
        for i, problem_data in enumerate(self.problems):
            # Create environment builder
            env_builder = create_aime_env_builder(
                problem_data,
                self.renderer,
                convo_prefix=self.convo_prefix,
                group_size=self.group_size,
            )

            # Enable logging for first few problems
            enable_logging = i < 5

            # Run rollout
            with logtree.optional_enable_logging(enable=enable_logging):
                traj_group = await do_group_rollout(env_builder, policy)
                trajectory_groups.append(traj_group)

            # Log progress
            if (i + 1) % 5 == 0:
                logger.info(f"Completed {i + 1}/{len(self.problems)} problems")

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = _compute_trajectory_metrics(trajectory_groups)

        # Add AIME-specific metadata
        metrics["eval/num_problems"] = len(self.problems)
        metrics["eval/group_size"] = self.group_size

        return metrics


# ============================================================================
# Main Evaluation Script
# ============================================================================


@chz.chz
class EvalConfig:
    """Configuration for AIME evaluation."""

    model_path: str | None = "tinker://a95b9543-d0b1-46c7-a5ab-7baa62d92906/sampler_weights/final"
    model_name: str = "Qwen/Qwen3-30B-A3B"
    renderer_name: str = "qwen3"
    max_tokens: int = 2048
    group_size: int = 4
    use_fewshot: bool = True
    num_problems: int | None = None
    log_path: str | None = None
    save_detailed_results: bool = True


def setup_logging(log_path: str | None) -> Path | None:
    """Setup logging configuration."""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Setup file logging if log_path is provided
    if log_path is not None:
        log_dir = Path(log_path)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"aime_eval_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

        logger.info(f"Logging to {log_file}")
        return log_dir
    return None


def print_metrics_table(metrics: dict[str, float]) -> None:
    """Print metrics in a beautiful table format using rich."""
    table = Table(title="AIME 2024 Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=40)
    table.add_column("Value", justify="right", style="green", width=20)

    # Group metrics by category
    categories = {
        "Accuracy": [],
        "Rewards": [],
        "Format": [],
        "Tokens": [],
        "Episodes": [],
        "By Group": [],
        "Evaluation": [],
        "Other": [],
    }

    for key, value in sorted(metrics.items()):
        if "correct" in key:
            categories["Accuracy"].append((key, value))
        elif "reward" in key:
            categories["Rewards"].append((key, value))
        elif "format" in key:
            categories["Format"].append((key, value))
        elif "token" in key:
            categories["Tokens"].append((key, value))
        elif "episode" in key or "turn" in key:
            categories["Episodes"].append((key, value))
        elif "by_group" in key:
            categories["By Group"].append((key, value))
        elif "eval" in key:
            categories["Evaluation"].append((key, value))
        else:
            categories["Other"].append((key, value))

    # Add rows by category
    for category, items in categories.items():
        if items:
            table.add_row(f"[bold]{category}[/bold]", "", style="bold yellow")
            for key, value in items:
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}" if abs(value) < 100 else f"{value:.2f}"
                else:
                    formatted_value = str(value)
                table.add_row(f"  {key}", formatted_value)

    console.print(table)


def save_results(metrics: dict[str, float], log_dir: Path, config: EvalConfig) -> None:
    """Save evaluation results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = log_dir / f"aime_results_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "config": {
            "model_path": config.model_path,
            "model_name": config.model_name,
            "renderer_name": config.renderer_name,
            "max_tokens": config.max_tokens,
            "group_size": config.group_size,
            "use_fewshot": config.use_fewshot,
            "num_problems": config.num_problems,
        },
        "metrics": metrics,
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")


async def main(config: EvalConfig):
    """Main evaluation function."""
    console.print("\n[bold cyan]AIME 2024 Evaluation Script[/bold cyan]\n")

    # Setup logging
    log_dir = setup_logging(config.log_path)

    # Create service client
    logger.info("Initializing Tinker service client...")
    service_client = tinker.ServiceClient()

    # Create sampling client
    logger.info(f"Creating sampling client for model: {config.model_name}")
    if config.model_path:
        logger.info(f"Loading model from path: {config.model_path}")
        sampling_client = service_client.create_sampling_client(
            model_path=config.model_path, base_model=config.model_name
        )
    else:
        logger.info("Using base model (no fine-tuned weights)")
        sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    # Create evaluator
    evaluator = AIMEEvaluator(
        model_name=config.model_name,
        renderer_name=config.renderer_name,
        max_tokens=config.max_tokens,
        group_size=config.group_size,
        use_fewshot=config.use_fewshot,
        num_problems=config.num_problems,
    )

    # Run evaluation
    console.print("\n[bold yellow]Running evaluation...[/bold yellow]\n")
    metrics = await evaluator(sampling_client)

    # Display results
    console.print("\n")
    print_metrics_table(metrics)

    # Save results
    if log_dir and config.save_detailed_results:
        save_results(metrics, log_dir, config)

    # Print summary
    console.print("\n[bold green]Evaluation Complete![/bold green]")
    console.print(
        f"\n[bold]Key Results:[/bold]\n"
        f"  • Accuracy: {metrics.get('correct', 0.0):.2%}\n"
        f"  • Format Valid: {metrics.get('format', 0.0):.2%}\n"
        f"  • Average Reward: {metrics.get('reward/total', 0.0):.4f}\n"
        f"  • Problems Evaluated: {metrics.get('eval/num_problems', 0):.0f}\n"
    )

    return metrics


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
