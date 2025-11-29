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


def load_aime2025_dataset() -> list[dict]:
    """Load the AIME 2025 dataset from HuggingFace."""
    dataset = load_dataset("yentinglin/aime_2025", split="train")
    return [
        {
            "id": item["id"],
            "problem": item["problem"],
            "answer": item["answer"],
            "solution": item["solution"],  # Same as answer for AIME 2025
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
        name: str = "aime_2024",
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
            name: Name identifier for this evaluator (used for metric prefixing)
        """
        self.name = name
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
                logger.info(f"Completed {i + 1}/{len(self.problems)} problems in {self.name}")

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = _compute_trajectory_metrics(trajectory_groups)

        # Add AIME-specific metadata
        metrics["eval/num_problems"] = len(self.problems)
        metrics["eval/group_size"] = self.group_size

        return metrics


class AIME2025Evaluator(SamplingClientEvaluator):
    """
    Custom evaluator for AIME 2025 dataset.
    Follows the same pattern as AIMEEvaluator.
    """

    def __init__(
        self,
        model_name: str,
        renderer_name: str = "llama3",
        max_tokens: int = 2048,
        group_size: int = 1,
        use_fewshot: bool = True,
        num_problems: int | None = None,
        name: str = "aime_2025",
    ):
        """
        Initialize the AIME 2025 evaluator.

        Args:
            model_name: Name of the model for tokenizer
            renderer_name: Name of the renderer to use
            max_tokens: Maximum tokens to generate
            group_size: Number of samples per problem (for computing variance)
            use_fewshot: Whether to use few-shot examples
            num_problems: Number of problems to evaluate (None = all)
            name: Name identifier for this evaluator (used for metric prefixing)
        """
        self.name = name
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_tokens = max_tokens
        self.group_size = group_size
        self.use_fewshot = use_fewshot

        # Load dataset
        logger.info("Loading AIME 2025 dataset...")
        self.problems = load_aime2025_dataset()
        if num_problems is not None:
            self.problems = self.problems[:num_problems]
        logger.info(f"Loaded {len(self.problems)} AIME 2025 problems")

        # Setup tokenizer and renderer
        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)

        # Setup conversation prefix
        self.convo_prefix = AIMEEnv.standard_fewshot_prefix() if use_fewshot else None

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run evaluation on the AIME 2025 dataset.

        Args:
            sampling_client: The sampling client to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Starting evaluation on {len(self.problems)} AIME 2025 problems...")

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
                logger.info(f"Completed {i + 1}/{len(self.problems)} problems in {self.name}")

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = _compute_trajectory_metrics(trajectory_groups)

        # Add AIME 2025-specific metadata
        metrics["eval/num_problems"] = len(self.problems)
        metrics["eval/group_size"] = self.group_size

        return metrics


# ============================================================================
# Parallel Evaluation Support
# ============================================================================


def _get_evaluator_name(evaluator: SamplingClientEvaluator) -> str:
    """Get the name of an evaluator, if available."""
    return getattr(evaluator, "name", "")


async def run_single_evaluation(
    evaluator: SamplingClientEvaluator,
    sampling_client: tinker.SamplingClient,
    log_path: str | None,
    num_groups_to_log: int,
) -> dict[str, float]:
    """
    Run a single evaluator and return its metrics with proper prefixing.
    
    Args:
        evaluator: The evaluator to run
        sampling_client: The sampling client to use
        log_path: Path for logging (optional)
        num_groups_to_log: Number of groups to log
        
    Returns:
        Dictionary of metrics prefixed with evaluator name
    """
    ev_name = _get_evaluator_name(evaluator)
    logger.info(f"Running evaluation: {ev_name or 'unnamed'}")
    
    # Run the evaluator
    eval_metrics = await evaluator(sampling_client)
    
    # Prefix metrics with evaluator name if available
    if ev_name:
        prefixed_metrics = {f"{ev_name}/{k}": v for k, v in eval_metrics.items()}
    else:
        prefixed_metrics = eval_metrics
    
    return prefixed_metrics


async def run_evaluations_parallel(
    evaluators: list[SamplingClientEvaluator],
    sampling_client: tinker.SamplingClient,
    log_path: str | None = None,
    num_groups_to_log: int = 4,
) -> dict[str, float]:
    """
    Run all evaluators in parallel and return aggregated metrics.
    
    Args:
        evaluators: List of evaluators to run
        sampling_client: The sampling client to use
        log_path: Path for logging (optional)
        num_groups_to_log: Number of groups to log
        
    Returns:
        Dictionary containing:
        - Per-evaluator metrics (prefixed with evaluator name)
        - Aggregate metrics (prefixed with "aggregate/")
    """
    if not evaluators:
        logger.warning("No evaluators provided")
        return {}
    
    logger.info(f"Running {len(evaluators)} evaluators in parallel...")
    
    # Create tasks for all evaluators
    tasks = []
    for i, evaluator in enumerate(evaluators):
        ev_name = _get_evaluator_name(evaluator)
        task = asyncio.create_task(
            run_single_evaluation(evaluator, sampling_client, log_path, num_groups_to_log),
            name=f"eval_{ev_name or i}",
        )
        tasks.append(task)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    
    # Merge all per-evaluator metrics
    all_metrics = {}
    for result in results:
        all_metrics.update(result)
    
    # Compute aggregate metrics across evaluators
    aggregate_metrics = _compute_aggregate_metrics(results)
    all_metrics.update(aggregate_metrics)
    
    return all_metrics


def _compute_aggregate_metrics(evaluator_results: list[dict[str, float]]) -> dict[str, float]:
    """
    Compute aggregate metrics across all evaluators.
    
    Args:
        evaluator_results: List of metric dictionaries from each evaluator
        
    Returns:
        Dictionary of aggregate metrics prefixed with "aggregate/"
    """
    if not evaluator_results:
        return {}
    
    # Collect all metric keys (without prefixes) that appear across evaluators
    all_keys = set()
    for result in evaluator_results:
        for key in result.keys():
            # Remove the evaluator name prefix to get the base metric name
            if "/" in key:
                base_key = key.split("/", 1)[1]
                all_keys.add(base_key)
    
    aggregate = {}
    
    # Compute mean across evaluators for each metric
    for base_key in all_keys:
        values = []
        for result in evaluator_results:
            # Try to find this metric in the result
            for full_key, value in result.items():
                if "/" in full_key and full_key.split("/", 1)[1] == base_key:
                    if isinstance(value, (int, float)):
                        values.append(value)
                    break
        
        if values:
            aggregate[f"aggregate/{base_key}"] = np.mean(values).item() #FIXME: NEED WEIGHTED AVERAGING BECAUSE DATASET SIZES WOULD BE DIFFERENT
    
    return aggregate


# ============================================================================
# Evaluator Configuration
# ============================================================================


@chz.chz
class EvaluatorConfig:
    """Configuration for a single evaluator."""
    
    dataset: str  # Name of the dataset (e.g., "aime_2024", "math", "deepmath")
    num_problems: int | None = None  # Number of problems to evaluate (None = all)
    use_fewshot: bool = True  # Whether to use few-shot examples
    
    def build(
        self,
        model_name: str,
        renderer_name: str,
        max_tokens: int,
        group_size: int,
    ) -> SamplingClientEvaluator:
        """
        Build the evaluator with the given shared parameters.
        
        Args:
            model_name: Model name for tokenizer
            renderer_name: Renderer name
            max_tokens: Max tokens to generate
            group_size: Number of samples per problem
            
        Returns:
            Configured SamplingClientEvaluator
        """
        if self.dataset == "aime_2024":
            return AIMEEvaluator(
                model_name=model_name,
                renderer_name=renderer_name,
                max_tokens=max_tokens,
                group_size=group_size,
                use_fewshot=self.use_fewshot,
                num_problems=self.num_problems,
                name="aime_2024",
            )
        elif self.dataset == "aime_2025":
            return AIME2025Evaluator(
                model_name=model_name,
                renderer_name=renderer_name,
                max_tokens=max_tokens,
                group_size=group_size,
                use_fewshot=self.use_fewshot,
                num_problems=self.num_problems,
                name="aime_2025",
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")


# ============================================================================
# Main Evaluation Configuration
# ============================================================================


@chz.chz
class EvalConfig:
    """Configuration for multi-evaluator evaluation."""

    model_path: str | None = "tinker://a95b9543-d0b1-46c7-a5ab-7baa62d92906/sampler_weights/final"
    model_path: str | None = None
    model_name: str = "Qwen/Qwen3-30B-A3B"
    renderer_name: str = "qwen3"
    max_tokens: int = 2048
    group_size: int = 2
    log_path: str | None = None
    save_detailed_results: bool = True
    num_groups_to_log: int = 4
    
    # Evaluator configurations
    evaluators: list[EvaluatorConfig] = chz.field(
        default_factory=lambda: [
            EvaluatorConfig(dataset="aime_2024", num_problems=None, use_fewshot=True),
            EvaluatorConfig(dataset="aime_2025", num_problems=None, use_fewshot=True),
        ]
    )


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


def print_metrics_table(metrics: dict[str, float], title: str = "Evaluation Results") -> None:
    """Print metrics in a beautiful table format using rich."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=50)
    table.add_column("Value", justify="right", style="green", width=20)

    # Separate metrics by evaluator
    evaluator_metrics = {}
    aggregate_metrics = {}
    
    for key, value in metrics.items():
        if key.startswith("aggregate/"):
            aggregate_metrics[key] = value
        else:
            # Extract evaluator name
            if "/" in key:
                eval_name = key.split("/", 1)[0]
                if eval_name not in evaluator_metrics:
                    evaluator_metrics[eval_name] = {}
                evaluator_metrics[eval_name][key] = value
            else:
                # No prefix, treat as misc
                if "other" not in evaluator_metrics:
                    evaluator_metrics["other"] = {}
                evaluator_metrics["other"][key] = value
    
    # Display aggregate metrics first
    if aggregate_metrics:
        table.add_row(f"[bold cyan]AGGREGATE METRICS[/bold cyan]", "", style="bold cyan")
        for key, value in sorted(aggregate_metrics.items()):
            formatted_value = f"{value:.4f}" if isinstance(value, float) and abs(value) < 100 else f"{value:.2f}"
            display_key = key.replace("aggregate/", "")
            table.add_row(f"  {display_key}", formatted_value)
        table.add_row("", "")  # Separator
    
    # Display per-evaluator metrics
    for eval_name in sorted(evaluator_metrics.keys()):
        table.add_row(f"[bold yellow]{eval_name.upper()}[/bold yellow]", "", style="bold yellow")
        
        eval_data = evaluator_metrics[eval_name]
        
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
        
        for key, value in sorted(eval_data.items()):
            # Remove evaluator prefix for display
            display_key = key.split("/", 1)[1] if "/" in key else key
            
            if "correct" in key:
                categories["Accuracy"].append((display_key, value))
            elif "reward" in key:
                categories["Rewards"].append((display_key, value))
            elif "format" in key:
                categories["Format"].append((display_key, value))
            elif "token" in key:
                categories["Tokens"].append((display_key, value))
            elif "episode" in key or "turn" in key:
                categories["Episodes"].append((display_key, value))
            elif "by_group" in key:
                categories["By Group"].append((display_key, value))
            elif "eval" in key:
                categories["Evaluation"].append((display_key, value))
            else:
                categories["Other"].append((display_key, value))
        
        # Add rows by category
        for category, items in categories.items():
            if items:
                table.add_row(f"  [italic]{category}[/italic]", "", style="dim")
                for key, value in items:
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}" if abs(value) < 100 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    table.add_row(f"    {key}", formatted_value)
        
        table.add_row("", "")  # Separator between evaluators

    console.print(table)


def save_results(metrics: dict[str, float], log_dir: Path, config: EvalConfig) -> None:
    """Save evaluation results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = log_dir / f"eval_results_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "config": {
            "model_path": config.model_path,
            "model_name": config.model_name,
            "renderer_name": config.renderer_name,
            "max_tokens": config.max_tokens,
            "group_size": config.group_size,
            "evaluators": [
                {
                    "dataset": ev_cfg.dataset,
                    "num_problems": ev_cfg.num_problems,
                    "use_fewshot": ev_cfg.use_fewshot,
                }
                for ev_cfg in config.evaluators
            ],
        },
        "metrics": metrics,
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")


async def main(config: EvalConfig):
    """Main evaluation function."""
    console.print("\n[bold cyan]Multi-Dataset Evaluation Script[/bold cyan]\n")

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

    # Build evaluators from configurations
    logger.info(f"Building {len(config.evaluators)} evaluators...")
    evaluators = []
    for ev_cfg in config.evaluators:
        evaluator = ev_cfg.build(
            model_name=config.model_name,
            renderer_name=config.renderer_name,
            max_tokens=config.max_tokens,
            group_size=config.group_size,
        )
        evaluators.append(evaluator)
        logger.info(f"  - {_get_evaluator_name(evaluator)}: {ev_cfg.dataset}")

    # Run evaluations in parallel
    console.print("\n[bold yellow]Running evaluations in parallel...[/bold yellow]\n")
    metrics = await run_evaluations_parallel(
        evaluators=evaluators,
        sampling_client=sampling_client,
        log_path=config.log_path,
        num_groups_to_log=config.num_groups_to_log,
    )

    # Display results
    console.print("\n")
    print_metrics_table(metrics, title="Multi-Dataset Evaluation Results")

    # Save results
    if log_dir and config.save_detailed_results:
        save_results(metrics, log_dir, config)

    # Print summary
    console.print("\n[bold green]Evaluation Complete![/bold green]")
    
    # Extract aggregate metrics for summary
    aggregate_correct = metrics.get('aggregate/correct', 0.0)
    aggregate_format = metrics.get('aggregate/format', 0.0)
    aggregate_reward = metrics.get('aggregate/reward/total', 0.0)
    
    console.print(
        f"\n[bold]Aggregate Results (across all datasets):[/bold]\n"
        f"  • Average Accuracy: {aggregate_correct:.2%}\n"
        f"  • Average Format Valid: {aggregate_format:.2%}\n"
        f"  • Average Reward: {aggregate_reward:.4f}\n"
        f"  • Datasets Evaluated: {len(evaluators)}\n"
    )
    
    # Print per-dataset summaries
    console.print("\n[bold]Per-Dataset Summary:[/bold]")
    for evaluator in evaluators:
        ev_name = _get_evaluator_name(evaluator)
        correct = metrics.get(f'{ev_name}/correct', 0.0)
        format_valid = metrics.get(f'{ev_name}/format', 0.0)
        reward = metrics.get(f'{ev_name}/reward/total', 0.0)
        console.print(
            f"  • {ev_name}: Accuracy={correct:.2%}, Format={format_valid:.2%}, Reward={reward:.4f}"
        )

    return metrics


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
