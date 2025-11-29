import sys
sys.path.append('./tinker-cookbook/tinker_cookbook/')
from functools import partial
from typing import Literal, Sequence, cast, Any
import math
import os
import json
import random
import tinker
import chz

from datasets import load_dataset, Dataset
from tinker_cookbook.recipes.math_rl.math_env import MathEnv, safe_grade, MathDataset
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed
)
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


#TODO:rename
class CustomMathEnv(MathEnv):
    def __init__(self, solution: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solution = solution

    def check_answer(self, sample_str: str) -> float:  # type: ignore[override]
        """
        Override to return float reward instead of bool.
        
        The parent class returns bool, but since problem_env.py's step() function
        converts it to float anyway (float(self.check_answer(...))), returning
        float directly works correctly at runtime and provides more granular rewards.
        """
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return 0.0
        
        return float(safe_grade(
                answer, self.answer, self.grader, self.timeout
        ))
    
    def get_reference_answer(self) -> str:
        return self.solution
    
    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]: #TODO: Come up with better fewshot prefix?
        return [
            {
                "role": "user",
                "content": "How many r's are in strawberry?" + MathEnv.question_suffix(),
            },
            {
                "role": "assistant",
                "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
            },
        ]


class CustomMathDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        train_size: int | None = None,  # None means use all samples
    ):
        # Call parent constructor to load and shuffle the dataset
        super().__init__(
            batch_size=batch_size,
            group_size=group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            split=split,
            seed=seed,
        )
        
        # For train split, optionally limit to train_size samples
        if split == "train" and train_size is not None:
            if train_size > len(self.ds):
                logger.warning(
                    f"Requested train_size={train_size} but only {len(self.ds)} samples available. "
                    f"Using all {len(self.ds)} samples."
                )
            else:
                # Select first train_size samples (deterministic since dataset was shuffled with seed)
                self.ds = self.ds.select(range(train_size))
                logger.info(f"CustomMathDataset (train): Limited to {train_size} samples")
    
    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            answer = extract_boxed(x["solution"])
        except ValueError:  # not sure if this happens
            logger.warning(f"No answer found for {x['solution']}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                CustomMathEnv, problem=x["problem"], solution=x["solution"], answer=answer, renderer=self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="math",
        )

@chz.chz
class CustomMathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    train_size: int | None = None  # None means use all ~12000 samples

    async def __call__(self) -> tuple[CustomMathDataset, CustomMathDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = CustomMathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        train_dataset = CustomMathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            split="train",
            seed=self.seed,
            train_size=self.train_size,
        )
        
        test_dataset = CustomMathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            split="test",
            seed=self.seed,
            train_size=None,  # Always use full test set (500 samples)
        )
        
        return (train_dataset, test_dataset)


def get_math_dataset_builder(
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
    train_size: int | None = None,  # For "math": None = all ~12000, for "deepmath": ignored (uses 8000)
    # Additional parameters for mixed dataset
    math_train_size: int | None = None,
    deepmath_train_size: int = 8000,
    deepmath_test_size: int = 500,
    deepmath_seed: int = 42,
) -> RLDatasetBuilder:
    """
    Get math dataset builder.
    
    Args:
        dataset_name: "math", "deepmath", or "mixed"
        batch_size: Number of groups per batch
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
        seed: Random seed for data shuffling (default: 0)
        train_size: Number of training samples to use. For "math" dataset,
                   None means use all ~12000 samples. For "deepmath", this
                   parameter is ignored (always uses 8000).
        math_train_size: For "mixed" mode - size of Math training set (None = all)
        deepmath_train_size: For "mixed" mode - size of DeepMath training set
        deepmath_test_size: For "mixed" mode - size of DeepMath test set
        deepmath_seed: For "mixed" mode - seed for DeepMath index creation
        math_ratio: For "mixed" mode - fraction of samples from Math (0.5 = equal)
    Returns:
        Dataset builder instance
    """
    if dataset_name == "math":
        return CustomMathDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name_for_tokenizer,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
            train_size=train_size,
        )
    elif dataset_name == "deepmath":
        return CustomDeepMathDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name_for_tokenizer,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    elif dataset_name == "mixed":
        return MixedMathDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name_for_tokenizer,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
            math_train_size=math_train_size,
            deepmath_train_size=deepmath_train_size,
            deepmath_test_size=deepmath_test_size,
            deepmath_seed=deepmath_seed,
        )
    else:
        raise ValueError(
            f"Unknown math dataset: {dataset_name}. Available: 'math', 'deepmath', 'mixed'"
        )


# ======================================================================
# Custom DeepMath Dataset Implementation
# ======================================================================

# Default path for storing indices
DEEPMATH_INDICES_DIR = os.path.join(os.path.dirname(__file__), "deepmath_indices")

# Difficulty levels to include (6.0, 6.5, 7.0, 7.5, 8.0, 8.5)
DEEPMATH_TARGET_DIFFICULTIES = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5]

# Original counts from deepmath-counts.csv for difficulties 6-8.5
DEEPMATH_DIFFICULTY_COUNTS = {
    8.5: 3989,
    8.0: 11686,
    7.5: 6746,
    7.0: 8561,
    6.5: 7750,
    6.0: 17488,
}

# Total samples in the target difficulty range
DEEPMATH_TOTAL_IN_RANGE = sum(DEEPMATH_DIFFICULTY_COUNTS.values())  # 56220


class CustomDeepMathEnv(MathEnv):
    """
    Custom environment for DeepMath problems.
    Uses r1_solution_1 as the reference solution for reasoning evaluation.
    """
    def __init__(self, solution: str, difficulty: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solution = solution
        self.difficulty = difficulty

    def check_answer(self, sample_str: str) -> float:  # type: ignore[override]
        """
        Override to return float reward instead of bool.
        """
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return 0.0
        
        return float(safe_grade(
            answer, self.answer, self.grader, self.timeout
        ))
    
    def get_reference_answer(self) -> str:
        return self.solution
    
    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": "How many r's are in strawberry?" + MathEnv.question_suffix(),
            },
            {
                "role": "assistant",
                "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
            },
        ]


def _compute_samples_per_difficulty(total_samples: int) -> dict[float, int]:
    """
    Compute the number of samples to draw from each difficulty level
    to maintain the original ratios.
    
    Args:
        total_samples: Total number of samples to draw
        
    Returns:
        Dictionary mapping difficulty level to sample count
    """
    samples_per_diff = {}
    remaining = total_samples
    
    # Calculate samples for each difficulty maintaining ratios
    sorted_diffs = sorted(DEEPMATH_DIFFICULTY_COUNTS.keys())
    for i, diff in enumerate(sorted_diffs):
        ratio = DEEPMATH_DIFFICULTY_COUNTS[diff] / DEEPMATH_TOTAL_IN_RANGE
        if i == len(sorted_diffs) - 1:
            # Last difficulty gets remaining samples to ensure exact total
            samples_per_diff[diff] = remaining
        else:
            count = round(total_samples * ratio)
            samples_per_diff[diff] = count
            remaining -= count
    
    return samples_per_diff


def _create_deepmath_indices(
    seed: int = 42,
    train_size: int = 8000,
    test_size: int = 500,
    indices_dir: str = DEEPMATH_INDICES_DIR,
    force_recreate: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Create or load train/test indices for the DeepMath dataset.
    
    Ensures:
    1. Train and test splits are completely disjoint
    2. Only difficulties 6.0-8.5 are included
    3. Difficulty ratios are maintained in both splits
    4. Indices are saved for reproducibility
    
    Args:
        seed: Random seed for reproducibility
        train_size: Number of training samples (default: 8000)
        test_size: Number of test samples (default: 500)
        indices_dir: Directory to save/load indices
        force_recreate: If True, recreate indices even if file exists
        
    Returns:
        Tuple of (train_indices, test_indices)
    """
    os.makedirs(indices_dir, exist_ok=True)
    indices_file = os.path.join(
        indices_dir, 
        f"deepmath_indices_train{train_size}_test{test_size}_seed{seed}.json"
    )
    
    # Try to load existing indices
    if not force_recreate and os.path.exists(indices_file):
        logger.info(f"Loading existing DeepMath indices from {indices_file}")
        with open(indices_file, 'r') as f:
            data = json.load(f)
        return data['train_indices'], data['test_indices']
    
    logger.info(f"Creating new DeepMath indices (train={train_size}, test={test_size}, seed={seed})")
    
    # Load the full dataset
    full_ds = cast(Dataset, load_dataset("zwhe99/DeepMath-103K", split="train"))
    
    # Group indices by difficulty
    difficulty_to_indices: dict[float, list[int]] = {diff: [] for diff in DEEPMATH_TARGET_DIFFICULTIES}
    
    for idx in range(len(full_ds)):
        diff = full_ds[idx]['difficulty']
        if diff in difficulty_to_indices:
            difficulty_to_indices[diff].append(idx)
    
    # Verify counts match expected
    for diff, indices in difficulty_to_indices.items():
        expected = DEEPMATH_DIFFICULTY_COUNTS[diff]
        actual = len(indices)
        if actual != expected:
            logger.warning(f"Difficulty {diff}: expected {expected} samples, got {actual}")
    
    # Set random seed for reproducibility
    rng = random.Random(seed)
    
    # Shuffle indices within each difficulty
    for diff in difficulty_to_indices:
        rng.shuffle(difficulty_to_indices[diff])
    
    # Calculate samples per difficulty for train and test
    train_per_diff = _compute_samples_per_difficulty(train_size)
    test_per_diff = _compute_samples_per_difficulty(test_size)
    
    train_indices = []
    test_indices = []
    
    for diff in DEEPMATH_TARGET_DIFFICULTIES:
        available = difficulty_to_indices[diff]
        n_train = train_per_diff[diff]
        n_test = test_per_diff[diff]
        
        # Ensure we have enough samples
        total_needed = n_train + n_test
        if len(available) < total_needed:
            raise ValueError(
                f"Not enough samples for difficulty {diff}: "
                f"need {total_needed}, have {len(available)}"
            )
        
        # Take first n_train for train, next n_test for test (no overlap)
        train_indices.extend(available[:n_train])
        test_indices.extend(available[n_train:n_train + n_test])
    
    # Shuffle final indices
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    
    # Verify no overlap
    train_set = set(train_indices)
    test_set = set(test_indices)
    overlap = train_set & test_set
    if overlap:
        raise ValueError(f"Train and test sets have {len(overlap)} overlapping indices!")
    
    # Save indices to file
    indices_data = {
        'train_indices': train_indices,
        'test_indices': test_indices,
        'metadata': {
            'seed': seed,
            'train_size': train_size,
            'test_size': test_size,
            'target_difficulties': DEEPMATH_TARGET_DIFFICULTIES,
            'train_per_difficulty': {str(k): v for k, v in train_per_diff.items()},
            'test_per_difficulty': {str(k): v for k, v in test_per_diff.items()},
        }
    }
    
    with open(indices_file, 'w') as f:
        json.dump(indices_data, f, indent=2)
    
    logger.info(f"Saved DeepMath indices to {indices_file}")
    logger.info(f"  Train: {len(train_indices)} samples")
    logger.info(f"  Test: {len(test_indices)} samples")
    
    return train_indices, test_indices


class CustomDeepMathDataset(RLDataset):
    """
    Custom DeepMath dataset with:
    - Disjoint train/test splits
    - Only difficulties 6.0-8.5
    - Maintained difficulty ratios
    - Fixed indices for reproducibility
    """
    
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 42,
        train_size: int = 8000,
        test_size: int = 500,
    ):
        # Get or create indices
        train_indices, test_indices = _create_deepmath_indices(
            seed=seed,
            train_size=train_size,
            test_size=test_size,
        )
        
        # Load the full dataset
        full_ds = cast(Dataset, load_dataset("zwhe99/DeepMath-103K", split="train"))
        
        # Select the appropriate indices
        if split == "train":
            indices = train_indices
        else:
            indices = test_indices
        
        # Create a subset dataset
        self.ds = full_ds.select(indices)
        
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.split = split
        
        logger.info(f"CustomDeepMathDataset ({split}): {len(self.ds)} samples, "
                   f"batch_size={batch_size}, group_size={self.group_size}")

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(cast(dict[str, Any], row), self.group_size)) is not None
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, Any], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract fields from DeepMath format
        problem = x.get("question", "")
        answer = x.get("final_answer", "")
        difficulty = float(x.get("difficulty", 0.0))
        # Use r1_solution_1 as the reference solution for reasoning evaluation
        solution = x.get("r1_solution_1", "")
        
        if not (problem and answer):
            logger.warning(f"Missing problem or answer in DeepMath entry")
            return None
        
        return ProblemGroupBuilder(
            env_thunk=partial(
                CustomDeepMathEnv,
                solution=solution,
                difficulty=difficulty,
                problem=problem,
                answer=answer,
                renderer=self.renderer,
                convo_prefix=self.convo_prefix,
            ),
            num_envs=group_size,
            dataset_name="deepmath",
        )


@chz.chz
class CustomDeepMathDatasetBuilder(RLDatasetBuilder):
    """
    Builder for CustomDeepMathDataset.
    
    Creates train (8000 samples) and test (500 samples) splits with:
    - Only difficulties 6.0-8.5
    - Maintained difficulty ratios
    - Disjoint splits (no overlap)
    - Fixed indices for reproducibility
    """
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 42
    train_size: int = 8000
    test_size: int = 500

    async def __call__(self) -> tuple[CustomDeepMathDataset, CustomDeepMathDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = CustomDeepMathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        train_dataset = CustomDeepMathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            split="train",
            seed=self.seed,
            train_size=self.train_size,
            test_size=self.test_size,
        )
        
        test_dataset = CustomDeepMathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            split="test",
            seed=self.seed,
            train_size=self.train_size,
            test_size=self.test_size,
        )
        
        return (train_dataset, test_dataset)


# ======================================================================
# Mixed Math + DeepMath Dataset Implementation
# ======================================================================

class MixedMathDataset(RLDataset):
    """
    A dataset that combines samples from both Math and DeepMath datasets,
    shuffling them together so each batch can contain samples from both.
    
    Note: The ratio of Math to DeepMath samples is determined by the
    sizes of the input datasets. Individual batches may have varying
    ratios, but the overall epoch will reflect the dataset size ratio.
    """
    
    def __init__(
        self,
        math_dataset: CustomMathDataset,
        deepmath_dataset: CustomDeepMathDataset,
        batch_size: int,
        seed: int = 0,
    ):
        """
        Args:
            math_dataset: The Math dataset
            deepmath_dataset: The DeepMath dataset
            batch_size: Number of groups per batch
            seed: Random seed for shuffling
        """
        self.math_dataset = math_dataset
        self.deepmath_dataset = deepmath_dataset
        self.batch_size = batch_size
        self.seed = seed
        
        # Collect all env group builders from both datasets
        self._all_builders: list[tuple[EnvGroupBuilder, str]] = []  # (builder, source)
        
        # Get all builders from math dataset
        for i in range(len(math_dataset)):
            for builder in math_dataset.get_batch(i):
                self._all_builders.append((builder, "math"))
        
        # Get all builders from deepmath dataset
        for i in range(len(deepmath_dataset)):
            for builder in deepmath_dataset.get_batch(i):
                self._all_builders.append((builder, "deepmath"))
        
        # Shuffle the combined list
        rng = random.Random(seed)
        rng.shuffle(self._all_builders)
        
        logger.info(f"MixedMathDataset: Combined {len(self._all_builders)} samples "
                   f"(Math: {sum(1 for _, s in self._all_builders if s == 'math')}, "
                   f"DeepMath: {sum(1 for _, s in self._all_builders if s == 'deepmath')})")
    
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of env group builders from the shuffled combined dataset."""
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self._all_builders))
        
        if batch_start >= len(self._all_builders):
            raise IndexError(f"Batch index {index} out of range")
        
        # Return just the builders (not the source labels)
        return [builder for builder, _ in self._all_builders[batch_start:batch_end]]
    
    def __len__(self) -> int:
        return math.ceil(len(self._all_builders) / self.batch_size)


@chz.chz
class MixedMathDatasetBuilder(RLDatasetBuilder):
    """
    Builder for mixed Math + DeepMath training.
    
    Creates a combined training dataset that shuffles samples from both
    Math and DeepMath, and returns BOTH test datasets for separate evaluation.
    
    The __call__ method returns:
    - Training: MixedMathDataset (shuffled combination)
    - Testing: List of test datasets [MathTest, DeepMathTest]
    
    Note: This builder returns a list of test datasets instead of a single
    test dataset, which requires corresponding handling in the training loop.
    """
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    
    # Math dataset config
    math_train_size: int | None = None  # None = use all ~12000 samples
    
    # DeepMath dataset config
    deepmath_train_size: int = 8000
    deepmath_test_size: int = 500
    deepmath_seed: int = 42  # Separate seed for DeepMath index creation

    async def __call__(self) -> tuple[MixedMathDataset, list[RLDataset]]:  # type: ignore[override]
        """
        Build the mixed training dataset and separate test datasets.
        
        Returns:
            Tuple of (train_dataset, [math_test, deepmath_test])
            
        Note: This returns a list of test datasets instead of a single optional dataset,
        which is handled specially in the training loop.
        """
        # Determine convo_prefix
        if self.convo_prefix == "standard":
            # Use the same few-shot prefix for both (they're compatible)
            convo_prefix = CustomMathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        # Create Math datasets
        math_train = CustomMathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            split="train",
            seed=self.seed,
            train_size=self.math_train_size,
        )
        
        math_test = CustomMathDataset(
            batch_size=self.batch_size,
            group_size=1,  # Test with group_size=1 for proper evaluation
            renderer=renderer,
            convo_prefix=convo_prefix,
            split="test",
            seed=self.seed,
            train_size=None,
        )
        
        # Create DeepMath datasets
        deepmath_train = CustomDeepMathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            split="train",
            seed=self.deepmath_seed,
            train_size=self.deepmath_train_size,
            test_size=self.deepmath_test_size,
        )
        
        deepmath_test = CustomDeepMathDataset(
            batch_size=self.batch_size,
            group_size=1,  # Test with group_size=1 for proper evaluation
            renderer=renderer,
            convo_prefix=convo_prefix,
            split="test",
            seed=self.deepmath_seed,
            train_size=self.deepmath_train_size,
            test_size=self.deepmath_test_size,
        )
        
        # Create mixed training dataset
        mixed_train = MixedMathDataset(
            math_dataset=math_train,
            deepmath_dataset=deepmath_train,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        
        logger.info(f"Created MixedMathDataset:")
        logger.info(f"  - Training: {len(mixed_train)} batches (mixed from Math + DeepMath)")
        logger.info(f"  - Math Test: {len(math_test)} batches ({len(math_test) * self.batch_size} samples)")
        logger.info(f"  - DeepMath Test: {len(deepmath_test)} batches ({len(deepmath_test) * self.batch_size} samples)")
        
        # Return train dataset and list of test datasets
        return (mixed_train, [math_test, deepmath_test])
