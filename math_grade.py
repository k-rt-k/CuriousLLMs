import sys
sys.path.append('./tinker-cookbook/tinker_cookbook/')
from functools import partial
from typing import Literal
import tinker
import chz

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
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    
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
        )

@chz.chz
class CustomMathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0

    async def __call__(self) -> tuple[CustomMathDataset, CustomMathDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = CustomMathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            CustomMathDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


def get_math_dataset_builder(
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
) -> RLDatasetBuilder:
    """
    Get math dataset builder. Returns CustomMathDatasetBuilder for "math" dataset.
    Args:
        dataset_name: Should be "math" for this function
        batch_size: Number of groups per batch
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
        seed: Random seed for data shuffling (default: 0)
    Returns:
        CustomMathDatasetBuilder instance
    """
    if dataset_name == "math":
        return CustomMathDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name_for_tokenizer,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown math dataset: {dataset_name}. This function only supports 'math' dataset."
        )
