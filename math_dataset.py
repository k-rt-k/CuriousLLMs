import math
from functools import partial
from typing import Literal, Sequence

from datasets import Dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_env import (
    _get_hendrycks_math_test,
    _get_hendrycks_math_train,
)
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset
from math_grade import CustomMathEnv

#FIXME: CONNECT THIS TO OTHER THINGS IN THE SUBMODULE SO THAT THIS IS INVOKED INSTEAD OF THE DEFAULT MATH DATASET
class CustomMathDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        if split == "train":
            self.ds = _get_hendrycks_math_train().shuffle(seed=seed)
        elif split == "test":
            self.ds = _get_hendrycks_math_test()
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

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
                CustomMathEnv, solution=x['solution'], use_tinker=False, problem=x["problem"], answer=answer, renderer=self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
        )