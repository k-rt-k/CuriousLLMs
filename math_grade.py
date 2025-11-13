import tinker
from llm_as_a_judge import GeminiJudge, TinkerJudge, JudgeResult

import sys
sys.path.append('./tinker-cookbook/tinker_cookbook/')
from tinker_cookbook.recipes.math_rl.math_env import MathEnv, safe_grade
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed
)

verdict_to_reward = {
    "CORRECT": 1.0,
    "INCORRECT": 0.0,
    "UNCERTAIN": 0.5,
}

#TODO:rename
class CustomMathEnv(MathEnv):
    def __init__(self, solution: str, use_tinker: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solution = solution
        ## initialize hyperparam (weighting between terminal and reasoning)
        ## initialize judge
        self.judge = TinkerJudge() if use_tinker else GeminiJudge() # FIXME: NEED DEFAULT SYSTEM PROMPT, POTENTIALLY BASED ON DATASET?
        self.reasoning_weight = 0.8

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

        judge_result: JudgeResult = self.judge.judge_single(
            self.get_question(),
            sample_str,
            self.solution
        )
        
        judge_reward = verdict_to_reward.get(judge_result.verdict, 0.0)  # Default to 0.0 for unknown verdicts
        
        return self.reasoning_weight * judge_reward + float(safe_grade(
                answer, self.answer, self.grader, self.timeout
        )) * (1 - self.reasoning_weight)


