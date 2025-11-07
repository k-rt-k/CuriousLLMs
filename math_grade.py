
import tinker
from llm_as_a_judge import GeminiJudge, TinkerJudge

import sys
sys.path.append('tinker-cookbook')
from tinker_cookbook.recipes.math_rl.math_env import MathEnv

#TODO:rename
class CustomMathEnv(MathEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ## initialize hyperparam (weighting between terminal and reasoning)
        ## initialize judge
        self.judge = TinkerJudge() if config.use_tinker else GeminiJudge()
        self.reasoning_weight = ???

    def grade_answer(self, question:str, answer: str)->float:
        return self.judge.grade(???)*self.reasoning_weight + float(self.check_answer(answer))


