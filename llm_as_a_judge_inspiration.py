from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Optional
import json
import json

"""
/home/hpoonia/CuriousLLMs/llm_as_a_judge.py

Template classes for "LLM as a judge".
Provides an abstract base LLMJudge and two concrete templates:
- TinkerJudge
- GeminiJudge

These templates are lightweight and expect an injected `model_client`
callable with signature: model_client(prompt: str, **kwargs) -> str
"""



ModelClient = Callable[[str, Dict[str, Any]], str]
CaseData = Dict[str, Any]
EvalResult = Dict[str, Any]


class LLMJudge(ABC):
    """
    Abstract base for LLM-based judges.

    model_client: a callable that accepts (prompt: str, options: dict) and returns the model text response.
    """

    def __init__(self, model_client: Optional[ModelClient] = None, default_options: Optional[Dict[str, Any]] = None):
        self.model_client = model_client
        self.default_options = default_options or {}

    def _call_llm(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        if not self.model_client:
            raise RuntimeError("No model_client provided. Inject a callable to send prompts to an LLM.")
        opts = dict(self.default_options)
        if options:
            opts.update(options)
        return self.model_client(prompt, opts)

    @abstractmethod
    def prepare_prompt(self, case: CaseData) -> str:
        """Build the prompt for the LLM from the given case data."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, case: CaseData) -> EvalResult:
        """
        Evaluate the case and return a structured result.
        Expected keys: verdict (str), score (float), explanation (str)
        """
        raise NotImplementedError

    def score_text(self, text: str, scale: float = 1.0) -> float:
        """
        Optional helper to derive a numeric score from freeform text.
        Default naive implementation maps presence of keywords to a score.
        Subclasses can override for more advanced heuristics or LLM-based scoring.
        """
        lowered = text.lower()
        if "strong" in lowered or "convinc" in lowered:
            return 1.0 * scale
        if "weak" in lowered or "uncertain" in lowered:
            return 0.5 * scale
        return 0.0


class TinkerJudge(LLMJudge):
    """
    A conservative, structured judge template.
    Produces short verdicts and a justification with bullet points.
    """

    PROMPT_TEMPLATE = (
        "You are a precise and conservative judge. Given the case below, produce:\n"
        "1) verdict: one-word label (Guilty/Not Guilty/Undetermined)\n"
        "2) score: float between 0.0 and 1.0 representing confidence\n"
        "3) explanation: 2-5 concise bullet points with reasons and evidence.\n\n"
        "Case:\n{case_text}\n\nRespond in JSON with keys: verdict, score, explanation."
    )

    def prepare_prompt(self, case: CaseData) -> str:
        case_text = case.get("text") or str(case)
        return self.PROMPT_TEMPLATE.format(case_text=case_text)

    def evaluate(self, case: CaseData) -> EvalResult:
        prompt = self.prepare_prompt(case)
        response = self._call_llm(prompt)
        # The template expects JSON. We keep parsing minimal and tolerant.
        try:
            parsed = json.loads(response)
            # enforce expected keys
            verdict = parsed.get("verdict", "Undetermined")
            score = float(parsed.get("score", 0.0))
            explanation = parsed.get("explanation", "")
            return {"verdict": verdict, "score": score, "explanation": explanation, "raw": response}
        except Exception:
            # Fallback: wrap raw response
            return {"verdict": "Undetermined", "score": 0.0, "explanation": response, "raw": response}


class GeminiJudge(LLMJudge):
    """
    A more creative/flexible judge template inspired by multi-step reasoning.
    Produces a verdict with chain-of-thought style explanation split into sections:
    - Facts
    - Inference
    - Conclusion
    """

    PROMPT_TEMPLATE = (
        "You are an LLM judge that reasons step-by-step. For the case below, output a JSON object with:\n"
        "facts: list of extracted facts\n"
        "inference: short chain-of-thought (concise)\n"
        "verdict: Accept / Reject / Inconclusive\n"
        "confidence: 0.0-1.0\n\n"
        "Case:\n{case_text}\n\nRespond ONLY with JSON."
    )

    def prepare_prompt(self, case: CaseData) -> str:
        # Allow optional instructions override from case
        extras = case.get("instructions", "")
        base = self.PROMPT_TEMPLATE.format(case_text=case.get("text") or str(case))
        if extras:
            base += "\n\nAdditional instructions:\n" + extras
        return base

    def evaluate(self, case: CaseData) -> EvalResult:
        prompt = self.prepare_prompt(case)
        response = self._call_llm(prompt)
        try:
            parsed = json.loads(response)
            verdict = parsed.get("verdict", "Inconclusive")
            confidence = float(parsed.get("confidence", parsed.get("confidence_score", 0.0)))
            facts = parsed.get("facts", [])
            inference = parsed.get("inference", "")
            explanation = f"Facts: {facts}\nInference: {inference}"
            return {"verdict": verdict, "score": confidence, "explanation": explanation, "raw": response}
        except Exception:
            return {"verdict": "Inconclusive", "score": 0.0, "explanation": response, "raw": response}


# Minimal usage example (replace `my_model_client` with your LLM caller):
#
# def my_model_client(prompt: str, options: dict) -> str:
#     # send prompt to your LLM and return text response
#     ...
#
# judge = TinkerJudge(model_client=my_model_client)
# result = judge.evaluate({"text": "Defendant A was seen leaving the scene at 10pm."})
# print(result)