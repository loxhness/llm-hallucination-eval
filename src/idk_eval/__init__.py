"""idk-eval: Test whether your LLM knows when to abstain."""

__version__ = "0.1.0"

from idk_eval.dataset import Dataset, Question
from idk_eval.evaluator import Evaluator, Results, RunResult
from idk_eval.scorer import Score, score_response
from idk_eval.strategy import Strategy

__all__ = [
    "Dataset",
    "Question",
    "Evaluator",
    "Results",
    "RunResult",
    "Score",
    "score_response",
    "Strategy",
]
