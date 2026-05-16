"""Main Evaluator class — orchestrates running strategies over a dataset."""

from dataclasses import dataclass
from typing import List

import pandas as pd
from tqdm import tqdm

from idk_eval.dataset import Dataset
from idk_eval.models import call_model
from idk_eval.scorer import Score, score_response
from idk_eval.strategy import Strategy


@dataclass
class RunResult:
    strategy_name: str
    scores: List[Score]


class Results:
    """Holds and summarizes results from one or more strategies."""

    def __init__(self, runs: List[RunResult], dataset: Dataset):
        self.runs = runs
        self.dataset = dataset

    def summary(self) -> str:
        """Return a printable summary table."""
        lines = ["Strategy comparison:\n"]
        for run in self.runs:
            total = len(run.scores)
            labels = [s.label for s in run.scores]
            lines.append(f"  Strategy: {run.strategy_name}")
            for label in ["correct", "abstained", "hallucinated", "incorrect"]:
                count = labels.count(label)
                pct = 100 * count / total if total else 0
                lines.append(f"    {label:15s} {count:4d} ({pct:.1f}%)")
            lines.append("")
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """Export raw scores to CSV."""
        rows = []
        for run in self.runs:
            for score in run.scores:
                rows.append({
                    "strategy": run.strategy_name,
                    "question_id": score.question_id,
                    "label": score.label,
                    "response": score.raw_response,
                    "notes": score.notes,
                })
        pd.DataFrame(rows).to_csv(path, index=False)


class Evaluator:
    """Runs a set of strategies over a dataset against a given model."""

    def __init__(
        self,
        model: str,
        strategies: List[Strategy],
        temperature: float = 0.0,
    ):
        self.model = model
        self.strategies = strategies
        self.temperature = temperature

    def run(self, dataset: Dataset) -> Results:
        runs = []
        for strategy in self.strategies:
            scores = []
            iterator = tqdm(
                dataset.questions,
                desc=f"[{strategy.name}]",
            )
            for question in iterator:
                response = call_model(
                    model=self.model,
                    system_prompt=strategy.system_prompt,
                    user_prompt=question.text,
                    temperature=self.temperature,
                )
                scores.append(score_response(question, response))
            runs.append(RunResult(strategy_name=strategy.name, scores=scores))
        return Results(runs=runs, dataset=dataset)
