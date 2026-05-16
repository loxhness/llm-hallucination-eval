"""Command-line interface for idk-eval."""

import click
from dotenv import load_dotenv

from idk_eval import Evaluator, Dataset, Strategy
from idk_eval.paths import project_root

load_dotenv(project_root() / ".env")

STRATEGY_MAP = {
    "baseline": Strategy.baseline,
    "abstain": Strategy.abstain_if_unsure,
    "abstain_if_unsure": Strategy.abstain_if_unsure,
    "cite_or_abstain": Strategy.cite_or_abstain,
    "chain_of_thought": Strategy.chain_of_thought,
    "confident": Strategy.confident,
}


def _resolve_strategies(strategy: str) -> list[Strategy]:
    names = [s.strip() for s in strategy.split(",") if s.strip()]
    unknown = [n for n in names if n not in STRATEGY_MAP]
    if unknown:
        available = ", ".join(sorted(STRATEGY_MAP))
        raise click.ClickException(
            f"Unknown strateg(ies): {', '.join(unknown)}. Available: {available}"
        )
    return [STRATEGY_MAP[name]() for name in names]


@click.group()
def main():
    """idk-eval: Test whether your LLM knows when to abstain."""
    pass


@main.command()
@click.option("--model", required=True, help="LiteLLM model string, e.g. anthropic/claude-sonnet-4")
@click.option(
    "--strategy",
    default="baseline",
    help="Comma-separated strategies: baseline, abstain_if_unsure, cite_or_abstain, ...",
)
@click.option("--dataset", default="builtin:mixed_v1", help="Dataset: builtin:<name> or path to JSONL")
@click.option("--out", default="results.csv", help="Output CSV path")
def run(model, strategy, dataset, out):
    """Run an evaluation (generate responses + rule-based scoring)."""
    if dataset.startswith("builtin:"):
        ds = Dataset.builtin(dataset.split(":", 1)[1])
    else:
        ds = Dataset.from_jsonl(dataset)

    strategies = _resolve_strategies(strategy)
    evaluator = Evaluator(model=model, strategies=strategies)
    results = evaluator.run(ds)

    print(results.summary())
    results.to_csv(out)
    click.echo(f"Detailed results saved to: {out}")


@main.command("download-truthfulqa")
@click.option("--output", default=None, help="Output JSONL path")
@click.option("--target", default=175, show_default=True, help="Approximate number of questions")
def download_truthfulqa(output, target):
    """Download a TruthfulQA subset in package JSONL format."""
    from pathlib import Path

    from idk_eval.dataset import download_truthfulqa as _download

    path = _download(output=Path(output) if output else None, target=target)
    click.echo(f"Wrote {path}")


if __name__ == "__main__":
    main()
