"""
Full hallucination-eval pipeline in one command.

    python eval.py                                       # baseline, model from .env
    python eval.py --all-conditions
    python eval.py --models anthropic:claude-haiku-4-5-20251001 openai:gpt-4o-mini --all-conditions
    python eval.py --dataset data/questions_truthfulqa.jsonl --all-conditions
    python eval.py --limit 10 --all-conditions
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# ── subprocess helpers ────────────────────────────────────────────────────────

def _step(label: str) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  {label}")
    print(f"{'─' * width}")


def _run(label: str, cmd: list[str]) -> None:
    """Print a step header, run the command live, abort the pipeline on failure."""
    _step(label)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(
            f"\nAborted: '{label}' exited with code {result.returncode}.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


# ── terminal summary ──────────────────────────────────────────────────────────

def _print_summary(summary_path: Path) -> None:
    """Read summary.csv and print a compact results table per model."""
    import pandas as pd

    if not summary_path.exists():
        print(f"(summary not found at {summary_path})")
        return

    df = pd.read_csv(summary_path)

    # One aggregated row per model across all conditions.
    agg = (
        df.groupby("model_name")[["accuracy", "hallucination_rate", "abstain_rate"]]
        .mean()
        .reset_index()
        .sort_values("model_name")
    )

    model_col = max(agg["model_name"].str.len().max(), len("Model")) + 2
    width = model_col + 30

    print(f"\n{'═' * width}")
    print("  RESULTS SUMMARY")
    print(f"{'═' * width}")
    print(
        f"  {'Model':<{model_col}}"
        f"{'Accuracy':>10}"
        f"{'Halluc.':>10}"
        f"{'Abstain':>10}"
    )
    print(f"  {'─' * (width - 2)}")
    for _, row in agg.iterrows():
        print(
            f"  {row['model_name']:<{model_col}}"
            f"  {row['accuracy']:>7.1%}"
            f"  {row['hallucination_rate']:>7.1%}"
            f"  {row['abstain_rate']:>7.1%}"
        )
    print(f"{'═' * width}")
    print(f"  Full breakdown: {summary_path}")
    print(f"{'═' * width}\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full eval pipeline: generate → score → analyze.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Generation
    parser.add_argument("--provider", default=None, help="openai | anthropic (default: from env)")
    parser.add_argument("--model", default=None, help="Single model name (default: from env)")
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="[PROVIDER:]MODEL",
        help=(
            "One or more models. Each entry is a bare model name (uses --provider) "
            "or 'provider:model', e.g. 'anthropic:claude-haiku-4-5-20251001'. "
            "Overrides --model."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="PATH",
        help="Questions JSONL to evaluate (default: data/questions.jsonl).",
    )
    parser.add_argument(
        "--condition",
        choices=["baseline", "abstain", "cite_or_abstain", "chain_of_thought", "confident"],
        default=None,
        help="Single prompting condition (default: baseline).",
    )
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run all five prompting conditions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only evaluate the first N questions from the dataset (passed through to run_eval).",
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help=(
            "Skip step 1 (run_eval.py) and re-score the existing raw_generations.jsonl. "
            "Useful for changing the judge model without re-calling the generation model."
        ),
    )

    # Scoring (defaults are independent of --provider / LLM_PROVIDER so OpenAI runs are not judged
    # with a Claude model id on the wrong API.)
    parser.add_argument(
        "--judge-provider",
        default=os.getenv("JUDGE_PROVIDER", "anthropic"),
        choices=["anthropic", "openai"],
        help="Provider for the LLM judge (default: env JUDGE_PROVIDER or anthropic).",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001"),
        help="Model for the LLM judge (default: env JUDGE_MODEL or claude-haiku-4-5-20251001).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip re-judging rows already present in scored.csv.",
    )

    # Paths
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "results",
        type=Path,
        metavar="DIR",
        help="Root directory for all pipeline outputs (default: results/).",
    )

    args = parser.parse_args()

    out = Path(args.output_dir)
    raw_path = out / "raw_generations.jsonl"
    scored_path = out / "scored.csv"
    summary_path = out / "summary.csv"
    plots_dir = out / "plots"

    py = sys.executable

    # ── 1 / 3  generate ──────────────────────────────────────────────────────
    if args.skip_eval:
        if not raw_path.exists():
            print(f"Error: --skip-eval requires {raw_path} to exist.", file=sys.stderr)
            sys.exit(1)
        _step("1 / 3 — generate responses  [skipped, using existing raw_generations.jsonl]")
    else:
        cmd1 = [py, "src/run_eval.py", "--output", str(raw_path)]
        if args.provider:
            cmd1 += ["--provider", args.provider]
        if args.models:
            cmd1 += ["--models"] + args.models
        elif args.model:
            cmd1 += ["--model", args.model]
        if args.dataset:
            cmd1 += ["--input", args.dataset]
        if args.all_conditions:
            cmd1 += ["--all-conditions"]
        elif args.condition:
            cmd1 += ["--condition", args.condition]
        if args.limit is not None:
            cmd1 += ["--limit", str(args.limit)]

        _run("1 / 3 — generate responses  (run_eval.py)", cmd1)

    # ── 2 / 3  score ─────────────────────────────────────────────────────────
    cmd2 = [
        py,
        "src/score.py",
        "--input",
        str(raw_path),
        "--output",
        str(scored_path),
        "--judge-provider",
        args.judge_provider,
        "--judge-model",
        args.judge_model,
    ]
    if args.resume:
        cmd2 += ["--resume"]

    _run("2 / 3 — score with LLM judge  (score.py)", cmd2)

    # ── 3 / 3  analyze ───────────────────────────────────────────────────────
    cmd3 = [
        py, "src/analyze.py",
        "--input", str(scored_path),
        "--summary", str(summary_path),
        "--plots-dir", str(plots_dir),
    ]

    _run("3 / 3 — compute metrics and plots  (analyze.py)", cmd3)

    # ── terminal summary ──────────────────────────────────────────────────────
    _print_summary(summary_path)


if __name__ == "__main__":
    main()
