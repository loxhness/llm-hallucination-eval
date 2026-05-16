"""Full generate -> LLM judge -> analyze pipeline (legacy CLI)."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from idk_eval.paths import default_dataset, project_root

load_dotenv(project_root() / ".env")


def _step(label: str) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  {label}")
    print(f"{'─' * width}")


def _run(label: str, cmd: list[str], cwd: Path) -> None:
    _step(label)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\nAborted: '{label}' exited with code {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Full eval pipeline with LLM judge and plots.")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--models", nargs="+", metavar="[PROVIDER:]MODEL")
    parser.add_argument("--dataset", default=None, type=Path)
    parser.add_argument(
        "--condition",
        choices=["baseline", "abstain", "cite_or_abstain", "chain_of_thought", "confident"],
        default=None,
    )
    parser.add_argument("--all-conditions", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--judge-provider",
        default=os.getenv("JUDGE_PROVIDER", "anthropic"),
        choices=["anthropic", "openai"],
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=root / "results", type=Path)
    args = parser.parse_args()

    out = Path(args.output_dir)
    raw_path = out / "raw_generations.jsonl"
    scored_path = out / "scored.csv"
    summary_path = out / "summary.csv"
    plots_dir = out / "plots"
    py = sys.executable

    if not args.skip_eval:
        cmd1 = [py, "-m", "idk_eval.generate", "--output", str(raw_path)]
        if args.provider:
            cmd1 += ["--provider", args.provider]
        if args.models:
            cmd1 += ["--models"] + args.models
        elif args.model:
            cmd1 += ["--model", args.model]
        if args.dataset:
            cmd1 += ["--input", str(args.dataset)]
        elif default_dataset().exists():
            cmd1 += ["--input", str(default_dataset())]
        if args.all_conditions:
            cmd1 += ["--all-conditions"]
        elif args.condition:
            cmd1 += ["--condition", args.condition]
        if args.limit is not None:
            cmd1 += ["--limit", str(args.limit)]
        _run("1 / 3 — generate responses", cmd1, root)

    cmd2 = [
        py, "-m", "idk_eval.judge",
        "--input", str(raw_path),
        "--output", str(scored_path),
        "--judge-provider", args.judge_provider,
        "--judge-model", args.judge_model,
    ]
    if args.resume:
        cmd2.append("--resume")
    _run("2 / 3 — score with LLM judge", cmd2, root)

    cmd3 = [
        py, "-m", "idk_eval.analyzer",
        "--input", str(scored_path),
        "--summary", str(summary_path),
        "--plots-dir", str(plots_dir),
    ]
    _run("3 / 3 — analyze", cmd3, root)
    print(f"Done. See {summary_path} and {plots_dir}/")


if __name__ == "__main__":
    main()
