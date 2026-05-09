"""
Download a subset of TruthfulQA and convert it to the project's questions.jsonl format.

Output fields
-------------
id        : unique string, e.g. "tqf001" / "tqa001" / "tqu001"
category  : "factual" | "ambiguous" | "unanswerable"  (compatible with run_eval.py)
type      : original TruthfulQA category label (e.g. "Misconceptions")
question  : question text
answer    : ground-truth string, or "UNANSWERABLE" for unknowable questions

Usage
-----
    python src/load_dataset.py                         # writes data/questions_truthfulqa.jsonl
    python src/load_dataset.py --target 200
    python src/load_dataset.py --output data/custom.jsonl
"""

import argparse
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "questions_truthfulqa.jsonl"
DEFAULT_TARGET = 175
_SEED = 42

# ── Category mapping ──────────────────────────────────────────────────────────

# These TruthfulQA categories ask things that depend on who/where/when the
# asker is — no model can answer them correctly, so we mark UNANSWERABLE.
_UNANSWERABLE_CATEGORIES = {
    "Indexical Error: Identity",
    "Indexical Error: Location",
    "Indexical Error: Other",
    "Indexical Error: Time",
}

# Genuinely subjective or socially contested questions where multiple answers
# are defensible.
_AMBIGUOUS_CATEGORIES = {
    "Stereotypes",
    "Subjective",
}

# Some best_answer strings themselves signal unknowability regardless of
# category label (e.g., "I have no comment.").
_ABSTAIN_RE = re.compile(
    r"^\s*("
    r"i\s+have\s+no\s+(comment|idea)"
    r"|i\s+don'?t\s+know"
    r"|i\s+cannot\s+(answer|say|tell)"
    r"|i\s+am\s+not\s+sure"
    r"|it'?s?\s+(not\s+possible|impossible)\s+to\s+(know|say)"
    r"|there\s+(is|are)\s+no\s+(single\s+|correct\s+|definitive\s+)?answer"
    r"|this\s+(question\s+)?cannot\s+be\s+answered"
    r")",
    re.IGNORECASE,
)


def _classify(tq_category: str, best_answer: str) -> tuple[str, str]:
    """Return (our_category, answer_value) for one TruthfulQA item."""
    if tq_category in _UNANSWERABLE_CATEGORIES or _ABSTAIN_RE.match(best_answer):
        return "unanswerable", "UNANSWERABLE"
    if tq_category in _AMBIGUOUS_CATEGORIES:
        return "ambiguous", best_answer
    return "factual", best_answer


# ── Sampling ──────────────────────────────────────────────────────────────────

def _proportional_sample(by_category: dict[str, list], target: int, seed: int) -> list:
    """
    Sample ~target items proportionally across TruthfulQA categories.
    Uses a fixed seed so the output is reproducible.
    """
    rng = random.Random(seed)
    total = sum(len(v) for v in by_category.values())
    ratio = target / total

    sampled: list = []
    for items in by_category.values():
        n = max(1, round(len(items) * ratio))
        sampled.extend(rng.sample(items, min(n, len(items))))

    rng.shuffle(sampled)
    return sampled


# ── Main ──────────────────────────────────────────────────────────────────────

def build_records(ds, target: int) -> list[dict]:
    """Convert a HuggingFace TruthfulQA dataset into our record format."""
    by_category: dict[str, list] = {}
    for item in ds:
        by_category.setdefault(item["category"], []).append(item)

    sample = _proportional_sample(by_category, target, _SEED)

    counters: dict[str, int] = {"factual": 0, "ambiguous": 0, "unanswerable": 0}
    prefixes = {"factual": "tqf", "ambiguous": "tqa", "unanswerable": "tqu"}
    records: list[dict] = []

    for item in sample:
        best = (item.get("best_answer") or "").strip()
        if not best:
            continue

        category, answer = _classify(item["category"], best)
        counters[category] += 1
        records.append({
            "id": f"{prefixes[category]}{counters[category]:03d}",
            "category": category,
            "type": item["category"],
            "question": item["question"],
            "answer": answer,
        })

        if len(records) >= target:
            break

    return records


def main(output: Path = DEFAULT_OUTPUT, target: int = DEFAULT_TARGET) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' library is required.\n"
            "Install it with:  pip install datasets"
        )

    print(f"Downloading TruthfulQA (first run fetches ~3 MB, then cached)...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    print(f"Loaded {len(ds)} questions from TruthfulQA.")

    records = build_records(ds, target)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(records)
    print(f"\nWrote {total} questions → {output}")
    for cat in ("factual", "ambiguous", "unanswerable"):
        n = sum(1 for r in records if r["category"] == cat)
        pct = n / total * 100 if total else 0
        print(f"  {cat:>12s}: {n:>3d}  ({pct:.0f}%)")

    if not (150 <= total <= 200):
        print(f"\nWarning: got {total} questions (target 150–200). "
              f"Adjust --target or check the dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download TruthfulQA and convert to questions.jsonl format."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help=f"Output path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        type=int,
        metavar="N",
        help=f"Approximate number of questions to sample (default: {DEFAULT_TARGET})",
    )
    args = parser.parse_args()
    main(output=args.output, target=args.target)
