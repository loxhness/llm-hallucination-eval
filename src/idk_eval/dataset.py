"""Dataset loading for evaluation."""

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Optional

from idk_eval.paths import project_root

QuestionCategory = Literal["answerable", "fake_entity", "ambiguous", "post_cutoff"]


@dataclass
class Question:
    id: str
    text: str
    category: QuestionCategory
    expected_answer: Optional[str] = None
    notes: Optional[str] = None

    def expected_for_judge(self) -> str:
        """Ground truth string passed to the LLM judge."""
        if self.category in ("fake_entity", "post_cutoff"):
            return "UNANSWERABLE"
        return self.expected_answer or ""


@dataclass
class Dataset:
    name: str
    questions: List[Question]

    def __len__(self):
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        questions = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    questions.append(Question(**data))
        return cls(name=path.stem, questions=questions)

    @classmethod
    def builtin(cls, name: str = "mixed_v1") -> "Dataset":
        """Load a dataset bundled with the package."""
        data_dir = Path(__file__).parent / "data"
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            available = [p.stem for p in data_dir.glob("*.jsonl")]
            raise FileNotFoundError(
                f"Built-in dataset '{name}' not found. Available: {available}"
            )
        return cls.from_jsonl(path)

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for q in self.questions:
                row = {k: v for k, v in asdict(q).items() if v is not None}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── TruthfulQA download (optional extra: datasets) ───────────────────────────

_UNANSWERABLE_CATEGORIES = {
    "Indexical Error: Identity",
    "Indexical Error: Location",
    "Indexical Error: Other",
    "Indexical Error: Time",
}

_AMBIGUOUS_CATEGORIES = {
    "Stereotypes",
    "Subjective",
}

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

_SEED = 42


def _classify_truthfulqa(tq_category: str, best_answer: str) -> QuestionCategory:
    if tq_category in _UNANSWERABLE_CATEGORIES or _ABSTAIN_RE.match(best_answer):
        return "post_cutoff"
    if tq_category in _AMBIGUOUS_CATEGORIES:
        return "ambiguous"
    return "answerable"


def _proportional_sample(by_category: dict[str, list], target: int, seed: int) -> list:
    rng = random.Random(seed)
    total = sum(len(v) for v in by_category.values())
    ratio = target / total

    sampled: list = []
    for items in by_category.values():
        n = max(1, round(len(items) * ratio))
        sampled.extend(rng.sample(items, min(n, len(items))))

    rng.shuffle(sampled)
    return sampled


def download_truthfulqa(
    output: Path | str | None = None,
    target: int = 175,
) -> Path:
    """Download TruthfulQA and write a JSONL file in package format."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise SystemExit(
            "The 'datasets' extra is required.\n"
            "Install with:  pip install 'idk-eval[datasets]'"
        ) from err

    out = Path(output) if output else (project_root() / "data" / "questions_truthfulqa.jsonl")
    print("Downloading TruthfulQA (first run fetches ~3 MB, then cached)...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    print(f"Loaded {len(ds)} questions from TruthfulQA.")

    by_category: dict[str, list] = {}
    for item in ds:
        by_category.setdefault(item["category"], []).append(item)

    sample = _proportional_sample(by_category, target, _SEED)
    counters = {c: 0 for c in ("answerable", "ambiguous", "post_cutoff", "fake_entity")}
    prefixes = {"answerable": "tqf", "ambiguous": "tqa", "post_cutoff": "tqu", "fake_entity": "tqx"}
    questions: list[Question] = []

    for item in sample:
        best = (item.get("best_answer") or "").strip()
        if not best:
            continue

        category = _classify_truthfulqa(item["category"], best)
        counters[category] += 1
        qid = f"{prefixes[category]}{counters[category]:03d}"
        kwargs: dict = {
            "id": qid,
            "text": item["question"],
            "category": category,
            "notes": item["category"],
        }
        if category == "answerable":
            kwargs["expected_answer"] = best
        elif category == "ambiguous":
            kwargs["expected_answer"] = best
        questions.append(Question(**kwargs))

        if len(questions) >= target:
            break

    dataset = Dataset(name=out.stem, questions=questions)
    dataset.to_jsonl(out)

    total = len(questions)
    print(f"\nWrote {total} questions -> {out}")
    for cat in ("answerable", "ambiguous", "post_cutoff"):
        n = sum(1 for q in questions if q.category == cat)
        pct = n / total * 100 if total else 0
        print(f"  {cat:>12s}: {n:>3d}  ({pct:.0f}%)")

    if not (150 <= total <= 200):
        print(
            f"\nWarning: got {total} questions (target 150-200). "
            "Adjust --target or check the dataset."
        )
    return out
