import argparse
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ABSTAIN_PATTERNS = [
    r"i don't know",
    r"i do not know",
    r"i'm not sure",
    r"i am not sure",
    r"not sure",
    r"cannot (?:answer|determine|say)",
    r"can't (?:answer|determine|say)",
    r"unable to (?:answer|determine)",
    r"no (?:way to )?know",
    r"doesn't (?:have|contain) (?:enough )?information",
    r"don't have (?:enough )?information",
    r"insufficient information",
    r"cannot be (?:answered|determined)",
    r"unanswerable",
]


def is_abstained(model_answer):
    text = (model_answer or "").strip().lower()
    for pat in ABSTAIN_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def is_correct(model_answer, expected):
    if expected.upper() == "UNANSWERABLE":
        return is_abstained(model_answer)

    exp = expected.strip().lower()
    ans = (model_answer or "").strip().lower()
    return exp in ans


def score_record(record):
    model_answer = record.get("model_answer", "")
    expected = record.get("expected", "")

    correct = is_correct(model_answer, expected)
    abstained = is_abstained(model_answer)
    hallucinated = (not correct) and (not abstained)

    out = dict(record)
    out["correct"] = correct
    out["abstained"] = abstained
    out["hallucinated"] = hallucinated
    return out


def load_raw_generations(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=PROJECT_ROOT / "results" / "raw_generations.jsonl",
        type=Path,
        help="Input JSONL (raw generations)",
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "results" / "scored.csv",
        type=Path,
        help="Output CSV (scored)",
    )
    args = parser.parse_args()

    records = load_raw_generations(args.input)
    scored = [score_record(r) for r in records]

    df = pd.DataFrame(scored)
    cols = ["id", "category", "condition", "expected", "model_answer", "confidence", "correct", "abstained", "hallucinated"]
    existing = [c for c in cols if c in df.columns]
    df = df[existing]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Scored {len(df)} records. Output: {args.output}")


if __name__ == "__main__":
    main()
