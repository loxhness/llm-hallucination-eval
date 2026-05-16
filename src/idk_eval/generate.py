"""Generate model responses (CLI pipeline step 1)."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from dotenv import load_dotenv
from tqdm import tqdm

from idk_eval.dataset import Dataset
from idk_eval.models import get_client
from idk_eval.paths import default_dataset, project_root
from idk_eval.strategy import ALL_CONDITIONS, _CONFIDENCE_SUFFIX, get_strategy

load_dotenv(project_root() / ".env")


def _parse_model_spec(spec: str, default_provider: str) -> tuple[str, str]:
    if ":" in spec:
        provider, _, model = spec.partition(":")
        return provider, model
    return default_provider, spec


def run_eval(
    client,
    condition: str,
    dataset: Dataset,
    out_file: IO[str],
    model: str | None = None,
) -> None:
    strategy = get_strategy(condition)
    resolved_model = model or client.default_model

    for q in tqdm(dataset.questions, desc=f"{resolved_model} / {condition}", leave=True):
        parts = strategy.format(q.text)
        user_prompt = f"{parts['user']}{_CONFIDENCE_SUFFIX}"
        try:
            resp = client.complete_messages(
                parts["system"],
                user_prompt,
                model=model,
            )
            resp_text = resp.raw_text
            resp_model_answer = resp.model_answer
            resp_confidence = resp.confidence
        except Exception as e:
            resp_text = f"[ERROR: {e}]"
            resp_model_answer = resp_text
            resp_confidence = None

        record = {
            "id": q.id,
            "category": q.category,
            "condition": condition,
            "model_name": resolved_model,
            "question": q.text,
            "expected": q.expected_for_judge(),
            "model_answer": resp_model_answer,
            "confidence": resp_confidence,
            "raw_text": resp_text,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Generate model responses for evaluation.")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--models", nargs="+", metavar="[PROVIDER:]MODEL")
    parser.add_argument("--condition", choices=ALL_CONDITIONS, default=None)
    parser.add_argument("--all-conditions", action="store_true")
    parser.add_argument("--input", default=default_dataset(), type=Path)
    parser.add_argument("--output", default=root / "results" / "raw_generations.jsonl", type=Path)
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    args = parser.parse_args()

    default_provider = args.provider or os.getenv("LLM_PROVIDER", "openai")
    if args.models:
        model_specs = [_parse_model_spec(s, default_provider) for s in args.models]
    else:
        model_specs = [(default_provider, args.model)]

    conditions = ALL_CONDITIONS if args.all_conditions else [args.condition or "baseline"]
    dataset = Dataset.from_jsonl(args.input)
    if args.limit is not None:
        dataset = Dataset(name=dataset.name, questions=dataset.questions[: args.limit])

    total = len(model_specs) * len(conditions)
    print(
        f"Running {len(model_specs)} model(s) x {len(conditions)} condition(s) "
        f"= {total} eval run(s). Output: {args.output}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_file:
        for provider_name, model in model_specs:
            client = get_client(name=provider_name, model=model)
            for condition in conditions:
                run_eval(client, condition, dataset, out_file, model=model)

    print(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
