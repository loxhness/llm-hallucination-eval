import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import IO

from dotenv import load_dotenv
from tqdm import tqdm

from providers import get_provider

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CONDITION_PROMPTS = {
    "baseline": (
        "Answer the following question. At the end, provide a confidence score from 0-100.\n\n"
        "Format your response as:\nAnswer: <your answer>\nConfidence: <number>"
    ),
    "abstain": (
        "Answer the following question. If you are not sure, say 'I don't know'.\n"
        "At the end, provide a confidence score from 0-100.\n\n"
        "Format your response as:\nAnswer: <your answer>\nConfidence: <number>"
    ),
    "cite_or_abstain": (
        "Only answer if you are confident and can cite a source type (e.g., official site, encyclopedia). "
        "Otherwise say 'I don't know'.\n"
        "At the end, provide a confidence score from 0-100.\n\n"
        "Format your response as:\nAnswer: <your answer>\nConfidence: <number>"
    ),
    "chain_of_thought": (
        "Think step by step before answering. Work through your reasoning first, then state your final answer.\n"
        "At the end, provide a confidence score from 0-100.\n\n"
        "Format your response as:\nAnswer: <your final answer>\nConfidence: <number>"
    ),
    "confident": (
        "Always give a direct, confident answer. Never say you don't know or express uncertainty — "
        "commit to the single most likely answer based on your knowledge.\n"
        "At the end, provide a confidence score from 0-100.\n\n"
        "Format your response as:\nAnswer: <your answer>\nConfidence: <number>"
    ),
}

ALL_CONDITIONS = ["baseline", "abstain", "cite_or_abstain", "chain_of_thought", "confident"]


def load_questions(input_path: Path) -> list[dict]:
    questions = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def build_prompt(question: str, condition: str) -> str:
    instruction = CONDITION_PROMPTS.get(condition, CONDITION_PROMPTS["baseline"])
    return f"{instruction}\n\nQuestion: {question}"


def _parse_model_spec(spec: str, default_provider: str) -> tuple[str, str]:
    """Parse 'provider:model' or bare 'model' into (provider, model)."""
    if ":" in spec:
        provider, _, model = spec.partition(":")
        return provider, model
    return default_provider, spec


def run_eval(
    provider_name: str,
    model: str | None,
    condition: str,
    input_path: Path,
    out_file: IO[str],
    limit: int | None = None,
) -> None:
    provider = get_provider(name=provider_name, model=model)
    questions = load_questions(input_path)
    if limit is not None:
        questions = questions[:limit]
    resolved_model = model or provider.default_model

    for q in tqdm(questions, desc=f"{resolved_model} / {condition}", leave=True):
        prompt = build_prompt(q["question"], condition)
        try:
            resp = provider.complete(prompt, model=model)
            resp_text = resp.raw_text
            resp_model_answer = resp.model_answer
            resp_confidence = resp.confidence
        except Exception as e:
            resp_text = f"[ERROR: {e}]"
            resp_model_answer = resp_text
            resp_confidence = None

        record = {
            "id": q["id"],
            "category": q["category"],
            "condition": condition,
            "model_name": resolved_model,
            "question": q["question"],
            "expected": q["answer"],
            "model_answer": resp_model_answer,
            "confidence": resp_confidence,
            "raw_text": resp_text,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None, help="openai | anthropic (default: from env)")
    parser.add_argument("--model", default=None, help="Single model name (default: from env)")
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="[PROVIDER:]MODEL",
        help=(
            "One or more models to evaluate in the same run. "
            "Each entry is either a bare model name (uses --provider) "
            "or 'provider:model' for explicit routing, e.g. "
            "'anthropic:claude-haiku-4-5-20251001 openai:gpt-4o-mini'. "
            "Overrides --model when provided."
        ),
    )
    parser.add_argument(
        "--condition",
        choices=ALL_CONDITIONS,
        default=None,
        help="Single condition to run (ignored if --all-conditions)",
    )
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run all conditions for every specified model.",
    )
    parser.add_argument(
        "--input",
        default=PROJECT_ROOT / "data" / "questions.jsonl",
        type=Path,
        help="Input JSONL questions",
    )
    parser.add_argument(
        "--output",
        default=PROJECT_ROOT / "results" / "raw_generations.jsonl",
        type=Path,
        help="Output JSONL generations",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only evaluate the first N questions from the dataset (for quick runs).",
    )
    args = parser.parse_args()

    default_provider = args.provider or os.getenv("LLM_PROVIDER", "openai")

    if args.models:
        model_specs = [_parse_model_spec(s, default_provider) for s in args.models]
    else:
        model_specs = [(default_provider, args.model)]

    conditions = ALL_CONDITIONS if args.all_conditions else [args.condition or "baseline"]

    total = len(model_specs) * len(conditions)
    print(
        f"Running {len(model_specs)} model(s) × {len(conditions)} condition(s) "
        f"= {total} eval run(s). Output: {args.output}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_file:
        for provider_name, model in model_specs:
            for condition in conditions:
                run_eval(
                    provider_name,
                    model,
                    condition,
                    args.input,
                    out_file,
                    limit=args.limit,
                )

    print(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
