import argparse
import json
import os
from datetime import datetime
from pathlib import Path

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
}


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


def run_eval(provider_name, model, condition, input_path, output_path):
    provider = get_provider(name=provider_name, model=model)
    questions = load_questions(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for q in tqdm(questions, desc=f"Evaluating ({condition})"):
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
                "question": q["question"],
                "expected": q["answer"],
                "model_answer": resp_model_answer,
                "confidence": resp_confidence,
                "raw_text": resp_text,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None, help="openai | anthropic (default: from env)")
    parser.add_argument("--model", default=None, help="Model name (default: from env)")
    parser.add_argument(
        "--condition",
        choices=["baseline", "abstain", "cite_or_abstain"],
        default=None,
        help="Single condition to run (ignored if --all-conditions)",
    )
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run all 3 conditions and concatenate into one output file",
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
    args = parser.parse_args()

    if args.all_conditions:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        first = True
        for cond in ["baseline", "abstain", "cite_or_abstain"]:
            tmp_path = args.output.parent / f"_tmp_{cond}.jsonl"
            run_eval(
                provider_name=args.provider,
                model=args.model,
                condition=cond,
                input_path=args.input,
                output_path=tmp_path,
            )
            with open(tmp_path, encoding="utf-8") as f:
                content = f.read()
            tmp_path.unlink()
            mode = "w" if first else "a"
            with open(args.output, mode, encoding="utf-8") as out:
                out.write(content)
            first = False
        print(f"Done. Output: {args.output}")
    else:
        cond = args.condition or "baseline"
        run_eval(
            provider_name=args.provider,
            model=args.model,
            condition=cond,
            input_path=args.input,
            output_path=args.output,
        )
        print(f"Done. Output: {args.output}")


if __name__ == "__main__":
    main()
