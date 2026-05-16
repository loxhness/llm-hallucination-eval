"""LLM-as-judge scoring for the generate -> judge -> analyze CLI pipeline."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from idk_eval.paths import project_root

load_dotenv(project_root() / ".env")

_anthropic_client = None
_openai_client = None


def _judge_http_timeout() -> float:
    return float(os.getenv("JUDGE_HTTP_TIMEOUT", os.getenv("LLM_HTTP_TIMEOUT", "120")))


def _judge_max_retries() -> int:
    return int(os.getenv("JUDGE_MAX_RETRIES", "2"))


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic

        _anthropic_client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=_judge_http_timeout(),
            max_retries=_judge_max_retries(),
        )
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=_judge_http_timeout(),
            max_retries=_judge_max_retries(),
        )
    return _openai_client


JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge for a question-answering system.

Given a question, the expected answer, and a model's response, classify the response as one of:
- "correct": The response accurately answers the question, consistent with the expected answer.
- "abstained": The response explicitly declines to answer or expresses uncertainty \
(e.g., "I don't know", "I'm not sure", "I cannot determine").
- "hallucinated": The response provides a confident but incorrect or unsupported answer.

If the expected answer is "UNANSWERABLE", treat "abstained" as the correct verdict \
and any confident answer as hallucinated.

Respond with a JSON object only, no other text:
{"verdict": "<correct|hallucinated|abstained>", "reason": "<one sentence explanation>"}\
"""

_JUDGE_USER_TEMPLATE = (
    "Question: {question}\n"
    "Expected answer: {expected}\n"
    "Model response: {model_answer}"
)


def _call_anthropic(user_content: str, model: str) -> str:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=[{
            "type": "text",
            "text": JUDGE_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_content}],
    )
    return next((b.text for b in response.content if b.type == "text"), "")


def _call_openai(user_content: str, model: str) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=256,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _extract_json(text: str) -> str:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    return m.group() if m else text


def _is_judge_auth_failure(exc: BaseException) -> bool:
    s = str(exc).lower()
    return (
        "401" in s
        or "403" in s
        or "authentication_error" in s
        or "invalid x-api-key" in s
        or "incorrect api key" in s
        or "invalid_api_key" in s
        or "permission_denied" in s
    )


class JudgeAuthenticationError(RuntimeError):
    """Judge API rejected credentials."""


def call_judge(
    question: str, expected: str, model_answer: str, provider: str, model: str
) -> tuple[str, str]:
    user_content = _JUDGE_USER_TEMPLATE.format(
        question=question,
        expected=expected,
        model_answer=model_answer or "(no response)",
    )
    try:
        if provider == "anthropic":
            raw = _call_anthropic(user_content, model)
        else:
            raw = _call_openai(user_content, model)

        data = json.loads(_extract_json(raw))
        verdict = data.get("verdict", "").lower()
        reason = data.get("reason", "")
        if verdict not in ("correct", "hallucinated", "abstained"):
            verdict, reason = "hallucinated", f"Unexpected verdict from judge: {raw[:120]}"
    except Exception as exc:
        if _is_judge_auth_failure(exc):
            raise JudgeAuthenticationError(
                f"Judge ({provider}) authentication failed: {exc}\n"
                "Fix the API key in .env for that provider."
            ) from exc
        verdict, reason = "hallucinated", f"Judge error: {exc}"

    return verdict, reason


class LLMJudge:
    """Score raw generation records with an LLM judge."""

    def __init__(
        self,
        judge_provider: str | None = None,
        judge_model: str | None = None,
    ):
        self.judge_provider = judge_provider or os.getenv("JUDGE_PROVIDER", "anthropic")
        self.judge_model = judge_model or os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")

    def score(self, record: dict) -> dict:
        return score_record(record, self.judge_provider, self.judge_model)


def score_record(record: dict, judge_provider: str, judge_model: str) -> dict:
    verdict, reason = call_judge(
        question=record.get("question", ""),
        expected=record.get("expected", ""),
        model_answer=record.get("model_answer", ""),
        provider=judge_provider,
        model=judge_model,
    )
    out = dict(record)
    out["correct"] = verdict == "correct"
    out["abstained"] = verdict == "abstained"
    out["hallucinated"] = verdict == "hallucinated"
    out["judge_reason"] = reason
    return out


def load_raw_generations(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_SCORED_COLS = [
    "id", "category", "condition", "model_name", "expected", "model_answer",
    "confidence", "correct", "abstained", "hallucinated", "judge_reason",
]


def _write_scored_csv(scored: list, output_path: Path) -> None:
    df = pd.DataFrame(scored)
    existing = [c for c in _SCORED_COLS if c in df.columns]
    df = df[existing]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _read_resume_verdicts(path: Path) -> dict[tuple[str, str], dict]:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    need = {"id", "condition", "correct", "abstained", "hallucinated", "judge_reason"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"--resume CSV missing columns {missing}: {path}")
    out = {}
    for row in df.to_dict("records"):
        key = (str(row["id"]), str(row["condition"]))
        out[key] = {
            "correct": bool(row["correct"]),
            "abstained": bool(row["abstained"]),
            "hallucinated": bool(row["hallucinated"]),
            "judge_reason": "" if pd.isna(row.get("judge_reason")) else str(row["judge_reason"]),
        }
    return out


def _merge_saved_verdict(raw: dict, verdicts: dict) -> dict:
    out = dict(raw)
    out.update(verdicts)
    return out


def _require_judge_api_key(provider: str) -> None:
    if provider == "anthropic":
        key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        if not key:
            raise SystemExit("Missing ANTHROPIC_API_KEY in .env (required for --judge-provider anthropic).")
    else:
        key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise SystemExit("Missing OPENAI_API_KEY in .env (required for --judge-provider openai).")


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Score raw generations using an LLM judge.")
    parser.add_argument("--input", default=root / "results" / "raw_generations.jsonl", type=Path)
    parser.add_argument("--output", default=root / "results" / "scored.csv", type=Path)
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
    args = parser.parse_args()

    _require_judge_api_key(args.judge_provider)

    records = load_raw_generations(args.input)
    saved = {}
    if args.resume and args.output.exists():
        saved = _read_resume_verdicts(args.output)
        if saved:
            print(f"--resume: reusing {len(saved)} verdict(s) from {args.output}")

    pending = sum(1 for r in records if (str(r["id"]), str(r["condition"])) not in saved)
    print(
        f"Judging {len(records)} records with {args.judge_provider}/{args.judge_model} "
        f"({pending} API call(s)) ..."
    )

    scored: list = []
    partial_path = args.output.with_name(f"{args.output.stem}_partial{args.output.suffix}")
    try:
        for r in tqdm(records):
            key = (str(r["id"]), str(r["condition"]))
            if key in saved:
                scored.append(_merge_saved_verdict(r, saved[key]))
            else:
                try:
                    scored.append(score_record(r, args.judge_provider, args.judge_model))
                except JudgeAuthenticationError as err:
                    print(f"\n{err}", file=sys.stderr)
                    raise SystemExit(2) from err
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C) while waiting on the judge API.", flush=True)
        if scored:
            _write_scored_csv(scored, partial_path)
            print(f"Saved {len(scored)}/{len(records)} rows to:\n  {partial_path}", flush=True)
        raise SystemExit(130) from None

    _write_scored_csv(scored, args.output)
    print(f"Scored {len(scored)} records. Output: {args.output}")


if __name__ == "__main__":
    main()
