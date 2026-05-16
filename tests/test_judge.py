"""Unit tests for LLM judge parsing (no live API calls)."""

import json
from unittest.mock import patch

import pytest

from idk_eval.judge import call_judge, score_record


@pytest.mark.parametrize(
    "verdict,expected_flags",
    [
        ("correct", {"correct": True, "abstained": False, "hallucinated": False}),
        ("abstained", {"correct": False, "abstained": True, "hallucinated": False}),
        ("hallucinated", {"correct": False, "abstained": False, "hallucinated": True}),
    ],
)
def test_score_record_maps_verdicts(verdict, expected_flags):
    payload = json.dumps({"verdict": verdict, "reason": "test reason"})
    record = {
        "id": "f001",
        "question": "What is 2+2?",
        "expected": "4",
        "model_answer": "4",
        "condition": "baseline",
    }

    with patch("idk_eval.judge._call_anthropic", return_value=payload):
        scored = score_record(record, judge_provider="anthropic", judge_model="claude-test")

    for key, val in expected_flags.items():
        assert scored[key] is val
    assert scored["judge_reason"] == "test reason"


def test_call_judge_strips_preamble_before_json():
    raw = 'Here is my verdict:\n{"verdict": "correct", "reason": "matches"}'
    with patch("idk_eval.judge._call_openai", return_value=raw):
        verdict, reason = call_judge("Q", "A", "A", provider="openai", model="gpt-test")
    assert verdict == "correct"
    assert reason == "matches"
