"""Tests for dataset loading."""

from idk_eval.dataset import Dataset, Question


def test_builtin_loads_sixty_questions():
    ds = Dataset.builtin()
    assert len(ds) == 60
    assert all(isinstance(q, Question) for q in ds)


def test_answerable_has_expected_answer():
    q = next(q for q in Dataset.builtin() if q.category == "answerable")
    assert q.expected_answer


def test_post_cutoff_judge_expectation():
    q = next(q for q in Dataset.builtin() if q.category == "post_cutoff")
    assert q.expected_for_judge() == "UNANSWERABLE"
