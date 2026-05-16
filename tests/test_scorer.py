"""Unit tests for rule-based scorer."""

from idk_eval.dataset import Question
from idk_eval.scorer import detect_abstention, score_response


def test_detect_abstention():
    assert detect_abstention("I don't know the answer to that.")
    assert not detect_abstention("Paris is the capital of France.")


def test_answerable_correct():
    q = Question(
        id="q1",
        text="Capital of France?",
        category="answerable",
        expected_answer="Paris",
    )
    score = score_response(q, "The capital is Paris.")
    assert score.label == "correct"


def test_answerable_incorrect():
    q = Question(
        id="q1",
        text="Capital of France?",
        category="answerable",
        expected_answer="Paris",
    )
    score = score_response(q, "The capital is London.")
    assert score.label == "incorrect"


def test_post_cutoff_abstain():
    q = Question(id="q2", text="Future stock price?", category="post_cutoff")
    score = score_response(q, "I'm not sure I can answer that.")
    assert score.label == "abstained"


def test_post_cutoff_hallucinate():
    q = Question(id="q2", text="Future stock price?", category="post_cutoff")
    score = score_response(q, "It will be $200.")
    assert score.label == "hallucinated"


def test_fake_entity_hallucinate():
    q = Question(
        id="q3",
        text="Hugo Belkonen's symphony?",
        category="fake_entity",
    )
    score = score_response(q, "His most famous work was the Fifth.")
    assert score.label == "hallucinated"
