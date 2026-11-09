import pytest

from papertrail.evaluate import score_question


def test_first_result_right():
    scores = score_question(["Paper A", "Paper B"], ["paper a"], k=5)
    assert scores == {"hit": True, "recall": 1.0, "reciprocal_rank": 1.0, "first_rank": 1}


def test_partial_recall_and_rank():
    ranked = ["X", "Y", "Paper B", "Z", "W", "Paper A"]
    scores = score_question(ranked, ["Paper A", "Paper B"], k=5)
    assert scores["hit"] is True
    assert scores["recall"] == 0.5  # only Paper B made the top 5
    assert scores["first_rank"] == 3
    assert scores["reciprocal_rank"] == pytest.approx(1 / 3)


def test_miss():
    scores = score_question(["X", "Y"], ["Paper A"], k=5)
    assert scores == {"hit": False, "recall": 0.0, "reciprocal_rank": 0.0, "first_rank": None}
