from types import SimpleNamespace

import pytest

from papertrail import answer as answer_module
from papertrail.answer import AnswerError, answer, format_papers

PAPERS = [
    {"title": "Paper One", "year": 2024, "authors": [f"Author {i}" for i in range(7)], "abstract": "First."},
    {"title": "Paper Two", "year": None, "authors": [], "abstract": "Second."},
]


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(response):
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


@pytest.fixture
def with_key_and_papers(monkeypatch):
    monkeypatch.setattr(answer_module, "ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(answer_module, "search", lambda question, k: PAPERS[:k])


def test_format_papers_numbers_sources_and_truncates_authors():
    text = format_papers(PAPERS)
    assert text.startswith("[1] Paper One (2024)")
    assert "Author 4 et al." in text and "Author 5" not in text
    assert "[2] Paper Two (None)" in text


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.setattr(answer_module, "ANTHROPIC_API_KEY", "")
    with pytest.raises(AnswerError, match="ANTHROPIC_API_KEY"):
        answer("anything")


def test_answer_sends_papers_and_returns_text(monkeypatch, with_key_and_papers):
    response = SimpleNamespace(
        stop_reason="end_turn",
        model="claude-opus-5-5",
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="RAG helps [1].")],
    )
    client, messages = fake_client(response)
    monkeypatch.setattr(answer_module, "_client", lambda: client)

    result = answer("What helps?", k=2)

    assert result["answer"] == "RAG helps [1]."
    assert result["papers"] == PAPERS
    sent = messages.calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert "[1] Paper One" in sent["messages"][0]["content"]
    assert "Question: What helps?" in sent["messages"][0]["content"]


def test_refusal_raises(monkeypatch, with_key_and_papers):
    client, _ = fake_client(SimpleNamespace(stop_reason="refusal", model="x", content=[]))
    monkeypatch.setattr(answer_module, "_client", lambda: client)
    with pytest.raises(AnswerError, match="declined"):
        answer("question")
