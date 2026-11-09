import json

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from papertrail import api
from papertrail import answer as answer_module
from papertrail.answer import AnswerError

client = TestClient(api.app)


def test_search_returns_papers(monkeypatch):
    monkeypatch.setattr(api, "search", lambda q, k: [{"title": f"{q} #{i}"} for i in range(k)])
    response = client.get("/search", params={"q": "rag", "k": 2})
    assert response.status_code == 200
    assert response.json() == {"query": "rag", "papers": [{"title": "rag #0"}, {"title": "rag #1"}]}


def test_search_validates_input():
    assert client.get("/search").status_code == 422
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "rag", "k": 50}).status_code == 422


def test_ask_returns_answer(monkeypatch):
    monkeypatch.setattr(api, "answer", lambda q, k: {"question": q, "answer": "yes [1]", "papers": []})
    response = client.get("/ask", params={"q": "is rag good?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "yes [1]"


@pytest.mark.parametrize(
    ("question", "reply"),
    [
        ("What did the authors compare?", "They compared retrieval with keyword search [1]."),
        ("What was the sample size?", "The supplied abstract does not report a sample size."),
    ],
)
def test_ask_round_trip_through_claude_sdk(monkeypatch, question, reply):
    paper = {
        "title": "Comparing Search Methods",
        "authors": ["Example Author"],
        "year": 2024,
        "abstract": "We compared semantic retrieval with keyword search.",
        "url": "https://example.org/search-methods",
    }

    def respond(request):
        assert request.url.path == "/v1/messages"
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        assert f"[1] {paper['title']}" in prompt
        assert paper["abstract"] in prompt
        assert prompt.endswith(f"Question: {question}")
        assert "don't contain enough information" in body["system"]
        return httpx2.Response(200, json={
            "id": "msg_demo",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [{"type": "text", "text": reply}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 80, "output_tokens": 20},
        })

    monkeypatch.setattr(answer_module, "ANTHROPIC_API_KEY", "offline-test-key")
    monkeypatch.setattr(answer_module, "search", lambda q, k: [paper])
    with anthropic.Anthropic(
        api_key="offline-test-key",
        base_url="https://api.anthropic.test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as sdk:
        monkeypatch.setattr(answer_module, "_client", lambda: sdk)
        response = client.get("/ask", params={"q": question})

    assert response.status_code == 200
    assert response.json() == {
        "question": question,
        "answer": reply,
        "papers": [paper],
        "model": answer_module.CLAUDE_MODEL,
    }


def test_ask_turns_errors_into_502(monkeypatch):
    def broken(q, k):
        raise AnswerError("No Claude API key found.")

    monkeypatch.setattr(api, "answer", broken)
    response = client.get("/ask", params={"q": "anything"})
    assert response.status_code == 502
    assert response.json() == {"detail": "No Claude API key found."}


def test_home_serves_web_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Paper Trail</title>" in response.text
