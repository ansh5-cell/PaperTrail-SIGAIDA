from fastapi.testclient import TestClient

from papertrail import api
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
