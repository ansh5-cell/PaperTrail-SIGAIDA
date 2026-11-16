"""Unit tests for full-text PDF chunking, metadata filtering, deduplication, and multi-topic API."""

from pathlib import Path
from fastapi.testclient import TestClient

from papertrail import api
from papertrail import pdf_parser
from papertrail import search as search_module
from papertrail.schema import Paper

client = TestClient(api.app)


def test_split_text_respects_size_and_overlap():
    sentences = [f"Sentence number {i} discusses retrieval augmented generation." for i in range(25)]
    text = " ".join(sentences)
    chunks = pdf_parser.split_text(text, chunk_size=200, overlap=50)

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # Adjacent chunks should share overlapping content
    assert chunks[0][-30:] in chunks[1]


def test_chunk_paper_falls_back_to_abstract_without_pdf(monkeypatch):
    paper = Paper(
        title="OpenAlex Only Paper",
        abstract="This paper has no arXiv PDF link.",
        authors=["Alice"],
        year=2024,
        source="openalex",
        citation_count=12,
        url="https://doi.org/10.1234/example",
    )
    chunks = pdf_parser.chunk_paper(paper, pid="abc123", use_full_text=True)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "abc123:0"
    assert chunks[0].section == "abstract"
    assert chunks[0].text == paper.abstract
    assert chunks[0].citation_count == 12


def test_chunk_paper_extracts_sections_and_skips_references(monkeypatch, tmp_path):
    fake_pdf = tmp_path / "2401.00001.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(pdf_parser, "download_pdf", lambda url: fake_pdf)
    monkeypatch.setattr(
        pdf_parser,
        "extract_sections",
        lambda path: [
            ("abstract", "Duplicate abstract text."),
            ("introduction", "Introductory background on LLMs."),
            ("methodology", "We propose a two-stage hybrid retriever."),
            ("results", "Our method improves Hit@5 by 20 points."),
        ],
    )

    paper = Paper(
        title="Full Text RAG",
        abstract="Main abstract.",
        authors=["Bob"],
        year=2025,
        source="arxiv",
        url="https://arxiv.org/abs/2401.00001",
    )
    chunks = pdf_parser.chunk_paper(paper, pid="pid1", use_full_text=True)

    sections = [c.section for c in chunks]
    assert sections == ["abstract", "introduction", "methodology", "results"]
    assert chunks[0].text == "Main abstract."
    assert chunks[2].text == "We propose a two-stage hybrid retriever."


def test_build_where_filter_single_and_combined():
    assert search_module.build_where_filter() is None

    single = search_module.build_where_filter(min_year=2024)
    assert single == {"year": {"$gte": 2024}}

    combined = search_module.build_where_filter(
        min_year=2023,
        max_year=2025,
        min_citations=10,
        source="ArXiv",
        section="Methodology",
    )
    assert combined == {
        "$and": [
            {"year": {"$gte": 2023}},
            {"year": {"$lte": 2025}},
            {"citation_count": {"$gte": 10}},
            {"source": {"$eq": "arxiv"}},
            {"section": {"$eq": "methodology"}},
        ]
    }


def test_search_deduplicates_chunks_by_paper_id(monkeypatch):
    class FakeCollection:
        def __init__(self):
            self.last_kwargs = None

        def query(self, **kwargs):
            self.last_kwargs = kwargs
            return {
                "metadatas": [[
                    {
                        "paper_id": "p1",
                        "section": "methodology",
                        "title": "Paper One",
                        "abstract": "Abstract 1",
                        "authors": "Alice; Bob",
                        "year": 2025,
                        "source": "arxiv",
                        "citation_count": 15,
                        "url": "https://arxiv.org/abs/2501.00001",
                    },
                    {
                        "paper_id": "p1",
                        "section": "results",
                        "title": "Paper One",
                        "abstract": "Abstract 1",
                        "authors": "Alice; Bob",
                        "year": 2025,
                        "source": "arxiv",
                        "citation_count": 15,
                        "url": "https://arxiv.org/abs/2501.00001",
                    },
                    {
                        "paper_id": "p2",
                        "section": "abstract",
                        "title": "Paper Two",
                        "abstract": "Abstract 2",
                        "authors": "Carol",
                        "year": 2024,
                        "source": "openalex",
                        "citation_count": 40,
                        "url": "https://doi.org/10.1000/p2",
                    },
                ]],
                "documents": [[
                    "Methodology chunk from Paper One.",
                    "Results chunk from Paper One.",
                    "Abstract 2",
                ]],
                "distances": [[0.10, 0.15, 0.25]],
            }

    fake_col = FakeCollection()
    monkeypatch.setattr(search_module, "_collection", lambda: fake_col)
    monkeypatch.setattr(search_module, "embed", lambda texts: [[0.1, 0.2]])

    results = search_module.search("hybrid retrieval", k=2, rerank=False, min_year=2024, source="arxiv")

    assert len(results) == 2
    assert [r["paper_id"] for r in results] == ["p1", "p2"]
    assert results[0]["matched_section"] == "methodology"
    assert results[0]["matched_chunk"] == "Methodology chunk from Paper One."
    assert fake_col.last_kwargs["where"] == {
        "$and": [
            {"year": {"$gte": 2024}},
            {"source": {"$eq": "arxiv"}},
        ]
    }


def test_api_topics_search_and_ask_filters(monkeypatch):
    monkeypatch.setattr(
        api,
        "list_topics",
        lambda: [{"topic": "LLMs & RAG", "collection": "papers", "count": 42}],
    )
    topics_resp = client.get("/topics")
    assert topics_resp.status_code == 200
    assert topics_resp.json() == {"topics": [{"topic": "LLMs & RAG", "collection": "papers", "count": 42}]}

    captured_search = {}

    def fake_search(q, k, **kwargs):
        captured_search.update({"q": q, "k": k, **kwargs})
        return [{"title": "Filtered Paper"}]

    monkeypatch.setattr(api, "search", fake_search)
    search_resp = client.get(
        "/search",
        params={
            "q": "hallucination",
            "k": 3,
            "topic": "AI Agents & Reasoning",
            "min_year": 2024,
            "min_citations": 5,
            "source": "arxiv",
            "section": "results",
        },
    )
    assert search_resp.status_code == 200
    body = search_resp.json()
    assert body["papers"] == [{"title": "Filtered Paper"}]
    assert captured_search == {
        "q": "hallucination",
        "k": 3,
        "topic": "AI Agents & Reasoning",
        "min_year": 2024,
        "min_citations": 5,
        "source": "arxiv",
        "section": "results",
    }

    captured_ask = {}

    def fake_answer(q, k, **kwargs):
        captured_ask.update({"q": q, "k": k, **kwargs})
        return {"question": q, "answer": "Grounded answer [1].", "papers": [], "model": "claude-opus-5-5"}

    monkeypatch.setattr(api, "answer", fake_answer)
    ask_resp = client.get(
        "/ask",
        params={
            "q": "rag grounding",
            "k": 4,
            "topic": "LLMs & RAG",
            "min_year": 2023,
            "source": "openalex",
        },
    )
    assert ask_resp.status_code == 200
    assert ask_resp.json()["answer"] == "Grounded answer [1]."
    assert captured_ask == {
        "q": "rag grounding",
        "k": 4,
        "topic": "LLMs & RAG",
        "min_year": 2023,
        "source": "openalex",
    }