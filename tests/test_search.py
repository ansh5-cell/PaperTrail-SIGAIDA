from papertrail import search as search_module


class FakeCollection:
    def __init__(self, n):
        self.n = n
        self.asked_for = None

    def query(self, query_embeddings, n_results):
        self.asked_for = n_results
        count = min(n_results, self.n)
        return {
            "metadatas": [[{"title": f"Paper {i}", "authors": "A; B", "source": "arxiv", "url": f"u{i}",
                            "year": 2024} for i in range(count)]],
            "documents": [[f"abstract {i}" for i in range(count)]],
            "distances": [[0.1 * i for i in range(count)]],
        }


def setup(monkeypatch, n=40):
    collection = FakeCollection(n)
    monkeypatch.setattr(search_module, "_collection", lambda: collection)
    monkeypatch.setattr(search_module, "embed", lambda texts: [[0.0]])
    return collection


def test_rerank_reorders_candidates(monkeypatch):
    collection = setup(monkeypatch)
    # Reranker prefers higher-numbered papers, the opposite of the embedding order
    monkeypatch.setattr(search_module, "rerank_scores", lambda q, texts: [float(i) for i in range(len(texts))])

    results = search_module.search("q", k=3)

    assert collection.asked_for == search_module.RERANK_CANDIDATES
    assert [p["title"] for p in results] == ["Paper 29", "Paper 28", "Paper 27"]
    assert results[0]["authors"] == ["A", "B"]
    assert results[0]["similarity"] == round(1 - 2.9, 4)  # cosine score kept for display


def test_without_rerank_keeps_embedding_order(monkeypatch):
    collection = setup(monkeypatch)
    results = search_module.search("q", k=3, rerank=False)
    assert collection.asked_for == 3
    assert [p["title"] for p in results] == ["Paper 0", "Paper 1", "Paper 2"]
