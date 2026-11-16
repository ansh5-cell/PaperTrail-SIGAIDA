"""Find the papers most related to a question.

Two stages: cosine similarity in Chroma picks the top candidates, then a cross-encoder reranks them.
Supports multi-topic collections, full-text chunk deduplication, and metadata filters.
"""

import sys
from functools import lru_cache

import chromadb

from papertrail.config import CHROMA_DIR, COLLECTION_NAME, RERANK_CANDIDATES, TOPICS, topic_to_collection
from papertrail.embeddings import embed
from papertrail.index import get_collection
from papertrail.rerank import rerank_scores


@lru_cache(maxsize=1)
def _collection():
    return get_collection(chromadb.PersistentClient(path=str(CHROMA_DIR)), COLLECTION_NAME)


@lru_cache(maxsize=16)
def _named_collection(collection_name: str):
    return get_collection(chromadb.PersistentClient(path=str(CHROMA_DIR)), collection_name)


def list_topics() -> list[dict]:
    """Return available topics and how many vectors exist in each Chroma collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = {c.name: c.count() for c in client.list_collections()}
    topics_out = []
    seen_collections = set()

    for label in TOPICS:
        cname = topic_to_collection(label)
        seen_collections.add(cname)
        topics_out.append({"topic": label, "collection": cname, "count": existing.get(cname, 0)})

    for cname, count in existing.items():
        if cname not in seen_collections:
            label = cname.removeprefix("papers_").replace("_", " ").title()
            topics_out.append({"topic": label, "collection": cname, "count": count})
    return topics_out


def build_where_filter(
    min_year: int | None = None,
    max_year: int | None = None,
    min_citations: int | None = None,
    source: str | None = None,
    section: str | None = None,
) -> dict | None:
    """Translate optional query parameters into a ChromaDB `where` filter dict."""
    clauses: list[dict] = []
    if min_year is not None:
        clauses.append({"year": {"$gte": int(min_year)}})
    if max_year is not None:
        clauses.append({"year": {"$lte": int(max_year)}})
    if min_citations is not None:
        clauses.append({"citation_count": {"$gte": int(min_citations)}})
    if source:
        clauses.append({"source": {"$eq": source.strip().lower()}})
    if section:
        clauses.append({"section": {"$eq": section.strip().lower()}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def search(
    question: str,
    k: int = 5,
    rerank: bool = True,
    topic: str | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_citations: int | None = None,
    source: str | None = None,
    section: str | None = None,
) -> list[dict]:
    collection_name = topic_to_collection(topic)
    col = _collection() if collection_name == COLLECTION_NAME else _named_collection(collection_name)

    base_candidates = max(k, RERANK_CANDIDATES) if rerank else k
    if hasattr(col, "count"):
        total = col.count()
        if total == 0:
            return []
        n_candidates = min(max(base_candidates, k * 3), total)
    else:
        n_candidates = base_candidates

    where = build_where_filter(
        min_year=min_year,
        max_year=max_year,
        min_citations=min_citations,
        source=source,
        section=section,
    )

    query_kwargs: dict = {
        "query_embeddings": embed([question]),
        "n_results": n_candidates,
    }
    if where:
        query_kwargs["where"] = where

    results = col.query(**query_kwargs)
    if not results["metadatas"] or not results["metadatas"][0]:
        return []

    candidates = []
    for metadata, doc_text, distance in zip(
        results["metadatas"][0], results["documents"][0], results["distances"][0]
    ):
        abstract = metadata.get("abstract") or doc_text
        candidates.append(
            {
                "paper_id": metadata.get("paper_id") or metadata["title"],
                "title": metadata["title"],
                "abstract": abstract,
                "matched_chunk": doc_text,
                "matched_section": metadata.get("section", "abstract"),
                "authors": metadata["authors"].split("; ") if metadata["authors"] else [],
                "year": metadata.get("year"),
                "source": metadata["source"],
                "citation_count": metadata.get("citation_count"),
                "url": metadata["url"],
                "similarity": round(1 - distance, 4),
            }
        )

    if rerank and len(candidates) > 1:
        scores = rerank_scores(
            question,
            [f"{p['title']} [{p['matched_section']}]. {p['matched_chunk']}" for p in candidates],
        )
        candidates = [p for _, p in sorted(zip(scores, candidates), key=lambda pair: -pair[0])]

    unique_papers: list[dict] = []
    seen_ids: set[str] = set()
    for item in candidates:
        pid = item["paper_id"]
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        unique_papers.append(item)
        if len(unique_papers) >= k:
            break

    return unique_papers


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how do you reduce hallucination in RAG?"
    for i, paper in enumerate(search(question), 1):
        print(
            f"{i}. [{paper['similarity']:.3f}] {paper['title']} ({paper['year']}) "
            f"[{paper['matched_section']}]\n   {paper['url']}"
        )