"""Find the papers most related to a question.

Two stages: cosine similarity in Chroma picks the top candidates, then a cross-encoder reranks them.
On the eval set this raised Hit@5 from 65% to 95% (see eval/RESULTS.md).

Try it with:  python -m papertrail.search "how do you reduce hallucination in RAG?"
"""

import sys
from functools import lru_cache

import chromadb

from papertrail.config import CHROMA_DIR, RERANK_CANDIDATES
from papertrail.embeddings import embed
from papertrail.index import get_collection
from papertrail.rerank import rerank_scores


@lru_cache(maxsize=1)
def _collection():
    return get_collection(chromadb.PersistentClient(path=str(CHROMA_DIR)))


def search(question: str, k: int = 5, rerank: bool = True) -> list[dict]:
    n_candidates = max(k, RERANK_CANDIDATES) if rerank else k
    results = _collection().query(query_embeddings=embed([question]), n_results=n_candidates)

    papers = []
    for metadata, abstract, distance in zip(
        results["metadatas"][0], results["documents"][0], results["distances"][0]
    ):
        papers.append(
            {
                "title": metadata["title"],
                "abstract": abstract,
                "authors": metadata["authors"].split("; ") if metadata["authors"] else [],
                "year": metadata.get("year"),
                "source": metadata["source"],
                "citation_count": metadata.get("citation_count"),
                "url": metadata["url"],
                "similarity": round(1 - distance, 4),
            }
        )

    if rerank:
        scores = rerank_scores(question, [f"{p['title']}. {p['abstract']}" for p in papers])
        papers = [p for _, p in sorted(zip(scores, papers), key=lambda pair: -pair[0])]
    return papers[:k]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how do you reduce hallucination in RAG?"
    for i, paper in enumerate(search(question), 1):
        print(f"{i}. [{paper['similarity']:.3f}] {paper['title']} ({paper['year']})\n   {paper['url']}")
