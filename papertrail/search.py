"""Find the papers most related to a question using cosine similarity in Chroma.

Try it with:  python -m papertrail.search "how do you reduce hallucination in RAG?"
"""

import sys
from functools import lru_cache

import chromadb

from papertrail.config import CHROMA_DIR
from papertrail.embeddings import embed
from papertrail.index import get_collection


@lru_cache(maxsize=1)
def _collection():
    return get_collection(chromadb.PersistentClient(path=str(CHROMA_DIR)))


def search(question: str, k: int = 5) -> list[dict]:
    results = _collection().query(query_embeddings=embed([question]), n_results=k)

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
    return papers


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how do you reduce hallucination in RAG?"
    for i, paper in enumerate(search(question), 1):
        print(f"{i}. [{paper['similarity']:.3f}] {paper['title']} ({paper['year']})\n   {paper['url']}")
