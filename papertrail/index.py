"""Turn every paper into a vector and store it in the Chroma vector database.

Run it with:  python -m papertrail.index

Reads data/papers.jsonl (from papertrail.merge) and rebuilds data/chroma/ from scratch.
"""

import hashlib
import json

import chromadb

from papertrail.config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR
from papertrail.embeddings import embed
from papertrail.merge import normalize_title
from papertrail.schema import Paper

BATCH_SIZE = 64


def paper_text(paper: Paper) -> str:
    """What gets embedded: the title plus the abstract."""
    return f"{paper.title}. {paper.abstract}"


def paper_id(paper: Paper) -> str:
    return hashlib.sha1(normalize_title(paper.title).encode()).hexdigest()


def to_metadata(paper: Paper) -> dict:
    """Chroma metadata can't hold lists or None, so flatten those."""
    metadata = {
        "title": paper.title,
        "authors": "; ".join(paper.authors),
        "source": paper.source,
        "url": paper.url,
    }
    if paper.year is not None:
        metadata["year"] = paper.year
    if paper.citation_count is not None:
        metadata["citation_count"] = paper.citation_count
    return metadata


def get_collection(client: chromadb.ClientAPI):
    # Cosine distance, so search scores are cosine similarity (1 - distance)
    return client.get_or_create_collection(COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}})


def build_index() -> None:
    with (DATA_DIR / "papers.jsonl").open() as f:
        papers = [Paper(**json.loads(line)) for line in f if line.strip()]

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = get_collection(client)

    for start in range(0, len(papers), BATCH_SIZE):
        batch = papers[start : start + BATCH_SIZE]
        texts = [paper_text(p) for p in batch]
        collection.add(
            ids=[paper_id(p) for p in batch],
            embeddings=embed(texts),
            documents=[p.abstract for p in batch],
            metadatas=[to_metadata(p) for p in batch],
        )
        print(f"Indexed {min(start + BATCH_SIZE, len(papers))}/{len(papers)} papers")

    print(f"Vector database ready at {CHROMA_DIR} ({collection.count()} papers)")


if __name__ == "__main__":
    build_index()
