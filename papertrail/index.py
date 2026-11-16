"""Turn papers (abstracts + optional full-text PDF chunks) into vectors in Chroma.

Run with:
  python -m papertrail.index
  python -m papertrail.index --topic "Robotics & Embodied AI" --full-text
"""

import argparse
import hashlib
import json
from pathlib import Path

import chromadb

from papertrail.config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, topic_to_collection
from papertrail.embeddings import embed
from papertrail.merge import normalize_title
from papertrail.pdf_parser import chunk_paper
from papertrail.schema import Paper, PaperChunk

BATCH_SIZE = 64


def paper_text(paper: Paper) -> str:
    """What gets embedded for a paper's abstract."""
    return f"{paper.title}. {paper.abstract}"


def chunk_text(chunk: PaperChunk) -> str:
    """Prefix section and title so embeddings retain document context."""
    if chunk.section == "abstract":
        return f"{chunk.title}. {chunk.text}"
    return f"{chunk.title} [{chunk.section}]. {chunk.text}"


def paper_id(paper: Paper) -> str:
    return hashlib.sha1(normalize_title(paper.title).encode()).hexdigest()


def to_metadata(item: Paper | PaperChunk) -> dict:
    """Chroma metadata can't hold lists or None, so flatten those."""
    if isinstance(item, PaperChunk):
        metadata = {
            "paper_id": item.paper_id,
            "section": item.section,
            "chunk_index": item.chunk_index,
            "title": item.title,
            "abstract": item.abstract[:1500],
            "authors": "; ".join(item.authors),
            "source": item.source,
            "url": item.url,
        }
        if item.year is not None:
            metadata["year"] = item.year
        if item.citation_count is not None:
            metadata["citation_count"] = item.citation_count
        return metadata

    # Backwards-compatible Paper path
    metadata = {
        "paper_id": paper_id(item),
        "section": "abstract",
        "chunk_index": 0,
        "title": item.title,
        "abstract": item.abstract[:1500],
        "authors": "; ".join(item.authors),
        "source": item.source,
        "url": item.url,
    }
    if item.year is not None:
        metadata["year"] = item.year
    if item.citation_count is not None:
        metadata["citation_count"] = item.citation_count
    return metadata


def get_collection(client: chromadb.ClientAPI, collection_name: str = COLLECTION_NAME):
    # Cosine distance, so search scores are cosine similarity (1 - distance)
    return client.get_or_create_collection(collection_name, configuration={"hnsw": {"space": "cosine"}})


def build_index(
    topic: str | None = None,
    use_full_text: bool = False,
    input_path: Path | None = None,
) -> None:
    collection_name = topic_to_collection(topic)
    jsonl_path = input_path or (
        DATA_DIR / "papers.jsonl"
        if collection_name == COLLECTION_NAME
        else DATA_DIR / f"{collection_name}.jsonl"
    )

    with jsonl_path.open() as f:
        papers = [Paper(**json.loads(line)) for line in f if line.strip()]

    all_chunks: list[PaperChunk] = []
    for p in papers:
        pid = paper_id(p)
        all_chunks.extend(chunk_paper(p, pid, use_full_text=use_full_text))

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    collection = get_collection(client, collection_name)

    for start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[start : start + BATCH_SIZE]
        texts = [chunk_text(c) for c in batch]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embed(texts),
            documents=[c.text for c in batch],
            metadatas=[to_metadata(c) for c in batch],
        )
        print(f"Indexed {min(start + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks")

    print(
        f"Vector database ready at {CHROMA_DIR} "
        f"[collection={collection_name}] ({collection.count()} chunks from {len(papers)} papers)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index for PaperTrail.")
    parser.add_argument("--topic", default=None, help="Topic name (maps to a Chroma collection)")
    parser.add_argument("--full-text", action="store_true", help="Download and chunk arXiv PDFs")
    args = parser.parse_args()
    build_index(topic=args.topic, use_full_text=args.full_text)