"""Turns text into vectors with the pretrained all-MiniLM-L6-v2 model."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from papertrail.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    # Downloads the model (~90 MB) the first time, then loads it from the local cache
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = get_model().encode(texts, normalize_embeddings=True)
    return vectors.tolist()
