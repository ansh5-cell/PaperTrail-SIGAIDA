"""Re-order search candidates with a cross-encoder, which reads the question and each paper together.

The embedding search compares two separately-made vectors, which is fast but rough. A cross-encoder
looks at the question and the paper side by side, so it's slower but much more accurate. We use it
only on the top candidates from the embedding search.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from papertrail.config import RERANK_MODEL


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    # Downloads the model (~90 MB) the first time, then loads it from the local cache
    return CrossEncoder(RERANK_MODEL)


def rerank_scores(question: str, texts: list[str]) -> list[float]:
    """Relevance score for each text (higher = more relevant); only the order is meaningful."""
    if not texts:
        return []
    return [float(s) for s in get_reranker().predict([(question, text) for text in texts])]
