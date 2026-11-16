"""Shared settings for Paper Trail."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
PDF_CACHE_DIR = DATA_DIR / "pdfs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 30  # how many embedding-search results the reranker re-orders
COLLECTION_NAME = "papers"

CLAUDE_MODEL = "claude-opus-5-5"

load_dotenv(ROOT / ".env")

# Needed for /ask (Claude answers). Get one at https://console.anthropic.com
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Optional keys. Everything works without them, just with stricter rate limits.
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")

# Starting topic: LLMs & RAG. Each collector turns these into its own query syntax.
DEFAULT_TOPIC = "LLMs & RAG"
TOPIC_TERMS = [
    "retrieval augmented generation",
    "large language model",
    "LLM hallucination",
]

TOPICS: dict[str, list[str]] = {
    DEFAULT_TOPIC: TOPIC_TERMS,
    "Computer Vision & Multimodal": [
        "vision language model",
        "multimodal representation learning",
        "image segmentation diffusion",
    ],
    "Robotics & Embodied AI": [
        "robot manipulation reinforcement learning",
        "sim-to-real transfer robotics",
        "embodied AI planning",
    ],
    "AI Agents & Reasoning": [
        "autonomous LLM agents",
        "chain of thought reasoning",
        "tool use function calling LLM",
    ],
}

# Full-text PDF parsing & chunking settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MAX_PDF_PAGES = 25


def topic_to_collection(topic: str | None) -> str:
    """Map a topic name to a ChromaDB collection name (defaults to 'papers')."""
    if not topic or topic.strip().lower() in {"default", DEFAULT_TOPIC.lower(), COLLECTION_NAME}:
        return COLLECTION_NAME
    slug = re.sub(r"[^a-z0-9]+", "_", topic.strip().lower()).strip("_")
    return f"papers_{slug}"[:63].rstrip("_")