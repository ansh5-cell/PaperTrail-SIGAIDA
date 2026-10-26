"""Shared settings for Paper Trail."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "papers"

CLAUDE_MODEL = "claude-opus-5-5"

load_dotenv(ROOT / ".env")

# Needed for /ask (Claude answers). Get one at https://console.anthropic.com
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Optional keys. Everything works without them, just with stricter rate limits.
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")

# Starting topic: LLMs & RAG. Each collector turns these into its own query syntax.
TOPIC_TERMS = [
    "retrieval augmented generation",
    "large language model",
    "LLM hallucination",
]
