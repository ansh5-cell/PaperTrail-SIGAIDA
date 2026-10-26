"""Shared settings for Paper Trail."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Starting topic: LLMs & RAG. Each collector turns these into its own query syntax.
TOPIC_TERMS = [
    "retrieval augmented generation",
    "large language model",
    "LLM hallucination",
]
