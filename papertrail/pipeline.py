"""Build everything in one command: collect papers, merge them, and index them for search.

Run it with:  python -m papertrail.pipeline --max 300
Multi-topic + full-text PDFs:
  python -m papertrail.pipeline --topic "AI Agents & Reasoning" --full-text
Skip a source:  python -m papertrail.pipeline --skip semantic_scholar
Reuse already-collected data:  python -m papertrail.pipeline --no-collect
"""

import argparse
import inspect
import time

from papertrail.collectors import arxiv, openalex, semantic_scholar
from papertrail.collectors.common import save
from papertrail.config import COLLECTION_NAME, DATA_DIR, DEFAULT_TOPIC, TOPICS, topic_to_collection
from papertrail.index import build_index
from papertrail.merge import build_table, load_papers, save_csv

COLLECTORS = {
    "arxiv": arxiv.fetch_papers,
    "semantic_scholar": semantic_scholar.fetch_papers,
    "openalex": openalex.fetch_papers,
}


def resolve_topic_terms(topic: str | None) -> list[str]:
    """Look up preset terms for a topic, or treat a custom topic string as the search query."""
    if not topic or topic.strip().lower() in {"default", DEFAULT_TOPIC.lower(), COLLECTION_NAME}:
        return TOPICS[DEFAULT_TOPIC]
    for preset_name, terms in TOPICS.items():
        if preset_name.lower() == topic.strip().lower():
            return terms
    return [topic.strip()]


def run(
    max_per_source: int,
    skip: list[str],
    collect: bool = True,
    backfill: bool = True,
    topic: str | None = None,
    use_full_text: bool = False,
) -> None:
    started = time.time()
    collection_name = topic_to_collection(topic)
    is_default = collection_name == COLLECTION_NAME
    terms = resolve_topic_terms(topic)

    if collect:
        for name, fetch in COLLECTORS.items():
            if name in skip:
                print(f"\n== Skipping {name} ==")
                continue
            print(f"\n== Collecting from {name} (topic: {topic or DEFAULT_TOPIC}) ==")
            sig = inspect.signature(fetch)
            papers = fetch(max_per_source, terms=terms) if "terms" in sig.parameters else fetch(max_per_source)
            out_name = f"{name}.jsonl" if is_default else f"{collection_name}_{name}.jsonl"
            save(papers, out_name)

    print("\n== Cleaning, merging, and removing duplicates ==")
    if is_default:
        all_papers = load_papers()
        merged_filename = "papers.jsonl"
    else:
        # Load topic-prefixed collector files if load_papers() only scans default names
        sig = inspect.signature(load_papers)
        if "prefix" in sig.parameters:
            all_papers = load_papers(prefix=f"{collection_name}_")
        else:
            import json
            from papertrail.schema import Paper

            all_papers = []
            for path in DATA_DIR.glob(f"{collection_name}_*.jsonl"):
                with path.open() as f:
                    all_papers.extend(Paper(**json.loads(line)) for line in f if line.strip())
        merged_filename = f"{collection_name}.jsonl"

    combined = build_table(all_papers, backfill=backfill)
    print(f"{len(all_papers)} papers in, {len(combined)} unique papers out")
    save(combined, merged_filename)
    if is_default:
        save_csv(combined)

    print("\n== Building the vector database ==")
    build_index(
        topic=topic,
        use_full_text=use_full_text,
        input_path=DATA_DIR / merged_filename,
    )

    print(f"\nDone in {time.time() - started:.0f}s. Start the API with: uvicorn papertrail.api:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect, merge, and index papers.")
    parser.add_argument("--max", type=int, default=300, help="papers to fetch from each source")
    parser.add_argument("--skip", nargs="*", default=[], choices=list(COLLECTORS), help="sources to skip")
    parser.add_argument("--no-collect", action="store_true", help="reuse data already in data/")
    parser.add_argument("--no-backfill", action="store_true", help="don't look up missing abstracts on arXiv")
    parser.add_argument("--topic", default=None, help="Topic preset name or custom search topic")
    parser.add_argument("--full-text", action="store_true", help="Download and chunk full arXiv PDFs")
    args = parser.parse_args()
    run(
        args.max,
        args.skip,
        collect=not args.no_collect,
        backfill=not args.no_backfill,
        topic=args.topic,
        use_full_text=args.full_text,
    )