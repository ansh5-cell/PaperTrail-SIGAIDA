"""Build everything in one command: collect papers, merge them, and index them for search.

Run it with:  python -m papertrail.pipeline --max 300
Skip a source:  python -m papertrail.pipeline --skip semantic_scholar
Reuse already-collected data:  python -m papertrail.pipeline --no-collect
"""

import argparse
import time

from papertrail.collectors import arxiv, openalex, semantic_scholar
from papertrail.collectors.common import save
from papertrail.index import build_index
from papertrail.merge import load_papers, merge, save_csv

COLLECTORS = {
    "arxiv": arxiv.fetch_papers,
    "semantic_scholar": semantic_scholar.fetch_papers,
    "openalex": openalex.fetch_papers,
}


def run(max_per_source: int, skip: list[str], collect: bool = True) -> None:
    started = time.time()

    if collect:
        for name, fetch in COLLECTORS.items():
            if name in skip:
                print(f"\n== Skipping {name} ==")
                continue
            print(f"\n== Collecting from {name} ==")
            save(fetch(max_per_source), f"{name}.jsonl")

    print("\n== Merging and removing duplicates ==")
    all_papers = load_papers()
    combined = merge(all_papers)
    print(f"{len(all_papers)} papers in, {len(combined)} unique papers out")
    save(combined, "papers.jsonl")
    save_csv(combined)

    print("\n== Building the vector database ==")
    build_index()

    print(f"\nDone in {time.time() - started:.0f}s. Start the API with: uvicorn papertrail.api:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect, merge, and index papers.")
    parser.add_argument("--max", type=int, default=300, help="papers to fetch from each source")
    parser.add_argument("--skip", nargs="*", default=[], choices=list(COLLECTORS), help="sources to skip")
    parser.add_argument("--no-collect", action="store_true", help="reuse data already in data/")
    args = parser.parse_args()
    run(args.max, args.skip, collect=not args.no_collect)
