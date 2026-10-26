"""Collect papers from the Semantic Scholar API (https://api.semanticscholar.org/api-docs/graph).

Run it with:  python -m papertrail.collectors.semantic_scholar --max 300

Works without a key, but the shared free tier often answers "429 Too Many Requests".
Set SEMANTIC_SCHOLAR_API_KEY in .env to get your own rate limit.
"""

import argparse
import time

import requests

from papertrail.collectors.common import get_with_retry, save
from papertrail.config import SEMANTIC_SCHOLAR_API_KEY, TOPIC_TERMS
from papertrail.schema import Paper

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,authors,venue,citationCount,url,externalIds"
PAGE_SIZE = 100
DELAY_SECONDS = 1.5


def parse_item(item: dict) -> Paper:
    return Paper(
        title=" ".join((item.get("title") or "").split()),
        abstract=" ".join((item.get("abstract") or "").split()),
        authors=[a["name"] for a in item.get("authors") or [] if a.get("name")],
        year=item.get("year"),
        source="semantic_scholar",
        citation_count=item.get("citationCount"),
        url=item.get("url") or "",
    )


def fetch_papers(max_results: int, terms: list[str] = TOPIC_TERMS) -> list[Paper]:
    """Split the budget across the topic terms and skip papers already seen."""
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else None
    per_term = -(-max_results // len(terms))  # round up
    seen_ids: set[str] = set()
    papers: list[Paper] = []

    for term in terms:
        kept_for_term = 0
        offset = 0
        while kept_for_term < per_term and len(papers) < max_results:
            params = {"query": term, "fields": FIELDS, "offset": offset, "limit": PAGE_SIZE}
            try:
                data = get_with_retry(API_URL, params, headers).json()
            except requests.RequestException as error:
                print(f"Semantic Scholar stopped early: {error}")
                if not SEMANTIC_SCHOLAR_API_KEY:
                    print("  Tip: add SEMANTIC_SCHOLAR_API_KEY to .env for your own rate limit.")
                return papers
            items = data.get("data") or []
            if not items:
                break

            for item in items:
                # Papers with no abstract are useless for semantic search
                if item["paperId"] in seen_ids or not item.get("abstract"):
                    continue
                seen_ids.add(item["paperId"])
                papers.append(parse_item(item))
                kept_for_term += 1
                if kept_for_term >= per_term or len(papers) >= max_results:
                    break

            print(f'Semantic Scholar: {len(papers)} papers (searching "{term}")')
            offset += PAGE_SIZE
            if "next" not in data:
                break
            time.sleep(DELAY_SECONDS)

    return papers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect papers from Semantic Scholar.")
    parser.add_argument("--max", type=int, default=300, help="how many papers to fetch")
    args = parser.parse_args()
    save(fetch_papers(args.max), "semantic_scholar.jsonl")
