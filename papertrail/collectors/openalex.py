"""Collect papers from the OpenAlex API (https://docs.openalex.org/).

Run it with:  python -m papertrail.collectors.openalex --max 300

Works without a key. Set OPENALEX_API_KEY in .env for a bigger daily budget.
"""

import argparse
import time

import requests

from papertrail.collectors.common import get_with_retry, save
from papertrail.config import OPENALEX_API_KEY, TOPIC_TERMS
from papertrail.schema import Paper

API_URL = "https://api.openalex.org/works"
SELECT = "id,title,publication_year,authorships,cited_by_count,doi,abstract_inverted_index,primary_location"
PAGE_SIZE = 100
DELAY_SECONDS = 0.5


def rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, spots in inverted_index.items() for pos in spots]
    return " ".join(word for _, word in sorted(positions))


def parse_work(work: dict) -> Paper:
    landing_page = (work.get("primary_location") or {}).get("landing_page_url") or ""
    return Paper(
        title=" ".join((work.get("title") or "").split()),
        abstract=" ".join(rebuild_abstract(work.get("abstract_inverted_index")).split()),
        authors=[
            a["author"]["display_name"]
            for a in work.get("authorships") or []
            if (a.get("author") or {}).get("display_name")
        ],
        year=work.get("publication_year"),
        source="openalex",
        citation_count=work.get("cited_by_count"),
        url=work.get("doi") or landing_page or work.get("id", ""),
    )


def fetch_papers(max_results: int, terms: list[str] = TOPIC_TERMS) -> list[Paper]:
    """Split the budget across the topic terms and skip papers already seen."""
    per_term = -(-max_results // len(terms))  # round up
    seen_ids: set[str] = set()
    papers: list[Paper] = []

    for term in terms:
        kept_for_term = 0
        page = 1
        while kept_for_term < per_term and len(papers) < max_results:
            params = {
                "filter": f'title_and_abstract.search:"{term}",has_abstract:true',
                "sort": "relevance_score:desc",
                "select": SELECT,
                "per-page": PAGE_SIZE,
                # Plain page numbers: without a key, OpenAlex rejects cursor paging past page 1
                "page": page,
            }
            if OPENALEX_API_KEY:
                params["api_key"] = OPENALEX_API_KEY
            try:
                data = get_with_retry(API_URL, params).json()
            except requests.RequestException as error:
                print(f"OpenAlex stopped early: {error}")
                return papers
            works = data.get("results") or []
            if not works:
                break

            for work in works:
                if work["id"] in seen_ids or not work.get("title"):
                    continue
                seen_ids.add(work["id"])
                papers.append(parse_work(work))
                kept_for_term += 1
                if kept_for_term >= per_term or len(papers) >= max_results:
                    break

            print(f'OpenAlex: {len(papers)} papers (searching "{term}")')
            page += 1
            time.sleep(DELAY_SECONDS)

    return papers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect papers from OpenAlex.")
    parser.add_argument("--max", type=int, default=300, help="how many papers to fetch")
    args = parser.parse_args()
    save(fetch_papers(args.max), "openalex.jsonl")
