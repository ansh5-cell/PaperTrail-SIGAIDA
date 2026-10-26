"""Collect papers from the arXiv API (https://info.arxiv.org/help/api/).

Run it with:  python -m papertrail.collectors.arxiv --max 300
"""

import argparse
import time
import xml.etree.ElementTree as ET

from papertrail.collectors.common import get_with_retry, save
from papertrail.config import TOPIC_TERMS
from papertrail.schema import Paper

API_URL = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
PAGE_SIZE = 100
DELAY_SECONDS = 3  # arXiv asks for at least 3 seconds between requests


def build_query(terms: list[str]) -> str:
    """Match any topic term in the abstract, limited to AI/NLP/ML categories."""
    term_part = " OR ".join(f'abs:"{t}"' for t in terms)
    return f"({term_part}) AND (cat:cs.CL OR cat:cs.AI OR cat:cs.LG OR cat:cs.IR)"


def clean(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_entry(entry: ET.Element) -> Paper:
    pdf_link = ""
    for link in entry.findall("atom:link", NS):
        if link.get("title") == "pdf":
            pdf_link = link.get("href", "")
    published = entry.findtext("atom:published", "", NS)

    return Paper(
        title=clean(entry.findtext("atom:title", "", NS)),
        abstract=clean(entry.findtext("atom:summary", "", NS)),
        authors=[clean(a.findtext("atom:name", "", NS)) for a in entry.findall("atom:author", NS)],
        year=int(published[:4]) if published else None,
        source="arxiv",
        citation_count=None,
        url=pdf_link or entry.findtext("atom:id", "", NS),
    )


def fetch_papers(max_results: int, terms: list[str] = TOPIC_TERMS) -> list[Paper]:
    query = build_query(terms)
    papers: list[Paper] = []

    for start in range(0, max_results, PAGE_SIZE):
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(PAGE_SIZE, max_results - start),
            "sortBy": "relevance",
        }
        response = get_with_retry(API_URL, params)

        entries = ET.fromstring(response.content).findall("atom:entry", NS)
        if not entries:
            break
        papers.extend(parse_entry(e) for e in entries)
        print(f"arXiv: fetched {len(papers)} papers")

        if start + PAGE_SIZE < max_results:
            time.sleep(DELAY_SECONDS)

    return papers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect papers from arXiv.")
    parser.add_argument("--max", type=int, default=300, help="how many papers to fetch")
    args = parser.parse_args()
    save(fetch_papers(args.max), "arxiv.jsonl")
