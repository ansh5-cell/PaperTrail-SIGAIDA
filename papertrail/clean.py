"""Clean up abstracts, and recover real ones from arXiv when a source gave us junk.

Some sources (mostly OpenAlex) return a citation ("A. Author, B. Author. Proceedings of ... 2024.")
or repository boilerplate in place of the abstract. Those would pollute search results, so they're
blanked out here. Papers left without an abstract get one more chance: a title lookup on arXiv.
"""

import re
import time
import xml.etree.ElementTree as ET

import requests

from papertrail.collectors import arxiv
from papertrail.collectors.common import get_with_retry
from papertrail.schema import Paper

MIN_WORDS = 15

# Text that shows up instead of a real abstract
BOILERPLATE = re.compile(
    r"embargo|was exported from|this report was prepared as an account of|presents a panel discussion",
    re.IGNORECASE,
)
# "Name, Name. Proceedings of the ... Conference ... 2024." or "... Findings of the Association for ..."
CITATION = re.compile(r"^[^.]{0,600}\.\s+(Proceedings|Findings) of\b", re.IGNORECASE)
ABSTRACT_PREFIX = re.compile(r"^\s*abstract\b[\s:.\-–—]*", re.IGNORECASE)


def clean_abstract(text: str) -> str:
    """Return a tidied abstract, or "" if the text isn't really an abstract."""
    text = ABSTRACT_PREFIX.sub("", " ".join(text.split()))
    if len(text.split()) < MIN_WORDS or BOILERPLATE.search(text) or CITATION.search(text):
        return ""
    return text


def clean_papers(papers: list[Paper]) -> int:
    """Clean every abstract in place; return how many were blanked out as junk."""
    blanked = 0
    for paper in papers:
        cleaned = clean_abstract(paper.abstract)
        if paper.abstract and not cleaned:
            blanked += 1
        paper.abstract = cleaned
    return blanked


def find_arxiv_abstract(title: str) -> str:
    """Look up a paper on arXiv by exact title; return its abstract or ""."""
    # Imported here to avoid a circular import (merge uses this module)
    from papertrail.merge import normalize_title

    phrase = re.sub(r"[^A-Za-z0-9 ]+", " ", title).strip()
    params = {"search_query": f'ti:"{phrase}"', "max_results": 5}
    response = get_with_retry(arxiv.API_URL, params, tries=3)
    for entry in ET.fromstring(response.content).findall("atom:entry", arxiv.NS):
        candidate = arxiv.parse_entry(entry)
        if normalize_title(candidate.title) == normalize_title(title):
            return clean_abstract(candidate.abstract)
    return ""


def backfill_from_arxiv(papers: list[Paper]) -> int:
    """Fill in missing abstracts from arXiv; return how many were recovered."""
    missing = [p for p in papers if not p.abstract]
    recovered = 0
    for i, paper in enumerate(missing):
        try:
            paper.abstract = find_arxiv_abstract(paper.title)
        except requests.RequestException as error:
            print(f"  arXiv lookup failed for {paper.title[:60]!r}: {error}")
        if paper.abstract:
            recovered += 1
            print(f"  recovered abstract from arXiv: {paper.title[:70]}")
        if i < len(missing) - 1:
            time.sleep(arxiv.DELAY_SECONDS)
    return recovered
