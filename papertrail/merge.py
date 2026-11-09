"""Combine every collected source into one table, removing duplicate papers by title.

Run it with:  python -m papertrail.merge
Skip the arXiv abstract lookups (offline):  python -m papertrail.merge --no-backfill

Reads whichever of data/arxiv.jsonl, data/semantic_scholar.jsonl, data/openalex.jsonl exist
and writes data/papers.jsonl (used by the search) plus data/papers.csv (easy to open in Excel/Sheets).
"""

import argparse
import csv
import json
import re

from papertrail.clean import backfill_from_arxiv, clean_papers
from papertrail.collectors.common import save
from papertrail.config import DATA_DIR
from papertrail.schema import Paper

# Earlier sources win for the link: arXiv PDFs are free to open, DOIs can be paywalled.
SOURCE_FILES = ["arxiv.jsonl", "semantic_scholar.jsonl", "openalex.jsonl"]


def normalize_title(title: str) -> str:
    """Lowercase and drop punctuation/extra spaces so 'RAG: A Survey' == 'rag a survey'."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def load_papers() -> list[Paper]:
    papers = []
    for filename in SOURCE_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"Skipping {filename} (not collected yet)")
            continue
        with path.open() as f:
            rows = [Paper(**json.loads(line)) for line in f if line.strip()]
        print(f"Loaded {len(rows)} papers from {filename}")
        papers.extend(rows)
    return papers


def combine(existing: Paper, duplicate: Paper) -> Paper:
    """Merge two copies of the same paper, keeping the most complete value for each field."""
    sources = existing.source.split(",")
    if duplicate.source not in sources:
        sources.append(duplicate.source)
    citations = [c for c in (existing.citation_count, duplicate.citation_count) if c is not None]
    years = [y for y in (existing.year, duplicate.year) if y]

    return Paper(
        title=existing.title,
        abstract=max(existing.abstract, duplicate.abstract, key=len),
        authors=max(existing.authors, duplicate.authors, key=len),
        year=min(years) if years else None,  # earliest version, e.g. the arXiv preprint
        source=",".join(sources),
        citation_count=max(citations) if citations else None,
        url=existing.url or duplicate.url,
    )


def merge(papers: list[Paper]) -> list[Paper]:
    """Combine duplicates. Papers without an abstract are kept so another copy can supply one."""
    merged: dict[str, Paper] = {}
    for paper in papers:
        key = normalize_title(paper.title)
        if not key:
            continue
        merged[key] = combine(merged[key], paper) if key in merged else paper
    return list(merged.values())


def build_table(papers: list[Paper], backfill: bool = True) -> list[Paper]:
    """Clean abstracts, merge duplicates, recover missing abstracts, and drop what's still unusable."""
    blanked = clean_papers(papers)
    print(f"Cleaned abstracts ({blanked} were citations or boilerplate, not real abstracts)")
    combined = merge(papers)

    if backfill and any(not p.abstract for p in combined):
        print("Looking up missing abstracts on arXiv...")
        print(f"Recovered {backfill_from_arxiv(combined)} abstracts from arXiv")

    usable = [p for p in combined if p.abstract]
    print(f"Dropped {len(combined) - len(usable)} papers with no usable abstract")
    return usable


def save_csv(papers: list[Paper], filename: str = "papers.csv") -> None:
    path = DATA_DIR / filename
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Paper.__dataclass_fields__))
        writer.writeheader()
        for paper in papers:
            row = paper.to_dict()
            row["authors"] = "; ".join(paper.authors)
            writer.writerow(row)
    print(f"Saved {len(papers)} papers to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge collected papers into one table.")
    parser.add_argument("--no-backfill", action="store_true", help="don't look up missing abstracts on arXiv")
    args = parser.parse_args()

    all_papers = load_papers()
    combined = build_table(all_papers, backfill=not args.no_backfill)
    print(f"{len(all_papers)} papers in, {len(combined)} unique papers out")
    save(combined, "papers.jsonl")
    save_csv(combined)
