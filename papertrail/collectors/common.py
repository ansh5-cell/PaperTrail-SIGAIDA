"""Helpers shared by all collectors."""

import json
import time

import requests

from papertrail.config import DATA_DIR
from papertrail.schema import Paper


def get_with_retry(url: str, params: dict, headers: dict | None = None, tries: int = 6) -> requests.Response:
    """GET a URL, retrying on timeouts, server errors, and "too many requests" (HTTP 429)."""
    wait = 5
    for attempt in range(tries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=60)
        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == tries - 1:
                raise
            print(f"  {type(error).__name__}, retrying in {wait}s...")
            time.sleep(wait)
            wait *= 2
            continue
        if response.status_code != 429 and response.status_code < 500:
            response.raise_for_status()
            return response
        if attempt < tries - 1:
            print(f"  got HTTP {response.status_code}, retrying in {wait}s...")
            time.sleep(wait)
            wait *= 2
    response.raise_for_status()
    return response


def save(papers: list[Paper], filename: str) -> None:
    if not papers:
        print(f"No papers collected, so {filename} was left unchanged")
        return
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / filename
    with path.open("w") as f:
        for paper in papers:
            f.write(json.dumps(paper.to_dict()) + "\n")
    print(f"Saved {len(papers)} papers to {path}")
