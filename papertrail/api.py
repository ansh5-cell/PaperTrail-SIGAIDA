"""The Paper Trail web API.

Run it with:  uvicorn papertrail.api:app --reload
Then open http://127.0.0.1:8000 for the web page, or http://127.0.0.1:8000/docs to try the endpoints.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from papertrail.answer import AnswerError, answer
from papertrail.search import list_topics, search

app = FastAPI(title="Paper Trail", description="Look up real research papers for an LLM.")

INDEX_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(INDEX_PAGE)


@app.get("/topics")
def get_topics() -> dict:
    return {"topics": list_topics()}


def _filter_kwargs(
    topic: str | None,
    min_year: int | None,
    max_year: int | None,
    min_citations: int | None,
    source: str | None,
    section: str | None,
) -> dict:
    raw = {
        "topic": topic,
        "min_year": min_year,
        "max_year": max_year,
        "min_citations": min_citations,
        "source": source,
        "section": section,
    }
    return {k: v for k, v in raw.items() if v is not None and v != ""}


@app.get("/search")
def search_papers(
    q: str = Query(..., min_length=1, description="The question or topic to search for"),
    k: int = Query(5, ge=1, le=20, description="How many papers to return"),
    topic: str | None = Query(None, description="Research topic collection to search"),
    min_year: int | None = Query(None, ge=1900, le=2100, description="Minimum publication year"),
    max_year: int | None = Query(None, ge=1900, le=2100, description="Maximum publication year"),
    min_citations: int | None = Query(None, ge=0, description="Minimum citation count"),
    source: str | None = Query(None, description="Filter by source (arxiv, semantic_scholar, openalex)"),
    section: str | None = Query(None, description="Filter by matched section (abstract, methodology, results, etc.)"),
) -> dict:
    filters = _filter_kwargs(topic, min_year, max_year, min_citations, source, section)
    payload: dict = {"query": q, "papers": search(q, k, **filters)}
    if filters:
        payload["filters"] = filters
    return payload

@app.get("/ask")
def ask(
    q: str = Query(..., min_length=1, description="The question to answer"),
    k: int = Query(5, ge=1, le=20, description="How many papers to give Claude"),
    topic: str | None = Query(None, description="Research topic collection to search"),
    min_year: int | None = Query(None, ge=1900, le=2100, description="Minimum publication year"),
    max_year: int | None = Query(None, ge=1900, le=2100, description="Maximum publication year"),
    min_citations: int | None = Query(None, ge=0, description="Minimum citation count"),
    source: str | None = Query(None, description="Filter by source (arxiv, semantic_scholar, openalex)"),
    section: str | None = Query(None, description="Filter by matched section (abstract, methodology, results, etc.)"),
) -> dict:
    filters = _filter_kwargs(topic, min_year, max_year, min_citations, source, section)
    try:
        return answer(q, k, **filters)
    except AnswerError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error