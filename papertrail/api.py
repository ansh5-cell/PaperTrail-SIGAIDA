"""The Paper Trail web API.

Run it with:  uvicorn papertrail.api:app --reload
Then open http://127.0.0.1:8000 for the web page, or http://127.0.0.1:8000/docs to try the endpoints.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from papertrail.answer import AnswerError, answer
from papertrail.search import search

app = FastAPI(title="Paper Trail", description="Look up real research papers for an LLM.")

INDEX_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(INDEX_PAGE)


@app.get("/search")
def search_papers(
    q: str = Query(..., min_length=1, description="The question or topic to search for"),
    k: int = Query(5, ge=1, le=20, description="How many papers to return"),
) -> dict:
    return {"query": q, "papers": search(q, k)}


@app.get("/ask")
def ask(
    q: str = Query(..., min_length=1, description="The question to answer"),
    k: int = Query(5, ge=1, le=20, description="How many papers to give Claude"),
) -> dict:
    try:
        return answer(q, k)
    except AnswerError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
