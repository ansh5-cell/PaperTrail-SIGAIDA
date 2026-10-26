"""The Paper Trail web API.

Run it with:  uvicorn papertrail.api:app --reload
Then open http://127.0.0.1:8000/docs to try the endpoints in your browser.
"""

from fastapi import FastAPI, Query

from papertrail.search import search

app = FastAPI(title="Paper Trail", description="Look up real research papers for an LLM.")


@app.get("/search")
def search_papers(
    q: str = Query(..., min_length=1, description="The question or topic to search for"),
    k: int = Query(5, ge=1, le=20, description="How many papers to return"),
) -> dict:
    return {"query": q, "papers": search(q, k)}
