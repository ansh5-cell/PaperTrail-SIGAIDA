# Paper Trail

APIs that let an LLM look up real research papers instead of relying only on what it already knows.

Paper Trail collects papers (titles, abstracts, authors, and more) from free research APIs, combines them into
one table, and makes them searchable by **meaning** rather than exact keywords. When someone asks a question,
the most related papers are handed to Claude, which answers using those papers and cites them. This is
retrieval-augmented generation (RAG).

The starting topic is **LLMs and RAG**. You can change it in `papertrail/config.py`.

## How it works

```
 arXiv ─────────┐
 Semantic Scholar ┼─► collectors ─► merge + dedupe ─► embeddings ─► Chroma vector DB
 OpenAlex ──────┘    (one schema)    (by title)      (MiniLM)          │
                                                                        ▼
                         question ─► embed question ─► top 5 by cosine similarity
                                                                        │
                                                         /search ◄──────┤
                                                                        ▼
                                                         /ask ◄── Claude answers with [n] citations
```

1. **Collect.** Each collector pulls papers for the topic terms and converts them into one shared format:
   title, abstract, authors, year, source, citation count, and link.
2. **Merge.** Papers from all sources go into one table. Duplicates are matched by normalized title, keeping
   the most complete value for each field (for example, the arXiv PDF link and the OpenAlex citation count).
3. **Index.** Each paper's title and abstract are turned into a vector with the pretrained
   [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model and stored in a local
   [Chroma](https://www.trychroma.com/) vector database.
4. **Search.** A question is embedded the same way and compared to every paper with cosine similarity.
5. **Answer.** The top papers and the question go to Claude, which answers using only those papers.

## Data sources

| Source | Fields used | Notes |
|---|---|---|
| [arXiv API](https://info.arxiv.org/help/api/) | title, summary, authors, published date, categories, PDF link | No citation counts. Limited to cs.CL, cs.AI, cs.LG, and cs.IR. |
| [Semantic Scholar API](https://api.semanticscholar.org/api-docs/graph) | title, abstract, year, authors, venue, citationCount | The free tier is often rate-limited, so a key is recommended. |
| [OpenAlex](https://docs.openalex.org/) | title, publication_year, authorships, cited_by_count, doi | Abstracts are rebuilt from OpenAlex's word-position index. |

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/ansh5-cell/PaperTrail-SIGAIDA.git
cd PaperTrail-SIGAIDA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Then open `.env` and add your keys:

- `ANTHROPIC_API_KEY`: needed for `/ask`. Get one at [console.anthropic.com](https://console.anthropic.com).
- `SEMANTIC_SCHOLAR_API_KEY` and `OPENALEX_API_KEY`: optional, for higher rate limits.

`.env` is in `.gitignore`, so your keys never get committed.

## Usage

**Build the database** (collect, merge, and index in one go):

```bash
python -m papertrail.pipeline --max 300
```

Useful options:

- `--skip semantic_scholar` skips a source.
- `--no-collect` rebuilds from data you've already collected.

Each step can also run on its own:

```bash
python -m papertrail.collectors.arxiv --max 300
python -m papertrail.collectors.openalex --max 300
python -m papertrail.collectors.semantic_scholar --max 300
python -m papertrail.merge
python -m papertrail.index
```

**Try it from the command line:**

```bash
python -m papertrail.search "how do you reduce hallucination in RAG?"
python -m papertrail.answer "how do you reduce hallucination in RAG?"
```

**Run the API and web page:**

```bash
uvicorn papertrail.api:app --reload
```

Then open:

- http://127.0.0.1:8000 for the web page
- http://127.0.0.1:8000/docs for interactive API docs

## API

| Endpoint | What it returns |
|---|---|
| `GET /search?q=...&k=5` | The `k` most related papers (1–20). Each has title, abstract, authors, year, source, citation count, URL, and similarity score. |
| `GET /ask?q=...&k=5` | Claude's answer with inline `[n]` citations, plus the `k` papers it was given. Returns 502 with a readable message if Claude can't answer (for example, a missing key). |

Example:

```bash
curl "http://127.0.0.1:8000/search?q=graph%20retrieval&k=3"
```

## Project layout

```
papertrail/
  collectors/      arxiv.py, semantic_scholar.py, openalex.py, common.py (retries, saving)
  schema.py        the shared Paper format
  merge.py         combine sources and remove duplicates
  embeddings.py    all-MiniLM-L6-v2 wrapper
  index.py         build the Chroma vector database
  search.py        cosine-similarity search
  answer.py        RAG answers with Claude
  api.py           FastAPI app (/, /search, /ask)
  pipeline.py      run everything in one command
  static/          the web page
tests/             pytest suite (run with: pytest)
data/              collected papers and the vector DB (generated, not committed)
```

## Tests

```bash
pytest
```

The tests use fakes for the external APIs and Claude, so they run offline without any keys.

## Status

- [x] arXiv, OpenAlex, and Semantic Scholar collectors
- [x] Merge and dedupe into one table
- [x] Embeddings and Chroma vector database
- [x] `/search` endpoint
- [x] `/ask` endpoint with Claude and citations
- [x] Web page
- [ ] Collect Semantic Scholar data (waiting on an API key)
- [ ] Test `/ask` against the live Claude API
- [ ] Evaluate retrieval quality on a set of test questions
