# Paper Trail

APIs that let an LLM look up real research papers instead of relying only on what it already knows.

Paper Trail collects papers (titles, abstracts, authors, and more) from free research APIs, combines them into
one table, and makes them searchable by **meaning** rather than exact keywords. When someone asks a question,
the most related papers are handed to Claude, which answers using those papers and cites them. This is
retrieval-augmented generation (RAG).

The starting topic is **LLMs and RAG**. You can change it in `papertrail/config.py`.

## How it works

```
arXiv ────────────┐
Semantic Scholar ─┼─► collect ─► clean + merge ─► embed (MiniLM) ─► Chroma vector DB
OpenAlex ─────────┘                                                      │
                                                                         ▼
question ─► embed question ─► top 30 by cosine similarity ─► rerank ─► top 5 papers
                                                                         │
                                                      /search ◄──────────┤
                                                                         ▼
                                                      /ask ◄── Claude answers with [n] citations
```

1. **Collect.** Each collector pulls papers for the topic terms and converts them into one shared format:
   title, abstract, authors, year, source, citation count, and link.
2. **Clean and merge.** Papers from all sources go into one table. Duplicates are matched by normalized title,
   keeping the most complete value for each field (for example, the arXiv PDF link and the OpenAlex citation
   count). Fake abstracts (citation text or repository boilerplate that some sources return) are removed, and
   the real abstract is looked up on arXiv by title when possible. Papers with no usable abstract are dropped.
3. **Index.** Each paper's title and abstract are turned into a vector with the pretrained
   [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model and stored in a local
   [Chroma](https://www.trychroma.com/) vector database.
4. **Search.** A question is embedded the same way, and the 30 closest papers by cosine similarity become
   candidates. A cross-encoder reranker
   ([ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)) then reads the
   question next to each candidate and re-orders them. This is slower than comparing vectors, but much more
   accurate.
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
- `--no-backfill` skips the arXiv abstract lookups (works offline).

Each step can also run on its own:

```bash
python -m papertrail.collectors.arxiv --max 300
python -m papertrail.collectors.openalex --max 300
python -m papertrail.collectors.semantic_scholar --max 300
python -m papertrail.merge
python -m papertrail.index
```

**Measure search quality:**

```bash
python -m papertrail.evaluate
```

This runs the test questions in `eval/questions.json`, each listing the papers a good search should find, and
writes [`eval/RESULTS.md`](eval/RESULTS.md). Add `--no-rerank` to compare against embedding search alone.

| Search | Hit@5 | Recall@5 | MRR |
|---|---|---|---|
| Embeddings only (all-MiniLM-L6-v2) | 65% | 59% | 0.39 |
| Keyword only (BM25) | 70% | 57% | 0.50 |
| Hybrid (embeddings + BM25) | 85% | 67% | 0.62 |
| **Embeddings + reranker (current)** | **95%** | **78%** | **0.77** |

Reranking costs about 0.3 seconds per search on a laptop CPU, compared with about 0.03 seconds without it.

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
  clean.py         remove fake abstracts and recover real ones from arXiv
  merge.py         combine sources and remove duplicates
  embeddings.py    all-MiniLM-L6-v2 wrapper
  index.py         build the Chroma vector database
  search.py        cosine-similarity search, then reranking
  rerank.py        cross-encoder reranker
  answer.py        RAG answers with Claude
  api.py           FastAPI app (/, /search, /ask)
  pipeline.py      run everything in one command
  evaluate.py      measure search quality on test questions
  static/          the web page
eval/             test questions and the latest results report
tests/             pytest suite (run with: pytest)
data/              collected papers and the vector DB (generated, not committed)
```

## Tests

```bash
pytest
```

The tests use fakes for the external APIs and Claude, so they run offline without any keys.

## Verify the live demo

1. Set `SEMANTIC_SCHOLAR_API_KEY` and `ANTHROPIC_API_KEY` in your local `.env`. In a fresh checkout,
   collect a small batch first.

   ```bash
   python -m papertrail.collectors.semantic_scholar --max 30
   ```

   Check that `data/semantic_scholar.jsonl` contains records with titles, abstracts, and paper links.
   If the collector reports HTTP 429 or saves no papers, resolve that before proceeding.

2. Build the index from the collected papers and start the app.

   ```bash
   python -X utf8 -m papertrail.pipeline --no-collect --no-backfill
   uvicorn papertrail.api:app
   ```

3. Search for a topic covered by the abstracts, then ask a specific question the returned papers can
   answer. Check that each citation opens the corresponding source card and that its abstract supports the
   claim. Ask a second question whose answer is absent from the retrieved abstracts, such as the name of
   a paper's internal code-review tool. Check that the answer says the sources lack that detail.

The offline tests cover the collector and the `/ask` request and response path with simulated provider
replies. The live steps check provider access and the quality of actual answers.

## Status

- [x] arXiv, OpenAlex, and Semantic Scholar collectors
- [x] Merge and dedupe into one table
- [x] Embeddings and Chroma vector database
- [x] `/search` endpoint
- [x] `/ask` endpoint with Claude and citations
- [x] Web page
- [x] Clean fake abstracts and recover real ones from arXiv
- [x] Evaluate retrieval quality on a set of test questions
- [ ] Collect Semantic Scholar data (waiting on an API key)
- [ ] Test `/ask` against the live Claude API
- [x] Improve search quality with a reranker (Hit@5 65% → 95%)
