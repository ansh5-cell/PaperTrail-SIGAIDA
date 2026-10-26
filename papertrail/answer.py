"""Answer a question with Claude, using the most related papers as sources (RAG).

Try it with:  python -m papertrail.answer "how do you reduce hallucination in RAG?"
Needs ANTHROPIC_API_KEY in .env.
"""

import sys
from functools import lru_cache

import anthropic

from papertrail.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from papertrail.search import search

SYSTEM_PROMPT = """You answer research questions using only the numbered papers provided.
Cite papers inline with their number in square brackets, like [1] or [2][4], right after the claim they support.
If the papers don't contain enough information to answer, say so plainly instead of filling the gap from memory.
Keep the answer concise: a few short paragraphs at most."""


class AnswerError(Exception):
    """Raised when Claude can't produce an answer (bad key, refusal, network trouble)."""


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def format_papers(papers: list[dict]) -> str:
    blocks = []
    for i, paper in enumerate(papers, 1):
        authors = ", ".join(paper["authors"][:5]) + (" et al." if len(paper["authors"]) > 5 else "")
        blocks.append(
            f"[{i}] {paper['title']} ({paper['year']})\nAuthors: {authors}\nAbstract: {paper['abstract']}"
        )
    return "\n\n".join(blocks)


def answer(question: str, k: int = 5) -> dict:
    if not ANTHROPIC_API_KEY:
        raise AnswerError("No Claude API key found. Add ANTHROPIC_API_KEY to your .env file (see .env.example).")

    papers = search(question, k)
    prompt = f"Papers:\n\n{format_papers(papers)}\n\nQuestion: {question}"

    try:
        response = _client().beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "medium"},
            # If the model declines for policy reasons, let the API retry on its default fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as error:
        raise AnswerError("Claude rejected the API key. Check ANTHROPIC_API_KEY in .env.") from error
    except anthropic.RateLimitError as error:
        raise AnswerError("Claude rate limit hit. Wait a minute and try again.") from error
    except anthropic.APIStatusError as error:
        raise AnswerError(f"Claude API error ({error.status_code}): {error.message}") from error
    except anthropic.APIConnectionError as error:
        raise AnswerError("Couldn't reach the Claude API. Check your internet connection.") from error

    if response.stop_reason == "refusal":
        raise AnswerError("Claude declined to answer this question.")

    text = "".join(block.text for block in response.content if block.type == "text")
    return {"question": question, "answer": text, "papers": papers, "model": response.model}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how do you reduce hallucination in RAG?"
    result = answer(question)
    print(result["answer"], "\n\nSources:")
    for i, paper in enumerate(result["papers"], 1):
        print(f"[{i}] {paper['title']} ({paper['year']}) {paper['url']}")
