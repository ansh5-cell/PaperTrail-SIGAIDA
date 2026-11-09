from papertrail.merge import build_table, combine, merge, normalize_title
from papertrail.schema import Paper


def make(title="RAG: A Survey", source="arxiv", **overrides):
    fields = dict(title=title, abstract="An abstract.", authors=["A"], year=2024, source=source, url="")
    fields.update(overrides)
    return Paper(**fields)


def test_normalize_title_ignores_case_and_punctuation():
    assert normalize_title("RAG: A  Survey!") == normalize_title("rag a survey")
    assert normalize_title("Retrieval-Augmented Generation") == "retrieval augmented generation"


def test_combine_keeps_most_complete_values():
    arxiv = make(source="arxiv", year=2023, citation_count=None, url="https://arxiv.org/pdf/1", authors=["A"])
    openalex = make(
        source="openalex", year=2024, citation_count=42, url="https://doi.org/x",
        authors=["A", "B"], abstract="A much longer abstract.",
    )
    merged = combine(arxiv, openalex)
    assert merged.source == "arxiv,openalex"
    assert merged.year == 2023  # earliest version wins
    assert merged.citation_count == 42
    assert merged.url == "https://arxiv.org/pdf/1"  # first source's link wins
    assert merged.authors == ["A", "B"]
    assert merged.abstract == "A much longer abstract."


def test_combine_same_source_does_not_repeat_it():
    assert combine(make(), make()).source == "arxiv"


def test_merge_dedupes_by_title_and_skips_blank_titles():
    papers = [
        make("RAG: A Survey", "arxiv"),
        make("rag - a survey", "openalex"),
        make("Another Paper", "openalex"),
        make("", "openalex"),
    ]
    merged = merge(papers)
    assert [p.title for p in merged] == ["RAG: A Survey", "Another Paper"]
    assert merged[0].source == "arxiv,openalex"


GOOD = "Retrieval augmented generation grounds a language model in documents it retrieves at answer time, " \
       "which reduces hallucination and lets it cite sources."
CITATION = "Ada Lovelace, Alan Turing. Proceedings of the 2024 Conference on Things (Volume 1: Long Papers). 2024."


def test_build_table_uses_good_copy_when_one_source_has_junk():
    papers = [
        make("ARES: An Evaluation Framework", "openalex", abstract=CITATION),
        make("ARES: an evaluation framework", "arxiv", abstract=GOOD),
        make("Only Junk", "openalex", abstract=CITATION),
    ]
    table = build_table(papers, backfill=False)
    assert [p.title for p in table] == ["ARES: An Evaluation Framework"]
    assert table[0].abstract == GOOD
