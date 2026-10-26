from papertrail.merge import combine, merge, normalize_title
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


def test_merge_dedupes_by_title_and_drops_unusable_rows():
    papers = [
        make("RAG: A Survey", "arxiv"),
        make("rag - a survey", "openalex"),
        make("Another Paper", "openalex"),
        make("No Abstract", "openalex", abstract=""),
        make("", "openalex"),
    ]
    merged = merge(papers)
    assert [p.title for p in merged] == ["RAG: A Survey", "Another Paper"]
    assert merged[0].source == "arxiv,openalex"
