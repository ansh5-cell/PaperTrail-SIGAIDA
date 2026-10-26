import xml.etree.ElementTree as ET

from papertrail.collectors import arxiv
from papertrail.collectors.openalex import parse_work, rebuild_abstract
from papertrail.collectors.semantic_scholar import parse_item

ARXIV_ENTRY = """
<entry xmlns="http://www.w3.org/2005/Atom">
  <id>http://arxiv.org/abs/2401.00001v1</id>
  <published>2024-01-02T00:00:00Z</published>
  <title>A  Paper
    About RAG</title>
  <summary>  Line one
  line two. </summary>
  <author><name>Ada Lovelace</name></author>
  <author><name>Alan Turing</name></author>
  <link href="http://arxiv.org/abs/2401.00001v1" rel="alternate" type="text/html"/>
  <link title="pdf" href="https://arxiv.org/pdf/2401.00001v1" rel="related" type="application/pdf"/>
</entry>
"""


def test_arxiv_query_includes_every_term_and_categories():
    query = arxiv.build_query(["large language model", "rag"])
    assert 'abs:"large language model" OR abs:"rag"' in query
    assert "cat:cs.CL" in query


def test_arxiv_parse_entry():
    paper = arxiv.parse_entry(ET.fromstring(ARXIV_ENTRY))
    assert paper.title == "A Paper About RAG"
    assert paper.abstract == "Line one line two."
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.year == 2024
    assert paper.source == "arxiv"
    assert paper.citation_count is None
    assert paper.url == "https://arxiv.org/pdf/2401.00001v1"


def test_openalex_rebuild_abstract_orders_words():
    assert rebuild_abstract({"world": [1], "hello": [0], "again": [3], "hello,": [2]}) == "hello world hello, again"
    assert rebuild_abstract(None) == ""


def test_openalex_parse_work_prefers_doi_link():
    work = {
        "id": "https://openalex.org/W1",
        "title": "Graph RAG",
        "publication_year": 2025,
        "authorships": [{"author": {"display_name": "Grace Hopper"}}, {"author": {}}],
        "cited_by_count": 7,
        "doi": "https://doi.org/10.1/x",
        "abstract_inverted_index": {"Graphs": [0], "help.": [1]},
        "primary_location": {"landing_page_url": "https://example.org/paper"},
    }
    paper = parse_work(work)
    assert paper.abstract == "Graphs help."
    assert paper.authors == ["Grace Hopper"]
    assert paper.citation_count == 7
    assert paper.url == "https://doi.org/10.1/x"

    work["doi"] = None
    assert parse_work(work).url == "https://example.org/paper"


def test_semantic_scholar_parse_item():
    paper = parse_item({
        "paperId": "abc", "title": " RAG  for\nQA ", "abstract": "An abstract.", "year": 2024,
        "authors": [{"name": "A B"}, {"name": None}], "citationCount": 12,
        "url": "https://www.semanticscholar.org/paper/abc",
    })
    assert paper.title == "RAG for QA"
    assert paper.authors == ["A B"]
    assert paper.citation_count == 12
    assert paper.source == "semantic_scholar"
