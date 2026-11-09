from papertrail import clean
from papertrail.clean import backfill_from_arxiv, clean_abstract, clean_papers
from papertrail.schema import Paper

REAL = ("Large language models hallucinate when they lack knowledge, so we retrieve supporting passages "
        "and condition generation on them to improve factual accuracy.")


def test_keeps_real_abstracts():
    assert clean_abstract(REAL) == REAL


def test_strips_abstract_prefix_and_extra_spaces():
    assert clean_abstract("Abstract   " + REAL) == REAL
    assert clean_abstract("ABSTRACT: " + REAL) == REAL


def test_rejects_citations_instead_of_abstracts():
    assert clean_abstract(
        "Shahul Es, Jithin James, Luis Espinosa Anke, Steven Schockaert. Proceedings of the 18th Conference "
        "of the European Chapter of the Association for Computational Linguistics: System Demonstrations. 2024."
    ) == ""
    assert clean_abstract(
        "Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu, Yue Xing, Han Xu, Jie Ren. "
        "Findings of the Association for Computational Linguistics: ACL 2024. 2024."
    ) == ""


def test_rejects_boilerplate_and_too_short_text():
    assert clean_abstract(
        "Submission original under an indefinite embargo labeled 'Open Access'. "
        "The submission was exported from vireo on 2024-09-16 without embargo terms"
    ) == ""
    assert clean_abstract("This report was prepared as an account of") == ""
    assert clean_abstract("Too short to be useful.") == ""


def test_clean_papers_counts_blanked():
    papers = [Paper(title="a", abstract=REAL), Paper(title="b", abstract="tiny"), Paper(title="c", abstract="")]
    assert clean_papers(papers) == 1
    assert [p.abstract for p in papers] == [REAL, "", ""]


def test_backfill_only_looks_up_missing(monkeypatch):
    looked_up = []

    def fake_lookup(title):
        looked_up.append(title)
        return REAL if title == "Findable" else ""

    monkeypatch.setattr(clean, "find_arxiv_abstract", fake_lookup)
    monkeypatch.setattr(clean.time, "sleep", lambda s: None)
    papers = [Paper(title="Has One", abstract=REAL), Paper(title="Findable", abstract=""),
              Paper(title="Lost", abstract="")]

    assert backfill_from_arxiv(papers) == 1
    assert looked_up == ["Findable", "Lost"]
    assert papers[1].abstract == REAL
