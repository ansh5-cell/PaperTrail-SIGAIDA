"""The single table every source gets converted into."""

from dataclasses import asdict, dataclass, field


@dataclass
class Paper:
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    source: str = ""  # "arxiv", "semantic_scholar", or "openalex"
    citation_count: int | None = None  # arXiv doesn't provide this
    url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PaperChunk:
    """A section-aware text chunk from a paper's abstract or full PDF."""

    chunk_id: str
    paper_id: str
    text: str
    section: str = "abstract"
    chunk_index: int = 0
    title: str = ""
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    source: str = ""
    citation_count: int | None = None
    url: str = ""