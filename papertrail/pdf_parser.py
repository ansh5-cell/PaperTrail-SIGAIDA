"""Download arXiv PDFs and split full text into section-aware chunks."""

import re
from pathlib import Path
from urllib.request import Request, urlopen

from papertrail.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_PDF_PAGES, PDF_CACHE_DIR
from papertrail.schema import Paper, PaperChunk

SECTION_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?"
    r"(abstract|introduction|related work|background|preliminaries|"
    r"method|methodology|approach|architecture|model|framework|"
    r"experiments?|experimental setup|evaluation|results?|analysis|"
    r"discussion|limitations?|conclusion|conclusions|future work)\s*:?\s*$",
    re.IGNORECASE,
)
REFERENCES_RE = re.compile(r"^(?:\d+\.?\s+)?(references|bibliography|acknowledg(?:e)?ments?)\s*$", re.IGNORECASE)


def arxiv_pdf_url(url: str) -> str | None:
    """Normalize an arXiv abs or pdf URL to a direct .pdf download URL."""
    if not url or "arxiv.org" not in url:
        return None
    normalized = url.replace("/abs/", "/pdf/")
    if not normalized.endswith(".pdf"):
        normalized = f"{normalized}.pdf"
    return normalized


def download_pdf(pdf_url: str, cache_dir: Path = PDF_CACHE_DIR, timeout: float = 15.0) -> Path | None:
    """Download a PDF to local cache if not already present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^a-zA-Z0-9._-]+", "_", pdf_url.split("/")[-1])
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    dest = cache_dir / filename
    if dest.exists() and dest.stat().st_size > 1024:
        return dest

    try:
        req = Request(pdf_url, headers={"User-Agent": "PaperTrail-SIGAIDA/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data.startswith(b"%PDF"):
            return None
        dest.write_bytes(data)
        return dest
    except Exception:
        return None


def extract_sections(pdf_path: Path, max_pages: int = MAX_PDF_PAGES) -> list[tuple[str, str]]:
    """Extract (section_name, section_text) pairs from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []

    sections: list[tuple[str, str]] = []
    current_section = "body"
    buffer: list[str] = []

    with fitz.open(pdf_path) as doc:
        for page in doc[:max_pages]:
            text = page.get_text("text")
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if REFERENCES_RE.match(line):
                    if buffer:
                        sections.append((current_section, " ".join(buffer)))
                    return sections
                if len(line) < 60 and SECTION_RE.match(line):
                    if buffer:
                        sections.append((current_section, " ".join(buffer)))
                        buffer = []
                    current_section = SECTION_RE.match(line).group(1).lower()
                else:
                    # De-hyphenate line breaks
                    if buffer and buffer[-1].endswith("-"):
                        buffer[-1] = buffer[-1][:-1] + line
                    else:
                        buffer.append(line)

    if buffer:
        sections.append((current_section, " ".join(buffer)))
    return sections


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character windows aligned on sentence/word boundaries."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            # Prefer breaking at a period or space near the end of the window
            period = cleaned.rfind(". ", start + chunk_size // 2, end)
            space = cleaned.rfind(" ", start + chunk_size // 2, end)
            cut = period + 1 if period != -1 else (space if space != -1 else end)
            end = cut
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_paper(paper: Paper, pid: str, use_full_text: bool = True) -> list[PaperChunk]:
    """Turn a Paper into an abstract chunk plus full-text PDF section chunks when available."""
    chunks: list[PaperChunk] = [
        PaperChunk(
            chunk_id=f"{pid}:0",
            paper_id=pid,
            text=paper.abstract,
            section="abstract",
            chunk_index=0,
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.authors,
            year=paper.year,
            source=paper.source,
            citation_count=paper.citation_count,
            url=paper.url,
        )
    ]

    if not use_full_text:
        return chunks

    pdf_url = arxiv_pdf_url(paper.url)
    if not pdf_url:
        return chunks

    pdf_path = download_pdf(pdf_url)
    if not pdf_path:
        return chunks

    idx = 1
    for section_name, section_text in extract_sections(pdf_path):
        if section_name == "abstract":
            continue
        for piece in split_text(section_text):
            chunks.append(
                PaperChunk(
                    chunk_id=f"{pid}:{idx}",
                    paper_id=pid,
                    text=piece,
                    section=section_name,
                    chunk_index=idx,
                    title=paper.title,
                    abstract=paper.abstract,
                    authors=paper.authors,
                    year=paper.year,
                    source=paper.source,
                    citation_count=paper.citation_count,
                    url=paper.url,
                )
            )
            idx += 1

    return chunks