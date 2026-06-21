"""Document parsing and hierarchy-preserving chunking utilities.

Full reconstruction uses pdfplumber as the primary PDF parser so text and
Fornas-style tables keep their row/column relationships. The chunk writer keeps
the current evaluator-compatible fields while also emitting the canonical
`source`, `metadata`, and `content` schema used by the production RAG path.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover - exercised in minimal environments
    pd = None

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional production dependency
    pdfplumber = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional fallback dependency
    PdfReader = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional extraction dependency
    BeautifulSoup = None


STRUCTURAL_HEADING_RE = re.compile(r"^(BAB\s+[IVXLC]+|[0-9]+[.)]\s+|Panduan\s+|Cara\s+|Fitur\s+)", flags=re.I)
UPPER_HEADING_RE = re.compile(r"^[A-Z0-9 ,/()\-]{8,}$")

TABLE_NOISE_RE = re.compile(r"^[\s|:/\\._\-–—•·]+$")

TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_tolerance": 5,
}


def clean_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(STRUCTURAL_HEADING_RE.search(stripped) or UPPER_HEADING_RE.search(stripped))


def sniff_kind(path: Path) -> str:
    raw = path.read_bytes()[:4096].lstrip()
    head = raw[:32].lower()
    if raw.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"<!doctype") or head.startswith(b"<html") or b"<html" in raw.lower():
        return "html"
    if path.suffix.lower() in {".html", ".htm"}:
        return "html"
    return "text"


def load_catalog(catalog_path: Path) -> List[Dict]:
    if pd is not None:
        return pd.read_csv(catalog_path).fillna("").to_dict(orient="records")
    with open(catalog_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def common_metadata(row: Dict) -> Dict:
    return {
        "doc_id": row.get("doc_id"),
        "title": row.get("title"),
        "institution": row.get("institution"),
        "source_type": row.get("source_type"),
        "source_url": row.get("source_url"),
        "version": row.get("version"),
        "accessed_at": row.get("accessed_at"),
        "format": row.get("format"),
        "year": row.get("year"),
        "scope": row.get("scope"),
    }


def extract_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if BeautifulSoup is not None:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return clean_text(soup.get_text("\n", strip=True))
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<[^>]+>", "\n", raw)
    return clean_text(raw)


def normalize_table_row(row: Sequence[object]) -> List[str]:
    return [clean_text(str(cell or "")) for cell in row]


def normalize_table(table: Sequence[Sequence[object]]) -> List[List[str]]:
    rows = [normalize_table_row(row) for row in table]
    return [row for row in rows if any(cell and not TABLE_NOISE_RE.fullmatch(cell) for cell in row)]


def infer_table_headers(table: Sequence[Sequence[object]]) -> tuple[List[str], int]:
    """Return usable headers and the first data-row index for a PDF table."""
    rows = normalize_table(table)
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return [], 0

    for idx, row in enumerate(rows[:3]):
        filled = [cell for cell in row if cell and not TABLE_NOISE_RE.fullmatch(cell)]
        alpha_cells = [cell for cell in filled if re.search(r"[A-Za-zÀ-ÿ]", cell)]
        if len(alpha_cells) >= max(2, min(4, width // 2)):
            headers = []
            for col, cell in enumerate(row):
                header = clean_text(cell) if cell else ""
                headers.append(header[:80] or f"col_{col + 1}")
            return headers, idx + 1
    return [f"col_{idx + 1}" for idx in range(width)], 0


def markdown_escape(cell: str) -> str:
    return clean_text(cell).replace("|", "\\|").replace("\n", "<br>")


def table_to_markdown(table: Sequence[Sequence[object]]) -> str:
    """Serialize a PDF table as Markdown while preserving column alignment."""
    rows = normalize_table(table)
    if not rows:
        return ""
    headers, start_idx = infer_table_headers(rows)
    width = max(len(headers), max((len(row) for row in rows), default=0))
    headers = (headers + [f"col_{idx + 1}" for idx in range(len(headers), width)])[:width]
    body_rows = rows[start_idx:] if start_idx < len(rows) else rows

    header_line = "| " + " | ".join(markdown_escape(header or f"col_{idx + 1}") for idx, header in enumerate(headers)) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in body_rows:
        cells = (list(row) + [""] * width)[:width]
        if any(cell and not TABLE_NOISE_RE.fullmatch(cell) for cell in cells):
            body.append("| " + " | ".join(markdown_escape(cell) for cell in cells) + " |")
    return "\n".join([header_line, divider, *body]) if body else ""


def table_to_text(table: Sequence[Sequence[object]]) -> str:
    """Serialize a relational PDF table as one key-value line per data row."""
    rows = normalize_table(table)
    if not rows:
        return ""

    headers, start_idx = infer_table_headers(rows)
    serialized = []
    for row_no, row in enumerate(rows[start_idx:], start=1):
        cells = list(row) + [""] * max(0, len(headers) - len(row))
        pairs = []
        for header, cell in zip(headers, cells):
            if not cell or TABLE_NOISE_RE.fullmatch(cell):
                continue
            pairs.append(f"{header}: {cell}")
        if pairs:
            serialized.append(f"Baris {row_no}: " + " | ".join(pairs))
    return "\n".join(serialized)


def table_to_structured_text(table: Sequence[Sequence[object]]) -> str:
    """Return Markdown plus row-level key-value text for reliable retrieval."""
    markdown = table_to_markdown(table)
    row_text = table_to_text(table)
    return clean_text("\n".join(part for part in (markdown, row_text) if part))


def extract_page_tables(page) -> List[Dict]:
    """Extract relational tables from a pdfplumber page with a robust fallback."""
    tables = page.extract_tables(TABLE_SETTINGS) or []
    if not tables:
        tables = page.extract_tables() or []

    extracted: List[Dict] = []
    seen: set[str] = set()
    for table_i, table in enumerate(tables, start=1):
        content = table_to_structured_text(table)
        if not content:
            continue
        key = re.sub(r"\s+", " ", content.lower())
        if key in seen:
            continue
        seen.add(key)
        extracted.append({"table_index": table_i, "content": content})
    return extracted


def extract_pdf_elements_with_pdfplumber(path: Path) -> Iterator[Dict]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for primary PDF extraction")
    with pdfplumber.open(str(path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = clean_text(page.extract_text(x_tolerance=1, y_tolerance=3) or "")
            if page_text:
                yield {
                    "page": page_no,
                    "section": None,
                    "subtitle": None,
                    "chunk_type": "text",
                    "text": page_text,
                    "content": page_text,
                }
            for table in extract_page_tables(page):
                table_i = table["table_index"]
                content = f"Tabel {table_i} halaman {page_no}\n{table['content']}"
                yield {
                    "page": page_no,
                    "section": f"Tabel {table_i} halaman {page_no}",
                    "subtitle": f"Tabel {table_i}",
                    "chunk_type": "table",
                    "table_index": table_i,
                    "text": content,
                    "content": content,
                }


def extract_pdf_with_pdfplumber(path: Path) -> Iterator[tuple[int, str]]:
    for element in extract_pdf_elements_with_pdfplumber(path):
        yield int(element["page"]), str(element["content"])


def extract_pdf_with_pypdf(path: Path) -> Iterator[tuple[int, str]]:
    if PdfReader is None:
        raise RuntimeError("pypdf is required when pdfplumber is unavailable")
    reader = PdfReader(str(path))
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            yield page_no, text


def extract_pdf_elements_with_pypdf(path: Path) -> Iterator[Dict]:
    for page_no, text in extract_pdf_with_pypdf(path):
        yield {
            "page": page_no,
            "section": None,
            "subtitle": None,
            "chunk_type": "text",
            "text": text,
            "content": text,
        }


def extract_pdf(path: Path) -> Iterator[tuple[int, str]]:
    if pdfplumber is not None:
        yield from extract_pdf_with_pdfplumber(path)
        return
    yield from extract_pdf_with_pypdf(path)


def extract_pdf_elements(path: Path) -> Iterator[Dict]:
    """Yield separate text and table elements for clean downstream chunking."""
    if pdfplumber is not None:
        yield from extract_pdf_elements_with_pdfplumber(path)
        return
    yield from extract_pdf_elements_with_pypdf(path)


def extract_documents_from_catalog(catalog_path: Path, root: Path) -> List[Dict]:
    docs: List[Dict] = []
    for row in load_catalog(catalog_path):
        path = root / str(row.get("local_path", ""))
        if not path.exists() or path.stat().st_size == 0:
            print(f"MISSING raw file, skipped: {path}")
            continue
        common = common_metadata(row)
        try:
            kind = sniff_kind(path)
            if kind == "html":
                text = extract_html(path)
                docs.append({**common, "page": None, "section": None, "subtitle": None, "chunk_type": "text", "text": text, "content": text})
            elif kind == "pdf":
                try:
                    for element in extract_pdf_elements(path):
                        docs.append({**common, **element})
                except Exception as exc:
                    print(f"PDF parse failed, falling back to HTML/text for {path}: {exc}", file=sys.stderr)
                    raw = path.read_text(encoding="utf-8", errors="ignore")
                    text = extract_html(path) if "<html" in raw.lower() or "<!doctype" in raw.lower() else clean_text(raw)
                    docs.append({**common, "page": None, "section": None, "subtitle": None, "chunk_type": "text", "text": text, "content": text})
            else:
                text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
                docs.append({**common, "page": None, "section": None, "subtitle": None, "chunk_type": "text", "text": text, "content": text})
        except Exception as exc:
            print(f"FAILED extraction {path}: {exc}", file=sys.stderr)
    return docs


def read_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_id(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value or "chunk")).strip("_").upper()[:48]


def split_by_headings(text: str) -> List[tuple[str, str]]:
    lines = [line.strip() for line in clean_text(text).splitlines() if line.strip()]
    sections: List[tuple[str, str]] = []
    current_title = "General"
    current: List[str] = []
    for line in lines:
        if is_heading(line):
            if current:
                sections.append((current_title, "\n".join(current)))
            current_title = line[:140]
            current = []
        else:
            current.append(line)
    if current:
        sections.append((current_title, "\n".join(current)))
    return sections


def window_text(text: str, max_words: int = 220, overlap: int = 40) -> Iterator[str]:
    words = text.split()
    if len(words) <= max_words:
        if text.strip():
            yield text
        return
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start = max(0, end - overlap)


def base_section(doc: Dict) -> str:
    if doc.get("section"):
        return str(doc["section"])
    if doc.get("page"):
        return f"page_{doc['page']}"
    return "General"


def chunk_source(doc: Dict) -> str:
    return str(doc.get("title") or doc.get("doc_id") or doc.get("source_url") or "unknown_source")


def build_chunks(documents: Sequence[Dict], *, max_words: int = 220, overlap: int = 40) -> List[Dict]:
    chunks: List[Dict] = []
    for doc_i, doc in enumerate(documents, start=1):
        title = doc.get("title") or doc.get("doc_id")
        page = doc.get("page")
        default_section = base_section(doc)
        chunk_type = str(doc.get("chunk_type") or "text")
        source_text = str(doc.get("content") or doc.get("text") or "")
        if chunk_type == "table" or doc.get("section"):
            detected_sections = [(str(doc.get("section") or default_section), source_text)]
        else:
            detected_sections = split_by_headings(source_text)
        for detected_section, section_text in detected_sections:
            section = default_section if detected_section == "General" else detected_section
            subtitle = None if detected_section == "General" else detected_section
            for part_i, part in enumerate(window_text(section_text, max_words=max_words, overlap=overlap), start=1):
                hierarchy = {
                    "title": title,
                    "section": section,
                    "subtitle": subtitle,
                    "page": page,
                }
                if doc.get("section") and page is None and part_i == 1:
                    chunk_id = f"{doc.get('doc_id')}_{norm_id(section)}_{doc_i:03d}"
                else:
                    chunk_id = f"{doc.get('doc_id')}_{norm_id(section)}_{doc_i:04d}_{part_i:02d}"
                metadata = {"page": page, "section": section, "subsection": subtitle, "chunk_type": chunk_type}
                if doc.get("table_index") is not None:
                    metadata["table_index"] = doc.get("table_index")
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc.get("doc_id"),
                    "source": chunk_source(doc),
                    "metadata": metadata,
                    "chunk_type": chunk_type,
                    "table_index": doc.get("table_index"),
                    "title": title,
                    "subtitle": subtitle,
                    "section": section,
                    "subsection": subtitle,
                    "source_type": doc.get("source_type"),
                    "institution": doc.get("institution"),
                    "source_url": doc.get("source_url"),
                    "version": doc.get("version"),
                    "page": page,
                    "page_start": page,
                    "page_end": page,
                    "hierarchy": hierarchy,
                    "hierarchy_path": [str(v) for v in (title, section, subtitle) if v],
                    "text": part,
                    "content": part,
                    "token_count_approx": len(re.findall(r"\w+", part)),
                })
    return chunks
