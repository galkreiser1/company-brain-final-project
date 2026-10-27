"""Turn the NovaOps documents into records ready to index. Pure functions: no AWS.

Each document is split at its "## " headings, and each piece (a "chunk") becomes one
record. The script scripts/ingest_documents.py adds the vectors and loads them.
"""

from pathlib import Path


# The collections (folders under documents/) that Maya searches.
COLLECTIONS = ("employment", "policies", "it_kb", "contracts", "internal_memos")


# 1. FIND THE DOCUMENTS -------------------------------------------------------

def document_paths(documents_dir: Path) -> list[Path]:
    """Every .md file in the five collections, in a fixed order."""

    paths = []
    for collection in COLLECTIONS:
        paths.extend(sorted((documents_dir / collection).glob("*.md")))
    return paths


# 2. SPLIT ONE DOCUMENT -------------------------------------------------------

def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a document at its "## " headings. Returns (heading, text) pairs.

    - The text before the first heading (title and header lines) is its own piece,
      with heading "".
    - Each piece's text keeps its heading line, so the heading is searchable too.
    - A document with no headings (the memos) comes back as one piece.
    """

    sections = []
    heading, lines = "", []
    for line in text.splitlines():
        if line.startswith("## "):
            sections.append((heading, lines))
            heading, lines = line.removeprefix("## ").strip(), []
        lines.append(line)
    sections.append((heading, lines))

    # Join each piece's lines, and drop pieces with no text at all.
    pieces = [(heading, "\n".join(lines).strip()) for heading, lines in sections]
    return [(heading, body) for heading, body in pieces if body]


# 3. BUILD THE RECORDS --------------------------------------------------------

def document_records(path: Path, documents_dir: Path) -> list[dict]:
    """One record per chunk of a document. Vectors are added later, by the script."""

    doc_id = path.relative_to(documents_dir).as_posix()  # e.g. "policies/expense_policy.md"
    sections = split_sections(path.read_text(encoding="utf-8"))

    records = []
    for number, (section, text) in enumerate(sections):
        records.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}#{number}",
            "collection": path.parent.name,
            "section": section,
            "text": text,
        })
    return records


def read_records(data_dir: Path) -> list[dict]:
    """Every chunk of every document, as records."""

    documents_dir = data_dir / "documents"
    records = []
    for path in document_paths(documents_dir):
        records.extend(document_records(path, documents_dir))
    return records
