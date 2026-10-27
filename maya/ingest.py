"""Turn the NovaOps documents into records ready to index. Pure functions: no AWS.

Each document is split at its "## " headings, and each piece (a "chunk") becomes one
record. The script scripts/ingest_documents.py adds the vectors and loads them.
"""

import json
import re
from pathlib import Path


# The collections (folders under documents/) that Maya searches.
COLLECTIONS = ("employment", "policies", "it_kb", "contracts", "internal_memos")

# Header lines that name the employee a document is about.
EMPLOYEE_KEYS = ("Employee", "Employee name")
# Header lines that hold a document's date. Each collection uses its own key.
DATE_KEYS = ("Effective date", "Last updated", "Date", "contract_start_date")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")  # e.g. 2026-06-01


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


# 3. READ THE METADATA --------------------------------------------------------
# Metadata belongs to the whole document, so every chunk of a document shares it.

def header_value(text: str, keys: tuple[str, ...]) -> str | None:
    """The value of the first "Key: value" line whose key is one of `keys`."""

    for line in text.splitlines():
        key, colon, value = line.partition(":")
        if colon and key.strip() in keys:
            return value.strip()
    return None


def load_employee_ids(data_dir: Path) -> dict[str, str]:
    """Map each employee's full name to their id, from the database seed file."""

    seed = json.loads((data_dir / "database" / "seed.json").read_text(encoding="utf-8"))
    return {person["full_name"]: person["employee_id"] for person in seed["employees"]}


class UnknownEmployee(ValueError):
    """A document names an employee who isn't in the database."""


def subject_employee_id(text: str, employee_ids: dict[str, str]) -> str | None:
    """The id of the employee the document is about, or None if it names nobody.

    Raises UnknownEmployee if the "Employee:" name isn't exactly an employee's full
    name. Guessing "all" there could show one person's private file to everyone.
    """

    name = header_value(text, EMPLOYEE_KEYS)
    if name is None:
        return None
    if name not in employee_ids:
        raise UnknownEmployee(f"'{name}' is not in seed.json")
    return employee_ids[name]


def updated_date(text: str) -> str | None:
    """The document's date as YYYY-MM-DD, or None if it has no date line."""

    value = header_value(text, DATE_KEYS)
    if value and ISO_DATE.fullmatch(value):
        return value
    return None


# 4. BUILD THE RECORDS --------------------------------------------------------

def document_records(path: Path, documents_dir: Path, employee_ids: dict) -> list[dict]:
    """One record per chunk of a document. Vectors are added later, by the script."""

    doc_id = path.relative_to(documents_dir).as_posix()  # e.g. "policies/expense_policy.md"
    text = path.read_text(encoding="utf-8")
    subject = subject_employee_id(text, employee_ids)
    # A document about one employee is for HR or that employee; the rest is for all.
    audience = "hr_or_subject" if subject else "all"
    updated = updated_date(text)

    records = []
    for number, (section, chunk_text) in enumerate(split_sections(text)):
        records.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}#{number}",
            "collection": path.parent.name,
            "section": section,
            "subject_employee_id": subject,
            "audience": audience,
            "updated": updated,
            "text": chunk_text,
        })
    return records


def read_records(data_dir: Path) -> tuple[list[dict], list[str]]:
    """Every chunk of every document, as records, plus the documents skipped.

    A document naming an unknown employee is skipped (with the reason), so one bad
    file doesn't stop the others from being indexed.
    """

    documents_dir = data_dir / "documents"
    employee_ids = load_employee_ids(data_dir)
    records, skipped = [], []
    for path in document_paths(documents_dir):
        try:
            records.extend(document_records(path, documents_dir, employee_ids))
        except UnknownEmployee as error:
            skipped.append(f"{path.relative_to(documents_dir)}: {error}")
    return records, skipped
