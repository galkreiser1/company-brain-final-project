"""Turn the NovaOps documents into records ready to index. Pure functions: no AWS.

Each document is split at its "## " headings, and each piece (a "chunk") becomes one
record. The script scripts/ingest_documents.py adds the vectors and loads them.
"""

import json
import re
from pathlib import Path


# The collections (folders under documents/) that Maya searches. The handbook is for
# everyone; the manager playbook is for managers only (each file's front matter says so).
COLLECTIONS = ("employment", "policies", "it_kb", "contracts", "internal_memos",
               "handbook", "manager_playbook")

# The audiences a document's front matter may declare. Anything else is skipped at
# ingest: a label we don't understand could mean "restricted", so we don't guess.
DECLARED_AUDIENCES = ("all", "manager")

# Header lines that name the employee a document is about.
EMPLOYEE_KEYS = ("Employee", "Employee name")
# Header lines that hold a document's date. Each collection uses its own key.
DATE_KEYS = ("Effective date", "Last updated", "Date", "contract_start_date")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")  # e.g. 2026-06-01


# 1. FIND THE DOCUMENTS -------------------------------------------------------

def document_paths(documents_dir: Path) -> list[Path]:
    """Every .md file in the collections, in a fixed order."""

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


def front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split off a document's front matter: the "key: value" lines between two "---"
    lines at the very top (the handbook and the manager playbook have it).

    Returns: (the keys and values, the text after it). No front matter: ({}, text).
    """

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            fields = {}
            for field in lines[1:end]:
                key, colon, value = field.partition(":")
                if colon:
                    fields[key.strip()] = value.strip()
            return fields, "\n".join(lines[end + 1:])
    return {}, text  # no closing "---": not front matter


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


def load_manager_ids(data_dir: Path) -> frozenset[str]:
    """The employees who manage someone: every manager_id in the database seed file.
    Only they may read the manager playbook."""

    seed = json.loads((data_dir / "database" / "seed.json").read_text(encoding="utf-8"))
    return frozenset(person["manager_id"] for person in seed["employees"] if person.get("manager_id"))


class UnknownEmployee(ValueError):
    """A document names an employee who isn't in the database."""


class UnknownAudience(ValueError):
    """A document's front matter declares an audience we don't know."""


def document_audience(declared: str | None, subject: str | None) -> str:
    """Who may read a document:
      - "hr_or_subject": it's about one employee (HR or that employee);
      - "manager": its front matter says audience: manager (managers only);
      - "all": everyone (front matter says all, or there's no front matter).
    Raises UnknownAudience for any other declared value.
    """

    if declared is not None and declared not in DECLARED_AUDIENCES:
        raise UnknownAudience(f"audience '{declared}' is not one of {DECLARED_AUDIENCES}")
    if subject:
        return "hr_or_subject"
    return declared or "all"


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


def updated_date(text: str, fields: dict[str, str] | None = None) -> str | None:
    """The document's date as YYYY-MM-DD, or None if it has no date line. The front
    matter's last_updated counts first."""

    value = (fields or {}).get("last_updated") or header_value(text, DATE_KEYS)
    if value and ISO_DATE.fullmatch(value):
        return value
    return None


# 4. BUILD THE RECORDS --------------------------------------------------------

def document_records(path: Path, documents_dir: Path, employee_ids: dict) -> list[dict]:
    """One record per chunk of a document. Vectors are added later, by the script."""

    doc_id = path.relative_to(documents_dir).as_posix()  # e.g. "policies/expense_policy.md"
    fields, text = front_matter(path.read_text(encoding="utf-8"))
    subject = subject_employee_id(text, employee_ids)
    audience = document_audience(fields.get("audience"), subject)
    updated = updated_date(text, fields)

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

    A document naming an unknown employee, or declaring an unknown audience, is
    skipped (with the reason), so one bad file doesn't stop the others from being
    indexed, and nothing is shown to people it may not be meant for.
    """

    documents_dir = data_dir / "documents"
    employee_ids = load_employee_ids(data_dir)
    records, skipped = [], []
    for path in document_paths(documents_dir):
        try:
            records.extend(document_records(path, documents_dir, employee_ids))
        except (UnknownEmployee, UnknownAudience) as error:
            skipped.append(f"{path.relative_to(documents_dir)}: {error}")
    return records, skipped
