"""Maya's permission rule: a document about one employee is for HR and that employee.

    .venv/bin/python -m pytest                    # the fast tests: no network, no model
    .venv/bin/python -m pytest -m integration     # the real index and Bedrock

Three callers:
  Sara   - E004, HR: may see every employee's file.
  Maya   - E001, regular: may see her own file.
  Rachel - E009, regular, unrelated to Maya: must never see Maya's file.
"""

import pytest

from maya.config import DATA_DIR
from maya.ingest import document_records, load_employee_ids, read_records
from maya.retrieval import (
    OpenSearchRetriever, access_filter, authorize, knn_candidates, search_about,
)
from maya.schemas import CallerContext, Evidence


SARA_HR = CallerContext(caller_id="E004", user_group="UG_HR")
MAYA = CallerContext(caller_id="E001", user_group="UG_REGULAR")
RACHEL = CallerContext(caller_id="E009", user_group="UG_REGULAR")


# 1. THE RULE -----------------------------------------------------------------

@pytest.mark.parametrize("caller, expected", [
    (SARA_HR, True),   # HR sees everyone's file
    (MAYA, True),      # Maya sees her own
    (RACHEL, False),   # Rachel is unrelated, so she doesn't
], ids=["hr", "herself", "unrelated"])
def test_authorize_document_about_maya(caller, expected):
    assert authorize(caller, "E001") is expected


@pytest.mark.parametrize("caller", [SARA_HR, MAYA, RACHEL], ids=["hr", "herself", "unrelated"])
def test_authorize_document_about_nobody(caller):
    # A document with no subject (a policy, a KB article) is for everyone.
    assert authorize(caller, None) is True


def test_access_filter_hr_has_no_limit():
    assert access_filter(SARA_HR) == []


@pytest.mark.parametrize("caller", [MAYA, RACHEL], ids=["herself", "unrelated"])
def test_access_filter_regular_sees_shared_docs_and_their_own(caller):
    assert access_filter(caller) == [{
        "bool": {
            "should": [
                {"term": {"audience": "all"}},
                {"term": {"subject_employee_id": caller.caller_id}},
            ],
            "minimum_should_match": 1,
        }
    }]


# 2. INGEST METADATA ----------------------------------------------------------
# Built from the real files on disk; no network.

DOCUMENTS_DIR = DATA_DIR / "documents"


def records_for(doc_id: str) -> list[dict]:
    """Every chunk record of one document, as the ingest builds them."""

    employee_ids = load_employee_ids(DATA_DIR)
    return document_records(DOCUMENTS_DIR / doc_id, DOCUMENTS_DIR, employee_ids)


def test_ingest_maya_offer_letter_is_about_maya():
    records = records_for("employment/maya_cohen_offer_letter.md")

    assert records  # the file was split into at least one chunk
    for record in records:  # every chunk carries the document's subject, not just the first
        assert record["subject_employee_id"] == "E001"
        assert record["audience"] == "hr_or_subject"


def test_ingest_onboarding_policy_is_for_everyone():
    records = records_for("policies/onboarding_policy.md")

    assert records
    for record in records:
        assert record["subject_employee_id"] is None
        assert record["audience"] == "all"
        assert record["updated"] == "2026-06-01"  # from its "Effective date:" line


# 3. A FAKE RETRIEVER ---------------------------------------------------------
# Stands in for OpenSearchRetriever (it fits the EvidenceRetriever Protocol), and
# writes down every search it's asked to run.

MAYA_CHUNK = Evidence(
    doc_id="employment/maya_cohen_offer_letter.md",
    chunk_id="employment/maya_cohen_offer_letter.md#0",
    collection="employment",
    section="",
    updated=None,
    text="Employee: Maya Cohen ...",
    score=1.0,
)


class FakeRetriever:
    """Returns one chunk about Maya for any search, and records each call."""

    def __init__(self):
        self.calls = []

    def search(self, query: str, caller: CallerContext, k: int = 4) -> list[Evidence]:
        self.calls.append((query, caller.caller_id))
        return [MAYA_CHUNK]


def test_denied_caller_triggers_no_search_and_gets_nothing():
    retriever = FakeRetriever()

    evidence = search_about(retriever, "What is Maya's salary?", RACHEL, "E001")

    assert evidence == []
    assert retriever.calls == []  # the retriever was never even asked


@pytest.mark.parametrize("caller", [SARA_HR, MAYA], ids=["hr", "herself"])
def test_allowed_caller_searches_once(caller):
    # The other half: proves the fake is wired up, so "zero calls" above means
    # the search was blocked, not that the fake never runs.
    retriever = FakeRetriever()

    evidence = search_about(retriever, "What is Maya's salary?", caller, "E001")

    assert evidence == [MAYA_CHUNK]
    assert retriever.calls == [("What is Maya's salary?", caller.caller_id)]


# 4. INTEGRATION: THE REAL INDEX ----------------------------------------------
# Calls OpenSearch and Bedrock, so these are skipped unless run with -m integration.

MAYA_QUERIES = [
    "What is in Maya Cohen's offer letter?",
    "When does Maya Cohen start, and who is her manager?",
    "What equipment and systems does Maya Cohen get for onboarding?",
]


@pytest.fixture(scope="module")
def retriever():
    """One real retriever (one OpenSearch connection) for all the tests below."""

    return OpenSearchRetriever()


@pytest.fixture(scope="module")
def maya_doc_ids():
    """The documents about Maya (subject E001), taken from the ingest's own records.

    Evidence doesn't carry subject_employee_id, so a chunk is recognised as Maya's
    by its doc_id.
    """

    records, _ = read_records(DATA_DIR)
    return {r["doc_id"] for r in records if r["subject_employee_id"] == "E001"}


@pytest.mark.integration
@pytest.mark.parametrize("query", MAYA_QUERIES)
def test_unrelated_caller_never_gets_mayas_chunks(retriever, maya_doc_ids, query):
    # The OpenSearch filter on its own: none of the 20 candidates is about Maya.
    # (Checked separately, because search() would also hide a broken filter.)
    candidates = knn_candidates(retriever.client, query, RACHEL)
    assert all(c["subject_employee_id"] != "E001" for c in candidates)

    # The full search: filter, Python check, reranker.
    evidence = retriever.search(query, RACHEL)
    assert evidence  # she still gets an answer from the shared documents
    assert not {e.doc_id for e in evidence} & maya_doc_ids


@pytest.mark.integration
@pytest.mark.parametrize("caller", [SARA_HR, MAYA], ids=["hr", "herself"])
@pytest.mark.parametrize("query", MAYA_QUERIES)
def test_allowed_caller_gets_mayas_chunks(retriever, maya_doc_ids, caller, query):
    evidence = retriever.search(query, caller)
    assert {e.doc_id for e in evidence} & maya_doc_ids
