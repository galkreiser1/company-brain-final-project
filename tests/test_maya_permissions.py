"""Maya's permission rules: a document about one employee is for HR and that employee;
the manager playbook is for managers only; everything else is for everyone.

    .venv/bin/python -m pytest                    # the fast tests: no network, no model
    .venv/bin/python -m pytest -m integration     # the real index and Bedrock

Callers:
  Sara   - E004, HR and a manager: may see every employee's file and the playbook.
  Maya   - E001, regular: may see her own file, never the playbook.
  Rachel - E009, regular, unrelated to Maya: must never see Maya's file.
  Yael   - E018, regular but a manager (Maya reports to her): may see the playbook.
  HR_NOT_MANAGER - an HR caller nobody reports to: every file, but not the playbook.
"""

import pytest

from maya.config import DATA_DIR
from maya.graph import caller_from_config
from maya.ingest import (
    UnknownAudience, document_audience, document_records, front_matter, load_employee_ids,
    load_manager_ids, read_records,
)
from maya.retrieval import (
    OpenSearchRetriever, access_filter, authorize, knn_candidates, may_read, search_about,
)
from maya.schemas import CallerContext, Evidence


SARA_HR = CallerContext(caller_id="E004", user_group="UG_HR", is_manager=True)
MAYA = CallerContext(caller_id="E001", user_group="UG_REGULAR")
RACHEL = CallerContext(caller_id="E009", user_group="UG_REGULAR")
YAEL_MANAGER = CallerContext(caller_id="E018", user_group="UG_REGULAR", is_manager=True)
HR_NOT_MANAGER = CallerContext(caller_id="E099", user_group="UG_HR")


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


def test_access_filter_hr_manager_sees_every_file_and_the_playbook():
    assert access_filter(SARA_HR) == [{"bool": {"should": [
        {"term": {"audience": "all"}},
        {"term": {"audience": "hr_or_subject"}},
        {"term": {"audience": "manager"}},
    ], "minimum_should_match": 1}}]


def test_access_filter_hr_who_manages_nobody_gets_no_playbook():
    assert access_filter(HR_NOT_MANAGER) == [{"bool": {"should": [
        {"term": {"audience": "all"}},
        {"term": {"audience": "hr_or_subject"}},
    ], "minimum_should_match": 1}}]


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


def test_access_filter_regular_manager_also_sees_the_playbook():
    assert {"term": {"audience": "manager"}} in access_filter(YAEL_MANAGER)[0]["bool"]["should"]


PLAYBOOK_CHUNK = {"audience": "manager", "subject_employee_id": None}


@pytest.mark.parametrize("caller, expected", [
    (SARA_HR, True), (YAEL_MANAGER, True),
    (MAYA, False), (RACHEL, False), (HR_NOT_MANAGER, False),
], ids=["hr-manager", "manager", "maya", "rachel", "hr-not-manager"])
def test_only_managers_may_read_the_playbook(caller, expected):
    assert may_read(caller, PLAYBOOK_CHUNK) is expected


@pytest.mark.parametrize("audience", [None, "", "managers", "secret"])
def test_a_chunk_with_no_known_audience_is_hidden_from_everyone(audience):
    # Fail closed: even HR and managers get nothing whose audience we don't know.
    chunk = {"audience": audience, "subject_employee_id": None}
    assert not any(may_read(caller, chunk) for caller in (SARA_HR, YAEL_MANAGER, MAYA))


def test_who_is_a_manager_comes_from_the_database_not_the_config():
    # Maya reports to Yael (E018); nobody reports to Maya (E001).
    assert {"E004", "E018"} <= load_manager_ids(DATA_DIR)
    assert "E001" not in load_manager_ids(DATA_DIR)
    config = {"configurable": {"caller_id": "E001", "user_group": "UG_REGULAR",
                               "is_manager": True}}  # a config can't make her one
    assert caller_from_config(config).is_manager is False
    config = {"configurable": {"caller_id": "E018", "user_group": "UG_REGULAR"}}
    assert caller_from_config(config).is_manager is True


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


def test_ingest_manager_playbook_is_for_managers_only():
    records = records_for("manager_playbook/promotion.md")

    assert records
    for record in records:
        assert record["audience"] == "manager"
        assert record["collection"] == "manager_playbook"
        assert not record["text"].startswith("---")  # the front matter isn't chunk text
    assert records[0]["updated"] == "2024-08-05"  # from its front matter's last_updated


def test_ingest_handbook_is_for_everyone():
    records = records_for("handbook/making-a-career.md")

    assert records
    assert all(record["audience"] == "all" for record in records)


def test_every_playbook_file_is_indexed_and_restricted():
    records, skipped = read_records(DATA_DIR)
    playbook = DOCUMENTS_DIR / "manager_playbook"
    indexed = {r["doc_id"] for r in records if r["collection"] == "manager_playbook"}
    assert indexed == {f"manager_playbook/{path.name}" for path in playbook.glob("*.md")}
    assert {r["audience"] for r in records if r["collection"] == "manager_playbook"} == {"manager"}
    assert skipped == []


def test_front_matter_is_read_and_removed():
    fields, rest = front_matter("---\naudience: manager\nlast_updated: 2024-01-02\n---\n# Title")
    assert fields == {"audience": "manager", "last_updated": "2024-01-02"}
    assert rest == "# Title"
    assert front_matter("# No front matter") == ({}, "# No front matter")


def test_an_unknown_declared_audience_is_refused():
    with pytest.raises(UnknownAudience):
        document_audience("managers-and-friends", None)
    assert document_audience("manager", None) == "manager"
    assert document_audience(None, None) == "all"
    assert document_audience(None, "E001") == "hr_or_subject"


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


# The S9 permission traps: the handbook and the manager playbook both answer these,
# so only the search filter can tell right from wrong.
PLAYBOOK_QUERIES = [
    "How do I get promoted here? What does the path look like for a CSM?",
    "What does the manager's guide say about giving feedback in someone's first 90 days?",
]


@pytest.mark.integration
@pytest.mark.parametrize("query", PLAYBOOK_QUERIES)
def test_a_non_manager_never_gets_playbook_chunks(retriever, query):
    # The OpenSearch filter on its own: none of the 20 candidates is from the playbook.
    for caller in (MAYA, HR_NOT_MANAGER):
        candidates = knn_candidates(retriever.client, query, caller)
        assert candidates
        assert all(c["collection"] != "manager_playbook" for c in candidates)


@pytest.mark.integration
def test_the_promotion_question_finds_the_handbook(retriever):
    evidence = retriever.search(PLAYBOOK_QUERIES[0], MAYA)
    assert any(e.doc_id.startswith("handbook/") for e in evidence)


@pytest.mark.integration
def test_a_manager_does_get_playbook_chunks(retriever):
    # The other half: the playbook is indexed and reachable, so "none" above means filtered.
    candidates = knn_candidates(retriever.client, PLAYBOOK_QUERIES[1], YAEL_MANAGER)
    assert any(c["collection"] == "manager_playbook" for c in candidates)
