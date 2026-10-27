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
from maya.ingest import document_records, load_employee_ids
from maya.retrieval import access_filter, authorize
from maya.schemas import CallerContext


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
