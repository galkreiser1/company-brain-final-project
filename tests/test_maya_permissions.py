"""Maya's permission rule: a document about one employee is for HR and that employee.

    .venv/bin/python -m pytest                    # the fast tests: no network, no model
    .venv/bin/python -m pytest -m integration     # the real index and Bedrock

Three callers:
  Sara   - E004, HR: may see every employee's file.
  Maya   - E001, regular: may see her own file.
  Rachel - E009, regular, unrelated to Maya: must never see Maya's file.
"""

import pytest

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
