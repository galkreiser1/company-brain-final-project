"""Retrieval over the NovaOps documents, limited to what the caller may see.

Based on Lesson 7's homework (retrieval.py and reranker.py).

Access rule: a document about one employee (subject_employee_id is set) is visible
only to HR and to that employee. Every other document is visible to everyone.
"""

from maya.schemas import CallerContext


HR_GROUP = "UG_HR"


# 1. WHO MAY SEE WHAT ---------------------------------------------------------

def authorize(caller: CallerContext, subject_employee_id: str | None) -> bool:
    """May this caller see a document about this employee?

    Gets: the caller, and the document's subject_employee_id (None if it's about
    nobody in particular).
    Returns: True if the document has no subject, the caller is in HR, or the
    caller is the subject.
    """

    if subject_employee_id is None:
        return True
    if caller.user_group == HR_GROUP:
        return True
    return caller.caller_id == subject_employee_id


def access_filter(caller: CallerContext) -> list[dict]:
    """The same rule as authorize(), written as OpenSearch filter clauses.

    Gets: the caller.
    Returns: [] for HR (no limit). For anyone else, one clause: audience is "all"
    OR the chunk is about the caller.
    """

    if caller.user_group == HR_GROUP:
        return []
    return [{
        "bool": {
            "should": [
                {"term": {"audience": "all"}},
                {"term": {"subject_employee_id": caller.caller_id}},
            ],
            "minimum_should_match": 1,  # at least one of the two must hold
        }
    }]
