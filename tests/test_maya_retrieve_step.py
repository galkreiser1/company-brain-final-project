"""The graph's retrieve step (maya/graph.py), with a fake retriever.

    .venv/bin/python -m pytest tests/test_maya_retrieve_step.py

No network and no model: retrieve() is called directly, with a hand-made state and
config, and the fake retriever writes down every search it's asked to run.
"""

import pytest
from langchain_core.messages import HumanMessage

from maya.graph import retrieve
from maya.planning import TurnPlan
from maya.schemas import CallerContext, Evidence


SARA_HR = {"caller_id": "E004", "user_group": "UG_HR"}
USER_MESSAGE = "What does her offer letter say she needs on day one?"


def chunk(doc_id: str, collection: str) -> Evidence:
    """A one-chunk Evidence from a document."""

    return Evidence(doc_id=doc_id, chunk_id=f"{doc_id}#0", collection=collection,
                    section="", updated=None, text="...", score=0.9)


OFFER_LETTER = chunk("employment/maya_cohen_offer_letter.md", "employment")
ONBOARDING_POLICY = chunk("policies/onboarding_policy.md", "policies")


class FakeRetriever:
    """Records each search. Returns the offer letter, or, for a search limited to
    some collections, `filtered` (the onboarding policy unless told otherwise)."""

    def __init__(self, filtered: list[Evidence] | None = None):
        self.calls = []
        self.filtered = [ONBOARDING_POLICY] if filtered is None else filtered

    def search(self, query: str, caller: CallerContext, k: int = 4,
               collections: list[str] | None = None) -> list[Evidence]:
        self.calls.append({"query": query, "caller": caller, "collections": collections})
        if collections:
            return self.filtered
        return [OFFER_LETTER]


def make_state(requires_tools: bool = True, search_query: str = "Maya Cohen offer letter",
               intent: str = "employee_lookup") -> dict:
    """A state just after the plan step: one user message and the plan.

    The default topic, employee_lookup, needs only "employment", which the offer
    letter covers, so by default one search is enough.
    """

    plan = TurnPlan(
        current_intent=intent,
        requires_tools=requires_tools,
        search_query=search_query,
    )
    return {"messages": [HumanMessage(USER_MESSAGE)], "plan": plan.model_dump()}


def make_config(**caller) -> dict:
    """A run config, as the code that starts a turn would pass it."""

    return {"configurable": {"thread_id": "test", **caller}}


# NO SEARCH -------------------------------------------------------------------

@pytest.mark.parametrize("caller", [
    {},                                              # nobody set
    {"caller_id": "E004"},                           # no group
    {"user_group": "UG_HR"},                         # no id
], ids=["no-caller", "no-group", "no-id"])
def test_no_caller_means_no_search(caller):
    retriever = FakeRetriever()

    result = retrieve(make_state(), make_config(**caller), retriever=retriever)

    assert result == {"evidence": []}
    assert retriever.calls == []


def test_no_lookup_needed_means_no_search():
    retriever = FakeRetriever()

    result = retrieve(make_state(requires_tools=False), make_config(**SARA_HR), retriever=retriever)

    assert result == {"evidence": []}
    assert retriever.calls == []


# ONE SEARCH ------------------------------------------------------------------
# The other half: proves the fake is wired up, so "no calls" above means skipped.

def test_searches_once_with_the_plans_query_as_the_caller():
    retriever = FakeRetriever()

    result = retrieve(make_state(), make_config(**SARA_HR), retriever=retriever)

    assert len(retriever.calls) == 1
    assert retriever.calls[0]["query"] == "Maya Cohen offer letter"
    assert retriever.calls[0]["caller"] == CallerContext(**SARA_HR)
    assert [chunk["chunk_id"] for chunk in result["evidence"]] == [
        "employment/maya_cohen_offer_letter.md#0"
    ]


def test_empty_search_query_falls_back_to_the_users_message():
    retriever = FakeRetriever()

    retrieve(make_state(search_query=""), make_config(**SARA_HR), retriever=retriever)

    assert retriever.calls[0]["query"] == USER_MESSAGE


# THE EXTRA SEARCH ------------------------------------------------------------
# onboarding_status needs "employment" and "policies". The first search returns
# only the offer letter (employment), so "policies" is missing.

def test_missing_collection_means_exactly_two_searches():
    retriever = FakeRetriever()

    result = retrieve(make_state(intent="onboarding_status"), make_config(**SARA_HR),
                      retriever=retriever)

    assert len(retriever.calls) == 2
    assert retriever.calls[0]["collections"] is None          # first: everything
    assert retriever.calls[1]["collections"] == ["policies"]  # then: only what's missing
    assert [c["chunk_id"] for c in result["evidence"]] == [
        OFFER_LETTER.chunk_id, ONBOARDING_POLICY.chunk_id,
    ]


def test_never_more_than_two_searches():
    # The extra search finds nothing in "policies" either. The step still stops.
    retriever = FakeRetriever(filtered=[])

    result = retrieve(make_state(intent="onboarding_status"), make_config(**SARA_HR),
                      retriever=retriever)

    assert len(retriever.calls) == 2
    assert [c["chunk_id"] for c in result["evidence"]] == [OFFER_LETTER.chunk_id]


def test_extra_search_adds_no_duplicates():
    # The extra search returns a chunk the first search already found.
    retriever = FakeRetriever(filtered=[OFFER_LETTER, ONBOARDING_POLICY])

    result = retrieve(make_state(intent="onboarding_status"), make_config(**SARA_HR),
                      retriever=retriever)

    assert [c["chunk_id"] for c in result["evidence"]] == [
        OFFER_LETTER.chunk_id, ONBOARDING_POLICY.chunk_id,
    ]
