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


class FakeRetriever:
    """Returns one chunk for any search, and records each call."""

    def __init__(self):
        self.calls = []

    def search(self, query: str, caller: CallerContext, k: int = 4) -> list[Evidence]:
        self.calls.append({"query": query, "caller": caller})
        return [Evidence(
            doc_id="employment/maya_cohen_offer_letter.md",
            chunk_id="employment/maya_cohen_offer_letter.md#0",
            collection="employment",
            section="",
            updated=None,
            text="Required systems: Okta, Slack, Webex, Salesforce",
            score=0.9,
        )]


def make_state(requires_tools: bool = True, search_query: str = "Maya Cohen offer letter") -> dict:
    """A state just after the plan step: one user message and the plan."""

    plan = TurnPlan(
        current_intent="onboarding_status",
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
