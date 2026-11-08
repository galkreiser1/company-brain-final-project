"""The Webex agent (webex/graph.py): its code-only steps, then the whole graph.

No server and no AWS: DbToolbox calls the server's functions (server/db.py)
directly, on a fresh copy of the seed database; the retriever and the model that
writes the answer are fakes.
"""

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from maya.schemas import CallerContext, Evidence
from server import db
from webex.graph import build_graph, decide, diagnose, lookup, run_webex
from webex.schemas import WebexRequest
from webex.tools import WEBEX_READ_TOOLS

RACHEL = CallerContext(caller_id="E010", user_group="UG_REGULAR")
SARA_HR = CallerContext(caller_id="E004", user_group="UG_HR")
RACHEL_GREEN = CallerContext(caller_id="E009", user_group="UG_REGULAR")  # a colleague, not her manager


class DbToolbox:
    """Calls server/db.py's functions in place of the MCP server."""

    async def call(self, name: str, **args) -> list:
        assert name in WEBEX_READ_TOOLS
        result = getattr(db, name)(**args)
        return result if isinstance(result, list) else [result]


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "novaops.db")
    db._conn.cache_clear()
    yield
    db._conn().close()
    db._conn.cache_clear()


def run_steps(caller, employee_id, wants_filing=False) -> dict:
    """lookup -> diagnose -> decide, the way the graph will chain them."""
    state = {"request": WebexRequest(
        caller=caller, employee_id=employee_id, message="...", wants_filing=wants_filing,
    )}
    state |= asyncio.run(lookup(state, DbToolbox()))
    if state["problem"]:
        return state
    state |= diagnose(state)
    state |= decide(state)
    return state


def test_rachel_reuses_ar001():
    state = run_steps(RACHEL, "E010")
    diagnosis = state["diagnosis"]
    assert diagnosis.allowed_by_role and diagnosis.over_seat_limit
    assert diagnosis.seat_reason == "42 of 40 seats in use (SUB001)"
    assert diagnosis.existing_request_id == "AR001"
    assert diagnosis.open_approval_ids == ["AP001", "AP002"]
    assert [t["ticket_id"] for t in state["records"]["tickets"]] == ["T001"]
    assert state["path"] == "reuse"


def test_rachel_asking_to_file_still_reuses():
    assert run_steps(RACHEL, "E010", wants_filing=True)["path"] == "reuse"


def test_sara_filing_for_maya_cohen_asks_for_approval():
    state = run_steps(SARA_HR, "E001", wants_filing=True)
    assert state["diagnosis"].existing_request_id is None  # AR004 is Salesforce
    assert state["path"] == "approve_then_file"


def test_a_question_about_maya_cohen_only_answers():
    assert run_steps(SARA_HR, "E001")["path"] == "answer_only"


def test_a_colleague_may_not_ask_for_rachel():
    state = run_steps(RACHEL_GREEN, "E010", wants_filing=True)
    assert not state["diagnosis"].caller_may_request
    assert state["path"] == "refuse"


def test_unknown_employee_stops_at_lookup():
    state = run_steps(SARA_HR, "E999")
    assert state["problem"] == "No employee E999."
    assert "diagnosis" not in state


# THE WHOLE GRAPH --------------------------------------------------------------------

WEBEX_ARTICLE = Evidence(
    doc_id="it_kb/webex_license_assignment.md", chunk_id="it_kb/webex_license_assignment.md#4",
    collection="it_kb", section="Escalation", updated=None, text="Escalate...", score=0.9,
)


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, caller, k=4, collections=None):
        self.queries.append((query, caller.caller_id, collections))
        return [WEBEX_ARTICLE]


def run_graph(caller, employee_id, wants_filing=False):
    retriever = FakeRetriever()
    writer = FakeListChatModel(responses=["Your request AR001 is already on file."])
    graph = build_graph(DbToolbox(), retriever, writer)
    request = WebexRequest(caller=caller, employee_id=employee_id,
                           message="I need Webex access.", wants_filing=wants_filing)
    return asyncio.run(run_webex(graph, request)), retriever


def test_rachel_ends_blocked_and_writes_nothing():
    decision, retriever = run_graph(RACHEL, "E010")
    assert decision.request_key == "webex-E010-S001"
    assert (decision.status, decision.outcome) == ("blocked", "existing_request_reused")
    assert decision.blocking_reason == "42 of 40 seats in use (SUB001)"
    assert decision.existing_records == ["T001", "AR001", "AP001", "AP002"]
    assert decision.actions_taken == []
    assert [s.split()[0] for s in decision.next_steps] == ["E006", "E011"]
    assert decision.answer == "Your request AR001 is already on file."
    # The search ran as Rachel, so her own document filter applied.
    assert retriever.queries[0][1] == "E010"


def test_rachel_twice_gives_the_same_decision_and_no_new_rows():
    before = len(db.list_access_requests("E010")), len(db.list_approvals("E010"))
    first, _ = run_graph(RACHEL, "E010")
    second, _ = run_graph(RACHEL, "E010")
    assert first == second
    assert (len(db.list_access_requests("E010")), len(db.list_approvals("E010"))) == before


def test_a_refused_caller_gets_refused():
    decision, _ = run_graph(RACHEL_GREEN, "E010", wants_filing=True)
    assert (decision.status, decision.outcome) == ("completed", "refused")


def test_filing_is_not_built_yet_so_it_needs_a_human():
    decision, _ = run_graph(SARA_HR, "E001", wants_filing=True)
    assert decision.status == "needs_human"
    assert decision.actions_taken == []


def test_unknown_employee_skips_search_and_the_model():
    decision, retriever = run_graph(SARA_HR, "E999")
    assert decision.status == "needs_human"
    assert "No employee E999." in decision.answer
    assert retriever.queries == []
