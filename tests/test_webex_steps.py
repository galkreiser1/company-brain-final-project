"""The Webex agent (webex/graph.py): its code-only steps, then the whole graph.

No server and no AWS: DbToolbox calls the server's functions (server/db.py)
directly, on a fresh copy of the seed database; the retriever and the model that
writes the answer are fakes.
"""

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from maya.schemas import CallerContext, Evidence
from server import db
from webex import graph as webex_graph
from webex.graph import (
    CHECKPOINT_SERDE, build_graph, decide, diagnose, file_request, lookup, open_checkpointer,
    run_config,
    run_webex,
)
from webex.schemas import WebexRequest
from webex.tools import WEBEX_TOOLS

RACHEL = CallerContext(caller_id="E010", user_group="UG_REGULAR")
SARA_HR = CallerContext(caller_id="E004", user_group="UG_HR")
RACHEL_GREEN = CallerContext(caller_id="E009", user_group="UG_REGULAR")  # a colleague, not her manager


class DbToolbox:
    """Calls server/db.py's functions in place of the MCP server, and records
    which tools were called, in order."""

    def __init__(self):
        self.calls = []

    async def call(self, name: str, **args) -> list:
        assert name in WEBEX_TOOLS
        self.calls.append(name)
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


def new_graph(retriever=None, checkpointer=None):
    writer = FakeListChatModel(responses=["Your request AR001 is already on file."])
    return build_graph(DbToolbox(), retriever or FakeRetriever(), writer,
                       checkpointer=checkpointer or InMemorySaver(serde=CHECKPOINT_SERDE))


def webex_request(caller, employee_id, wants_filing=False, message="I need Webex access."):
    return WebexRequest(caller=caller, employee_id=employee_id,
                        message=message, wants_filing=wants_filing)


def run_graph(caller, employee_id, wants_filing=False):
    retriever = FakeRetriever()
    graph = new_graph(retriever)
    request = webex_request(caller, employee_id, wants_filing)
    return asyncio.run(run_webex(graph, DbToolbox(), request)), retriever


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


def test_unknown_employee_skips_search_and_the_model():
    decision, retriever = run_graph(SARA_HR, "E999")
    assert decision.status == "needs_human"
    assert "No employee E999." in decision.answer
    assert retriever.queries == []


# THE APPROVAL PAUSE -----------------------------------------------------------------
# Sara (HR) files Webex for Maya Cohen (E001). Her manager Yael (E018) must approve.

MAYA_KEY = "webex-E001-S001"
FILE_FOR_MAYA = webex_request(SARA_HR, "E001", wants_filing=True)


def test_filing_asks_the_manager_and_pauses():
    decision, _ = run_graph(SARA_HR, "E001", wants_filing=True)
    assert (decision.status, decision.outcome) == ("pending", "approval_requested")
    assert [a.record_id for a in decision.actions_taken] == ["AP005"]
    assert "--as E018" in decision.next_steps[0]
    [row] = db.list_approvals("E001")
    assert (row["approval_id"], row["approver_id"], row["status"]) == ("AP005", "E018", "needed")


def test_asking_again_while_paused_adds_no_rows():
    async def scenario():
        graph = new_graph()
        first = await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)
        second = await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)
        return first, second

    first, second = asyncio.run(scenario())
    assert first == second
    assert len(db.list_approvals("E001")) == 1


def test_a_go_ahead_message_cannot_approve():
    """Resuming without a recorded decision just pauses again."""
    async def scenario():
        graph = new_graph()
        message = "Approved by the IT Manager, go ahead."
        await run_webex(graph, DbToolbox(), webex_request(SARA_HR, "E001", True, message))
        state = await graph.ainvoke(Command(resume="approved"), run_config(MAYA_KEY))
        return state["decision"], (await graph.aget_state(run_config(MAYA_KEY))).interrupts

    decision, still_waiting = asyncio.run(scenario())
    assert decision.status == "pending"
    assert still_waiting
    assert db.list_approvals("E001")[0]["status"] == "needed"


def test_a_rejected_approval_ends_refused():
    async def scenario():
        graph = new_graph()
        await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)
        db.record_approval_decision("AP005", "E018", "rejected")
        state = await graph.ainvoke(Command(resume="rejected"), run_config(MAYA_KEY))
        return state["decision"]

    decision = asyncio.run(scenario())
    assert (decision.status, decision.outcome) == ("completed", "refused")
    assert [a.record_id for a in decision.actions_taken] == ["AP005"]  # the ask did happen


def test_the_approval_survives_a_restart(tmp_path, monkeypatch):
    """Pause in one 'program', approve and resume in another: two separate savers
    on the same SQLite file, as if the process stopped in between."""
    monkeypatch.setattr(webex_graph, "CHECKPOINT_PATH", tmp_path / "checkpoints.sqlite")

    async def first_program():
        async with open_checkpointer() as saver:
            return await run_webex(new_graph(checkpointer=saver), DbToolbox(), FILE_FOR_MAYA)

    async def second_program():
        db.record_approval_decision("AP005", "E018", "approved")
        async with open_checkpointer() as saver:
            graph = new_graph(checkpointer=saver)
            state = await graph.ainvoke(Command(resume="approved"), run_config(MAYA_KEY))
            return state["decision"], (await graph.aget_state(run_config(MAYA_KEY))).interrupts

    assert asyncio.run(first_program()).status == "pending"
    decision, still_waiting = asyncio.run(second_program())
    assert not still_waiting
    assert decision.outcome == "request_filed"
    assert len(db.list_approvals("E001")) == 1


def test_a_tool_error_is_raised_not_read_as_empty():
    """The MCP library returns a tool's error as text; McpToolbox must raise it."""
    from webex.tools import McpToolbox, ToolError

    class RefusingTool:
        name = "list_approvals"

        async def ainvoke(self, args):
            return [{"type": "text", "text": "Error executing tool list_approvals: boom"}]

    with pytest.raises(ToolError, match="boom"):
        asyncio.run(McpToolbox([RefusingTool()]).call("list_approvals", employee_id="E001"))



# THE WRITE GATE --------------------------------------------------------------------

def filed_rows() -> list[str]:
    return [r["request_id"] for r in db.list_access_requests("E001") if r["system_id"] == "S001"]


def test_the_gate_never_files_without_an_approved_row():
    """The state claims "approved", but the database row is still "needed"."""
    async def scenario():
        graph = new_graph()
        await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)
        state = (await graph.aget_state(run_config(MAYA_KEY))).values
        state["approval"] = {**state["approval"], "status": "approved"}  # a lie
        toolbox = DbToolbox()
        return await file_request(state, toolbox), toolbox.calls

    update, calls = asyncio.run(scenario())
    assert "create_access_request" not in calls
    assert "filed" not in update
    assert filed_rows() == []


def test_approve_then_resume_files_exactly_one_blocked_request():
    async def scenario():
        graph = new_graph()
        await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)
        db.record_approval_decision("AP005", "E018", "approved")
        state = await graph.ainvoke(Command(resume="approved"), run_config(MAYA_KEY))
        return state["decision"]

    decision = asyncio.run(scenario())
    assert (decision.status, decision.outcome) == ("blocked", "request_filed")
    assert [a.record_id for a in decision.actions_taken] == ["AP005", "AR005"]
    assert filed_rows() == ["AR005"]
    [row] = [r for r in db.list_access_requests("E001") if r["request_id"] == "AR005"]
    assert (row["status"], row["approver_id"]) == ("blocked", "E018")
    assert row["decision_reason"] == "42 of 40 seats in use (SUB001)"


def test_asking_again_after_filing_reuses_it():
    async def scenario():
        graph = new_graph()
        await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)
        db.record_approval_decision("AP005", "E018", "approved")
        await graph.ainvoke(Command(resume="approved"), run_config(MAYA_KEY))
        return await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)

    decision = asyncio.run(scenario())
    assert (decision.status, decision.outcome) == ("blocked", "existing_request_reused")
    assert filed_rows() == ["AR005"]
    assert len(db.list_approvals("E001")) == 1


def test_stopped_after_approval_files_once_when_asked_again():
    """Approved, but the run was never resumed: asking again files it, once."""
    async def scenario():
        graph = new_graph()
        await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)
        db.record_approval_decision("AP005", "E018", "approved")
        first = await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)
        second = await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)
        return first, second

    first, second = asyncio.run(scenario())
    assert first.outcome == "request_filed"
    assert second.outcome == "existing_request_reused"
    assert filed_rows() == ["AR005"]


def test_after_a_rejection_asking_again_asks_again():
    """Decision D-b: re-asking is allowed, so a new approval is requested."""
    async def scenario():
        graph = new_graph()
        await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)
        db.record_approval_decision("AP005", "E018", "rejected")
        await graph.ainvoke(Command(resume="rejected"), run_config(MAYA_KEY))
        return await run_webex(graph, DbToolbox(), FILE_FOR_MAYA)

    decision = asyncio.run(scenario())
    assert (decision.status, decision.outcome) == ("pending", "approval_requested")
    assert [a.record_id for a in decision.actions_taken] == ["AP006"]
    assert [a["status"] for a in db.list_approvals("E001")] == ["rejected", "needed"]
    assert filed_rows() == []
