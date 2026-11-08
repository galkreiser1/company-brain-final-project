"""The Webex agent's code-only steps (webex/graph.py): lookup, diagnose, decide.

No server and no AWS: DbToolbox calls the server's functions (server/db.py)
directly, on a fresh copy of the seed database.
"""

import asyncio

import pytest

from maya.schemas import CallerContext
from server import db
from webex.graph import decide, diagnose, lookup
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
