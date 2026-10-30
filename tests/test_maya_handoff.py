"""Maya's Webex handoff (maya/graph.py): sent exactly once, and honestly reported.

    .venv/bin/python -m pytest -v tests/test_maya_handoff.py

No network and no model: the steps are called directly, with a hand-made S2 turn 10
state and FakeWebexPort, which records every handoff it gets (it doesn't remove
repeats, so a second send would show).
"""

import asyncio

from langchain_core.messages import HumanMessage

from maya.graph import authorize_or_deny, handoff
from maya.planning import TurnPlan, build_context_plan
from maya.ports import FakeWebexPort
from maya.schemas import make_idempotency_key


THREAD_ID = "test-s2"
GO_AHEAD = "Go ahead and file the Webex access request for her now."
SARA_HR = {"caller_id": "E004", "user_group": "UG_HR"}
RACHEL = {"caller_id": "E009", "user_group": "UG_REGULAR"}
WEBEX_KEY = make_idempotency_key(THREAD_ID, "E001", "Webex")


def turn_10_state(message: str = GO_AHEAD, action_confirmed: bool = True) -> dict:
    """The state just after the plan step of S2 turn 10: a go-ahead about Maya (E001)."""

    turn_plan = TurnPlan(
        current_intent="access_request",
        needs_documents=False,
        needs_records=False,
        action_confirmed=action_confirmed,
        subject_name="Maya Cohen",
    )
    plan = build_context_plan(turn_plan, {"Maya Cohen": "E001"})
    return {"messages": [HumanMessage(message)], "plan": plan.model_dump()}


def make_config(caller: dict = SARA_HR) -> dict:
    """A run config, as the code that starts a turn would pass it."""

    return {"configurable": {"thread_id": THREAD_ID, **caller}}


def run_handoff(state: dict, port) -> dict:
    """Run the async handoff step to the end, and return its state update."""

    return asyncio.run(handoff(state, make_config(), webex=port))


# SENT EXACTLY ONCE -----------------------------------------------------------

def test_sends_one_correct_handoff():
    port = FakeWebexPort()

    update = run_handoff(turn_10_state(), port)

    assert len(port.calls) == 1
    sent = port.calls[0]
    assert (sent.employee_id, sent.software) == ("E001", "Webex")
    assert (sent.caller.caller_id, sent.caller.user_group) == ("E004", "UG_HR")
    assert sent.thread_id == THREAD_ID
    assert sent.idempotency_key == WEBEX_KEY
    assert sent.reason
    assert update["handoffs"][WEBEX_KEY]["status"] == "pending"


def test_replay_sends_nothing_new():
    port = FakeWebexPort()
    state = turn_10_state()
    first = run_handoff(state, port)

    # The checkpointer saves the first update; the replayed turn starts from it.
    second = run_handoff({**state, **first}, port)

    assert len(port.calls) == 1
    request_id = first["handoffs"][WEBEX_KEY]["request_id"]
    assert request_id in second["reply"]["answer"]


def test_reply_says_pending_never_granted():
    answer = run_handoff(turn_10_state(), FakeWebexPort())["reply"]["answer"].lower()

    assert "pending" in answer
    assert "granted" not in answer


# ONLY A CONFIRMED WEBEX REQUEST, FROM SOMEONE ALLOWED ------------------------

def test_only_a_webex_go_ahead_routes_to_handoff():
    assert authorize_or_deny(turn_10_state(), make_config()) == "handoff"

    okta = turn_10_state(message="File Okta access for her")
    not_confirmed = turn_10_state(action_confirmed=False)
    assert authorize_or_deny(okta, make_config()) == "retrieve"
    assert authorize_or_deny(not_confirmed, make_config()) == "retrieve"


def test_denied_caller_never_reaches_the_port():
    # Rachel is a regular employee asking to file access for Maya.
    port = FakeWebexPort()

    route = authorize_or_deny(turn_10_state(), make_config(RACHEL))

    assert route == "deny"
    assert port.calls == []


# THE WORKFLOW IS DOWN --------------------------------------------------------

class BrokenWebexPort:
    """A workflow that can't be reached."""

    async def request_access(self, handoff):
        raise ConnectionError("Webex workflow unreachable")


def test_port_failure_saves_nothing_and_says_so():
    update = run_handoff(turn_10_state(), BrokenWebexPort())

    assert update["handoffs"] == {}
    assert "nothing was sent" in update["reply"]["answer"]
