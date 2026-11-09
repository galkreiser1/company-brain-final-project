"""Maya passes Webex questions to the Webex scope (maya/graph.py: is_webex_question,
ask_webex), and shows Webex's own answer after a handoff.

No network and no model: the steps are called directly with FakeWebexPort or a
small port that returns a fixed Webex result.
"""

import asyncio

from langchain_core.messages import HumanMessage

from maya.graph import ask_webex, authorize_or_deny, handoff
from maya.planning import TurnPlan, build_context_plan
from maya.ports import FakeWebexPort
from maya.schemas import EvidenceCitation, PendingAccessResult

EMPLOYEE_IDS = {"Maya Cohen": "E001", "Rachel Stein": "E010"}
RACHEL = {"caller_id": "E010", "user_group": "UG_REGULAR"}
SARA_HR = {"caller_id": "E004", "user_group": "UG_HR"}
RACHEL_GREEN = {"caller_id": "E009", "user_group": "UG_REGULAR"}


def state_for(message: str, intent: str, subject: str = "",
              confirmed: bool = False) -> dict:
    """The state just after the plan step, for one message."""
    turn_plan = TurnPlan(current_intent=intent, needs_documents=False, needs_records=True,
                         action_confirmed=confirmed, subject_name=subject)
    plan = build_context_plan(turn_plan, EMPLOYEE_IDS)
    return {"messages": [HumanMessage(message)], "plan": plan.model_dump()}


def config(caller: dict) -> dict:
    return {"configurable": {"thread_id": "test-webex", **caller}}


RACHEL_ASKS = state_for("I need Webex access. Webex says my account is not licensed.",
                        "access_request")


def test_webex_access_and_ticket_questions_go_to_webex():
    ticket = state_for("What is the status of my Webex login ticket?", "ticket_status")
    assert authorize_or_deny(RACHEL_ASKS, config(RACHEL)) == "webex"
    assert authorize_or_deny(ticket, config(RACHEL)) == "webex"


def test_other_systems_stay_with_maya():
    salesforce = state_for("I need Salesforce access.", "access_request")
    laptop = state_for("What's the status of my laptop ticket?", "ticket_status")
    subscription = state_for("How many Webex seats are in use?", "subscription_review")
    for state in (salesforce, laptop, subscription):
        assert authorize_or_deny(state, config(RACHEL)) == "retrieve"


def test_an_unknown_named_person_stays_with_maya():
    # HR may ask about anyone, but a name that matches nobody can't go to Webex.
    unknown = state_for("Does Dana Fox have Webex access?", "access_request", "Dana Fox")
    assert authorize_or_deny(unknown, config(SARA_HR)) == "retrieve"


def test_a_denied_caller_never_reaches_webex():
    about_maya = state_for("Is Maya Cohen's Webex access done?", "access_request", "Maya Cohen")
    assert authorize_or_deny(about_maya, config(RACHEL_GREEN)) == "deny"


def test_the_webex_node_asks_about_the_caller_and_shows_the_answer():
    port = FakeWebexPort()
    update = asyncio.run(ask_webex(RACHEL_ASKS, config(RACHEL), webex=port))

    [question] = port.questions
    assert (question.employee_id, question.caller.caller_id) == ("E010", "E010")
    assert port.calls == []  # a question never files
    assert update["reply"]["answer"] == "(fake Webex answer about E010)"


def test_the_webex_node_asks_about_the_named_person():
    port = FakeWebexPort()
    about_maya = state_for("Is Maya Cohen's Webex access done?", "access_request", "Maya Cohen")
    asyncio.run(ask_webex(about_maya, config(SARA_HR), webex=port))
    assert port.questions[0].employee_id == "E001"


class ReusingWebexPort:
    """Webex answers a go-ahead by reusing an existing request (W-S-01 turn 4)."""

    async def request_access(self, handoff):
        return PendingAccessResult(
            workflow="webex_access", status="blocked", request_id="AR001",
            idempotency_key=handoff.idempotency_key,
            answer="AR001 is already on file; 42 of 40 seats are in use (SUB001).",
            citations=[EvidenceCitation(source="list_access_requests", record_id="AR001")],
        )


def test_a_handoff_shows_webex_own_answer():
    """Decision O2: not a fixed "pending" text, which would be wrong here."""
    go_ahead = state_for("Go ahead and file Rachel Stein's Webex access request.",
                         "access_request", "Rachel Stein", confirmed=True)
    update = asyncio.run(handoff(go_ahead, config(SARA_HR), webex=ReusingWebexPort()))

    assert update["reply"]["answer"].startswith("AR001 is already on file")
    assert [c["record_id"] for c in update["reply"]["citations"]] == ["AR001"]
    [saved] = update["handoffs"].values()
    assert (saved["status"], saved["request_id"]) == ("blocked", "AR001")
