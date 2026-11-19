"""A pasted email can be cited, for that turn only. (No model calls.)

    .venv/bin/python -m pytest -v tests/test_maya_pasted_message.py

Maya's own session, turn 2: the user pastes an email thread and asks what is theirs
to do. The answer comes from the thread, not from any document or record, so the
thread itself is the source. An ordinary question gets no citable ID, so it can never
back up a claim.
"""

from langchain_core.messages import AIMessage, HumanMessage

from maya.planning import TurnPlan, has_pasted_material, pasted_message_id, respond_messages
from maya.respond import enforce_evidence

THREAD = """Sara sent me this thread - what here is mine to do?

---------- Forwarded message ----------
From: Sara Ben-David <sara@novaops.example>
> On Tue, Amir wrote:
> Okta groups still pending.
Maya, sign the equipment responsibility form."""


def test_a_forwarded_email_is_pasted_material():
    assert has_pasted_material(THREAD)
    assert has_pasted_material("can you read this?\n> quoted reply line")


def test_an_ordinary_question_is_not():
    assert not has_pasted_material("How do I get my software licenses sorted?")
    # "from:" inside a sentence isn't an email header line.
    assert not has_pasted_material("What did I get from: the offer letter?")


def test_the_id_is_per_turn():
    first = [HumanMessage("hi"), AIMessage("hello")]
    assert pasted_message_id(first + [HumanMessage(THREAD)]) == "user_message#2"
    # The next turn is an ordinary question: no ID, even though turn 2 pasted.
    later = first + [HumanMessage(THREAD), AIMessage("ok"), HumanMessage("Thanks. And licenses?")]
    assert pasted_message_id(later) is None


def test_the_responder_sees_the_paste_with_its_id():
    state = {"messages": [HumanMessage(THREAD), AIMessage("Sign the form.")], "evidence": []}
    content = respond_messages(state, TurnPlan(current_intent="onboarding_status", needs_documents=False, needs_records=False), True)[1].content
    assert "--- PASTED BY THE USER [user_message#1] ---" in content
    plain = {"messages": [HumanMessage("Licenses?"), AIMessage("Ask IT.")], "evidence": []}
    assert "PASTED BY THE USER" not in respond_messages(
        plain, TurnPlan(current_intent="onboarding_status", needs_documents=False, needs_records=False), True)[1].content


CHECKLIST = {"employee_id": "E001", "name": "Maya Cohen", "role": "Customer Success Manager",
             "start_date": "2026-08-01", "location": "Israel", "status": "in_progress"}
ITEM = {"name": "Equipment responsibility form", "kind": "task", "status": "pending",
        "citations": [{"source": "user_message", "chunk_id": "user_message#2"}]}


def test_a_citation_to_the_paste_survives_when_seen():
    reply = enforce_evidence({"answer": "Sign the form.", "checklist": {**CHECKLIST, "items": [ITEM]}},
                             ["user_message#2"], True, "E001")
    assert reply.checklist.items[0].status == "pending"
    assert reply.checklist.items[0].citations[0].chunk_id == "user_message#2"


def test_a_citation_to_a_paste_from_no_turn_is_dropped():
    # Not in sources_seen (no paste this turn): dropped, and the item is unresolved.
    reply = enforce_evidence({"answer": "Sign the form.", "checklist": {**CHECKLIST, "items": [ITEM]}},
                             [], True, "E001")
    assert reply.checklist.items[0].status == "unresolved"


def test_the_planner_is_told_a_pasted_answer_needs_no_lookup():
    # A prompt rule, so the real check is the runner (Maya's own session, turn 2:
    # "plan needs no lookup"). This only guards against the rule being deleted.
    from maya.memory import PLANNER_PROMPT
    assert "pasted text: needs_records and needs_documents are False" in PLANNER_PROMPT
