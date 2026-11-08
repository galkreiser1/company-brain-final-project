"""The respond step's parts: code builds the AccessDecision, a model writes only `answer`.

Every fact, record id and next step here comes from tool results or retrieved
chunks, so the reply can't claim something that didn't happen.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from maya.schemas import Evidence, EvidenceCitation
from webex.schemas import (
    AccessDecision, ActionTaken, Diagnosis, ObservedFact, WebexRequest,
)

# How many retrieved chunks to list as observed facts.
EVIDENCE_FACTS = 3

# The Webex help article and memo, not an approval row, say what to do about a full
# subscription (docs/webex-design.md, section 6).
SEAT_NEXT_STEP = "IT Manager to reclaim unused seats, or get Finance to approve more"

OPEN_APPROVAL_STATUSES = {"needed", "pending"}

# Paths that end the run without writing anything.
OUTCOMES = {"answer_only": "answered", "reuse": "existing_request_reused", "refuse": "refused"}


def observed_facts(diagnosis: Diagnosis, records: dict,
                   evidence: list[Evidence]) -> list[ObservedFact]:
    """The facts behind the decision, each with its source."""

    def record(tool: str, record_id: str, fact: str) -> ObservedFact:
        return ObservedFact(fact=fact, citation=EvidenceCitation(source=tool, record_id=record_id))

    allowed = "is" if diagnosis.allowed_by_role else "is not"
    facts = [
        record("get_employee", diagnosis.employee_id,
               f"{diagnosis.employee_id} {allowed} allowed {diagnosis.system_name} by role"),
        record("check_software_subscription", diagnosis.subscription_id,
               f"{diagnosis.system_name}: {diagnosis.seat_reason}"),
    ]
    for ticket in system_tickets(diagnosis, records):
        facts.append(record("list_employee_tickets", ticket["ticket_id"],
                            f"Ticket {ticket['ticket_id']} is {ticket['status']}: {ticket['subject']}"))
    for request in records["access_requests"]:
        if request["request_id"] == diagnosis.existing_request_id:
            facts.append(record("list_access_requests", request["request_id"],
                                f"Request {request['request_id']} is {request['status']}"))
    for approval in open_approvals(diagnosis, records):
        facts.append(record("list_approvals", approval["approval_id"],
                            f"Approval {approval['approval_id']} by {approval['approver_id']} "
                            f"is {approval['status']}"))
    for chunk in evidence[:EVIDENCE_FACTS]:
        facts.append(ObservedFact(
            fact=f"{chunk.doc_id}, section {chunk.section or 'top'}",
            citation=EvidenceCitation(source=chunk.doc_id, chunk_id=chunk.chunk_id,
                                      section=chunk.section or None),
        ))
    return facts


def system_tickets(diagnosis: Diagnosis, records: dict) -> list[dict]:
    return [t for t in records["tickets"] if t["related_system_id"] == diagnosis.system_id]


def open_approvals(diagnosis: Diagnosis, records: dict) -> list[dict]:
    return [a for a in records["approvals"] if a["approval_id"] in diagnosis.open_approval_ids]


def next_steps(diagnosis: Diagnosis, records: dict) -> list[str]:
    """Each waiting approval, then the seat problem if nobody is on it yet."""
    steps = [
        f"{a['approver_id']} to decide approval {a['approval_id']} ({a['decision_reason']})"
        for a in open_approvals(diagnosis, records)
    ]
    if diagnosis.over_seat_limit and not steps:
        steps.append(SEAT_NEXT_STEP)
    return steps


def approval_result(approval: dict, diagnosis: Diagnosis) -> tuple:
    """(status, outcome, next_steps) for the approve-then-file path."""
    approval_id, approver = approval["approval_id"], approval["approver_id"]
    if approval["status"] in OPEN_APPROVAL_STATUSES:
        steps = [f"{approver} to decide approval {approval_id}: python -m webex.approve "
                 f"{approval_id} --as {approver} --decision approved (or rejected)"]
        return "pending", "approval_requested", steps
    if approval["status"] == "rejected":
        return "completed", "refused", []
    # Approved. Filing the request is the next build step (the write gate).
    return "pending", "approval_requested", ["File the access request (not built yet)"]


def approval_action(approval: dict) -> ActionTaken:
    return ActionTaken(
        tool="request_approval", record_id=approval["approval_id"],
        summary=f"Asked {approval['approver_id']} to approve {approval['request_id']}",
    )


def build_decision(request: WebexRequest, path: str, diagnosis: Diagnosis, records: dict,
                   evidence: list[Evidence], request_key: str,
                   approval: dict | None = None) -> AccessDecision:
    """The full AccessDecision, with a placeholder answer the model then replaces."""
    steps, actions = next_steps(diagnosis, records), []
    reason = diagnosis.seat_reason if diagnosis.over_seat_limit else None
    if path == "approve_then_file":
        status, outcome, steps = approval_result(approval, diagnosis)
        actions = [approval_action(approval)]
        if reason:
            steps.append(SEAT_NEXT_STEP)  # filing won't free a seat
    elif path == "refuse":
        status, outcome, reason = "completed", "refused", None
    else:
        # Something outside the person's control stands in the way, or nothing does.
        status, outcome = ("blocked" if reason else "completed"), OUTCOMES[path]
    if status != "blocked":
        reason = None  # only a blocked decision names a blocker

    existing = [t["ticket_id"] for t in system_tickets(diagnosis, records)]
    existing += [diagnosis.existing_request_id] if diagnosis.existing_request_id else []
    existing += diagnosis.open_approval_ids
    return AccessDecision(
        request_key=request_key, employee_id=request.employee_id,
        system=diagnosis.system_name, status=status, outcome=outcome,
        blocking_reason=reason, observed_facts=observed_facts(diagnosis, records, evidence),
        actions_taken=actions, existing_records=existing, next_steps=steps, answer="",
    )


WRITER_PROMPT = (
    "You write the reply for NovaOps IT support. You get the caller's message and a "
    "decision as JSON. Write 2-4 plain sentences to the caller, using only the facts, "
    "record ids and next steps in the JSON. Never say access was granted or a seat was "
    "assigned. Don't promise anything that isn't in next_steps."
)


async def write_answer(answer_model, request: WebexRequest, decision: AccessDecision) -> str:
    """Ask the model for the reply text. The decision it reads is already final."""
    facts = decision.model_dump(exclude={"answer"})
    reply = await answer_model.ainvoke([
        SystemMessage(WRITER_PROMPT),
        HumanMessage(f"Message: {request.message}\n\nDecision:\n{json.dumps(facts, indent=1)}"),
    ])
    return reply.text
