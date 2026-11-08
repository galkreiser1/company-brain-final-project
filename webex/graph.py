"""The Webex agent: works out why someone can't use a system, and what to do.

The steps (docs/webex-design.md, section 3), each its own graph node:
  lookup   -> calls the read tools, in a fixed order
  search   -> searches the help articles, policies, contracts and memos
  diagnose -> turns the records into a Diagnosis
  decide   -> picks one path: answer_only, reuse, approve_then_file or refuse
  request_approval -> (approve_then_file only) saves an approval for the manager
  respond  -> code builds the AccessDecision; a model writes only its `answer`
  wait     -> pauses until resumed, then re-reads the approval from the database
  file_request -> the write gate: files the access request only if the approval
              row in the database says approved

Each run is saved under its request key (webex-E001-S001) in
runs/webex_checkpoints.sqlite, so a paused run survives the program stopping.

Each step gets the state and returns only the keys it changes, as in maya/graph.py.
"""

import json
from functools import partial
from typing import TypedDict

import aiosqlite
from contextlib import asynccontextmanager
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from maya.config import REPO_ROOT
from maya.model import get_model
from maya.retrieval import EvidenceRetriever
from maya.schemas import CallerContext, Evidence
from webex.respond import build_decision, write_answer
from webex.schemas import (
    AccessDecision, DecisionPath, Diagnosis, WebexRequest, make_request_key,
)
from webex.tools import Toolbox

HR_GROUP = "UG_HR"
IT_GROUP = "UG_IT"
# A request in one of these states is over, so it can't be reused.
CLOSED_REQUEST_STATUSES = {"rejected", "denied", "cancelled", "revoked"}
OPEN_APPROVAL_STATUSES = {"needed", "pending"}
# Where paused runs are saved. runs/ is git-ignored.
CHECKPOINT_PATH = REPO_ROOT / "runs" / "webex_checkpoints.sqlite"
# Our own types that a saved run holds. LangGraph will only load listed types back
# from the file, so a tampered checkpoint can't make it build anything else.
CHECKPOINT_SERDE = JsonPlusSerializer(allowed_msgpack_modules=[
    ("webex.schemas", "WebexRequest"), ("webex.schemas", "Diagnosis"),
    ("webex.schemas", "AccessDecision"), ("maya.schemas", "Evidence"),
])
# The document collections search looks in (Maya also searches employment files).
SEARCH_COLLECTIONS = ["it_kb", "policies", "contracts", "internal_memos"]


class WebexState(TypedDict, total=False):
    request: WebexRequest
    records: dict           # the read tools' results, by name
    problem: str | None     # why the run can't go on, e.g. "No employee E999."
    evidence: list[Evidence]
    diagnosis: Diagnosis
    path: DecisionPath
    approval: dict | None   # the approvals row this run asked for
    filed: dict | None      # the access request file_request created
    decision: AccessDecision


# 1. LOOKUP ------------------------------------------------------------------------

async def lookup(state: WebexState, toolbox: Toolbox) -> dict:
    """Call the read tools for this person and system, always in the same order.

    Returns: {"records": ...}, or {"problem": ...} if the person or the
    subscription can't be found exactly (then nothing else is looked up).
    """
    request = state["request"]
    # get_employee also matches names, so keep only the exact id.
    people = await toolbox.call("get_employee", query=request.employee_id)
    employee = [p for p in people if p["employee_id"] == request.employee_id]
    if not employee:
        return {"problem": f"No employee {request.employee_id}."}

    subscriptions = await toolbox.call("check_software_subscription", software=request.software)
    if len(subscriptions) != 1:
        return {"problem": f"No single subscription matches {request.software!r}."}

    employee_id = request.employee_id
    records = {
        "employee": employee[0],
        "subscription": subscriptions[0],
        "tickets": await toolbox.call("list_employee_tickets", employee_id=employee_id),
        "access_requests": await toolbox.call("list_access_requests", employee_id=employee_id),
        "approvals": await toolbox.call("list_approvals", employee_id=employee_id),
    }
    return {"records": records, "problem": None}


# 2. DIAGNOSE ----------------------------------------------------------------------

def caller_may_request(caller: CallerContext, employee: dict) -> bool:
    """The person themself, their manager, HR or IT may ask for someone's access."""
    return (
        caller.caller_id in (employee["employee_id"], employee["manager_id"])
        or caller.user_group in (HR_GROUP, IT_GROUP)
    )


def diagnose(state: WebexState) -> dict:
    """Turn the lookup's records into one Diagnosis."""
    request, records = state["request"], state["records"]
    employee, subscription = records["employee"], records["subscription"]
    system_id = subscription["system_id"]

    # A request for the same system that isn't over yet is the one to reuse.
    existing = [
        r for r in records["access_requests"]
        if r["system_id"] == system_id and r["status"] not in CLOSED_REQUEST_STATUSES
    ]
    existing_request_id = existing[0]["request_id"] if existing else None

    # Approvals still waiting on that request. (The approval this run asks for
    # itself is kept in state["approval"] and reported as an action, not here,
    # so a replay reports the same thing as the first run.)
    open_approval_ids = [
        a["approval_id"] for a in records["approvals"]
        if existing_request_id and a["request_id"] == existing_request_id
        and a["status"] in OPEN_APPROVAL_STATUSES
    ]

    diagnosis = Diagnosis(
        employee_id=employee["employee_id"],
        system_id=system_id,
        system_name=subscription["system_name"],
        # systems_allowed_by_role is stored as JSON text, e.g. '["Webex", "Slack"]'.
        allowed_by_role=subscription["system_name"]
        in json.loads(employee["systems_allowed_by_role"]),
        caller_may_request=caller_may_request(request.caller, employee),
        subscription_id=subscription["subscription_id"],
        seats_in_use=subscription["active_seats"],
        seat_limit=subscription["seat_limit"],
        existing_request_id=existing_request_id,
        open_approval_ids=open_approval_ids,
    )
    return {"diagnosis": diagnosis}


# 3. DECIDE ------------------------------------------------------------------------

def decide(state: WebexState) -> dict:
    """Pick one path from the diagnosis and the request."""
    diagnosis, request = state["diagnosis"], state["request"]
    if not (diagnosis.caller_may_request and diagnosis.allowed_by_role):
        path = "refuse"
    elif diagnosis.existing_request_id:
        path = "reuse"  # never file twice, even if the caller asked to
    elif request.wants_filing:
        path = "approve_then_file"
    else:
        path = "answer_only"
    return {"path": path}


# 2. SEARCH ------------------------------------------------------------------------

def search(state: WebexState, retriever: EvidenceRetriever) -> dict:
    """Find the help articles, policies, contract and memos about this system."""
    request = state["request"]
    system_name = state["records"]["subscription"]["system_name"]
    query = f"{system_name} not licensed, seat limit, access request and approval"
    evidence = retriever.search(query, request.caller, collections=SEARCH_COLLECTIONS)
    return {"evidence": evidence}


# 4b. ASK FOR APPROVAL, AND WAIT -----------------------------------------------------

async def request_approval(state: WebexState, toolbox: Toolbox) -> dict:
    """Ask the employee's manager to approve. The tool returns the existing row
    instead of adding one if this request key already has one, so a replay is safe."""
    employee, diagnosis = state["records"]["employee"], state["diagnosis"]
    if not employee["manager_id"]:
        return {"problem": f"{employee['employee_id']} has no manager to approve."}
    [approval] = await toolbox.call(
        "request_approval",
        request_key=make_request_key(diagnosis.employee_id, diagnosis.system_id),
        approver_id=employee["manager_id"],
        reason=f"Manager approval for {diagnosis.system_name} access for {diagnosis.employee_id}.",
    )
    return {"approval": approval}


async def wait_for_approval(state: WebexState, toolbox: Toolbox) -> dict:
    """Pause the run here. When resumed, read the approval row again.

    interrupt() saves the run and stops. On resume, LangGraph runs this node again
    from the top, and interrupt() then returns the resume value instead of stopping.
    We don't use that value: a message saying "approved" can't approve anything.
    """
    approval = state["approval"]
    interrupt({"approval_id": approval["approval_id"], "approver_id": approval["approver_id"]})
    rows = await toolbox.call("list_approvals", employee_id=state["request"].employee_id)
    [fresh] = [row for row in rows if row["approval_id"] == approval["approval_id"]]
    return {"approval": fresh}


# 4c. THE WRITE GATE --------------------------------------------------------------

async def file_request(state: WebexState, toolbox: Toolbox) -> dict:
    """File the access request, but only if the approval row says approved now.

    It reads the row again itself instead of trusting the state, so nothing that
    resumed the run, and no message text, can open the gate."""
    request, diagnosis = state["request"], state["diagnosis"]
    approval_id = state["approval"]["approval_id"]
    rows = await toolbox.call("list_approvals", employee_id=request.employee_id)
    [approval] = [row for row in rows if row["approval_id"] == approval_id]
    if approval["status"] != "approved":
        return {"approval": approval}  # the gate stays shut: no write

    # No duplicates: a request for this system filed since the lookup wins.
    requests = await toolbox.call("list_access_requests", employee_id=request.employee_id)
    for existing in requests:
        same_system = existing["system_id"] == diagnosis.system_id
        if same_system and existing["status"] not in CLOSED_REQUEST_STATUSES:
            return {"problem": f"{existing['request_id']} is already filed; nothing new was filed."}

    # Over the seat limit, the request is filed but blocked (decision D-a).
    blocked = diagnosis.over_seat_limit
    [filed] = await toolbox.call(
        "create_access_request",
        employee_id=request.employee_id, software=diagnosis.system_name,
        business_justification=f"Approved by {approval['approver_id']} in {approval_id}.",
        status="blocked" if blocked else "approved",
        approver_id=approval["approver_id"],
        decision_reason=diagnosis.seat_reason if blocked else f"Approved in {approval_id}.",
    )
    return {"approval": approval, "filed": filed}


# 5. RESPOND -----------------------------------------------------------------------

async def respond(state: WebexState, answer_model) -> dict:
    """Build the AccessDecision. A failed lookup ends here as needs_human."""
    request = state["request"]
    if state.get("problem"):
        decision = AccessDecision(
            request_key=make_request_key(request.employee_id, request.software),
            employee_id=request.employee_id, system=request.software,
            status="needs_human", outcome="answered",
            answer=f"I couldn't check this request: {state['problem']}",
        )
        return {"decision": decision}

    diagnosis = state["diagnosis"]
    decision = build_decision(
        request, state["path"], diagnosis, state["records"], state["evidence"],
        make_request_key(diagnosis.employee_id, diagnosis.system_id),
        state.get("approval"), state.get("filed"),
    )
    decision.answer = await write_answer(answer_model, request, decision)
    return {"decision": decision}


# THE GRAPH ------------------------------------------------------------------------

def after_lookup(state: WebexState) -> str:
    return "respond" if state.get("problem") else "search"


def after_decide(state: WebexState) -> str:
    return "request_approval" if state["path"] == "approve_then_file" else "respond"


def by_approval(state: WebexState) -> str:
    """approved -> file_request; still open or rejected -> respond (which then
    pauses or ends). Used after request_approval too, so a run stopped after the
    approval but before filing files when asked again."""
    approval = state.get("approval")
    if not state.get("problem") and approval and approval["status"] == "approved":
        return "file_request"
    return "respond"


def after_respond(state: WebexState) -> str:
    approval = state.get("approval")
    waiting = (state.get("path") == "approve_then_file" and not state.get("problem")
               and approval and approval["status"] in OPEN_APPROVAL_STATUSES)
    return "wait" if waiting else END


def build_graph(toolbox: Toolbox, retriever: EvidenceRetriever, answer_model=None,
                checkpointer=None):
    """lookup -> search -> diagnose -> decide -> respond, except on approve_then_file:
    request_approval, then by the approval's status: open -> respond -> wait (pause),
    approved -> file_request -> respond, rejected -> respond.
    A lookup that finds nothing goes straight to respond.

    `answer_model` words the answer (a fake in tests). Not called `writer`: LangGraph
    fills a node argument with that name itself.
    `checkpointer` saves each run (an open_checkpointer(), or InMemorySaver in tests);
    the pause needs one."""
    graph = StateGraph(WebexState)
    graph.add_node("lookup", partial(lookup, toolbox=toolbox))
    graph.add_node("search", partial(search, retriever=retriever))
    graph.add_node("diagnose", diagnose)
    graph.add_node("decide", decide)
    graph.add_node("request_approval", partial(request_approval, toolbox=toolbox))
    graph.add_node("respond", partial(respond, answer_model=answer_model or get_model()))
    graph.add_node("wait", partial(wait_for_approval, toolbox=toolbox))
    graph.add_node("file_request", partial(file_request, toolbox=toolbox))

    graph.add_edge(START, "lookup")
    graph.add_conditional_edges("lookup", after_lookup, {"search": "search", "respond": "respond"})
    graph.add_edge("search", "diagnose")
    graph.add_edge("diagnose", "decide")
    graph.add_conditional_edges(
        "decide", after_decide, {"request_approval": "request_approval", "respond": "respond"},
    )
    approval_routes = {"file_request": "file_request", "respond": "respond"}
    graph.add_conditional_edges("request_approval", by_approval, approval_routes)
    graph.add_conditional_edges("respond", after_respond, {"wait": "wait", END: END})
    graph.add_conditional_edges("wait", by_approval, approval_routes)
    graph.add_edge("file_request", "respond")
    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_checkpointer():
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    async with aiosqlite.connect(CHECKPOINT_PATH) as conn:
        yield AsyncSqliteSaver(conn, serde=CHECKPOINT_SERDE)


async def request_key_for(toolbox: Toolbox, request: WebexRequest) -> str:
    """e.g. "webex-E001-S001". Falls back to the software name if it matches no
    single subscription (the run then stops at lookup anyway)."""
    subscriptions = await toolbox.call("check_software_subscription", software=request.software)
    system_id = subscriptions[0]["system_id"] if len(subscriptions) == 1 else request.software
    return make_request_key(request.employee_id, system_id)


def run_config(request_key: str) -> dict:
    """LangGraph saves and finds a run by its thread_id: here, the request key."""
    return {"configurable": {"thread_id": request_key}}


async def run_webex(graph, toolbox: Toolbox, request: WebexRequest) -> AccessDecision:
    """Run one request and return its AccessDecision (status "pending" if it paused).

    Asking again lands on the same saved run and starts it over from lookup, so it
    sees what's in the database now: the approval it already asked for is reused."""
    config = run_config(await request_key_for(toolbox, request))
    final_state = await graph.ainvoke({"request": request, "approval": None, "filed": None}, config)
    return final_state["decision"]
