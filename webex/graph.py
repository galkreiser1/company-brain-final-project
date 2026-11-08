"""The Webex agent: works out why someone can't use a system, and what to do.

The steps (docs/webex-design.md, section 3), each its own graph node:
  lookup   -> calls the read tools, in a fixed order
  search   -> searches the help articles, policies, contracts and memos
  diagnose -> turns the records into a Diagnosis
  decide   -> picks one path: answer_only, reuse, approve_then_file or refuse
  respond  -> code builds the AccessDecision; a model writes only its `answer`
Read-only so far: no step writes anything.

Each step gets the state and returns only the keys it changes, as in maya/graph.py.
"""

import json
from functools import partial
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

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
# The document collections search looks in (Maya also searches employment files).
SEARCH_COLLECTIONS = ["it_kb", "policies", "contracts", "internal_memos"]


class WebexState(TypedDict, total=False):
    request: WebexRequest
    records: dict           # the read tools' results, by name
    problem: str | None     # why the run can't go on, e.g. "No employee E999."
    evidence: list[Evidence]
    diagnosis: Diagnosis
    path: DecisionPath
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

    # Approvals still waiting, for that request or for this run's own key.
    request_key = make_request_key(employee["employee_id"], system_id)
    related_ids = {existing_request_id, request_key}
    open_approval_ids = [
        a["approval_id"] for a in records["approvals"]
        if a["request_id"] in related_ids and a["status"] in OPEN_APPROVAL_STATUSES
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
    )
    decision.answer = await write_answer(answer_model, request, decision)
    return {"decision": decision}


# THE GRAPH ------------------------------------------------------------------------

def after_lookup(state: WebexState) -> str:
    return "respond" if state.get("problem") else "search"


def build_graph(toolbox: Toolbox, retriever: EvidenceRetriever, answer_model=None):
    """lookup -> search -> diagnose -> decide -> respond, or lookup -> respond
    when the person or subscription wasn't found.

    `answer_model` words the answer (a fake in tests). Not called `writer`: LangGraph
    fills a node argument with that name itself."""
    graph = StateGraph(WebexState)
    graph.add_node("lookup", partial(lookup, toolbox=toolbox))
    graph.add_node("search", partial(search, retriever=retriever))
    graph.add_node("diagnose", diagnose)
    graph.add_node("decide", decide)
    graph.add_node("respond", partial(respond, answer_model=answer_model or get_model()))

    graph.add_edge(START, "lookup")
    graph.add_conditional_edges("lookup", after_lookup, {"search": "search", "respond": "respond"})
    graph.add_edge("search", "diagnose")
    graph.add_edge("diagnose", "decide")
    graph.add_edge("decide", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


async def run_webex(graph, request: WebexRequest) -> AccessDecision:
    """Run one request through the graph and return its AccessDecision."""
    final_state = await graph.ainvoke({"request": request})
    return final_state["decision"]
