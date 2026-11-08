"""The Webex agent: works out why someone can't use a system, and what to do.

Steps so far (docs/webex-design.md, section 3), each a graph node later:
  lookup   -> calls the read tools, in a fixed order
  diagnose -> turns their results into a Diagnosis
  decide   -> picks one path: answer_only, reuse, approve_then_file or refuse
All three are plain code: no model decides anything here.

Each step gets the state and returns only the keys it changes, as in maya/graph.py.
"""

import json
from typing import TypedDict

from maya.schemas import CallerContext
from webex.schemas import DecisionPath, Diagnosis, WebexRequest, make_request_key
from webex.tools import Toolbox

HR_GROUP = "UG_HR"
IT_GROUP = "UG_IT"
# A request in one of these states is over, so it can't be reused.
CLOSED_REQUEST_STATUSES = {"rejected", "denied", "cancelled", "revoked"}
OPEN_APPROVAL_STATUSES = {"needed", "pending"}


class WebexState(TypedDict, total=False):
    request: WebexRequest
    records: dict           # the read tools' results, by name
    problem: str | None     # why the run can't go on, e.g. "No employee E999."
    diagnosis: Diagnosis
    path: DecisionPath


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
