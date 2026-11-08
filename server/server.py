"""
NovaOps MCP server — policies + knowledge base + HR documents + database.

Copied from Lesson 10 (code/server/) so this project owns its tool server: Webex
needs tools Lesson 10 doesn't have. The only change so far is where it reads data:
data/novaops/ in this repo. It keeps port 9877, so Maya connects with no change
(stop Lesson 10's server first; both can't hold the port).

Lesson 10's tool list:

  files    list_policies · get_policy
  FAISS    search_knowledge_base · search_hr_documents          <- hr_docs is new
  SQLite   get_employee · check_software_subscription
           list_onboarding_tasks · check_asset_inventory
           list_employee_tickets                                <- three new reads
  SQLite   create_access_request                                <- the only WRITE

Ten tools. That number is deliberate: every tool's name, description, and JSON
schema is sent to the model on every call that has tools bound, so ten schemas is
a standing input-token bill you pay whether or not the turn needs a single one of
them. Lab 02 is about not paying it.

Added here for Webex (14 tools in all):
  SQLite   list_access_requests · list_approvals                <- reads
  SQLite   request_approval · record_approval_decision          <- WRITES
Maya loads only her five reads (maya/loadout.py), so none of these reach her.

Run from the project root:
  python -m server.server        # serves at http://127.0.0.1:9877/mcp
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Settings as _FastMCPSettings

from server import db, rag

POLICIES_DIR = Path(__file__).resolve().parent.parent / "data" / "novaops" / "documents" / "policies"

# Port 9877, not Lesson 9's 9876, so both lessons' servers can run side by side.
# FastMCP's settings model annotates `lifespan` with a forward reference to FastMCP
# itself, which pydantic-settings 2.15+ warns about on every construction. Resolving
# the reference once — which is exactly what the warning asks for — silences it without
# suppressing anything. Harmless to call on older versions.
_FastMCPSettings.model_rebuild()

mcp = FastMCP("novaops-assistant", host="127.0.0.1", port=9877)


@mcp.tool()
def list_policies() -> list[str]:
    """List the names of all available NovaOps company policies.

    Call this first to discover which policies exist before fetching one.
    """
    return sorted(p.stem for p in POLICIES_DIR.glob("*.md"))


@mcp.tool()
def get_policy(name: str) -> str:
    """Return the full text of one NovaOps policy.

    Args:
        name: A policy name as returned by list_policies, e.g. 'equipment_policy'.
    """
    path = POLICIES_DIR / f"{name}.md"
    if not path.is_file():
        raise ValueError(f"No policy named {name!r}. Call list_policies for valid names.")
    return path.read_text(encoding="utf-8")


@mcp.tool()
def search_knowledge_base(query: str) -> list[dict]:
    """Search the NovaOps IT knowledge base for articles relevant to a question.

    Use this for how-to and troubleshooting questions — password reset, VPN access,
    MFA, laptop provisioning, and the like. Returns the most relevant articles.

    Args:
        query: A natural-language description of the problem or question.
    """
    return rag.search("it_kb", query)


@mcp.tool()
def search_hr_documents(query: str) -> list[dict]:
    """Search employment documents and internal memos (offer letters, addenda, memos).

    Use this for what was agreed or announced about a specific person or spending
    rule — an offer letter's required systems and equipment, a remote-work addendum,
    or a memo that changed an approval threshold.

    Args:
        query: A natural-language description of what you need to find.
    """
    return rag.search("hr_docs", query)


@mcp.tool()
def get_employee(query: str) -> list[dict]:
    """Look up NovaOps employees by employee id, email, or (partial) name.

    Args:
        query: An employee id (e.g. 'E001'), an email, or a name to match.
    """
    return db.get_employee(query)


@mcp.tool()
def check_software_subscription(software: str) -> list[dict]:
    """Check a SaaS subscription's seat usage — is it at or over its seat limit?

    Use this when someone can't get access to a tool (e.g. 'Webex says I'm not
    licensed') to see whether seats are available or the subscription is full.
    Also returns the annual cost and renewal date.

    Args:
        software: The software or vendor name, e.g. 'Webex' or 'Salesforce'.
    """
    return db.check_software_subscription(software)


@mcp.tool()
def list_onboarding_tasks(employee_id: str) -> list[dict]:
    """List an employee's onboarding checklist tasks and their statuses.

    The checklist is the system of record for onboarding — use it to answer
    "what's left?" and to see which tasks are blocked.

    Args:
        employee_id: The employee id, e.g. 'E001'.
    """
    return db.list_onboarding_tasks(employee_id)


@mcp.tool()
def check_asset_inventory(asset_type: str = "", location: str = "") -> list[dict]:
    """Check hardware inventory — what equipment is on hand and where.

    Use this for equipment questions (laptops, monitors, docks). Both arguments are
    optional filters; omit them to list everything.

    Args:
        asset_type: e.g. 'Laptop' or 'Monitor'. Omit for all types.
        location: e.g. 'Israel'. Omit for all locations.
    """
    return db.check_asset_inventory(asset_type, location)


@mcp.tool()
def list_employee_tickets(employee_id: str, status: str = "") -> list[dict]:
    """List the IT tickets raised by or for one employee.

    Use this to see what has already been reported before opening anything new.

    Args:
        employee_id: The employee id, e.g. 'E010'.
        status: Optional status filter, e.g. 'open'. Omit for all statuses.
    """
    return db.list_employee_tickets(employee_id, status)


@mcp.tool()
def create_access_request(
    employee_id: str,
    software: str,
    business_justification: str,
    status: str = "pending_approval",
    approver_id: str | None = None,
    decision_reason: str | None = None,
) -> dict:
    """File an access request for an employee to a system.

    This WRITES to the database. Use it only after confirming who the employee is
    and that access is warranted. It does not grant access on its own.

    Args:
        employee_id: The employee id the access is for, e.g. 'E001'.
        software: The software or system to request, e.g. 'Webex'.
        business_justification: A short reason for the request.
        status: 'pending_approval' (default: waits for a decision), or, once an
            approval is recorded, 'blocked' (e.g. no free seat) or 'approved'.
        approver_id: Who approved it, if anyone, e.g. 'E018'.
        decision_reason: Why it has this status, e.g. '42 of 40 seats in use'.
    """
    return db.create_access_request(
        employee_id, software, business_justification,
        status=status, approver_id=approver_id, decision_reason=decision_reason,
    )


# --- Webex tools (new in this repo) ---------------------------------------------

@mcp.tool()
def list_access_requests(employee_id: str) -> list[dict]:
    """List one employee's access requests (any status), with the system name.

    Check this before filing anything: if a request for the same system already
    exists, reuse it instead of filing a new one.

    Args:
        employee_id: The employee id, e.g. 'E010'.
    """
    return db.list_access_requests(employee_id)


@mcp.tool()
def list_approvals(employee_id: str) -> list[dict]:
    """List every approval about one employee: who must approve, and the status.

    Status is 'needed' or 'pending' (waiting), 'approved' or 'rejected'.

    Args:
        employee_id: The employee id, e.g. 'E010'.
    """
    return db.list_approvals(employee_id)


@mcp.tool()
def request_approval(request_key: str, approver_id: str, reason: str) -> dict:
    """Ask a person to approve a new access request (status: needed).

    This WRITES to the database. Calling it again with the same request_key
    returns the existing approval instead of adding another.

    Args:
        request_key: The Webex run key, e.g. 'webex-E001-S001'.
        approver_id: Who must approve, e.g. the employee's manager 'E018'.
        reason: Why the approval is needed.
    """
    return db.request_approval(request_key, approver_id, reason)


@mcp.tool()
def record_approval_decision(
    approval_id: str, approver_id: str, decision: str, reason: str = ""
) -> dict:
    """Record the named approver's decision on an open approval.

    This WRITES to the database. Only the approver named on the row may decide,
    and a decided approval can't be changed.

    Args:
        approval_id: The approval to decide, e.g. 'AP005'.
        approver_id: Who is deciding; must match the approval's approver.
        decision: 'approved' or 'rejected'.
        reason: Optional note saved with the decision.
    """
    return db.record_approval_decision(approval_id, approver_id, decision, reason)


if __name__ == "__main__":
    # Build both retrieval indexes up front so the first question isn't slow. The
    # server calls Bedrock (Titan) here — capability, including model calls, lives
    # on the server side, not in the agent.
    print("Building retrieval indexes...")
    for corpus, count in rag.warm_index().items():
        print(f"  {corpus}: {count} documents")
    print("NovaOps MCP server → http://127.0.0.1:9877/mcp  (Ctrl-C to stop)")
    mcp.run(transport="streamable-http")
