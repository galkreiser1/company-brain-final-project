"""
The database tools' engine (server side).

A tiny SQLite layer over the NovaOps seed data. The MCP server calls these plain
functions; the agent never touches SQL. Six tools live here:
  - get_employee()                 read  — look up people
  - check_software_subscription()  read  — seat usage for a SaaS (the over-limit signal)
  - list_onboarding_tasks()        read  — an employee's onboarding checklist
  - check_asset_inventory()        read  — laptops/monitors on hand
  - list_employee_tickets()        read  — that person's IT tickets
  - create_access_request()        WRITE — file a request, status pending_approval

The last one is the only tool that changes anything, which is why this lesson only
exposes it at the workflow stage where it belongs.

Lesson 9 shipped the first two reads and the write; the three middle reads are new
here. They exist so the tool catalogue is big enough for tool *selection* to matter:
ten schemas on every call is a real, measurable cost.

The DB is a file, runs/novaops.db, built from data/novaops/database/{schema,seed}.sql
the first time it's needed. (Lesson 10 kept it in memory, so every restart wiped
it.) A file keeps writes across restarts, which Webex needs: a run paused for
approval must find its rows after the program stops, and a replay must be able to
check that it added none. `python -m server.reset_db` puts it back to the seed.
"""

import functools
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "data" / "novaops" / "database"
# runs/ is git-ignored: this file holds employee data.
DB_PATH = PROJECT_ROOT / "runs" / "novaops.db"

# The statuses create_access_request may file a request with.
FILED_STATUSES = ("pending_approval", "blocked", "approved")

# Only non-sensitive profile columns. A production version of this tool would also
# filter by the caller's user group (HR-only fields exist) — that guardrail is the
# subject of the homework, not this lab.
EMPLOYEE_FIELDS = (
    "employee_id, full_name, email, role, department_id, "
    "manager_id, location, employment_type, status, systems_allowed_by_role"
)


def reset_database() -> None:
    """Delete runs/novaops.db and rebuild it from the seed SQL.

    Stop the server first: it keeps its connection open, and would go on using
    the deleted file."""
    DB_PATH.parent.mkdir(exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript((DB_DIR / "schema.sql").read_text(encoding="utf-8"))
    conn.executescript((DB_DIR / "seed.sql").read_text(encoding="utf-8"))
    conn.commit()
    conn.close()


@functools.lru_cache(maxsize=1)
def _conn() -> sqlite3.Connection:
    """Open the database file once, building it from the seed if it's missing.

    check_same_thread=False because FastMCP runs sync tools in a worker thread,
    which may not be the thread that opened this connection. The agent calls
    tools one at a time, so no concurrent-write guarding is needed here — a real
    multi-user server would need it."""
    if not DB_PATH.exists():
        reset_database()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_employee(query: str) -> list[dict]:
    """Look up NovaOps employees by id, email, or (partial) name.

    Returns EVERY match — the seed data intentionally has duplicate first names
    (two Daniels, two Mayas, two Saras), so a name search can return more than one
    person, which the agent then has to disambiguate.
    """
    like = f"%{query}%"
    rows = _conn().execute(
        f"SELECT {EMPLOYEE_FIELDS} FROM employees "
        "WHERE employee_id = ? OR lower(email) = lower(?) OR lower(full_name) LIKE lower(?) "
        "ORDER BY full_name",
        (query, query, like),
    ).fetchall()
    return [dict(row) for row in rows]


def check_software_subscription(software: str) -> list[dict]:
    """Look up a SaaS subscription's seat usage by software or vendor name.

    Resolves the name against both the system catalog and the vendor list, so
    'Webex', 'webex', or a partial match all work. Returns seat_limit against the
    seats currently in use, plus a computed `over_limit` flag — the signal an
    access workflow reads to decide whether a new seat needs approval.

    `annual_cost` is included because approval thresholds are stated in money, not
    seats. A tool that returns only what its immediate caller needs quietly forces
    the model to guess at the rest, and it will guess confidently.
    """
    like = f"%{software}%"
    rows = _conn().execute(
        "SELECT sub.subscription_id, sub.system_id, sys.system_name, v.vendor_name, "
        "       sub.seat_limit, sub.active_seats, "
        "       sub.status, sub.renewal_date, sub.annual_cost "
        "FROM software_subscriptions sub "
        "JOIN systems sys ON sub.system_id = sys.system_id "
        "JOIN vendors  v  ON sub.vendor_id = v.vendor_id "
        "WHERE lower(sys.system_name) LIKE lower(?) OR lower(v.vendor_name) LIKE lower(?) "
        "ORDER BY sys.system_name",
        (like, like),
    ).fetchall()
    results = []
    for row in rows:
        sub = dict(row)
        sub["seats_available"] = sub["seat_limit"] - sub["active_seats"]
        sub["over_limit"] = sub["active_seats"] >= sub["seat_limit"]
        results.append(sub)
    return results


def list_onboarding_tasks(employee_id: str) -> list[dict]:
    """Return one employee's onboarding checklist, earliest due date first.

    The checklist is the system of record for onboarding: a task is done only when
    this table says so. Statuses that matter to the workflow are `blocked` (something
    is in the way) and `planned`/`pending` (still open).
    """
    rows = _conn().execute(
        "SELECT task_id, task_type, description, owner_group, status, due_date "
        "FROM onboarding_tasks WHERE employee_id = ? ORDER BY due_date, task_id",
        (employee_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def check_asset_inventory(asset_type: str = "", location: str = "") -> list[dict]:
    """Return hardware assets, optionally narrowed to a type and/or a location.

    Both filters are optional substring matches, so 'Laptop' / 'israel' / '' all work.
    `status` is the operative field: 'available' means it can be assigned now.
    """
    rows = _conn().execute(
        "SELECT asset_id, asset_type, model, status, location, employee_id, notes "
        "FROM assets "
        "WHERE lower(asset_type) LIKE lower(?) AND lower(location) LIKE lower(?) "
        "ORDER BY asset_type, asset_id",
        (f"%{asset_type}%", f"%{location}%"),
    ).fetchall()
    return [dict(row) for row in rows]


def list_employee_tickets(employee_id: str, status: str = "") -> list[dict]:
    """Return an employee's IT tickets, newest first; optionally filter by status.

    Used to answer 'what's already been reported?' — and, in this lesson, to show
    how a *resolved* earlier issue keeps polluting the model's context if nothing
    ever drops it.
    """
    rows = _conn().execute(
        "SELECT ticket_id, category, subcategory, priority, status, subject, "
        "       created_at, assigned_team, related_system_id "
        "FROM tickets WHERE employee_id = ? AND lower(status) LIKE lower(?) "
        "ORDER BY created_at DESC",
        (employee_id, f"%{status}%"),
    ).fetchall()
    return [dict(row) for row in rows]


def create_access_request(
    employee_id: str,
    software: str,
    business_justification: str,
    access_level: str = "standard_user",
    status: str = "pending_approval",
    approver_id: str | None = None,
    decision_reason: str | None = None,
) -> dict:
    """File a new access request for an employee to a system.

    It resolves the software name to a system id, inserts a row into
    access_requests, and returns the created record. By default (Lesson 10's
    behaviour) the row waits as pending_approval with no approver. Webex files only
    after a recorded approval, so it passes the status (blocked or approved), the
    approver and the reason, e.g. "42 of 40 seats in use (SUB001)".
    """
    if status not in FILED_STATUSES:
        raise ValueError(f"status must be one of {FILED_STATUSES}, got {status!r}.")
    conn = _conn()

    if conn.execute(
        "SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,)
    ).fetchone() is None:
        raise ValueError(f"No employee {employee_id!r}. Look them up with get_employee first.")

    system = conn.execute(
        "SELECT system_id, system_name FROM systems "
        "WHERE lower(system_name) LIKE lower(?) ORDER BY system_name LIMIT 1",
        (f"%{software}%",),
    ).fetchone()
    if system is None:
        raise ValueError(f"No system matching {software!r}.")

    # Human-readable ids (AR005, AR006, ...) continuing the seed's scheme, so demo
    # output reads cleanly — a real system would use a uuid or DB sequence.
    last = conn.execute(
        "SELECT request_id FROM access_requests ORDER BY request_id DESC LIMIT 1"
    ).fetchone()
    next_num = int(last["request_id"][2:]) + 1 if last else 1
    request_id = f"AR{next_num:03d}"

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    conn.execute(
        "INSERT INTO access_requests "
        "(request_id, employee_id, system_id, access_level, business_justification, "
        " status, approver_id, created_at, decision_reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (request_id, employee_id, system["system_id"], access_level,
         business_justification, status, approver_id, created_at, decision_reason),
    )
    conn.commit()
    return {
        "request_id": request_id,
        "employee_id": employee_id,
        "system_id": system["system_id"],
        "system_name": system["system_name"],
        "access_level": access_level,
        "status": status,
        "approver_id": approver_id,
        "decision_reason": decision_reason,
        "created_at": created_at,
    }


# --- Webex tools -------------------------------------------------------------
# An approval for a NEW request is saved before that request exists, so it can't
# point at an AR... id. It points at the Webex run key instead, e.g.
# "webex-E001-S001" (person + system), and gets its own request_type.
NEW_REQUEST_TYPE = "new_access_request"
OPEN_APPROVAL_STATUSES = ("needed", "pending")
DECISIONS = ("approved", "rejected")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def list_access_requests(employee_id: str) -> list[dict]:
    """Return one employee's access requests (any status), oldest first."""
    rows = _conn().execute(
        "SELECT ar.request_id, ar.employee_id, ar.system_id, sys.system_name, "
        "       ar.access_level, ar.status, ar.approver_id, ar.created_at, "
        "       ar.decision_reason "
        "FROM access_requests ar JOIN systems sys ON ar.system_id = sys.system_id "
        "WHERE ar.employee_id = ? ORDER BY ar.created_at, ar.request_id",
        (employee_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def list_approvals(employee_id: str) -> list[dict]:
    """Return every approval about one employee, oldest first.

    The approvals table has no employee column, so there are two ways to match:
    a seed approval points at one of the employee's access requests (AR001), and
    a new one points at a run key that contains the employee id (webex-E001-S001).
    """
    rows = _conn().execute(
        "SELECT approval_id, request_type, request_id, approver_id, status, "
        "       decision_reason, created_at "
        "FROM approvals "
        "WHERE request_id IN (SELECT request_id FROM access_requests WHERE employee_id = ?) "
        "   OR request_id LIKE ? "
        "ORDER BY created_at, approval_id",
        (employee_id, f"%-{employee_id}-%"),
    ).fetchall()
    return [dict(row) for row in rows]


def request_approval(request_key: str, approver_id: str, reason: str) -> dict:
    """WRITE: ask one person to approve a new access request (status 'needed').

    Safe to call twice: if this run key already has an approval that isn't
    rejected, that row is returned and nothing new is written. So a replay of
    the same run can't create a second approval.
    """
    conn = _conn()
    if conn.execute(
        "SELECT 1 FROM employees WHERE employee_id = ?", (approver_id,)
    ).fetchone() is None:
        raise ValueError(f"No employee {approver_id!r} to approve.")

    existing = conn.execute(
        "SELECT * FROM approvals WHERE request_id = ? AND status != 'rejected' "
        "ORDER BY created_at LIMIT 1",
        (request_key,),
    ).fetchone()
    if existing is not None:
        return dict(existing)

    # Same id scheme as the seed (AP001...). The seed already has AP001-AP004.
    last = conn.execute(
        "SELECT approval_id FROM approvals ORDER BY approval_id DESC LIMIT 1"
    ).fetchone()
    approval_id = f"AP{int(last['approval_id'][2:]) + 1:03d}"

    conn.execute(
        "INSERT INTO approvals (approval_id, request_type, request_id, approver_id, "
        " status, decision_reason, created_at) VALUES (?, ?, ?, ?, 'needed', ?, ?)",
        (approval_id, NEW_REQUEST_TYPE, request_key, approver_id, reason, _now()),
    )
    conn.commit()
    return dict(conn.execute(
        "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
    ).fetchone())


def record_approval_decision(
    approval_id: str, approver_id: str, decision: str, reason: str = ""
) -> dict:
    """WRITE: save a person's decision on an open approval.

    Only the approver named on the row may decide, and only once: a row that is
    already approved or rejected can't be changed.
    """
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}.")

    conn = _conn()
    row = conn.execute(
        "SELECT approver_id, status FROM approvals WHERE approval_id = ?", (approval_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"No approval {approval_id!r}.")
    if row["approver_id"] != approver_id:
        raise ValueError(f"{approval_id} must be decided by {row['approver_id']}, not {approver_id}.")
    if row["status"] not in OPEN_APPROVAL_STATUSES:
        raise ValueError(f"{approval_id} is already {row['status']}.")

    conn.execute(
        "UPDATE approvals SET status = ?, decision_reason = ? WHERE approval_id = ?",
        (decision, reason or None, approval_id),
    )
    conn.commit()
    return dict(conn.execute(
        "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
    ).fetchone())
