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

# Only non-sensitive profile columns. A production version of this tool would also
# filter by the caller's user group (HR-only fields exist) — that guardrail is the
# subject of the homework, not this lab.
EMPLOYEE_FIELDS = (
    "employee_id, full_name, email, role, department_id, "
    "manager_id, location, employment_type, status"
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
        "SELECT sys.system_name, v.vendor_name, sub.seat_limit, sub.active_seats, "
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
        "       created_at, assigned_team "
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
) -> dict:
    """File a new access request for an employee to a system; status pending_approval.

    The lab's one WRITE tool. It resolves the software name to a system id, inserts
    a row into access_requests, and returns the created record. It deliberately does
    NOT decide the request — approval is a separate human step, which is exactly the
    gate Lesson 9 builds on top of this tool.
    """
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
        "VALUES (?, ?, ?, ?, ?, 'pending_approval', NULL, ?, NULL)",
        (request_id, employee_id, system["system_id"], access_level,
         business_justification, created_at),
    )
    conn.commit()
    return {
        "request_id": request_id,
        "employee_id": employee_id,
        "system_id": system["system_id"],
        "system_name": system["system_name"],
        "access_level": access_level,
        "status": "pending_approval",
        "created_at": created_at,
    }
