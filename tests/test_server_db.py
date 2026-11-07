"""The tool server's Webex tools, run on a fresh copy of the seed database."""

import pytest

from server import db

KEY = "webex-E001-S001"  # Maya Cohen + Webex


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Point the server at a temporary database file built from the seed."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "novaops.db")
    db._conn.cache_clear()
    yield
    db._conn().close()
    db._conn.cache_clear()


def test_reads_find_rachels_seed_records():
    requests = db.list_access_requests("E010")
    assert [(r["request_id"], r["system_name"], r["status"]) for r in requests] == [
        ("AR001", "Webex", "blocked")
    ]
    approvals = db.list_approvals("E010")
    assert [(a["approval_id"], a["approver_id"]) for a in approvals] == [
        ("AP001", "E006"),
        ("AP002", "E011"),
    ]


def test_request_approval_writes_once():
    first = db.request_approval(KEY, "E018", "Manager approval for Webex.")
    second = db.request_approval(KEY, "E018", "Manager approval for Webex.")
    assert first["approval_id"] == second["approval_id"] == "AP005"
    assert first["status"] == "needed"
    assert [a["approval_id"] for a in db.list_approvals("E001")] == ["AP005"]


def test_only_the_named_approver_can_decide():
    approval_id = db.request_approval(KEY, "E018", "Manager approval.")["approval_id"]
    with pytest.raises(ValueError, match="must be decided by E018"):
        db.record_approval_decision(approval_id, "E006", "approved")

    decided = db.record_approval_decision(approval_id, "E018", "approved", "OK")
    assert decided["status"] == "approved"


def test_a_decision_cannot_be_changed():
    approval_id = db.request_approval(KEY, "E018", "Manager approval.")["approval_id"]
    db.record_approval_decision(approval_id, "E018", "rejected")
    with pytest.raises(ValueError, match="already rejected"):
        db.record_approval_decision(approval_id, "E018", "approved")


def test_unknown_decision_is_refused():
    approval_id = db.request_approval(KEY, "E018", "Manager approval.")["approval_id"]
    with pytest.raises(ValueError, match="decision must be"):
        db.record_approval_decision(approval_id, "E018", "granted")
