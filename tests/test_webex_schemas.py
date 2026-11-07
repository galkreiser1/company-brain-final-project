"""The Webex agent's types (webex/schemas.py): the rules a reply can't break."""

import pytest
from pydantic import ValidationError

from maya.schemas import EvidenceCitation
from webex.schemas import AccessDecision, ActionTaken, Diagnosis, make_request_key

SUB001 = EvidenceCitation(source="check_software_subscription", record_id="SUB001")


def rachel_diagnosis() -> Diagnosis:
    return Diagnosis(
        employee_id="E010", system_id="S001", system_name="Webex",
        allowed_by_role=True, caller_may_request=True,
        subscription_id="SUB001", seats_in_use=42, seat_limit=40,
        existing_request_id="AR001", open_approval_ids=["AP001", "AP002"],
    )


def decision(**changes) -> AccessDecision:
    """Rachel's expected decision (design section 4), with some fields changed."""
    fields = dict(
        request_key="webex-E010-S001", employee_id="E010", system="Webex",
        status="blocked", outcome="existing_request_reused",
        blocking_reason="42 of 40 seats in use (SUB001)",
        existing_records=["T001", "AR001", "AP001", "AP002"],
        answer="Your request AR001 is already on file.",
    )
    fields.update(changes)
    return AccessDecision(**fields)


APPROVAL = ActionTaken(tool="request_approval", record_id="AP005", summary="Asked E018")


def test_request_key_is_the_same_every_time():
    assert make_request_key("E010", "S001") == "webex-E010-S001"
    assert make_request_key("E010", "S001") == make_request_key("E010", "S001")


def test_diagnosis_works_out_the_seat_limit():
    diagnosis = rachel_diagnosis()
    assert diagnosis.over_seat_limit
    assert diagnosis.seat_reason == "42 of 40 seats in use (SUB001)"
    assert not diagnosis.model_copy(update={"seats_in_use": 39}).over_seat_limit


def test_rachels_decision_builds():
    rachel = decision(observed_facts=[{"fact": "42 of 40 Webex seats", "citation": SUB001}])
    assert rachel.actions_taken == []


def test_there_is_no_granted_outcome():
    with pytest.raises(ValidationError):
        decision(outcome="granted")


def test_blocked_needs_a_reason():
    with pytest.raises(ValidationError, match="blocking_reason"):
        decision(blocking_reason=None)


def test_approval_requested_goes_with_pending():
    decision(status="pending", outcome="approval_requested", actions_taken=[APPROVAL])
    with pytest.raises(ValidationError, match="approval_requested"):
        decision(status="completed", outcome="approval_requested", actions_taken=[APPROVAL])
    with pytest.raises(ValidationError, match="approval_requested"):
        decision(status="pending")


def test_a_reused_request_lists_no_actions():
    with pytest.raises(ValidationError, match="writes nothing"):
        decision(actions_taken=[APPROVAL])


def test_a_filed_request_lists_its_write():
    filed = ActionTaken(tool="create_access_request", record_id="AR005", summary="Filed")
    decision(outcome="request_filed", actions_taken=[filed])
    with pytest.raises(ValidationError, match="must list the write"):
        decision(outcome="request_filed")
