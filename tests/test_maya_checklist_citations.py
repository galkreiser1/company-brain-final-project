"""Two citation rules for checklist items. (No model calls.)

    .venv/bin/python -m pytest -v tests/test_maya_checklist_citations.py

1. An item that loses its citation when the model rewrites the checklist keeps the
   one the same item had last turn (if it's still real).
2. "Blocked" is a claim about the live state, so it needs a database record behind
   it; with only documents, the item becomes unresolved and says why.
"""

from maya.respond import NOT_CONFIRMED, checked_item, same_item

SEEN = {"user_message#2", "SUB001", "employment/maya_cohen_offer_letter.md#0"}
EMAIL = {"source": "user_message", "chunk_id": "user_message#2"}
OFFER = {"source": "employment/maya_cohen_offer_letter.md",
         "chunk_id": "employment/maya_cohen_offer_letter.md#0"}
RECORD = {"source": "check_software_subscription", "record_id": "SUB001"}


def ids(citations: list[dict]) -> list[str]:
    return [c.get("chunk_id") or c.get("record_id") for c in citations]
SAVED = {"items": [{"name": "Sign the equipment responsibility form", "status": "pending",
                    "citations": [EMAIL]}]}


# 1. A LOST CITATION IS CARRIED OVER ------------------------------------------

def test_an_item_that_lost_its_citation_keeps_last_turns():
    item = {"name": "Equipment responsibility form", "kind": "task", "status": "pending",
            "citations": []}
    checked = checked_item(item, SEEN, SAVED)
    assert checked["status"] == "pending"
    assert ids(checked["citations"]) == ["user_message#2"]


def test_last_turns_citation_must_still_be_real():
    item = {"name": "Equipment responsibility form", "status": "pending", "citations": []}
    checked = checked_item(item, {"SUB001"}, SAVED)  # the email isn't a seen source
    assert checked["status"] == "unresolved" and checked["citations"] == []


def test_an_item_with_its_own_citation_keeps_it():
    item = {"name": "Equipment responsibility form", "status": "pending", "citations": [OFFER]}
    assert ids(checked_item(item, SEEN, SAVED)["citations"]) == [OFFER["chunk_id"]]


def test_a_new_item_gets_nothing_carried_over():
    item = {"name": "Headset", "status": "pending", "citations": []}
    assert checked_item(item, SEEN, SAVED)["status"] == "unresolved"


def test_item_names_match_on_their_words():
    assert same_item("Equipment responsibility form", "Sign the equipment responsibility form")
    assert same_item("Webex", "Webex seats")
    assert not same_item("Headset", "Laptop")
    assert not same_item("", "Laptop")


# 2. BLOCKED NEEDS A RECORD -----------------------------------------------------

def test_blocked_with_a_record_stays_blocked():
    item = {"name": "Webex", "status": "blocked", "reason": "seat limit", "citations": [RECORD]}
    assert checked_item(item, SEEN)["status"] == "blocked"


def test_blocked_on_a_document_alone_becomes_unresolved():
    # The onboarding session, turn 2: "Webex blocked (seat limit)" citing only the
    # offer letter, which says Webex is required, not that it's blocked.
    item = {"name": "Webex", "status": "blocked", "reason": "seat limit", "citations": [OFFER]}
    checked = checked_item(item, SEEN)
    assert checked["status"] == "unresolved"
    assert checked["reason"] == f"seat limit; {NOT_CONFIRMED}"
    assert ids(checked["citations"]) == [OFFER["chunk_id"]]  # the document still backs that it's required


def test_blocked_can_borrow_last_turns_record():
    saved = {"items": [{"name": "Webex seats", "status": "blocked", "citations": [RECORD]}]}
    item = {"name": "Webex", "status": "blocked", "reason": "seat limit", "citations": []}
    checked = checked_item(item, SEEN, saved)
    assert checked["status"] == "blocked" and ids(checked["citations"]) == ["SUB001"]
