"""The Maya checks: grade one turn's result against the golden dataset.

Each check returns (label, passed) pairs. Used by evals/run_maya.py, for every Maya
item: the two sessions and the three quick questions. A check whose expectation the
golden data doesn't give is skipped, not failed.
"""

from maya.loadout import DOCUMENT_TOOLS, MAYA_READ_TOOLS
from maya.ports import FakeWebexPort


# THE TURN'S ANSWER, PLAN AND TOOLS -------------------------------------------

def arg_check_passes(check: dict, tool_calls: list) -> bool:
    """True if some call to check["tool"] has an argument that matches the check."""

    for call in tool_calls:
        if call["name"] != check["tool"]:
            continue
        value = str(call["args"].get(check["arg"], ""))
        if "equals" in check and value == check["equals"]:
            return True
        if "contains" in check and check["contains"].lower() in value.lower():
            return True
    return False


def covered_by_evidence(evidence: list) -> bool:
    """True if the retrieve step found documents this turn.

    The dataset expects the MCP document tools, which Maya no longer loads: its own
    search returns those documents instead, so its evidence stands in for them.
    """

    return bool(evidence)


def check_turn(turn: dict, result: dict, tool_calls: list, answer: str) -> list[tuple[str, bool]]:
    """Grade one turn against the golden dataset. Returns (label, passed) pairs."""

    plan = result["plan"]
    called = {call["name"] for call in tool_calls}
    by_evidence = covered_by_evidence(result.get("evidence"))
    checks = []

    # "a|b" in the dataset means either intent is accepted; null means not checked.
    intent = plan["current_intent"]
    if turn.get("expected_intent"):
        accepted = turn["expected_intent"].split("|")
        label = f"intent {intent}" + ("" if intent in accepted else f" (expected {turn['expected_intent']})")
        checks.append((label, intent in accepted))

    wants_tools = turn["should_use_tools"]
    plan_wants = plan["needs_documents"] or plan["needs_records"]
    checks.append(("plan needs a lookup" if wants_tools else "plan needs no lookup",
                   plan_wants == wants_tools))
    # A lookup is a tool call or a search that found documents.
    looked_up = bool(called) or by_evidence
    checks.append(("lookup done" if wants_tools else "no lookup", looked_up == wants_tools))

    # Maya loads only its five read tools, so a loadout can never hold anything else.
    loadout = result["loadout"]
    checks.append(("only read tools loaded", set(loadout) <= set(MAYA_READ_TOOLS)))

    missing = [
        name for name in turn.get("expected_tools", [])
        if name not in called and not (by_evidence and name in DOCUMENT_TOOLS)
    ]
    label = f"expected tools (missing {', '.join(missing)})" if missing else "expected tools"
    checks.append((label, not missing))

    if turn.get("arg_checks"):
        passed = all(arg_check_passes(check, tool_calls) for check in turn["arg_checks"])
        checks.append(("args", passed))

    # Each fact may list alternatives, "a|b": any one of them in the answer is enough.
    facts = turn.get("required_facts") or []
    missing = [
        fact for fact in facts
        if not any(option.lower() in answer.lower() for option in fact.split("|"))
    ]
    label = f"facts {', '.join(facts)}" + (f" (missing {', '.join(missing)})" if missing else "")
    checks.append((label, not missing))

    # Facts that must NOT appear, e.g. a closed tangent's ticket on a later turn.
    leaked = [fact for fact in turn.get("forbidden_facts") or [] if fact.lower() in answer.lower()]
    if turn.get("forbidden_facts"):
        checks.append((f"no forbidden facts" + (f" (found {', '.join(leaked)})" if leaked else ""),
                       not leaked))

    forbidden = [name for name in turn.get("forbidden_tools", []) if name in called]
    label = f"forbidden tool called: {', '.join(forbidden)}" if forbidden else "no forbidden tool"
    checks.append((label, not forbidden))
    return checks


# THE SOURCES THE TURN READ -------------------------------------------------

def check_sources(turn: dict, source_ids: list[str]) -> list[tuple[str, bool]]:
    """Which documents the turn read, when the golden data says.

    - forbidden sources: none of them may be read at all. This is the manager-playbook
      boundary: a regular employee's search must not return those chunks, even if
      the answer would look right with them.
    - expected sources: at least one of them read (a source is a file name, e.g.
      "equipment_policy.md"; a chunk ID is "policies/equipment_policy.md#2").
    """

    checks = []
    forbidden = turn.get("forbidden_sources") or []
    read = [source for source in forbidden if any(source in sid for sid in source_ids)]
    if forbidden:
        checks.append(("no forbidden source" + (f" (read {', '.join(read)})" if read else ""),
                       not read))
    expected = turn.get("expected_sources") or []
    if expected:
        found = any(source in sid for source in expected for sid in source_ids)
        checks.append(("an expected source read" + ("" if found else f" (none of {', '.join(expected)})"),
                       found))
    return checks


# THE WEBEX HANDOFF ----------------------------------------------------------

def turn_handoffs(webex: FakeWebexPort, first: int, result: dict) -> list[dict]:
    """The handoffs sent this turn, each with the result the graph saved for it.

    The fake port records what was really sent, so a handoff the graph reused (not
    resent) isn't counted.
    """

    saved = result.get("handoffs") or {}
    return [
        {"handoff": call, "result": saved.get(call.idempotency_key, {})}
        for call in webex.calls[first:]
    ]


def check_handoff(turn: dict, handoffs: list[dict]) -> list[tuple[str, bool]]:
    """For a turn with expected_handoff: exactly `count` handoffs, each one matching.

    Every other turn must send none.
    """

    expected = turn.get("expected_handoff")
    if not expected:
        return [("no handoff", not handoffs)] if handoffs else []

    matching = [
        sent for sent in handoffs
        if sent["handoff"].employee_id == expected["employee_id"]
        and sent["handoff"].software == expected["software"]
        and sent["result"].get("workflow") == expected["workflow"]
        and sent["result"].get("status") == expected["status"]
    ]
    passed = len(handoffs) == expected["count"] and len(matching) == len(handoffs)
    label = f"handoff {len(matching)}/{expected['count']} {expected['software']} {expected['status']}"
    return [(label + ("" if passed else f" (sent {len(handoffs)})"), passed)]


# THE HOMEWORK'S RULES, ON EVERY TURN -----------------------------------------
# Code only, no model judges: each rule is a string or field test anyone can rerun.

TANGENT_CLOSED_FROM = 6  # turn 4 asks about Rachel Stein's ticket; turn 5 moves on
TICKETS_CLOSED_FROM = 5  # from turn 5, nothing should look up tickets again
TANGENT_WORDS = ["rachel", "t001"]  # her name, and her Webex ticket's ID
# Words that make "granted" honest: "not granted", "isn't granted", "until granted"...
NEGATIONS = ["not", "n't", "never", "until", "before"]


def checklist_text(checklist: dict | None) -> str:
    """Every item's name and reason, as one lowercase text to search."""

    items = (checklist or {}).get("items") or []
    return " ".join(f"{item['name']} {item.get('reason') or ''}" for item in items).lower()


def check_tangent(number: int, answer: str, checklist: dict | None,
                  tool_calls: list) -> list[tuple[str, bool]]:
    """Once the Rachel tangent is over, it must stay out of answers and lookups."""

    checks = []
    if number >= TANGENT_CLOSED_FROM:
        text = answer.lower() + " " + checklist_text(checklist)
        found = [word for word in TANGENT_WORDS if word in text]
        checks.append(("tangent closed" + (f" (found {', '.join(found)})" if found else ""),
                       not found))
    if number >= TICKETS_CLOSED_FROM:
        called = any(call["name"] == "list_employee_tickets" for call in tool_calls)
        checks.append(("no ticket lookup", not called))
    return checks


def citation_ids(citations: list[dict]) -> list[str]:
    return [citation.get("chunk_id") or citation.get("record_id") for citation in citations]


def check_citations(checklist: dict | None, sources_seen: list[str]) -> list[tuple[str, bool]]:
    """On a checklist turn: every item is cited, and only with IDs Maya really saw.

    A blocked item must also cite a database record (e.g. OT005, SUB001): a document
    can say what someone needs, but only the live data shows it's blocked right now.
    """

    if not checklist:
        return []
    items = checklist.get("items") or []
    seen = set(sources_seen)

    uncited = [item["name"] for item in items if not item.get("citations")]
    unseen = [cid for item in items for cid in citation_ids(item.get("citations") or [])
              if cid not in seen]
    weak_blockers = [
        item["name"] for item in items
        if item["status"] == "blocked"
        and not (item.get("reason")
                 and any(c.get("record_id") for c in item.get("citations") or []))
    ]
    return [
        ("items cited" + (f" (none for {', '.join(uncited)})" if uncited else ""), not uncited),
        ("citations seen" + (f" (unseen {', '.join(unseen)})" if unseen else ""), not unseen),
        ("blockers cite a record" + (f" (not {', '.join(weak_blockers)})" if weak_blockers else ""),
         not weak_blockers),
    ]


def says_granted(answer: str) -> bool:
    """True if the answer says "granted" without a negation just before it."""

    text = answer.lower()
    start = text.find("granted")
    while start != -1:
        before = text[max(0, start - 15):start]
        if not any(word in before for word in NEGATIONS):
            return True
        start = text.find("granted", start + 1)
    return False


def check_homework_rules(turn: dict, result: dict, tool_calls: list, answer: str,
                         tangent: bool = True) -> list[tuple[str, bool]]:
    """The homework's per-turn rules: tangent closed, citations real, never "granted".

    The tangent (Rachel's ticket, closed after turn 5) exists only in the onboarding
    session: other items pass tangent=False.
    """

    checklist = (result.get("reply") or {}).get("checklist")
    return ((check_tangent(turn["n"], answer, checklist, tool_calls) if tangent else [])
            + check_citations(checklist, result.get("sources_seen") or [])
            + [("never granted", not says_granted(answer))])
