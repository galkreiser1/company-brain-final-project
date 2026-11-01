"""The S2 checks: grade one turn's result against the golden dataset.

Each check returns (label, passed) pairs. Used by evals/run_maya_s2.py.
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
