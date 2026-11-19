"""Which golden fields fail a turn and which only warn. (No model calls.)

    .venv/bin/python -m pytest -v tests/test_maya_checks.py

GOLDEN-DATASETS.json says some fields are binding (a real defect if missed) and some
illustrative (how the reference build did it). Only the illustrative ones may become
warnings; these tests pin that line, so a binding field can't quietly stop failing.
"""

from evals.maya_checks import check_sources, check_turn, is_warning

TURN = {
    "n": 1, "user": "q", "should_use_tools": True,
    "expected_intent": "access_request",              # illustrative
    "expected_tools": ["check_software_subscription"],  # illustrative
    "arg_checks": [{"tool": "check_software_subscription", "arg": "software", "equals": "Webex"}],
    "required_facts": ["Webex"],                       # binding
    "forbidden_facts": ["Datadog"],                    # binding
    "forbidden_tools": ["create_access_request"],      # binding
}
RESULT = {"plan": {"current_intent": "onboarding_status", "needs_documents": True,
                   "needs_records": True},
          "loadout": ["get_employee"], "evidence": []}


def checks_for(answer: str, tools: list[str]) -> dict[str, bool]:
    calls = [{"name": name, "args": {}} for name in tools]
    return dict(check_turn(TURN, RESULT, calls, answer))


def test_a_different_topic_label_and_tools_only_warn():
    checks = checks_for("Webex is over its seat limit.", ["get_employee"])
    warnings = [label for label in checks if is_warning(label)]
    assert len(warnings) == 3  # intent, expected tools, args
    assert all(checks.values())  # nothing fails the turn


def test_a_missing_required_fact_still_fails():
    assert not all(checks_for("Ask IT.", ["get_employee"]).values())


def test_a_forbidden_fact_still_fails():
    assert not all(checks_for("Webex is blocked; Datadog needs an owner.", ["get_employee"]).values())


def test_a_forbidden_tool_still_fails():
    assert not all(checks_for("Webex is blocked.", ["create_access_request"]).values())


def test_no_lookup_when_one_was_needed_still_fails():
    assert not all(checks_for("Webex is blocked.", []).values())


def test_a_forbidden_source_still_fails():
    turn = {"forbidden_sources": ["manager_playbook/promotion.md"]}
    [(label, passed)] = check_sources(turn, ["manager_playbook/promotion.md#2"])
    assert not passed and not is_warning(label)
