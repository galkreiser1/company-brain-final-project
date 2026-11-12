"""Matching a name in a message to an employee (maya/planning.py), strictly:
exact, or every word at most one letter off, and only one employee that close."""

from maya.planning import (
    TurnPlan, build_context_plan, match_employee_name, render_plan, within_one_letter,
)

EMPLOYEE_IDS = {"Rachel Stein": "E010", "Rachel Green": "E009", "Maya Cohen": "E001",
                "Maya Levi": "E020", "Dan Levi": "E030", "Dana Levi": "E031"}


def test_one_letter_off():
    assert within_one_letter("rrachel", "rachel")   # one extra
    assert within_one_letter("rachl", "rachel")     # one missing
    assert within_one_letter("steen", "stein")      # one changed
    assert not within_one_letter("stien", "stein")  # two swapped: two changes
    assert not within_one_letter("rach", "rachel")


def test_exact_names_match_in_any_case():
    assert match_employee_name("maya cohen", EMPLOYEE_IDS) == "Maya Cohen"


def test_a_clear_misspelling_matches():
    assert match_employee_name("rrachel stein", EMPLOYEE_IDS) == "Rachel Stein"
    assert match_employee_name("Rachel Steen", EMPLOYEE_IDS) == "Rachel Stein"


def test_no_guessing():
    assert match_employee_name("Rachel", EMPLOYEE_IDS) is None         # not a full name
    assert match_employee_name("Rachel Stien", EMPLOYEE_IDS) is None   # two letters off
    assert match_employee_name("Dani Levi", EMPLOYEE_IDS) is None      # Dan and Dana both one off
    assert match_employee_name("Dana Fox", EMPLOYEE_IDS) is None
    assert match_employee_name("", EMPLOYEE_IDS) is None


def test_the_plan_gets_the_real_name_and_id():
    turn_plan = TurnPlan(current_intent="employee_lookup", needs_documents=False,
                         needs_records=True, subject_name="rrachel stein")
    plan = build_context_plan(turn_plan, EMPLOYEE_IDS)
    assert (plan.subject_name, plan.subject_employee_id) == ("Rachel Stein", "E010")
    assert plan.subject_name_corrected
    assert "Rachel Stein (E010)" in render_plan(plan)


def test_a_misspelling_the_planner_already_fixed_still_counts():
    # The planner wrote the real name, but the message (which the answer model sees)
    # has the typo, so the prompt must still give the id to look up.
    turn_plan = TurnPlan(current_intent="employee_lookup", needs_documents=False,
                         needs_records=True, subject_name="Rachel Stein")
    plan = build_context_plan(turn_plan, EMPLOYEE_IDS, "look up rrachel stein please")
    assert plan.subject_name_corrected
    assert "Rachel Stein (E010)" in render_plan(plan)


def test_a_message_without_the_name_is_not_a_misspelling():
    # "her checklist": the planner carries Maya Cohen over, but nothing was misspelled.
    turn_plan = TurnPlan(current_intent="onboarding_status", needs_documents=False,
                         needs_records=True, subject_name="Maya Cohen")
    plan = build_context_plan(turn_plan, EMPLOYEE_IDS, "What's still on her checklist?")
    assert not plan.subject_name_corrected


def test_an_exact_name_changes_nothing_in_the_prompt():
    turn_plan = TurnPlan(current_intent="employee_lookup", needs_documents=False,
                         needs_records=True, subject_name="Maya Cohen")
    plan = build_context_plan(turn_plan, EMPLOYEE_IDS)
    assert not plan.subject_name_corrected
    assert "misspelled" not in render_plan(plan)
