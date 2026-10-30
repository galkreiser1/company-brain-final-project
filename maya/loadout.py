"""Which tools each turn may see: the loadout.

Every tool's schema (name, description, arguments) attached to a model call costs
input tokens, whether or not the turn uses it. So plain code, not a model, picks a
small set of tools from the plan's topic. The write tool is kept apart and added only
when the user has explicitly said to go ahead.
"""

from maya.planning import TurnPlan


# For each topic, the read tools whose schemas the model gets.
#   - Each list is a little wider than strictly needed. If the planner picks a nearby
#     topic by mistake, the right tool is often still in the list.
#   - None means "no fixed list": select() in graph.py then gives the model every
#     tool. (Note: "every tool" there includes the write tool; see select().)
#   - No write tool may appear in this table.
LOADOUTS: dict[str, list[str] | None] = {
    "policy_question": [
        "list_policies", "get_policy", "search_knowledge_base", "search_hr_documents",
    ],
    "employee_lookup": ["get_employee", "search_hr_documents"],
    "onboarding_status": [
        "get_employee", "list_onboarding_tasks", "search_hr_documents",
        "check_asset_inventory", "list_policies", "get_policy",
        "check_software_subscription",
    ],
    "equipment_request": [
        "check_asset_inventory", "list_policies", "get_policy", "search_hr_documents",
    ],
    "subscription_review": [
        "check_software_subscription", "search_hr_documents", "get_policy",
    ],
    "access_request": [
        "get_employee", "check_software_subscription", "search_hr_documents",
        "list_policies", "get_policy",
    ],
    "ticket_status": ["get_employee", "list_employee_tickets"],
    "other": None,
}

# Tools that change a real system. They are never chosen by topic. They are added only
# when the user has approved the action, so when in doubt they stay hidden ("fail
# closed"). Read tools do the opposite: when in doubt, show them ("fail open").
WRITE_TOOLS = {"create_access_request"}


def select_tools(plan: TurnPlan) -> list[str] | None:
    """Turn the plan into the list of tools this turn may see.

    This is the only function in the policy that can add the write tool.

    Returns:
      - []   the turn needs no records (and no write was approved), so show none;
      - None the topic has no fixed list (only "other"); select() in graph.py then
             shows every tool;
      - a list of tool names otherwise. The write tool is added only for an
        access_request turn where the user approved the action.
    """

    if not plan.needs_records and not plan.action_confirmed:
        return []

    selected = LOADOUTS.get(plan.current_intent)
    if selected is None:
        return None

    selected = list(selected)  # a copy, so LOADOUTS itself is never changed
    if plan.current_intent == "access_request" and plan.action_confirmed:
        selected += sorted(WRITE_TOOLS)
    return selected
