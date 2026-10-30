"""Which tools Maya loads, and which of them each turn may see: the loadout.

Maya loads only the five database read tools (MAYA_READ_TOOLS). Documents come from
the retrieve step, which checks who may see what; the MCP server's document tools
don't, so they're never loaded. Neither is the write tool.

Every loaded tool's schema (name, description, arguments) attached to a model call
costs input tokens, whether or not the turn uses it. So plain code, not a model, picks
a small set of the five from the plan's topic.
"""

from maya.planning import TurnPlan


# The only tools Maya loads: an allowlist (name what's allowed, block the rest), so a
# new tool on the server stays out until it's added here. Order is fixed.
MAYA_READ_TOOLS = (
    "get_employee",
    "list_onboarding_tasks",
    "check_asset_inventory",
    "check_software_subscription",
    "list_employee_tickets",
)

# Never loaded. Named here so tests can check that none of them gets through.
WRITE_TOOLS = {"create_access_request"}  # changes a real system
DOCUMENT_TOOLS = {  # return documents without checking who is asking
    "search_hr_documents", "search_knowledge_base", "list_policies", "get_policy",
}


def maya_tools(tools: list) -> list:
    """The tools Maya may load: those in MAYA_READ_TOOLS, in that order.

    Gets: every tool the MCP server offers.
    Returns: only the allowed ones. Anything else (the write tool, the document
    tools, a tool added to the server later) is dropped.
    """

    by_name = {tool.name: tool for tool in tools}
    return [by_name[name] for name in MAYA_READ_TOOLS if name in by_name]


# For each topic, the read tools whose schemas the model gets.
#   - Each list is a little wider than strictly needed. If the planner picks a nearby
#     topic by mistake, the right tool is often still in the list.
#   - "other" gets all five: when in doubt, show the read tools ("fail open").
#   - policy_question gets none: its answers come from the retrieve step's search.
LOADOUTS: dict[str, list[str]] = {
    "policy_question": [],
    "employee_lookup": ["get_employee"],
    "onboarding_status": [
        "get_employee", "list_onboarding_tasks", "check_asset_inventory",
        "check_software_subscription",
    ],
    "equipment_request": ["check_asset_inventory"],
    "subscription_review": ["check_software_subscription"],
    "access_request": ["get_employee", "check_software_subscription"],
    "ticket_status": ["get_employee", "list_employee_tickets"],
    "other": list(MAYA_READ_TOOLS),
}


def select_tools(plan: TurnPlan) -> list[str]:
    """Turn the plan into the list of tools this turn may see.

    Returns: [] if the turn needs no records, else the topic's LOADOUTS list (a copy,
    so LOADOUTS itself is never changed). action_confirmed adds nothing: there is no
    write tool to add (it's kept for Milestone 4's handoff).
    """

    if not plan.needs_records:
        return []
    return list(LOADOUTS.get(plan.current_intent, MAYA_READ_TOOLS))
