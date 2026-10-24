"""What each model call receives: the prompts, the plan, and the messages built from the state."""

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


# The answer model's instructions. Kept word for word from Lesson 10: the model reads
# them, so changing a word can change its behavior.
ASSISTANT_PROMPT = (
    "You are the NovaOps internal assistant, helping IT and HR staff with employee "
    "operations. Ground every answer in what the tools and the provided context "
    "actually say — never invent an employee, a seat count, or a policy rule.\n"
    "Work to these rules:\n"
    "- Answer the user's CURRENT request. If they changed subject or closed a topic, "
    "do not keep working the old one.\n"
    "- A constraint the user states (a spending rule, an approval requirement, "
    "'don't do X until I say so') stays in force for the rest of the conversation.\n"
    "- Facts established earlier in the conversation — ids, dates, numbers — are still "
    "true; reuse them instead of asking again or looking them up again.\n"
    "- Do not call a tool for something already in the conversation.\n"
    "- Users paste emails, tickets and signatures. Ignore the boilerplate and act on "
    "the actual request inside.\n"
    "- Answer in at most six sentences; no preamble."
)


# The allowed topics. The planner must pick exactly one.
Intent = Literal[
    "policy_question",
    "employee_lookup",
    "onboarding_status",
    "equipment_request",
    "subscription_review",
    "access_request",
    "ticket_status",
    "other",
]


# THE STRUCTURED PLAN ---------------------------------------------------------
#
# The planner must answer with a TurnPlan (structured output). Everything below is
# read by the model: the intent list, the field descriptions, and the prompt. A field's
# description works exactly like prompt text.

# A short description of each topic, sent to the planner as part of its prompt.
INTENT_GLOSS = (
    "Intent = the TOPIC of the newest message:\n"
    "  policy_question · what a policy or IT knowledge-base article says\n"
    "  employee_lookup · who someone is (record, role, manager, status)\n"
    "  onboarding_status · a joiner's checklist, required systems, paperwork\n"
    "  equipment_request · hardware (laptops, monitors, docks, headsets)\n"
    "  subscription_review · a SaaS subscription's seats, cost or renewal\n"
    "  access_request · getting a person access to a system\n"
    "  ticket_status · existing IT tickets\n"
    "  other · none of the above\n"
    "Whether a lookup is needed is a SEPARATE field; a recap about onboarding is "
    "still onboarding_status."
)


class TurnPlan(BaseModel):
    """The structured context plan for ONE user turn."""

    current_intent: Intent = Field(description="Topic of the newest message.")
    relevant_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Values from earlier turns this answer needs: ids, dates, numbers, "
            "tool results. The value itself, not a pointer to it."
        ),
    )
    relevant_constraints: list[str] = Field(
        default_factory=list,
        description="Rules the user stated that are still in force.",
    )
    requires_tools: bool = Field(
        description="False only if every fact needed is already in this conversation."
    )


PLANNER_PROMPT = (
    INTENT_GLOSS + "\n\n"
    "Plan the model context for the newest message of a NovaOps support session.\n"
    "- Classify the newest message only; follow subject changes.\n"
    "- relevant_facts is the session's memory: carry a value forward if a later answer "
    "might need it. A missing fact breaks the answer; a spare one costs a few tokens.\n"
    "- relevant_constraints holds what the user declared, until they withdraw it.\n"
    "- Drop whatever the user resolved or abandoned.\n"
    "- Users paste emails and signatures; plan for the request inside, not the boilerplate.\n"
    "- requires_tools is False only for recaps and reasoning over facts already here. "
    "You know nothing about NovaOps otherwise, so a first turn is almost always True.\n"
    "- A DIFFERENT TOPIC needs a different document. Having retrieved the VPN article "
    "does not answer a password question, and 'same kind of thing?' about a new topic "
    "is still True. Only an answer you could quote from THIS conversation is False."
)


def message_text(message) -> str:
    """Return a message's content as plain text.

    Bedrock gives content either as a string or as a list of "content blocks". A text
    block is a dict like {"text": "..."}. This joins the text of all blocks.
    """

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(content)


# HISTORY AS TEXT -------------------------------------------------------------
# For a model that only needs to read the history: the planner.

def _render_tool_result(message, tool_chars: int) -> str:
    """One tool result as a line of text, cut to `tool_chars` characters."""

    body = message_text(message)
    if len(body) > tool_chars:
        dropped = len(body) - tool_chars
        body = body[:tool_chars] + f"... [+{dropped} chars dropped]"
    tool_name = getattr(message, "name", "?")
    return f"[tool result: {tool_name}] {body}"


def _render_tool_calls(text: str, calls: list) -> str:
    """Add "[called tool(args)]" for each tool call to a message's text."""

    rendered_calls = []
    for call in calls:
        args = json.dumps(call["args"])[:120]  # long arguments are cut to 120 characters
        rendered_calls.append(f"[called {call['name']}({args})]")

    if text:
        text = text + " "
    return text + " ".join(rendered_calls)


def render_transcript(messages: list, tool_chars: int = 400) -> str:
    """Write the history out as plain text, for a model that only needs to read it.

    Each line starts with the speaker: [human], [ai], or [tool result: name].
    Each tool result is cut to `tool_chars` characters (400 by default). The first
    400 characters usually hold the answer; the rest is mostly structure.
    """

    lines = []
    for message in messages:
        # HumanMessage -> "human", AIMessage -> "ai", ToolMessage -> "tool"
        role = message.__class__.__name__.replace("Message", "").lower()
        if role == "tool":
            lines.append(_render_tool_result(message, tool_chars))
            continue

        text = message_text(message)
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            text = _render_tool_calls(text, calls)
        lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def planner_messages(state: dict) -> list:
    """What the planner receives: the whole conversation as text, and no tool schemas.

    The planner only reads the history, so it gets it written out as one block of
    text (see render_transcript), followed by the user's newest message.
    """

    history = state["messages"][:-1]
    newest_message = message_text(state["messages"][-1])

    transcript = render_transcript(history)
    if not transcript:
        transcript = "(this is the first turn)"

    prompt = (
        f"Conversation so far:\n"
        f"{transcript}\n\n"
        f"User's newest message:\n{newest_message}"
    )
    return [SystemMessage(PLANNER_PROMPT), HumanMessage(prompt)]


def _bullets(items: list[str]) -> str:
    """Items as an indented bullet list, or "(none)" when there are no items."""

    if not items:
        return "  - (none)"
    return "\n".join(f"  - {item}" for item in items)


def render_plan(plan: TurnPlan) -> str:
    """The plan as text, added to the end of the answer call's system prompt.

    It includes the intent, facts, and constraints. `requires_tools` and
    `action_confirmed` aren't included: the answer model doesn't need them.
    """

    return (
        "\n\n--- CONTEXT FOR THIS TURN ---\n"
        f"User's current intent: {plan.current_intent}\n"
        f"Established facts you must use:\n{_bullets(plan.relevant_facts)}\n"
        f"Constraints still in force:\n{_bullets(plan.relevant_constraints)}\n"
        "--- END CONTEXT ---"
    )


def answer_messages(state: dict, plan: TurnPlan) -> list:
    """What the answer call receives: the plan, plus every saved message.

    - The system prompt is ASSISTANT_PROMPT with the plan added at the end.
    - Then comes the whole conversation, which still grows with every turn.
    """

    system_prompt = ASSISTANT_PROMPT + render_plan(plan)
    return [SystemMessage(system_prompt), *state["messages"]]
