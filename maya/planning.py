"""What each model call receives: the prompts, the plan, and the messages built from the state."""

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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

# Maya's own rules, added after ASSISTANT_PROMPT (which stays word for word from
# Lesson 10). Each rule fixes a mistake seen in the smoke test.
IMPORTANT_POINTERS = (
    "\n\nImportant pointers:\n"
    # With documents about a person in the prompt, the model treated her as already
    # looked up and passed her name where a tool needs her id (which finds nothing).
    "- An employee id (like E001) is not the person's name. If a tool needs an "
    "employee_id you don't have, look it up with get_employee first."
)


# How many recent turns the answer call receives as real messages.
KEEP_VERBATIM_TURNS = 2


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


# The docstring is sent to the planner as the schema's description, so it is kept word
# for word from Lesson 10's stage 3 (whose TurnPlan is stage 2's with this docstring).
# New here (not in Lesson 10): requires_tools is split into needs_documents (drives the
# search) and needs_records (drives the tools and the rearm), and search_query and
# subject_name feed the retrieve and authorize steps.
class TurnPlan(BaseModel):
    """Unchanged from stage 02 — what THIS turn needs."""

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
    needs_documents: bool = Field(
        description=(
            "True if the answer needs company documents (policies, contracts, offer "
            "letters, IT articles, memos) that are not already in this conversation"
        )
    )
    needs_records: bool = Field(
        description=(
            "True if the answer needs database records (employee, onboarding tasks, "
            "tickets, assets, subscription seats) that are not already in this "
            "conversation"
        )
    )
    action_confirmed: bool = Field(
        default=False,
        description=(
            "True only if THIS message authorises creating or filing a record in a "
            "system ('go ahead', 'file it'). Writing, drafting, summarising or "
            "'write it up' is composing text, NOT authorisation to file anything."
        ),
    )
    search_query: str = Field(
        default="",
        description=(
            "A standalone search query for the company documents, with names and "
            "pronouns resolved from the conversation, e.g. 'Maya Cohen offer letter "
            "day-one systems'. Empty if no documents are needed."
        ),
    )
    subject_name: str = Field(
        default="",
        description=(
            "Full name of the employee this message is about, e.g. 'Maya Cohen'; "
            "empty if none"
        ),
    )


# For each topic, the document collections a good answer needs. If the first search
# finds nothing from one of them, the retrieve step searches once more, limited to the
# missing ones (see retrieve() in graph.py). An empty list: any collection will do.
# (It lives here, not in loadout.py, because loadout.py imports this file.)
REQUIRED_COLLECTIONS: dict[str, list[str]] = {
    "policy_question": ["policies"],
    "employee_lookup": ["employment"],
    "onboarding_status": ["employment", "policies"],
    "equipment_request": ["policies"],
    # Turn 9's USD 7,500 approval threshold is in the Q3 memo, not a contract.
    "subscription_review": ["contracts", "internal_memos"],
    "access_request": ["policies"],
    "ticket_status": ["it_kb"],
    "other": [],
}


class ContextPlan(TurnPlan):
    """The planner's TurnPlan, plus two fields that code fills in (never the model)."""

    subject_employee_id: str | None = None  # e.g. "E001" for subject_name "Maya Cohen"
    required_evidence: list[str] = []       # collections to search, e.g. ["employment", "policies"]


def build_context_plan(turn_plan: TurnPlan, employee_ids: dict[str, str]) -> ContextPlan:
    """Add the code-filled fields to the planner's plan.

    Gets: the planner's TurnPlan, and {full name: employee id} (load_employee_ids).
    Returns: the ContextPlan.
      - subject_employee_id: the id whose full name matches subject_name exactly,
        ignoring case. None if there's no name, or no employee has it.
      - required_evidence: the topic's collections if documents are needed, else [].
    """

    ids_by_lower_name = {name.lower(): emp_id for name, emp_id in employee_ids.items()}
    subject_id = ids_by_lower_name.get(turn_plan.subject_name.strip().lower())

    required = []
    if turn_plan.needs_documents:
        required = list(REQUIRED_COLLECTIONS.get(turn_plan.current_intent, []))

    return ContextPlan(**turn_plan.model_dump(), subject_employee_id=subject_id,
                       required_evidence=required)


PLANNER_PROMPT = (
    INTENT_GLOSS + "\n\n"
    "Plan the model context for the newest message of a NovaOps support session.\n"
    "- Classify the newest message only; follow subject changes.\n"
    "- relevant_facts is the session's memory: carry a value forward if a later answer "
    "might need it. A missing fact breaks the answer; a spare one costs a few tokens.\n"
    "- relevant_constraints holds what the user declared, until they withdraw it.\n"
    "- Drop whatever the user resolved or abandoned.\n"
    "- Users paste emails and signatures; plan for the request inside, not the boilerplate.\n"
    "- needs_documents and needs_records are both False only for recaps and reasoning "
    "over facts already here. You know nothing about NovaOps otherwise, so a first turn "
    "almost always needs at least one of them.\n"
    "- A DIFFERENT TOPIC needs a different document. Having retrieved the VPN article "
    "does not answer a password question, and 'same kind of thing?' about a new topic "
    "still needs documents. Only an answer you could quote from THIS conversation "
    "needs none.\n"
    "- action_confirmed only when this message says to go ahead; then the intent is "
    "access_request and needs_records is True."
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
# For models that only need to read the history: the planner and the distiller.

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


# THE ANSWER CALL'S WINDOW ----------------------------------------------------
#
# A "turn" starts with a user message (HumanMessage) and includes everything after it:
# the model's tool calls, the tool results, and the model's answer.

def recent_turns(messages: list, keep: int = 2) -> list:
    """Return the last `keep` turns of the conversation.

    It cuts only at user messages. That keeps every tool call together with its
    result. If a cut separated them, Bedrock would reject the whole request, because
    each tool call must be followed by its result.
    """

    turn_starts = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]
    if not turn_starts:
        return list(messages)

    if len(turn_starts) >= keep:
        start = turn_starts[-keep]
    else:
        start = turn_starts[0]  # fewer turns than `keep`: start at the first one
    return list(messages[start:])


def _is_tool_traffic(message) -> bool:
    """True for a tool result, or for a model message that asks for tools."""

    if isinstance(message, ToolMessage):
        return True
    tool_calls = getattr(message, "tool_calls", None) or []
    return bool(tool_calls)


def conversation_window(messages: list, keep: int = 2) -> list:
    """The last `keep` turns, with tool calls and results removed from older turns.

    - The current turn stays complete. The model may be in the middle of using tools,
      and it needs its own tool calls and their results to finish.
    - In older turns, the tool calls and results are noise by now, so they are
      dropped. Each tool call is dropped together with its result, so no tool call is
      left without its result (which Bedrock would reject).
    """

    window = recent_turns(messages, keep)
    current_turn = recent_turns(messages, 1)
    older_turns = window[: len(window) - len(current_turn)]

    kept = []
    for message in older_turns:
        if not _is_tool_traffic(message):
            kept.append(message)
    return kept + current_turn


# THE ANSWER CALL'S MESSAGES --------------------------------------------------

def _bullets(items: list[str]) -> str:
    """Items as an indented bullet list, or "(none)" when there are no items."""

    if not items:
        return "  - (none)"
    return "\n".join(f"  - {item}" for item in items)


def render_plan(plan: TurnPlan) -> str:
    """The plan as text, added to the end of the answer call's system prompt.

    It includes the intent, facts, and constraints. The lookup flags, the subject and
    `action_confirmed` aren't included: the answer model doesn't need them.
    """

    return (
        "\n\n--- CONTEXT FOR THIS TURN ---\n"
        f"User's current intent: {plan.current_intent}\n"
        f"Established facts you must use:\n{_bullets(plan.relevant_facts)}\n"
        f"Constraints still in force:\n{_bullets(plan.relevant_constraints)}\n"
        "--- END CONTEXT ---"
    )


def render_evidence(evidence: list[dict]) -> str:
    """The retrieved document chunks as a labelled block, for the answer call.

    Each chunk is labelled with its chunk_id (and section and date), so the answer can
    cite exactly where a claim came from. No chunks: no block at all.
    """

    if not evidence:
        return ""

    chunks = []
    for chunk in evidence:
        section = chunk["section"] or "(top of document)"
        updated = chunk["updated"] or "no date"
        chunks.append(f"[{chunk['chunk_id']}] section: {section} · updated: {updated}\n"
                      f"{chunk['text']}")
    return (
        "\n\n--- DOCUMENTS FOR THIS TURN ---\n"
        "Excerpts from company documents, retrieved for this message. After each claim "
        "you take from them, cite its chunk_id in square brackets, e.g. "
        "[policies/expense_policy.md#2].\n\n"
        + "\n\n".join(chunks)
        + "\n--- END DOCUMENTS ---"
    )


def answer_messages(state: dict, plan: TurnPlan) -> list:
    """What the answer call receives: the plan, the documents, and the last two turns.

    - The system prompt is ASSISTANT_PROMPT, then IMPORTANT_POINTERS, then the plan,
      then this turn's retrieved chunks (see render_evidence).
    - Then come the last KEEP_VERBATIM_TURNS turns (see conversation_window): the
      previous turn without its tool calls, and the current turn in full.
    """

    evidence = state.get("evidence") or []
    system_prompt = (
        ASSISTANT_PROMPT + IMPORTANT_POINTERS + render_plan(plan) + render_evidence(evidence)
    )
    recent = conversation_window(state["messages"], keep=KEEP_VERBATIM_TURNS)
    return [SystemMessage(system_prompt), *recent]
