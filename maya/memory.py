"""The memory: short structured notes that stand in for old messages.

The planner reads the whole conversation on every turn, so its input keeps growing.
"Distillation" fixes that: once enough new messages pile up, a distiller model call
turns them into short notes (the memory). No message is ever deleted; the memory only
changes what the planner is sent.

Ordered by the graph step that uses each piece:
  check  - DISTIL_AFTER_* and needs_distillation: is it time to distil?
  distil - ConversationMemory, DISTIL_PROMPT, render_memory, distiller_messages and
           memory_is_safe: build the new memory, then check it before trusting it.
  plan   - PLANNER_PROMPT and planner_messages: the planner reads the memory plus only
           the newer messages.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from maya.planning import PLANNER_PROMPT as BASE_PLANNER_PROMPT
from maya.planning import _bullets, message_text, render_transcript


def estimate_tokens(text: str) -> int:
    """Rough token count for text we can't measure exactly: about 4 characters per token."""

    return len(text) // 4


# CHECK: IS IT TIME TO DISTIL? ------------------------------------------------

# Distil once the messages since the last distillation reach either limit.
DISTIL_AFTER_MESSAGES = 12
DISTIL_AFTER_TOKENS = 2500


def needs_distillation(state: dict) -> bool:
    """True when the "fresh" messages (after `distilled_upto`) reach 12 messages or
    about 2,500 tokens."""

    fresh = state["messages"][state.get("distilled_upto", 0):]
    if len(fresh) >= DISTIL_AFTER_MESSAGES:
        return True
    return estimate_tokens(render_transcript(fresh)) >= DISTIL_AFTER_TOKENS


# DISTIL: BUILD THE MEMORY, AND CHECK IT -------------------------------------
#
# The distiller must answer with a ConversationMemory (structured output). The field
# descriptions and DISTIL_PROMPT are read by the model, so they're kept word for word.

class ConversationMemory(BaseModel):
    """The memory: short notes that stand in for old messages.

    Five named fields instead of one summary string. One summary blurs things together:
    that's how "don't file anything until I say so" turns into "the user discussed
    filing". Named fields give each kind of information its own place, for the model
    to fill in and for you to check.
    """

    current_intent: str = Field(description="What the session is about NOW.")
    important_facts: list[str] = Field(
        default_factory=list,
        description="Durable values later turns may need: ids, names, dates, numbers.",
    )
    active_constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Rules the user imposed that are STILL in force. Never drop one unless "
            "the user withdrew it."
        ),
    )
    decisions: list[str] = Field(
        default_factory=list,
        description="Conclusions reached and actions taken, including things closed.",
    )
    unresolved_items: list[str] = Field(
        default_factory=list,
        description="Open questions and outstanding work. These must survive.",
    )


DISTIL_PROMPT = (
    "You maintain the durable memory of a long support conversation.\n"
    "You are given the memory so far and the conversation since it was last updated. "
    "Return the UPDATED memory.\n\n"
    "Rules:\n"
    "- Carry every still-relevant item forward. This memory replaces the raw messages, "
    "so anything you omit is gone.\n"
    "- Keep concrete values (E001, 2026-08-01, 42 of 40 seats), not descriptions of them.\n"
    "- NEVER silently drop an active constraint. Move it to decisions only if the user "
    "explicitly withdrew or satisfied it.\n"
    "- Move a topic the user closed into decisions, and drop its raw detail.\n"
    "- unresolved_items must keep anything still outstanding.\n"
    "- Be compact. This is a working memory, not minutes."
)


def render_memory(memory: dict) -> str:
    """The memory as readable text, one labelled section per field.

    Used by both the distiller (step 2) and the planner (step 3).
    """

    if not memory:
        return "(nothing distilled yet)"
    notes = ConversationMemory(**memory)

    return (
        f"Session is about: {notes.current_intent}\n"
        f"Established facts:\n{_bullets(notes.important_facts)}\n"
        f"Constraints in force:\n{_bullets(notes.active_constraints)}\n"
        f"Decisions made:\n{_bullets(notes.decisions)}\n"
        f"Still open:\n{_bullets(notes.unresolved_items)}"
    )


def distiller_messages(state: dict) -> list:
    """What the distiller receives: the old memory, plus every message since it.

    "Every message since it" includes the user's newest message, because distillation
    runs at the start of a turn, after that message arrives.
    """

    covered = state.get("distilled_upto", 0)
    fresh = state["messages"][covered:]
    old_memory = render_memory(state.get("memory") or {})

    prompt = (
        f"Memory so far:\n{old_memory}\n\n"
        f"Conversation since then:\n{render_transcript(fresh)}"
    )
    return [SystemMessage(DISTIL_PROMPT), HumanMessage(prompt)]


def memory_is_safe(old: dict, new: ConversationMemory) -> bool:
    """Check a new memory before using it. Returns False to reject it.

    Rejects it if:
      - it's empty (no facts, no decisions, no open items), or
      - it has fewer active constraints than the old memory (a sign it dropped one).
    A rejected memory loses nothing: the old memory and all raw messages stay.
    """

    if not (new.important_facts or new.decisions or new.unresolved_items):
        return False
    old_constraints = old.get("active_constraints", [])
    return len(new.active_constraints) >= len(old_constraints)


# PLAN: WHAT THE PLANNER RECEIVES ---------------------------------------------

# planning.py's planner prompt, plus two lines about reading the memory.
PLANNER_PROMPT = (
    BASE_PLANNER_PROMPT
    + "\n- You are given distilled memory plus the raw conversation since it was written. "
      "The memory's topic is the session's history, not this turn.\n"
    + "- Checklists, seat counts, tickets and inventory in memory are a SNAPSHOT; if the "
      "user asks what the state is NOW, needs_records is True."
)


def planner_messages(state: dict) -> list:
    """What the planner receives: the memory, the messages after it, and the newest one.

    Before the memory existed, the planner got the whole conversation as text. Here
    its input stays small however long the conversation gets.
    """

    covered = state.get("distilled_upto", 0)
    fresh = state["messages"][covered:-1]  # after the memory, minus the newest message
    memory_text = render_memory(state.get("memory") or {})
    newest_message = message_text(state["messages"][-1])

    transcript = render_transcript(fresh)
    if not transcript:
        transcript = "(nothing new)"

    prompt = (
        f"Session memory:\n{memory_text}\n\n"
        f"Conversation since then:\n{transcript}\n\n"
        f"User's newest message:\n{newest_message}"
    )
    return [SystemMessage(PLANNER_PROMPT), HumanMessage(prompt)]
