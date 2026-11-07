"""Typed records for the Webex agent: what it worked out, and what it returns.

As in maya/schemas.py, each rule lives in the type: a reply that breaks one
raises an error when it's built, instead of reading fine as prose.
"""

from typing import Literal

from pydantic import BaseModel, computed_field, model_validator

from maya.schemas import EvidenceCitation

# The path the decide step picks (docs/webex-design.md, section 3).
DecisionPath = Literal["answer_only", "reuse", "approve_then_file", "refuse"]


def make_request_key(employee_id: str, system_id: str) -> str:
    """The name of the Webex run for one person + one system, e.g. "webex-E010-S001".

    Asking again about the same person and system gives the same key, so it lands
    on the same saved run instead of starting a second one."""
    return f"webex-{employee_id}-{system_id}"


# WHAT THE DIAGNOSE STEP WORKS OUT ------------------------------------------------
# Filled by code from tool results only. The decide step reads it to pick a path.

class Diagnosis(BaseModel):
    """The facts that decide a request, e.g. Rachel (E010) and Webex (S001)."""

    employee_id: str              # e.g. "E010"
    system_id: str                # e.g. "S001"
    system_name: str              # e.g. "Webex"
    allowed_by_role: bool         # the system is in the person's systems_allowed_by_role
    caller_may_request: bool      # the caller may ask for this person (themself, HR, IT)
    subscription_id: str          # e.g. "SUB001"
    seats_in_use: int             # e.g. 42
    seat_limit: int               # e.g. 40
    existing_request_id: str | None = None  # e.g. "AR001"; reuse it, don't file again
    open_approval_ids: list[str] = []       # e.g. ["AP001", "AP002"]

    # Worked out from the seat numbers, never set by hand, so it can't disagree.
    @computed_field
    @property
    def over_seat_limit(self) -> bool:
        return self.seats_in_use >= self.seat_limit

    @property
    def seat_reason(self) -> str:
        """e.g. "42 of 40 seats in use (SUB001)"."""
        return f"{self.seats_in_use} of {self.seat_limit} seats in use ({self.subscription_id})"


# WHAT THE AGENT RETURNS -----------------------------------------------------------

class ObservedFact(BaseModel):
    """One thing the agent found, with where it found it."""

    fact: str                   # e.g. "Webex has 42 of 40 seats in use"
    citation: EvidenceCitation  # a record (SUB001) or a document chunk


class ActionTaken(BaseModel):
    """One write that really happened, copied from the tool's result."""

    tool: str        # e.g. "request_approval"
    record_id: str   # the row it wrote, e.g. "AP005"
    summary: str     # e.g. "Asked Yael Romano (E018) to approve Webex for E001"


# Where the workflow got to: exactly one per run.
Status = Literal["completed", "blocked", "pending", "needs_human", "failed"]

# What happened to the request. There is no "granted": the agent never grants access.
Outcome = Literal[
    "answered", "existing_request_reused", "approval_requested", "request_filed", "refused"
]

# These outcomes write nothing, so they must have no actions.
NO_WRITE_OUTCOMES = ("answered", "existing_request_reused", "refused")


class AccessDecision(BaseModel):
    """The Webex agent's answer. Code fills every field except `answer`."""

    request_key: str                   # e.g. "webex-E010-S001", same on every replay
    employee_id: str                   # e.g. "E010"
    system: str                        # e.g. "Webex"
    status: Status
    outcome: Outcome
    blocking_reason: str | None = None  # e.g. "42 of 40 seats in use (SUB001)"
    observed_facts: list[ObservedFact] = []
    actions_taken: list[ActionTaken] = []
    existing_records: list[str] = []   # e.g. ["T001", "AR001", "AP001", "AP002"]
    next_steps: list[str] = []         # e.g. ["IT Manager (E006) to approve AP001"]
    answer: str                        # the text the person reads; the model writes it

    @model_validator(mode="after")
    def status_outcome_and_actions_agree(self):
        # Blocked must say what blocks it, so someone can act on it.
        if self.status == "blocked" and not self.blocking_reason:
            raise ValueError("a blocked decision needs a blocking_reason")
        # Asking for approval is what pauses a run, and the only thing that does.
        if (self.outcome == "approval_requested") != (self.status == "pending"):
            raise ValueError("outcome approval_requested goes with status pending, and only it")
        # Actions come from real writes: a no-write outcome can't list any, and a
        # write outcome must list the write it made.
        if self.outcome in NO_WRITE_OUTCOMES and self.actions_taken:
            raise ValueError(f"outcome {self.outcome} writes nothing, so it can't list actions")
        if self.outcome not in NO_WRITE_OUTCOMES and not self.actions_taken:
            raise ValueError(f"outcome {self.outcome} must list the write it made")
        return self
