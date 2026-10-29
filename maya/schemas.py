"""Typed records shared across the agent."""

import hashlib
from typing import Literal

from pydantic import BaseModel, computed_field, model_validator


class CallerContext(BaseModel):
    """Who is asking. Set by code from the logged-in user, never from model output.

    The access rules trust these two fields, so a model must never be able to
    choose them (otherwise a prompt could say "I'm HR" and read anyone's file).
    """

    caller_id: str   # the employee id, e.g. "E001"
    user_group: str  # e.g. "UG_REGULAR", "UG_HR", "UG_IT"


class Evidence(BaseModel):
    """One retrieved chunk, as the agent receives it."""

    doc_id: str          # the file, e.g. "policies/expense_policy.md"
    chunk_id: str        # the chunk, e.g. "policies/expense_policy.md#2"
    collection: str
    section: str         # the chunk's "## " heading, or "" for the top of the file
    updated: str | None  # the document's date (YYYY-MM-DD), if it has one
    text: str
    score: float         # the reranker's relevance score, 0.0 to 1.0


# THE REPLY'S PARTS -------------------------------------------------------------
# Each rule lives in the type, so a bad reply raises an error when it's built,
# instead of reading fine as prose.

class EvidenceCitation(BaseModel):
    """Where one claim came from: a document chunk, or a database record."""

    # A doc_id ("policies/onboarding_policy.md"), or the read tool that returned a
    # record ("list_onboarding_tasks").
    source: str
    chunk_id: str | None = None   # for documents, e.g. "policies/onboarding_policy.md#2"
    record_id: str | None = None  # for records, e.g. "OT005"
    section: str | None = None    # e.g. "Day-one systems"

    @model_validator(mode="after")
    def one_kind_of_source(self):
        # A citation points at a chunk OR a record. Both or neither can't be checked.
        if (self.chunk_id is None) == (self.record_id is None):
            raise ValueError("set exactly one of chunk_id and record_id")
        return self


class ChecklistItem(BaseModel):
    """One line of an onboarding checklist, e.g. Maya Cohen's Webex licence."""

    name: str                                              # e.g. "Webex"
    kind: Literal["system", "equipment", "policy", "task"]
    status: Literal["complete", "pending", "proposed", "blocked", "unresolved"]
    reason: str | None = None  # e.g. "42 seats in use, contract limit 40"
    citations: list[EvidenceCitation] = []

    @model_validator(mode="after")
    def reason_and_citations(self):
        # Blocked and unresolved items must say why, so someone can act on them.
        if self.status in ("blocked", "unresolved") and not self.reason:
            raise ValueError(f"a {self.status} item needs a reason")
        # Any status except unresolved is a claim about the facts, so it needs a
        # source. Unresolved means "we couldn't find out", so it may have none.
        if self.status != "unresolved" and not self.citations:
            raise ValueError(f"a {self.status} item needs at least one citation")
        return self


class OnboardingChecklist(BaseModel):
    """Everything a new hire needs before day one, with each item's status."""

    employee_id: str         # e.g. "E001"
    name: str                # e.g. "Maya Cohen"
    role: str                # e.g. "Customer Success Manager"
    start_date: str          # e.g. "2026-08-01"
    location: str            # e.g. "Israel"
    items: list[ChecklistItem]

    # Worked out from the items, never set by hand, so it can't disagree with them.
    # @computed_field also includes it when the checklist is turned into JSON.
    @computed_field
    @property
    def status(self) -> Literal["blocked", "incomplete", "on_track"]:
        """'blocked' if any item is blocked, else 'incomplete' if any item is
        unresolved, else 'on_track'."""

        statuses = {item.status for item in self.items}
        if "blocked" in statuses:
            return "blocked"
        if "unresolved" in statuses:
            return "incomplete"
        return "on_track"

    @property
    def blocked_items(self) -> list[ChecklistItem]:
        """The items that are blocked, e.g. Webex (42 seats in use, limit 40)."""

        return [item for item in self.items if item.status == "blocked"]


class MayaReply(BaseModel):
    """What Maya returns each turn: the answer, its sources, and maybe a checklist."""

    answer: str  # the text the user reads
    citations: list[EvidenceCitation] = []
    checklist: OnboardingChecklist | None = None  # only for onboarding questions


# THE WEBEX HANDOFF -------------------------------------------------------------
# Not used until Milestone 4.

class WebexHandoff(BaseModel):
    """A blocked licence passed on to a person to sort out, e.g. Webex for E001."""

    employee_id: str        # e.g. "E001"
    software: str           # e.g. "Webex"
    caller: CallerContext   # who asked, e.g. E004 (Sara, HR)
    reason: str             # e.g. "42 seats in use, contract limit 40"
    thread_id: str          # the conversation it came from
    idempotency_key: str    # from make_idempotency_key(); see below


def make_idempotency_key(thread_id: str, employee_id: str, software: str) -> str:
    """A short key that's always the same for the same request.

    Gets: the conversation, the employee and the software.
    Returns: the first 16 hex characters of a sha256 of the three, joined.

    "Idempotent" means doing it twice has the same effect as once: if the same
    handoff is sent again (a retry, or the user asking twice), its key matches
    the first one, so the receiver can spot the repeat and ignore it.
    """

    joined = "|".join([thread_id, employee_id, software])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
