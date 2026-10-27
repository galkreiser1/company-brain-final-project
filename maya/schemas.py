"""Typed records shared across the agent."""

from pydantic import BaseModel


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
