"""The respond step's code checks: which sources are real, and which citations stay.

The model proposes citations; code decides. A citation survives only if it points at
an ID this conversation has actually seen: a chunk the retrieve step returned, or a
record a tool returned. Pure functions: no model, no AWS.

The check runs on the model's raw reply (a plain dict), BEFORE the MayaReply rules
run. Otherwise one item that breaks a rule (say, a proposed item with no citation)
would make the whole reply fail, and the turn would lose every citation.
"""

import json
import re

from langchain_core.messages import ToolMessage

from maya.planning import message_text
from maya.schemas import MayaReply


# The reason given to a checklist item whose citations were all dropped.
NO_EVIDENCE = "no supporting evidence found"
# Added to the reason of a "blocked" item no record backs: it becomes unresolved.
NOT_CONFIRMED = "not confirmed by a record"


# 1. WHICH SOURCES ARE REAL ---------------------------------------------------

def ids_in(value) -> list[str]:
    """Every value of a key ending in "_id", anywhere inside a JSON value.

    Gets: parsed JSON (a dict, a list, or a plain value).
    Returns: the IDs, e.g. ["OT005", "E001"] from a task record. Nested dicts and
    lists are searched too.
    """

    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("_id") and isinstance(item, str):
                found.append(item)
            else:
                found.extend(ids_in(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(ids_in(item))
    return found


def json_values(text: str) -> list:
    """Every JSON value in a text, even several in a row like '{...}{...}'.

    A tool that returns a list of records sends each one as its own block, so the
    text can hold several JSON objects back to back. Text that isn't JSON is skipped.
    """

    decoder = json.JSONDecoder()
    values, position = [], 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break  # the rest isn't JSON (an error message, say): stop here
        values.append(value)
    return values


def tool_result_ids(messages: list) -> list[str]:
    """The record IDs in these messages' tool results (e.g. "OT005", "E001")."""

    found = []
    for message in messages:
        if isinstance(message, ToolMessage):
            for value in json_values(message_text(message)):
                found.extend(ids_in(value))
    return found


def add_sources(seen: list[str], new: list[str]) -> list[str]:
    """`seen` plus the new IDs, in order, each only once. It only ever grows."""

    return seen + [source for source in dict.fromkeys(new) if source not in seen]


# 2. WHICH CITATIONS STAY -----------------------------------------------------

def cleaned_citation(citation: dict, sources_seen: set[str]) -> dict | None:
    """The citation in its proper shape, or None if it points at nothing seen.

    The model gets the shape wrong in two ways, and both are fixed here:
      - a chunk split into source + number: chunk_id "2" -> "policies/x.md#2";
      - an ID in the wrong field: chunk_id "OT005" is really a record_id, and a
        source "policies/x.md#2" is really a chunk_id.
    Fixing the shape is safe: the ID must still be in sources_seen to survive.
    """

    source = citation.get("source") or ""
    for raw_id in (citation.get("chunk_id"), citation.get("record_id"), source):
        if not raw_id:
            continue
        if raw_id.isdigit():
            raw_id = f"{source}#{raw_id}"
        if raw_id in sources_seen:
            if "#" in raw_id:  # every chunk_id has a "#", no record ID does
                return {"source": raw_id.split("#")[0], "chunk_id": raw_id,
                        "section": citation.get("section")}
            return {"source": source, "record_id": raw_id}
    return None


def real_citations(citations: list[dict], sources_seen: set[str]) -> list[dict]:
    """Only the citations that point at a chunk or record this conversation has seen."""

    cleaned = [cleaned_citation(citation, sources_seen) for citation in citations or []]
    return [citation for citation in cleaned if citation is not None]


def cited_in_answer(answer: str, sources_seen: set[str]) -> list[dict]:
    """Citations for the [chunk_id]s the answer itself writes, like [policies/x.md#2].

    The answer model was asked to cite in square brackets, so those are claims it
    already tied to a chunk. Only chunk_ids this conversation has seen count.
    """

    found = re.findall(r"\[([^\[\]\s]+#\d+)\]", answer)
    return [
        {"source": chunk_id.split("#")[0], "chunk_id": chunk_id}
        for chunk_id in dict.fromkeys(found)
        if chunk_id in sources_seen
    ]


def merge_citations(first: list[dict], extra: list[dict]) -> list[dict]:
    """The first citations, then the extra ones not already there."""

    seen = {c.get("chunk_id") or c.get("record_id") for c in first}
    return first + [c for c in extra if (c.get("chunk_id") or c.get("record_id")) not in seen]


def name_words(name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (name or "").lower()))


def same_item(name: str, other: str) -> bool:
    """True if two checklist item names are the same item, worded a little differently:
    one's words all appear in the other's ("Equipment responsibility form" and "Sign
    the equipment responsibility form")."""

    words, other_words = name_words(name), name_words(other)
    return bool(words and other_words) and (words <= other_words or other_words <= words)


def earlier_citations(item: dict, saved_checklist: dict | None) -> list[dict]:
    """The citations the same item had in the saved checklist (last checklist turn)."""

    for saved in (saved_checklist or {}).get("items") or []:
        if same_item(item.get("name", ""), saved.get("name", "")):
            return saved.get("citations") or []
    return []


def checked_item(item: dict, sources_seen: set[str], saved_checklist: dict | None = None) -> dict:
    """The item with its fake citations dropped; unresolved if none are left.

    - An item that comes back with no real citation keeps the ones the same item had
      in the saved checklist, if those are still real. (Maya's own session: the
      equipment form, cited to the email pasted in turn 2, came back uncited in
      turn 3 when the model rewrote the checklist, and was marked unresolved.)
    - A blocked item with no record behind it becomes unresolved: a document can say
      what someone needs, but only the live data shows it's blocked right now.
    """

    citations = real_citations(item.get("citations"), sources_seen)
    if not citations:
        citations = real_citations(earlier_citations(item, saved_checklist), sources_seen)
    if not citations and item.get("status") != "unresolved":
        # Its claim (complete, blocked, ...) has nothing behind it.
        return {**item, "status": "unresolved", "reason": NO_EVIDENCE, "citations": []}
    if item.get("status") == "blocked" and not any(c.get("record_id") for c in citations):
        reason = f"{item['reason']}; {NOT_CONFIRMED}" if item.get("reason") else f"blocked, {NOT_CONFIRMED}"
        item = {**item, "status": "unresolved", "reason": reason}
    if item.get("status") in ("blocked", "unresolved") and not item.get("reason"):
        item = {**item, "reason": NO_EVIDENCE}  # these must say why; we can't invent it
    return {**item, "citations": citations}


def enforce_evidence(raw: dict, sources_seen: list[str], checklist_wanted: bool,
                     subject_employee_id: str | None,
                     saved_checklist: dict | None = None) -> MayaReply:
    """The model's raw reply, keeping only citations code can back up, as a MayaReply.

    Gets: the responder's raw reply (a dict), every ID seen so far, whether this turn
    wants a checklist (code decides that from the topic, not the model), and the
    turn's subject (the checklist's employee_id: code knows it, the model may not),
    and the checklist saved by an earlier turn, if any.
    Returns: a MayaReply where
      - citations not in sources_seen are dropped, from the answer and every item;
      - an item left with none keeps the same item's citations from the saved
        checklist, if they are real;
      - a blocked item no record backs becomes unresolved ("not confirmed by a record");
      - the answer's own [chunk_id]s are added to its citations;
      - an item left with no citation becomes unresolved, with reason "no supporting
        evidence found" (an item that was already unresolved keeps its reason);
      - the checklist is None if none was wanted.
    Raises: pydantic's ValidationError if the reply still breaks a rule (the caller
    then falls back to the plain answer).
    """

    seen = set(sources_seen)
    checklist = raw.get("checklist")
    if not checklist_wanted or not isinstance(checklist, dict):
        checklist = None
    else:
        items = [checked_item(item, seen, saved_checklist) for item in checklist.get("items") or []]
        employee_id = subject_employee_id or checklist.get("employee_id")
        checklist = {**checklist, "employee_id": employee_id, "items": items}

    answer = raw.get("answer", "")
    citations = merge_citations(real_citations(raw.get("citations"), seen),
                                cited_in_answer(answer, seen))
    return MayaReply.model_validate({
        "answer": answer,
        "citations": citations,
        "checklist": checklist,
    })
