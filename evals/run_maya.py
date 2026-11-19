"""The Maya runner: send every Maya evaluation item through the agent and check each turn.

The five Maya items of data/EVALUATION-INPUTS.yaml (20 turns, one trace each):
  - three quick questions, each its own one-turn conversation:
      personal laptop (M-I-01), MFA reset (M-I-02), Maya's equipment (M-I-03);
  - Maya's own session (M-S-02, golden S9): 5 turns, Maya Cohen asking for herself;
  - the onboarding session (M-S-01, golden S2): 12 turns, HR onboarding Maya Cohen.
    Its turn 10 files a Webex request, so it runs last and is then replayed once
    more: the replay must send nothing new.

The messages and callers come from data/EVALUATION-INPUTS.yaml, the graded inputs.
What each turn is expected to do comes from data/GOLDEN-DATASETS.json (ITEMS below
pairs them); the checks live in evals/maya_checks.py.

Needs this repo's MCP server running (python -m server.server). Webex questions and
the turn 10 handoff go through the real Webex workflow (webex/), which reads and writes
runs/novaops.db. Run from the repo root:
    .venv/bin/python -m evals.run_maya                      # every item
    .venv/bin/python -m evals.run_maya --only onboarding    # one group: quick, own, onboarding
    .venv/bin/python -m evals.run_maya --only onboarding --turns 3   # a quick run

Each session's turns run on one thread_id, so they form one conversation. Outputs:
  - the terminal: a short summary per turn (tools called, checks, cost);
  - runs/maya.log: everything in full, including exactly what each model call
    received. Rewritten on every run;
  - runs/maya_<YYYYmmdd-HHMMSS>.json: the run's record;
  - one Langfuse trace per turn, tagged with the item's id.
"""

import argparse
import asyncio
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langfuse import get_client
from langchain_mcp_adapters.client import MultiServerMCPClient

from maya.config import MCP_SERVER_URL, REPO_ROOT
import maya.memory
from maya.graph import REFUSAL, build_graph
from maya.loadout import MAYA_READ_TOOLS
from maya.memory import DISTIL_PROMPT, PLANNER_PROMPT, estimate_tokens, render_memory
from maya.planning import ASSISTANT_PROMPT, IMPORTANT_POINTERS, message_text, render_transcript
from maya.respond import tool_result_ids
from maya.retrieval import OpenSearchRetriever
import yaml

from evals.maya_checks import (check_handoff, check_homework_rules, check_sources, check_turn,
                               is_warning, turn_handoffs)
from evals.tracing import RUN_STAMP, finish_turn, require_langfuse, traced_turn, turn_status
from evals.webex_recording import RecordingWebexPort, real_webex_port


EVALUATION_INPUTS = REPO_ROOT / "data" / "EVALUATION-INPUTS.yaml"
GOLDEN_DATASETS = REPO_ROOT / "data" / "GOLDEN-DATASETS.json"
LOG_FILE = REPO_ROOT / "runs" / "maya.log"
RUNS_DIR = REPO_ROOT / "runs"  # not committed: the records hold employee data
ONBOARDING = "M-S-01"  # the onboarding session: the only one with a tangent and a handoff
HANDOFF_TURN = 10  # its turn that files the Webex request; replayed at the end

# Each Maya item: its id in EVALUATION-INPUTS.yaml, the golden entry with its
# expectations, a plain name, and the --only group it belongs to. In run order: the
# onboarding session writes to the database (turn 10), so it goes last.
ITEMS = [
    ("M-I-01", "SIMPLE_001", "quick question: personal laptop", "quick"),
    ("M-I-02", "SIMPLE_002", "quick question: MFA reset", "quick"),
    ("M-I-03", "SIMPLE_007", "quick question: Maya's equipment", "quick"),
    ("M-S-02", "S9-maya-own-account", "Maya's own session", "own"),
    (ONBOARDING, "S2-onboarding-maya", "onboarding session", "onboarding"),
]
GROUPS = ("quick", "own", "onboarding")

# The evaluation file names groups; the agent's config uses their IDs.
GROUP_IDS = {"Regular Employee": "UG_REGULAR", "HR Group": "UG_HR", "IT Group": "UG_IT"}

# Long prompts that calls repeat. The log shows ‹NAME› instead of the full text, so
# what changes from call to call stands out.
KNOWN_PROMPTS = {
    "ASSISTANT_PROMPT": ASSISTANT_PROMPT,
    "IMPORTANT_POINTERS": IMPORTANT_POINTERS.strip(),
    "PLANNER_PROMPT": PLANNER_PROMPT,
    "DISTIL_PROMPT": DISTIL_PROMPT,
}

# Structured output works through a fake "tool" with the schema's name: the model
# "calls" it to give its reply in that shape. These are those names, not real tools.
OUTPUT_FORMATS = {"TurnPlan", "ConversationMemory", "MayaReply"}


# In the log, old answers and tool results are cut to this many characters.
SHORT = 200

# How many tools Maya loads (of the MCP server's ten). No loadout can be bigger.
TOOL_COUNT = len(MAYA_READ_TOOLS)


# LOADING ---------------------------------------------------------------------

async def load_tools() -> list:
    """Connect to the MCP server and get its NovaOps tools as LangChain tools."""

    client = MultiServerMCPClient(
        {"novaops": {"url": MCP_SERVER_URL, "transport": "streamable_http"}}
    )
    return await client.get_tools()


def same_text(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def quick_question_turn(case: dict) -> dict:
    """A golden single-turn case in the shape of a session turn, so the same checks run.

    should_use_tools is True for all three: each needs documents (its expected_sources),
    and the retrieve step's search counts as a lookup. (The case's requires_tool_call
    means a database tool, which only Maya's equipment question needs.)
    """

    return {"n": 1, "user": case["input"], "should_use_tools": True,
            "expected_sources": case["expected_sources"]}


def load_items(groups: list[str]) -> list[dict]:
    """The Maya items to run: caller and messages from EVALUATION-INPUTS.yaml, each turn's
    expectations from GOLDEN-DATASETS.json. Stops if the two files' messages differ."""

    workflows = yaml.safe_load(EVALUATION_INPUTS.read_text())["workflows"]
    maya = next(w for w in workflows if w["id"] == "maya")
    inputs = {item["id"]: item for item in maya["inputs"] + maya["sessions"]}
    golden = json.loads(GOLDEN_DATASETS.read_text())
    expectations = {s["id"]: s["turns"] for s in golden["sessions"]}
    expectations |= {c["case_id"]: [quick_question_turn(c)] for c in golden["single_turn_cases"]}

    items = []
    for item_id, golden_id, plain, group in ITEMS:
        if group not in groups:
            continue
        given = inputs[item_id]
        messages = [t["message"] for t in given["turns"]] if "turns" in given else [given["message"]]
        turns = expectations[golden_id]
        if len(turns) != len(messages) or not all(
                same_text(t["user"], m) for t, m in zip(turns, messages)):
            raise SystemExit(f"{item_id}: its messages differ from golden {golden_id}")
        caller = given["caller"]
        items.append({
            "id": item_id, "plain": plain, "group": group, "turns": turns,
            "session": "turns" in given,
            # The conversation's ID. The checkpointer saves the state under it, so every
            # turn sent with the same thread_id continues the same conversation.
            "thread_id": golden_id,
            "caller_id": caller["employee_id"],
            "caller_group": GROUP_IDS[caller["group"]],
        })
    return items


# RECORDING EVERY MODEL CALL --------------------------------------------------

class ModelCallRecorder(BaseCallbackHandler):
    """Records every model call: which step made it, what it received, its reply, its cost.

    A callback is a hook LangChain runs around each model call. Passed in the graph's
    config, it reaches every model call inside the graph, so the agent itself needs no
    logging code.
    """

    run_inline = True  # record calls in the order they happen

    def __init__(self):
        self.calls = []
        self._open_calls = {}  # run_id -> call, until its reply arrives

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        tools = (kwargs.get("invocation_params") or {}).get("tools") or []
        call = {
            "step": (kwargs.get("metadata") or {}).get("langgraph_node", "?"),
            "messages": messages[0],
            "tools": [tool["function"]["name"] for tool in tools],
            "reply": None,
            "input_tokens": 0,
        }
        self.calls.append(call)
        self._open_calls[run_id] = call

    def on_llm_end(self, response, *, run_id, **kwargs):
        call = self._open_calls.pop(run_id)
        call["reply"] = response.generations[0][0].message
        call["input_tokens"] = (call["reply"].usage_metadata or {}).get("input_tokens", 0)

    def on_llm_error(self, error, *, run_id, **kwargs):
        # A failed call (e.g. Bedrock's malformed_tool_use) has no reply. Keep the error
        # so the log can show it; the graph's own fallback handles the turn.
        call = self._open_calls.pop(run_id, None)
        if call is not None:
            call["error"] = f"{type(error).__name__}: {error}"


# RECORDING EVERY SEARCH ------------------------------------------------------

class RecordingRetriever:
    """Wraps the real retriever and records each search's collections filter.

    It has the same search() method, so the graph can't tell the difference.
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self.searches = []  # one entry per search: its collections (None: all)

    def search(self, query, caller, k=4, collections=None):
        self.searches.append(collections)
        return self.retriever.search(query, caller, k=k, collections=collections)


# RUNNING ONE TURN ------------------------------------------------------------

def latest_turn_start(messages: list) -> int:
    """The index where the latest turn starts: its user message."""

    return max(
        index for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    )


def turn_route(denied: bool, asked_webex: bool, handoffs: list) -> str:
    """Who answered the turn: "deny", "webex" (a Webex question), "handoff" (a Webex
    go-ahead that was sent), or "maya"."""

    if denied:
        return "deny"
    if handoffs:
        return "handoff"
    return "webex" if asked_webex else "maya"


def describe_handoffs(handoffs: list[dict]) -> str:
    """This turn's handoffs on one line, like "E001 Webex pending (WX-13acdeaf)"."""

    return ", ".join(
        f"{sent['handoff'].employee_id} {sent['handoff'].software} "
        f"{sent['result'].get('status', '?')} ({sent['result'].get('request_id', '?')})"
        for sent in handoffs
    ) or "(none)"


async def run_turn(graph, config: dict, recorder: ModelCallRecorder, turn: dict,
                   retriever: RecordingRetriever, webex: RecordingWebexPort, item: dict) -> dict:
    """Send one user message, then collect what happened in that turn."""

    first_call = len(recorder.calls)
    first_search = len(retriever.searches)
    first_handoff = len(webex.calls)
    first_question = len(webex.questions)
    first_webex_tool = len(webex.tool_calls)
    # Where the memory ended before this turn, to tell if a distil was kept or rejected.
    before = (await graph.aget_state(config)).values.get("distilled_upto", 0)
    # Only the new message is sent. The checkpointer adds the earlier ones.
    result = await graph.ainvoke({"messages": [HumanMessage(turn["user"])]}, config=config)

    messages = result["messages"]
    turn_start = latest_turn_start(messages)
    tool_calls = [
        call
        for message in messages[turn_start:]
        for call in getattr(message, "tool_calls", None) or []
    ]
    answer = message_text(messages[-1])
    handoffs = turn_handoffs(webex, first_handoff, result)
    asked_webex = len(webex.questions) > first_question
    # Tools the Webex workflow called for this turn. On a Webex question they count as
    # the turn's lookups (turn 4's list_employee_tickets: Webex answers it now). On a
    # handoff they don't: Maya only passes the request on (turn 10 needs no lookup).
    webex_tool_calls = webex.tool_calls[first_webex_tool:]
    checked_calls = tool_calls + (webex_tool_calls if asked_webex else [])
    evidence = result.get("evidence") or []
    source_ids = [chunk["chunk_id"] for chunk in evidence] + tool_result_ids(messages[turn_start:])
    return {
        "turn": turn,
        "answer": answer,
        # What this turn looked at: the chunks it retrieved, the records its tools returned.
        "source_ids": source_ids,
        "plan": result["plan"],
        "denied": answer == REFUSAL,  # the authorize step sent the turn down the deny path
        "route": turn_route(answer == REFUSAL, asked_webex, handoffs),
        "loadout": result["loadout"],
        "tool_calls": tool_calls,
        "webex_tool_calls": webex_tool_calls,
        "handoffs": handoffs,
        "checks": (check_turn(turn, result, checked_calls, answer) + check_handoff(turn, handoffs)
                   + check_sources(turn, source_ids)
                   + check_homework_rules(turn, result, checked_calls, answer,
                                          tangent=item["id"] == ONBOARDING)
                   + [("finished without error", True)]),
        "error": None,
        "model_calls": recorder.calls[first_call:],
        "saved_messages": len(messages),
        "memory": result.get("memory") or {},
        "distilled_upto": result.get("distilled_upto", 0),
        "distilled_upto_before": before,
        "evidence": result.get("evidence") or [],
        "reply": result.get("reply") or {},  # the respond step's MayaReply, as a dict
        "searches": retriever.searches[first_search:],
        # The messages the memory doesn't cover yet: what the next check counts.
        "fresh": messages[result.get("distilled_upto", 0):],
    }


def trace_name(item: dict, number: int, suffix: str = "") -> str:
    """Like "M-S-02 turn 3 · Maya's own session", or "M-I-01 · quick question: ..."."""

    turn = f" turn {number}{suffix}" if item["session"] else suffix
    return f"{item['id']}{turn} · {item['plain']}"


def trace_tags(item: dict) -> list[str]:
    # The onboarding session keeps the S2 runner's old tags, so its earlier traces
    # and the new ones are found by the same filter.
    old = ["s2-runner", "S2"] if item["id"] == ONBOARDING else []
    return ["maya-runner", item["id"], item["group"]] + old


async def run_traced_turn(graph, config: dict, recorder: ModelCallRecorder, number: int,
                          turn: dict, retriever: RecordingRetriever, webex: RecordingWebexPort,
                          item: dict) -> dict:
    """run_turn_safely inside its own Langfuse trace, named after the item and turn."""

    first_result = len(webex.results)
    with traced_turn(
        trace_name(item, number), turn["user"], conversation=item["thread_id"],
        user_id=item["caller_id"], tags=trace_tags(item),
        metadata={"case_id": item["id"], "turn": str(number), "caller_group": item["caller_group"]},
    ) as root:
        outcome = await run_turn_safely(graph, config, recorder, turn, retriever, webex, item)
        refused = outcome["answer"] == REFUSAL
        results = webex.results[first_result:]
        outcome["trace_url"] = finish_turn(root, outcome["answer"], {
            "status": "error" if outcome["error"] else turn_status(refused, results),
            "caller_group": item["caller_group"],
            "route": outcome["route"],
            "intent": (outcome.get("plan") or {}).get("current_intent"),
            # The scope: the tools this turn could use, then the ones it called.
            "loadout": outcome.get("loadout") or [],
            "tool_sequence": [call["name"] for call in outcome["tool_calls"]
                              + outcome["webex_tool_calls"]],
            "request_ids": [result.request_id for result in results if result.request_id],
            "passed": turn_passed(outcome),
        })
    return outcome


async def run_turn_safely(graph, config: dict, recorder: ModelCallRecorder, turn: dict,
                          retriever: RecordingRetriever, webex: RecordingWebexPort,
                          item: dict) -> dict:
    """run_turn, but a crash becomes a failed turn instead of ending the run.

    A crash on turn 6 (say, GraphRecursionError) must not hide turns 7-12. The next
    turn runs on the same thread, from whatever state the checkpointer last saved.
    """

    first_call = len(recorder.calls)
    try:
        return await run_turn(graph, config, recorder, turn, retriever, webex, item)
    except Exception as error:
        return {
            "turn": turn,
            "error": f"{type(error).__name__}: {error}",
            "answer": "",
            "tool_calls": [],
            "webex_tool_calls": [],
            "route": "?",
            "handoffs": [],
            "checks": [("finished without error", False)],
            "model_calls": recorder.calls[first_call:],  # the calls made before the crash
        }


def describe_error_turn(number: int, outcome: dict) -> str:
    """A crashed turn, on two lines: for the terminal and the log."""

    return (f"Turn {number}  FAIL  {one_line(outcome['turn']['user'], 70, count_dropped=False)}\n"
            f"  ERROR  {outcome['error']}")


# FORMATTING ------------------------------------------------------------------

def one_line(text: str, limit: int, count_dropped: bool = True) -> str:
    """The text on one line, cut to `limit` characters.

    With count_dropped, the cut says how much was dropped, like "… (+53 chars)".
    """

    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    if not count_dropped:
        return text[:limit] + "…"
    return text[:limit] + f"… (+{len(text) - limit} chars)"


def describe_tool_call(call: dict) -> str:
    """A tool call as text, like get_employee(query='Maya Cohen')."""

    args = ", ".join(f"{name}={value!r}" for name, value in call["args"].items())
    return f"{call['name']}({args})"


def describe_checks(checks: list) -> str:
    """Each check with ✓ (passed), ✗ (failed) or ⚠ (a warning: an illustrative field
    the reference build did differently; see evals/maya_checks.py)."""

    def mark(label: str, passed: bool) -> str:
        return "⚠" if is_warning(label) else ("✓" if passed else "✗")

    return " · ".join(f"{label} {mark(label, passed)}" for label, passed in checks)


def count_warnings(records: list[dict]) -> int:
    return sum(is_warning(check["label"]) for record in records for check in record["checks"])


def was_rearmed(call: dict, calls: list) -> bool:
    """True for an answer that rearm deleted: written without tools, but not the last
    answer call (respond runs after the last one, so it isn't the last call overall).

    Only rearm makes the graph call the model again after a plain answer.
    """

    answer_calls = [c for c in calls if c["step"] == "model"]
    return (call["step"] == "model" and call["reply"] is not None
            and not call["reply"].tool_calls
            and call is not answer_calls[-1])


def describe_memory(outcome: dict) -> str:
    """One line on the memory: distilled this turn, or how close the next check is."""

    if any(call["step"] == "distil" for call in outcome["model_calls"]):
        if outcome["distilled_upto"] == outcome["distilled_upto_before"]:
            return "distil REJECTED by memory_is_safe ✗ · old memory and all messages kept"
        return f"distilled this turn ✓ · covers messages 1-{outcome['distilled_upto']}"

    fresh = outcome["fresh"]
    tokens = estimate_tokens(render_transcript(fresh))
    status = "saved" if outcome["memory"] else "none yet"
    # Read the limits from the module, so a demo that lowers them shows the new values.
    max_messages = maya.memory.DISTIL_AFTER_MESSAGES
    max_tokens = maya.memory.DISTIL_AFTER_TOKENS
    # The next turn's check also counts that turn's user message, hence the + 1.
    next_turn = " → next turn distils" if len(fresh) + 1 >= max_messages or tokens >= max_tokens else ""
    return (
        f"{status} · since last distil: {len(fresh)}/{max_messages} messages, "
        f"~{tokens:,}/{max_tokens:,} tokens{next_turn}"
    )


def describe_searches(searches: list) -> str:
    """How many searches retrieve ran, like "2 searches (2nd only: policies)"."""

    if not searches:
        return "no search"
    if len(searches) == 1:
        return "1 search"
    return f"2 searches (2nd only: {', '.join(searches[1])})"


def citation_id(citation: dict) -> str:
    """A citation's ID: its chunk_id for a document, its record_id for a record."""

    return citation.get("chunk_id") or citation.get("record_id")


def reply_citation_ids(reply: dict) -> set[str]:
    """Every ID the reply cites: the answer's and every checklist item's."""

    ids = {citation_id(c) for c in reply.get("citations", [])}
    for item in (reply.get("checklist") or {}).get("items", []):
        ids |= {citation_id(c) for c in item["citations"]}
    return ids


def describe_evidence(evidence: list[dict], reply: dict) -> str:
    """The retrieved chunks on one line, with how many the reply cites.

    Like "3 chunks, 1 cited: policies/x.md#2 (0.9) CITED, ...". The number is the
    reranker's score. Citations come from the checked MayaReply, so only real ones count.
    """

    if not evidence:
        return "(none)"
    cited_ids = reply_citation_ids(reply)
    chunks, cited = [], 0
    for chunk in evidence:
        is_cited = chunk["chunk_id"] in cited_ids
        cited += is_cited
        chunks.append(f"{chunk['chunk_id']} ({chunk['score']:.1f})" + (" CITED" if is_cited else ""))
    return f"{len(evidence)} chunks, {cited} cited: {', '.join(chunks)}"


def describe_citations(reply: dict) -> str:
    """The answer's citations on one line, like "2: OT005, policies/x.md#2"."""

    ids = [citation_id(c) for c in reply.get("citations", [])]
    return f"{len(ids)}" + (f": {', '.join(ids)}" if ids else "")


def describe_checklist(reply: dict) -> list[str]:
    """The checklist as lines: its overall status, then one line per item."""

    checklist = reply.get("checklist")
    if not checklist:
        return ["(none)"]
    lines = [f"{checklist['status']} · {len(checklist['items'])} items"]
    for item in checklist["items"]:
        reason = f" ({item['reason']})" if item.get("reason") else ""
        cites = ", ".join(citation_id(c) for c in item["citations"]) or "no citation"
        lines.append(f"{item['status']:<10} {item['name']}{reason} [{cites}]")
    return lines


def turn_passed(outcome: dict) -> bool:
    return all(passed for _, passed in outcome["checks"])


def describe_message(message) -> str:
    """One message as the log shows it: [role] then its content."""

    text = message_text(message)
    if message.type == "system":
        for name, prompt in KNOWN_PROMPTS.items():
            text = text.replace(prompt, f"‹{name}›")
        return f"[system] {text}"
    if message.type == "human":
        return f"[human] {text}"
    if message.type == "tool":
        return f"[tool result: {message.name}] {one_line(text, SHORT)}"

    calls = [describe_tool_call(call) for call in message.tool_calls]
    if calls:
        return f"[ai] called {', '.join(calls)}"
    return f"[ai] {one_line(text, SHORT)}"


def describe_fields(fields: dict) -> str:
    """A structured reply (like the plan) as one field per line; lists as bullets."""

    lines = []
    for name, value in fields.items():
        if isinstance(value, list):
            lines.append(f"{name}:")
            lines += [f"  - {item}" for item in value] or ["  (none)"]
        else:
            lines.append(f"{name}: {value}")
    return "\n".join(lines)


def describe_reply(reply) -> str:
    """What a model call returned, in full: a structured reply, tool calls, or an answer."""

    if reply is None:
        return "(no reply: the call failed)"
    if reply.tool_calls and reply.tool_calls[0]["name"] in OUTPUT_FORMATS:
        return describe_fields(reply.tool_calls[0]["args"])
    if reply.tool_calls:
        return "\n".join(f"called {describe_tool_call(call)}" for call in reply.tool_calls)
    return message_text(reply)


def indent(text: str, spaces: int) -> str:
    return textwrap.indent(text, " " * spaces)


# THE TWO OUTPUTS -------------------------------------------------------------

ROUTE_NOTES = {
    "webex": " (Maya passed the question to the Webex workflow; its answer is shown)",
    "handoff": " (a confirmed Webex request, sent to the Webex workflow)",
}


def print_turn(number: int, outcome: dict) -> None:
    """The short terminal summary of one turn."""

    calls = outcome["model_calls"]
    tools = ", ".join(describe_tool_call(call) for call in outcome["tool_calls"])
    input_tokens = sum(call["input_tokens"] for call in calls)
    answer_calls = [call for call in calls if call["step"] == "model"]
    # default=0: a denied turn has no answer call.
    most_tools = max((len(call["tools"]) for call in answer_calls), default=0)
    # Messages the answer model saw, not counting its system prompt.
    most_seen = max((len(call["messages"]) - 1 for call in answer_calls), default=0)
    reply = outcome["reply"]
    answer = reply.get("answer", "")

    # How many calls each step made, like "plan 1, model 2".
    steps = [call["step"] for call in calls]
    per_step = ", ".join(f"{step} {steps.count(step)}" for step in dict.fromkeys(steps))

    plan = outcome["plan"]
    needed = [name for name in ("documents", "records") if plan[f"needs_{name}"]]
    needs = "needs " + " + ".join(needed) if needed else "needs nothing new"

    print(f"\nTurn {number}  {'PASS' if turn_passed(outcome) else 'FAIL'}  "
          f"{one_line(outcome['turn']['user'], 70, count_dropped=False)}")
    print(f"  plan:    {plan['current_intent']} · {needs} · "
          f"subject: {plan['subject_name'] or '(none)'} → {plan['subject_employee_id']} "
          f"({'DENIED' if outcome['denied'] else 'allowed'}) · "
          f"facts: {len(plan['relevant_facts'])} · constraints: {len(plan['relevant_constraints'])}")
    loadout = outcome["loadout"]
    print(f"  route:   {outcome['route']}" + ROUTE_NOTES.get(outcome["route"], ""))
    print(f"  loadout: {', '.join(loadout) or '(no tools)'} ({len(loadout)} of {TOOL_COUNT})")
    print(f"  tools:   {tools or '(none)'}")
    if outcome["webex_tool_calls"]:
        webex_tools = ", ".join(describe_tool_call(call) for call in outcome["webex_tool_calls"])
        print(f"  webex tools: {webex_tools}")
    print(f"  evidence: {describe_searches(outcome['searches'])} · "
          f"{describe_evidence(outcome['evidence'], reply)}")
    print(f"  cites:   {describe_citations(reply)}")
    first, *items = describe_checklist(reply)
    print(f"  checklist: {first}")
    for line in items:
        print(f"    {line}")
    print(f"  checks:  {describe_checks(outcome['checks'])}")
    print(f"  cost:    {len(calls)} model calls ({per_step}) · {input_tokens:,} input tokens · "
          f"up to {most_tools} tools shown")
    print(f"  history: {outcome['saved_messages']} messages saved · "
          f"answer model saw up to {most_seen}")
    print(f"  memory:  {describe_memory(outcome)}")
    rearms = sum(was_rearmed(call, calls) for call in calls)
    print(f"  rearms:  {rearms}" + (" (an answer without a lookup was deleted and retried)" if rearms else ""))
    print(f"  handoff: {describe_handoffs(outcome['handoffs'])}")
    print(f"  answer:  {one_line(answer, 90, count_dropped=False)}")


LOG_INTRO = """\
How to read this log:
  - Each turn lists every model call in the order it ran.
  - "Sent" is exactly what that call received; "Reply" is what it returned, in full.
  - step: distil is the distiller (only on turns that distil); step: plan is the
    planner; step: model is the answer model. (select picks the tools and rearm
    resets them with plain code, so they make no model call.)
  - retrieve searches the documents. Its reranker calls Bedrock directly, not
    through LangChain, so those calls aren't listed or counted here.
  - The answer call's system prompt ends with "DOCUMENTS FOR THIS TURN": the chunks
    retrieve found. EVIDENCE at the end of a turn says how many searches ran (a
    2nd one, limited to the missing collections, runs when a collection the topic
    needs is missing), then lists the chunks (reranker score) and marks the ones
    the answer CITED.
  - ‹ASSISTANT_PROMPT›, ‹IMPORTANT_POINTERS›, ‹PLANNER_PROMPT› and ‹DISTIL_PROMPT›
    stand for those prompts' full text (see maya/planning.py and maya/memory.py).
  - step: respond is the last call (Nova Pro): it turns the answer into a MayaReply.
    Code then drops every citation that points at nothing this conversation saw.
    CITES lists the answer's checked citations; CHECKLIST shows each item's status
    and citations (onboarding_status turns only). CITED in EVIDENCE counts these.
  - HANDOFF lists the Webex requests this turn really sent (to a fake workflow).
    A confirmed Webex request goes plan -> handoff -> END, so it makes only the
    planner's model call.
  - MEMORY at the end of a turn shows the saved notes, when there are any.
  - Maya loads only 5 of the MCP server's 10 tools (the database reads). Where the
    dataset expects a document tool, retrieve's evidence stands in for it.
  - Old answers and tool results are cut to {short} characters.
"""


def log_header(tool_count: int, items: list[dict]) -> str:
    started = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"Maya runner log · {started} · {', '.join(item['id'] for item in items)}\n"
        f"{tool_count} tools from {MCP_SERVER_URL}\n\n"
        + LOG_INTRO.format(short=SHORT)
    )


def describe_tools_shown(names: list[str], tool_count: int) -> str:
    """Which tools a call could see: "all 10", "none", or their names."""

    formats = [name for name in names if name in OUTPUT_FORMATS]
    if formats:
        return f"none (replies in the {', '.join(formats)} format)"
    if len(names) == tool_count:
        return f"all {tool_count}"
    return ", ".join(names) or "none"


def log_turn(number: int, outcome: dict, tool_count: int, thread_id: str) -> str:
    """Everything that happened in one turn, for the log file."""

    rule = "═" * 78
    status = "PASS" if turn_passed(outcome) else "FAIL"
    lines = [f"\n{rule}", f"TURN {number}  ·  {status}", rule]
    lines += ["USER", indent(outcome["turn"]["user"], 2)]

    calls = outcome["model_calls"]
    for index, call in enumerate(calls, start=1):
        lines += [
            "",
            f"MODEL CALL {index} of {len(calls)}  ·  step: {call['step']}  ·  "
            f"{call['input_tokens']:,} input tokens",
            f"  Tools shown: {describe_tools_shown(call['tools'], tool_count)}",
            f"  Sent ({len(call['messages'])} messages):",
        ]
        for position, message in enumerate(call["messages"], start=1):
            # Continuation lines sit deeper than the number, so each message stands out.
            first_line, *more_lines = describe_message(message).split("\n")
            lines.append(f"    {position}. {first_line}")
            lines += [indent(line, 8) for line in more_lines]
        reply_label = "  Reply:"
        if was_rearmed(call, calls):
            reply_label = "  Reply (DELETED by rearm: no lookup happened; the user never saw it):"
        reply_text = describe_reply(call["reply"])
        if call.get("error"):
            reply_text += f"\n{call['error']}"
        lines += [reply_label, indent(reply_text, 4)]

    lines += [
        "",
        f"ROUTE    {outcome['route']}",
        "WEBEX TOOLS " + (", ".join(describe_tool_call(call) for call in outcome["webex_tool_calls"])
                          or "(none)"),
        f"CHECKS   {describe_checks(outcome['checks'])}",
        f"EVIDENCE {describe_searches(outcome['searches'])} · "
        f"{describe_evidence(outcome['evidence'], outcome['reply'])}",
        f"CITES    {describe_citations(outcome['reply'])}",
        "CHECKLIST " + "\n  ".join(describe_checklist(outcome["reply"])),
        f"STATE    {outcome['saved_messages']} messages saved in thread {thread_id}",
        f"HANDOFF  {describe_handoffs(outcome['handoffs'])}",
        f"MEMORY   {describe_memory(outcome)}",
    ]
    if outcome["memory"]:
        lines.append(indent(render_memory(outcome["memory"]), 2))
    return "\n".join(lines) + "\n"


# THE RUN'S RECORD ------------------------------------------------------------

def turn_record(number: int, outcome: dict) -> dict:
    """One turn as plain JSON: what was asked, what Maya did, and how it was graded."""

    turn = outcome["turn"]
    calls = outcome["model_calls"]
    return {
        "turn": number,
        "user": turn["user"],
        "kind": turn.get("kind"),
        "plan": outcome.get("plan"),
        "answer": outcome["answer"],
        "reply": outcome.get("reply"),
        "checklist": (outcome.get("reply") or {}).get("checklist"),
        "source_ids": outcome.get("source_ids", []),
        "route": outcome.get("route"),
        "tool_calls": [{"name": call["name"], "args": call["args"]} for call in outcome["tool_calls"]],
        "webex_tool_calls": outcome.get("webex_tool_calls", []),
        "handoffs": [{"handoff": sent["handoff"].model_dump(), "result": sent["result"]}
                     for sent in outcome["handoffs"]],
        "error": outcome["error"],
        "checks": [{"label": label, "passed": passed} for label, passed in outcome["checks"]],
        "model_calls": len(calls),
        "input_tokens": sum(call["input_tokens"] for call in calls),
    }


def failure_lines(item_id: str, records: list[dict]) -> list[str]:
    """One line per failed turn, naming its failed checks, e.g.
    "M-S-01 turn 6: expected tools (missing get_policy)"."""

    lines = []
    for record in records:
        failed = [check["label"] for check in record["checks"] if not check["passed"]]
        if failed:
            lines.append(f"{item_id} turn {record['turn']}: {'; '.join(failed)}")
    return lines


def save_run(runs: list[dict]) -> Path:
    """Write the whole run to runs/maya_<YYYYmmdd-HHMMSS>.json. Returns its path."""

    path = RUNS_DIR / f"maya_{datetime.now():%Y%m%d-%H%M%S}.json"
    turns = [record for run in runs for record in run["turns"]]
    record = {
        "items": runs,
        "totals": {
            "turns": len(turns),
            "turns_passed": sum(all(c["passed"] for c in r["checks"]) for r in turns),
            "warnings": count_warnings(turns),
            "model_calls": sum(r["model_calls"] for r in turns),
            "input_tokens": sum(r["input_tokens"] for r in turns),
        },
    }
    path.write_text(json.dumps(record, indent=2, default=str))  # default=str: e.g. a date
    return path


# MAIN ------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Maya evaluation items.")
    parser.add_argument("--only", choices=GROUPS, action="append",
                        help="run only this group (repeatable): quick, own, onboarding")
    parser.add_argument("--turns", type=int, metavar="N",
                        help="run only each session's first N turns (a quick run)")
    return parser.parse_args()


async def replay_handoff_turn(graph, config: dict, turn: dict, webex: RecordingWebexPort,
                              handoffs_before: int) -> bool:
    """Send the handoff turn's message once more, on the same thread. Returns passed.

    The checkpointer already holds the first send's key, so nothing new may reach the
    workflow: this session must still have sent exactly one handoff, and the reply says
    pending.
    """

    try:
        result = await graph.ainvoke({"messages": [HumanMessage(turn["user"])]}, config=config)
    except Exception as error:
        print(f"replay turn {HANDOFF_TURN}: FAIL · {type(error).__name__}: {error}")
        return False
    answer = (result.get("reply") or {}).get("answer", "")
    sent = len(webex.calls) - handoffs_before
    passed = sent == 1 and "pending" in answer.lower()
    print(f"replay turn {HANDOFF_TURN}: {sent} handoff call(s) in this session · "
          f"{'PASS' if passed else 'FAIL'} · {one_line(answer, 90, count_dropped=False)}")
    return passed


async def main(groups: list[str], turn_limit: int | None) -> int:
    """Run the items, save the record, and return the exit code: 1 if anything failed."""

    require_langfuse()
    tools = await load_tools()
    retriever = RecordingRetriever(OpenSearchRetriever())
    items = load_items(groups)
    # The real Webex workflow, recorded. Its own searches use the plain retriever, so
    # `retriever.searches` stays Maya's only.
    async with real_webex_port(tools, OpenSearchRetriever()) as webex:
        graph = build_graph(tools, retriever, webex)
        recorder = ModelCallRecorder()
        print(f"Maya runner: {len(items)} items · {len(tools)} tools from MCP, Maya loads "
              f"{TOOL_COUNT} · Langfuse run {RUN_STAMP}")
        LOG_FILE.parent.mkdir(exist_ok=True)
        runs, failures = [], []
        with LOG_FILE.open("w") as log:
            log.write(log_header(len(tools), items))
            for item in items:
                run = await run_item(graph, recorder, item, retriever, webex, log, turn_limit,
                                     len(tools))
                runs.append(run)
                failures += failure_lines(item["id"], run["turns"])
                if run.get("replay_passed") is False:
                    failures.append(f"{item['id']} replay turn {HANDOFF_TURN}: "
                                    "sent a second handoff, or didn't say pending")

    turns = [record for run in runs for record in run["turns"]]
    passed = sum(all(c["passed"] for c in r["checks"]) for r in turns)
    input_tokens = sum(call["input_tokens"] for call in recorder.calls)
    print(f"\nTOTAL  {passed}/{len(turns)} turns pass · {count_warnings(turns)} warnings "
          f"(illustrative fields only, see evals/maya_checks.py) · "
          f"{len(recorder.calls)} model calls · {input_tokens:,} input tokens")
    print(f"Full log: {LOG_FILE.relative_to(REPO_ROOT)}")
    print(f"Run record: {save_run(runs).relative_to(REPO_ROOT)}")
    if failures:
        print("\nFAILED\n  " + "\n  ".join(failures))
    return 1 if failures else 0


async def run_item(graph, recorder: ModelCallRecorder, item: dict, retriever: RecordingRetriever,
                   webex: RecordingWebexPort, log, turn_limit: int | None,
                   tool_count: int) -> dict:
    """One item: its turns in order on one thread, then (onboarding only) the replay."""

    config = {
        # The caller is set here, by code: the retrieve step searches as this person.
        "configurable": {
            "thread_id": item["thread_id"],
            "caller_id": item["caller_id"],
            "user_group": item["caller_group"],
        },
        "callbacks": [recorder],
    }
    turns = item["turns"][:turn_limit] if item["session"] else item["turns"]
    handoffs_before = len(webex.calls)
    print(f"\n{'━' * 78}\n{item['id']} · {item['plain']} · {len(turns)} turn(s) · "
          f"caller {item['caller_id']} ({item['caller_group']})")
    log.write(f"\n{'█' * 78}\n{item['id']} · {item['plain']} · thread {item['thread_id']}\n")

    records = []
    for number, turn in enumerate(turns, start=1):
        outcome = await run_traced_turn(graph, config, recorder, number, turn, retriever,
                                        webex, item)
        if outcome["error"]:
            print(describe_error_turn(number, outcome))
            log.write("\n" + describe_error_turn(number, outcome) + "\n")
        else:
            print_turn(number, outcome)
            log.write(log_turn(number, outcome, tool_count, item["thread_id"]))
        print(f"  trace: {outcome['trace_url'] or '(none)'}")
        log.write(f"TRACE {outcome['trace_url'] or '(none)'}\n")
        records.append(turn_record(number, outcome) | {"trace_url": outcome["trace_url"]})

    run = {"id": item["id"], "plain": item["plain"], "thread_id": item["thread_id"],
           "turns": records}
    if item["id"] == ONBOARDING and len(turns) >= HANDOFF_TURN:
        replay = turns[HANDOFF_TURN - 1]
        with traced_turn(
            trace_name(item, HANDOFF_TURN, " (again)"), replay["user"],
            conversation=item["thread_id"], user_id=item["caller_id"], tags=trace_tags(item),
            metadata={"case_id": item["id"], "turn": f"{HANDOFF_TURN}-replay"},
        ) as root:
            run["replay_passed"] = await replay_handoff_turn(graph, config, replay, webex,
                                                             handoffs_before)
            print(f"  trace: {finish_turn(root, '', {'passed': run['replay_passed']})}")
    return run


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args.only or list(GROUPS), args.turns))
    # Langfuse sends traces in batches from a background thread. Flush before exiting,
    # or the last ones are lost.
    get_client().flush()
    sys.exit(exit_code)
