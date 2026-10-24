"""Smoke test: run the Maya agent on the first turns of golden session S2.

Needs Lesson 10's MCP server running. Run from the repo root:
    .venv/bin/python -m scripts.smoke_s2

All turns run on one thread_id, so they form one conversation. Two outputs:
  - the terminal: a short summary per turn (tools called, checks, cost);
  - runs/smoke_s2.log: everything in full, including exactly what each model call
    received. Rewritten on every run.

The turns come from data/GOLDEN-DATASETS.json (search for "S2-onboarding-maya").
The checks are a simple version of Lesson 10's evals, not the full thing.
"""

import asyncio
import json
import textwrap
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from maya.config import MCP_SERVER_URL, REPO_ROOT
from maya.graph import build_graph
from maya.planning import ASSISTANT_PROMPT, PLANNER_PROMPT, message_text


GOLDEN_DATASETS = REPO_ROOT / "data" / "GOLDEN-DATASETS.json"
LOG_FILE = REPO_ROOT / "runs" / "smoke_s2.log"
SESSION_ID = "S2-onboarding-maya"
TURNS_TO_RUN = 3

# The conversation's ID. The checkpointer saves the state under it, so every turn sent
# with the same thread_id continues the same conversation.
THREAD_ID = "smoke-s2"

# LangGraph counts every step it runs in a turn and stops the turn with an error past
# this limit. It stops a model that keeps asking for tools from looping forever.
RECURSION_LIMIT = 12

# Long prompts that calls repeat. The log shows ‹NAME› instead of the full text, so
# what changes from call to call stands out.
KNOWN_PROMPTS = {"ASSISTANT_PROMPT": ASSISTANT_PROMPT, "PLANNER_PROMPT": PLANNER_PROMPT}

# Structured output works through a fake "tool" with the schema's name: the model
# "calls" it to give its reply in that shape. These are those names, not real tools.
OUTPUT_FORMATS = {"TurnPlan"}

# In the log, old answers and tool results are cut to this many characters.
SHORT = 200


# LOADING ---------------------------------------------------------------------

async def load_tools() -> list:
    """Connect to the MCP server and get its NovaOps tools as LangChain tools."""

    client = MultiServerMCPClient(
        {"novaops": {"url": MCP_SERVER_URL, "transport": "streamable_http"}}
    )
    return await client.get_tools()


def load_turns() -> list[dict]:
    """The first turns of session S2: each user message, plus what a good answer needs."""

    sessions = json.loads(GOLDEN_DATASETS.read_text())["sessions"]
    session = next(s for s in sessions if s["id"] == SESSION_ID)
    return session["turns"][:TURNS_TO_RUN]


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


# CHECKS AGAINST THE GOLDEN DATASET -------------------------------------------

def arg_check_passes(check: dict, tool_calls: list) -> bool:
    """True if some call to check["tool"] has an argument that matches the check."""

    for call in tool_calls:
        if call["name"] != check["tool"]:
            continue
        value = str(call["args"].get(check["arg"], ""))
        if "equals" in check and value == check["equals"]:
            return True
        if "contains" in check and check["contains"].lower() in value.lower():
            return True
    return False


def check_turn(turn: dict, plan: dict, tool_calls: list, answer: str) -> list[tuple[str, bool]]:
    """Grade one turn against the golden dataset. Returns (label, passed) pairs."""

    called = {call["name"] for call in tool_calls}
    checks = []

    # "a|b" in the dataset means either intent is accepted.
    intent = plan["current_intent"]
    accepted = turn["expected_intent"].split("|")
    label = f"intent {intent}" + ("" if intent in accepted else f" (expected {turn['expected_intent']})")
    checks.append((label, intent in accepted))

    wants_tools = turn["should_use_tools"]
    checks.append(("tools used" if wants_tools else "no tools", bool(called) == wants_tools))

    missing = [name for name in turn.get("expected_tools", []) if name not in called]
    label = f"expected tools (missing {', '.join(missing)})" if missing else "expected tools"
    checks.append((label, not missing))

    if turn.get("arg_checks"):
        passed = all(arg_check_passes(check, tool_calls) for check in turn["arg_checks"])
        checks.append(("args", passed))

    facts = turn.get("required_facts", [])
    missing = [fact for fact in facts if fact.lower() not in answer.lower()]
    label = f"facts {', '.join(facts)}" + (f" (missing {', '.join(missing)})" if missing else "")
    checks.append((label, not missing))

    forbidden = [name for name in turn.get("forbidden_tools", []) if name in called]
    label = f"forbidden tool called: {', '.join(forbidden)}" if forbidden else "no forbidden tool"
    checks.append((label, not forbidden))
    return checks


# RUNNING ONE TURN ------------------------------------------------------------

def latest_turn_start(messages: list) -> int:
    """The index where the latest turn starts: its user message."""

    return max(
        index for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    )


async def run_turn(graph, config: dict, recorder: ModelCallRecorder, turn: dict) -> dict:
    """Send one user message, then collect what happened in that turn."""

    first_call = len(recorder.calls)
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
    return {
        "turn": turn,
        "plan": result["plan"],
        "tool_calls": tool_calls,
        "checks": check_turn(turn, result["plan"], tool_calls, answer),
        "model_calls": recorder.calls[first_call:],
        "saved_messages": len(messages),
    }


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
    return " · ".join(f"{label} {'✓' if passed else '✗'}" for label, passed in checks)


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

    if reply.tool_calls and reply.tool_calls[0]["name"] in OUTPUT_FORMATS:
        return describe_fields(reply.tool_calls[0]["args"])
    if reply.tool_calls:
        return "\n".join(f"called {describe_tool_call(call)}" for call in reply.tool_calls)
    return message_text(reply)


def indent(text: str, spaces: int) -> str:
    return textwrap.indent(text, " " * spaces)


# THE TWO OUTPUTS -------------------------------------------------------------

def print_turn(number: int, outcome: dict) -> None:
    """The short terminal summary of one turn."""

    calls = outcome["model_calls"]
    tools = ", ".join(describe_tool_call(call) for call in outcome["tool_calls"])
    input_tokens = sum(call["input_tokens"] for call in calls)
    answer_calls = [call for call in calls if call["step"] == "model"]
    most_tools = max(len(call["tools"]) for call in answer_calls)
    answer = describe_reply(calls[-1]["reply"])

    # How many calls each step made, like "plan 1, model 2".
    steps = [call["step"] for call in calls]
    per_step = ", ".join(f"{step} {steps.count(step)}" for step in dict.fromkeys(steps))

    plan = outcome["plan"]
    needs = "needs tools" if plan["requires_tools"] else "no tools needed"

    print(f"\nTurn {number}  {'PASS' if turn_passed(outcome) else 'FAIL'}  "
          f"{one_line(outcome['turn']['user'], 70, count_dropped=False)}")
    print(f"  plan:    {plan['current_intent']} · {needs} · "
          f"facts: {len(plan['relevant_facts'])} · constraints: {len(plan['relevant_constraints'])}")
    print(f"  tools:   {tools or '(none)'}")
    print(f"  checks:  {describe_checks(outcome['checks'])}")
    print(f"  cost:    {len(calls)} model calls ({per_step}) · {input_tokens:,} input tokens · "
          f"up to {most_tools} tools shown · {outcome['saved_messages']} messages saved")
    print(f"  answer:  {one_line(answer, 90, count_dropped=False)}")


LOG_INTRO = """\
How to read this log:
  - Each turn lists every model call in the order it ran.
  - "Sent" is exactly what that call received; "Reply" is what it returned, in full.
  - step: plan is the planner; step: model is the answer model.
  - ‹ASSISTANT_PROMPT› and ‹PLANNER_PROMPT› stand for those prompts' full text
    (see maya/planning.py).
  - Old answers and tool results are cut to {short} characters.
"""


def log_header(tool_count: int) -> str:
    started = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"Smoke test log · {started} · session {SESSION_ID} · thread {THREAD_ID}\n"
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


def log_turn(number: int, outcome: dict, tool_count: int) -> str:
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
        lines += ["  Reply:", indent(describe_reply(call["reply"]), 4)]

    lines += [
        "",
        f"CHECKS  {describe_checks(outcome['checks'])}",
        f"STATE   {outcome['saved_messages']} messages saved in thread {THREAD_ID}",
    ]
    return "\n".join(lines) + "\n"


# MAIN ------------------------------------------------------------------------

async def main() -> None:
    tools = await load_tools()
    graph = build_graph(tools)
    recorder = ModelCallRecorder()
    config = {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": RECURSION_LIMIT,
        "callbacks": [recorder],
    }
    turns = load_turns()
    print(f"Smoke test: {SESSION_ID}, turns 1-{len(turns)} · {len(tools)} tools from MCP")

    LOG_FILE.parent.mkdir(exist_ok=True)
    passed = 0
    with LOG_FILE.open("w") as log:
        log.write(log_header(len(tools)))
        for number, turn in enumerate(turns, start=1):
            outcome = await run_turn(graph, config, recorder, turn)
            print_turn(number, outcome)
            log.write(log_turn(number, outcome, len(tools)))
            passed += turn_passed(outcome)

    input_tokens = sum(call["input_tokens"] for call in recorder.calls)
    print(f"\nTOTAL  {passed}/{len(turns)} turns pass · {len(recorder.calls)} model calls · "
          f"{input_tokens:,} input tokens")
    print(f"Full log: {LOG_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
