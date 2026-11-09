"""The Webex runner: send the Webex evaluation inputs through the agent and check each.

Needs this repo's MCP server running (python -m server.server). For a graded run,
first reset the database to the seed: stop the server, run python -m server.reset_db,
start it again. Run from the repo root:
    .venv/bin/python -m evals.run_webex

The inputs come from data/EVALUATION-INPUTS.yaml (workflow "webex"). Each goes through
Maya's front door, exactly as a user's message would: the plan step decides whether
Maya answers or passes it to the Webex workflow (webex/).
  - W-I-01, W-I-02, W-I-03: one conversation each;
  - W-S-01: its four turns in one conversation (Sara, HR);
  - then W-I-03 again, in a new conversation: the same answer, and no new rows.
The checks live in evals/webex_checks.py.

Outputs: a short summary per input in the terminal, everything in full in
runs/webex.log, and a JSON record in runs/webex_<YYYYmmdd-HHMMSS>.json.
Exit code 1 if any check failed.
"""

import asyncio
import json
import sqlite3
import sys
from datetime import datetime

import yaml
from langchain_core.messages import HumanMessage

from evals.run_maya_s2 import describe_checks, describe_tool_call, load_tools, one_line, turn_route
from evals.webex_checks import COUNTED_TABLES, EXPECTED, check_input
from evals.webex_recording import RecordingWebexPort, real_webex_port
from maya.config import REPO_ROOT
from maya.graph import REFUSAL, build_graph
from maya.retrieval import OpenSearchRetriever
from server.db import DB_PATH

EVALUATION_INPUTS = REPO_ROOT / "data" / "EVALUATION-INPUTS.yaml"
LOG_FILE = REPO_ROOT / "runs" / "webex.log"
RUNS_DIR = REPO_ROOT / "runs"  # not committed: the records hold employee data
WORKFLOW_ID = "webex"
REPLAYED_INPUT = "W-I-03"
# The YAML names each caller's group; the agent uses the group's id.
GROUP_IDS = {"Regular Employee": "UG_REGULAR", "HR Group": "UG_HR", "IT Group": "UG_IT"}


def load_cases() -> list[dict]:
    """Every message to send, in order: {id, turn, caller, message, thread_id}."""

    workflows = yaml.safe_load(EVALUATION_INPUTS.read_text())["workflows"]
    webex = next(w for w in workflows if w["id"] == WORKFLOW_ID)
    cases = [
        {"id": item["id"], "turn": None, "caller": item["caller"],
         "message": item["message"], "thread_id": item["id"]}
        for item in webex["inputs"]
    ]
    for session in webex["sessions"]:
        cases += [
            {"id": session["id"], "turn": turn["n"], "caller": session["caller"],
             "message": turn["message"], "thread_id": session["id"]}
            for turn in session["turns"]
        ]
    replayed = next(case for case in cases if case["id"] == REPLAYED_INPUT)
    cases.append({**replayed, "thread_id": f"{REPLAYED_INPUT}-replay", "replay": True})
    return cases


def count_rows() -> dict[str, int]:
    """Rows in Webex's tables now. Read-only, so it can't disturb the server."""

    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in COUNTED_TABLES}


async def run_case(graph, webex: RecordingWebexPort, case: dict) -> dict:
    """Send one message, then collect what happened and check it."""

    before = count_rows()
    first_question, first_handoff = len(webex.questions), len(webex.calls)
    first_tool = len(webex.tool_calls)
    caller = case["caller"]
    config = {"configurable": {
        "thread_id": case["thread_id"],
        "caller_id": caller["employee_id"],
        "user_group": GROUP_IDS[caller["group"]],
    }}
    result = await graph.ainvoke({"messages": [HumanMessage(case["message"])]}, config=config)

    reply = result.get("reply") or {}
    answer = reply.get("answer", "")
    saved = result.get("handoffs") or {}
    handoff_results = [saved.get(sent.idempotency_key, {}) for sent in webex.calls[first_handoff:]]
    after = count_rows()
    outcome = {
        "answer": answer,
        "intent": result["plan"]["current_intent"],
        "route": turn_route(answer == REFUSAL, len(webex.questions) > first_question,
                            handoff_results),
        "citations": sorted({c.get("record_id") or c.get("chunk_id") or c.get("source")
                             for c in reply.get("citations", [])}),
        "handoff_results": handoff_results,
        "webex_tool_calls": webex.tool_calls[first_tool:],
        "rows_added": {table: after[table] - before[table] for table in COUNTED_TABLES},
        "error": None,
    }
    outcome["checks"] = check_input(EXPECTED[(case["id"], case["turn"])], outcome)
    return outcome


async def run_case_safely(graph, webex: RecordingWebexPort, case: dict) -> dict:
    """run_case, but a crash becomes a failed case instead of ending the run."""

    try:
        return await run_case(graph, webex, case)
    except Exception as error:
        return {"answer": "", "intent": "?", "route": "?", "citations": [], "handoff_results": [],
                "webex_tool_calls": [], "rows_added": {}, "error": f"{type(error).__name__}: {error}",
                "checks": [("finished without error", False)]}


def replay_checks(first: dict, again: dict) -> list[tuple[str, bool]]:
    """W-I-03 asked again: the same route and the same cited records."""

    return [("replay: same route", again["route"] == first["route"]),
            ("replay: same cited records", again["citations"] == first["citations"])]


def case_name(case: dict) -> str:
    name = case["id"] + (f" turn {case['turn']}" if case["turn"] else "")
    return name + (" (again)" if case.get("replay") else "")


def passed(outcome: dict) -> bool:
    return all(ok for _, ok in outcome["checks"])


def print_case(case: dict, outcome: dict) -> None:
    """The short terminal summary of one case."""

    rows = ", ".join(f"+{n} {t}" for t, n in outcome["rows_added"].items() if n) or "none"
    webex_tools = ", ".join(call["name"] for call in outcome["webex_tool_calls"]) or "(none)"
    print(f"\n{case_name(case)}  {'PASS' if passed(outcome) else 'FAIL'}  "
          f"{one_line(case['message'], 70, count_dropped=False)}")
    print(f"  route:  {outcome['route']} · intent {outcome['intent']} · rows added: {rows}")
    print(f"  webex tools: {webex_tools}")
    print(f"  cites:  {', '.join(outcome['citations']) or '(none)'}")
    print(f"  checks: {describe_checks(outcome['checks'])}")
    print(f"  answer: {outcome['error'] or one_line(outcome['answer'], 90, count_dropped=False)}")


def log_case(case: dict, outcome: dict) -> str:
    """Everything about one case, for the log file."""

    rule = "═" * 78
    lines = [f"\n{rule}", f"{case_name(case)}  ·  {'PASS' if passed(outcome) else 'FAIL'}", rule,
             f"CALLER   {case['caller']['name']} ({case['caller']['employee_id']}, "
             f"{case['caller']['group']})", "MESSAGE", case["message"],
             f"ROUTE    {outcome['route']} (intent {outcome['intent']})",
             "WEBEX TOOLS " + ("\n  ".join(describe_tool_call(c) for c in outcome["webex_tool_calls"])
                               or "(none)"),
             f"HANDOFFS {outcome['handoff_results'] or '(none)'}",
             f"ROWS     added {outcome['rows_added']}",
             f"CITES    {', '.join(outcome['citations']) or '(none)'}",
             f"CHECKS   {describe_checks(outcome['checks'])}",
             "ANSWER", outcome["error"] or outcome["answer"]]
    return "\n".join(lines) + "\n"


def save_run(cases: list[dict], outcomes: list[dict]):
    """Write the whole run to runs/webex_<YYYYmmdd-HHMMSS>.json. Returns its path."""

    path = RUNS_DIR / f"webex_{datetime.now():%Y%m%d-%H%M%S}.json"
    records = [
        {"case": case_name(case), "caller": case["caller"], "message": case["message"],
         **{k: v for k, v in outcome.items() if k != "checks"},
         "checks": [{"label": label, "passed": ok} for label, ok in outcome["checks"]]}
        for case, outcome in zip(cases, outcomes)
    ]
    run = {"workflow": WORKFLOW_ID, "cases": records,
           "totals": {"cases": len(records), "passed": sum(passed(o) for o in outcomes)}}
    path.write_text(json.dumps(run, indent=2, default=str))
    return path


async def main() -> int:
    """Run every case, save the record, and return the exit code: 1 if anything failed."""

    tools = await load_tools()
    cases = load_cases()
    print(f"Webex runner: {len(cases)} messages from {EVALUATION_INPUTS.relative_to(REPO_ROOT)} · "
          f"rows now: {count_rows()}")
    outcomes = []
    async with real_webex_port(tools, OpenSearchRetriever()) as webex:
        graph = build_graph(tools, OpenSearchRetriever(), webex)
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open("w") as log:
            for case in cases:
                outcome = await run_case_safely(graph, webex, case)
                if case.get("replay"):
                    first = outcomes[[c["id"] for c in cases].index(REPLAYED_INPUT)]
                    outcome["checks"] += replay_checks(first, outcome)
                outcomes.append(outcome)
                print_case(case, outcome)
                log.write(log_case(case, outcome))

    failures = [f"{case_name(c)}: {'; '.join(label for label, ok in o['checks'] if not ok)}"
                for c, o in zip(cases, outcomes) if not passed(o)]
    print(f"\nTOTAL  {len(cases) - len(failures)}/{len(cases)} pass · rows now: {count_rows()}")
    print(f"Full log: {LOG_FILE.relative_to(REPO_ROOT)}")
    print(f"Run record: {save_run(cases, outcomes).relative_to(REPO_ROOT)}")
    if failures:
        print("\nFAILED\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
