"""Run the five Webex cases, score every turn, and attach the scores to its Langfuse trace.

The answer key is evals/webex_cases.json; the rules are evals/case_checks.py. Each turn:
  1. runs inside its own labelled trace (evals/tracing.py), like the other runners:
     - a message turn goes through Maya's front door, as a user's message would;
     - an approval turn records the approver's decision in the database, then resumes
       the paused Webex run, as python -m webex.approve does;
  2. is scored from what was recorded during the run: Webex's tool calls, its answer
     and its status (no model, no Langfuse needed for that);
  3. gets its scores attached to its trace, each with its reason as the comment.

The rules run on what this script recorded, not on traces fetched back from Langfuse:
the same calls, without waiting for Langfuse to make them readable.

Needs this repo's MCP server running, on a fresh database (the cases expect the seed:
WX-01 files a request). Stop the server, python -m server.reset_db, start it again, then:
    .venv/bin/python -m evals.score_webex
"""

import asyncio
import json
import sqlite3
import sys

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from evals.case_checks import DATA_TYPES, roll_up, score_turn
from evals.run_maya_s2 import load_tools, one_line
from evals.tracing import RUN_STAMP, finish_turn, langfuse, require_langfuse, traced_turn, turn_status
from evals.webex_recording import RecordingWebexPort, real_webex_port
from maya.config import REPO_ROOT
from maya.graph import REFUSAL, build_graph
from maya.retrieval import OpenSearchRetriever
from server.db import DB_PATH
from webex.graph import traced_run_config
from webex.tools import WEBEX_WRITE_TOOLS

CASES_FILE = REPO_ROOT / "evals" / "webex_cases.json"
RUNS_DIR = REPO_ROOT / "runs"  # not committed: the records hold employee data
SEED_ACCESS_REQUESTS = 4  # AR001-AR004; more means an earlier run already wrote
RESULT_CHARS = 1500  # how much of each tool result the judges get; they pay per token
# Langfuse runs the two judges as soon as each turn's trace arrives. Seven turns in a
# burst made Bedrock answer "too many requests", and Langfuse dropped those grades
# after two retries. A pause between turns spreads the judges' calls out.
SECONDS_BETWEEN_TURNS = 8


# THE EVIDENCE FOR THE JUDGES ----------------------------------------------------------
# Langfuse's judges (Milestone 3) read only the turn's top step, not the tool steps
# inside it. So the evidence they grade the answer against is copied onto that step.

def shorten(result) -> str:
    text = json.dumps(result, default=str)
    return text if len(text) <= RESULT_CHARS else text[:RESULT_CHARS] + " …[cut]"


class DocumentRecorder:
    """Wraps the retriever Webex searches with, and keeps every chunk it returned.
    The same search() method, so Webex can't tell the difference."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.chunks = []  # Evidence, in order

    def search(self, query, caller, k=4, collections=None):
        found = self.retriever.search(query, caller, k=k, collections=collections)
        self.chunks.extend(found)
        return found


def evidence_for(tool_calls: list[dict], chunks: list, earlier: list[dict]) -> dict:
    """What Webex read and what it did this turn, plus what earlier turns read.

    - records_read: every tool result, as the evidence a claim must rest on;
    - documents_read: the help articles and policies Webex searched (next steps like
      "the IT Manager can reclaim unused seats" come from these);
    - actions_taken: only the writes (approval asked, request filed), so a judge can
      tell "requested" from "done";
    - earlier_turns: what earlier turns read (records and documents). An answer may
      repeat what an earlier turn found (WX-04 turn 2 runs no tools at all), and that
      isn't invented.
    """
    calls = [{"tool": call["name"], "args": call["args"], "result": shorten(call["result"])}
             for call in tool_calls]
    documents = [{"chunk": chunk.chunk_id, "text": shorten(chunk.text)} for chunk in chunks]
    return {
        "records_read": calls,
        "documents_read": documents,
        "actions_taken": [call for call in calls if call["tool"] in WEBEX_WRITE_TOOLS],
        "earlier_turns": earlier,
    }


# RUN ONE TURN -------------------------------------------------------------------------

async def send_message(graph, case: dict, turn: dict, webex: RecordingWebexPort) -> dict:
    """A message turn: through Maya, exactly as a user's message would go."""
    first_result = len(webex.results)
    config = {"configurable": {"thread_id": f"{case['id']}-{RUN_STAMP}",
                               "caller_id": case["caller"]["employee_id"],
                               "user_group": case["caller"]["group"]}}
    result = await graph.ainvoke({"messages": [HumanMessage(turn["user"])]}, config=config)
    answer = (result.get("reply") or {}).get("answer", "")
    return {"answer": answer,
            "status": turn_status(answer == REFUSAL, webex.results[first_result:])}


async def approve_and_resume(turn: dict, webex: RecordingWebexPort, waiting: str) -> dict:
    """An approval turn: the approver decides `waiting` (e.g. AP005), and the paused
    Webex run goes on. The decision is the human's, not the agent's, so it goes through
    the unrecorded toolbox: it isn't one of Webex's own tool calls."""
    decision = turn["approve"]
    if not waiting:  # the turn before didn't pause on an approval: nothing to decide
        return {"answer": "", "status": "no paused run"}
    [row] = await webex.toolbox.toolbox.call(
        "record_approval_decision", approval_id=waiting, approver_id=decision["approver_id"],
        decision=decision["decision"], reason="Decided in the Webex scoring run.",
    )
    # A new approval's request_id is the key the paused run is saved under.
    final_state = await webex.port.graph.ainvoke(Command(resume=decision["decision"]),
                                                 traced_run_config(row["request_id"]))
    return {"answer": final_state["decision"].answer, "status": final_state["decision"].status}


async def run_turn(graph, case: dict, turn: dict, webex: RecordingWebexPort,
                   documents: DocumentRecorder, waiting: str | None, earlier: list[dict]) -> dict:
    """One turn inside its own trace: run it, score it, attach the scores."""
    label = turn.get("user") or f"approval by {turn['approve']['approver_id']}"
    with traced_turn(f"{case['id']} turn {turn['n']}", label, conversation=case["id"],
                     user_id=case["caller"]["employee_id"], tags=["webex-scoring", case["id"]],
                     metadata={"case_id": case["id"], "turn": str(turn["n"])}) as root:
        first_tool, first_chunk = len(webex.tool_calls), len(documents.chunks)
        if "approve" in turn:
            outcome = await approve_and_resume(turn, webex, waiting)
        else:
            outcome = await send_message(graph, case, turn, webex)
        outcome["tool_calls"] = webex.tool_calls[first_tool:]

        checks = score_turn(turn, outcome["answer"], outcome["tool_calls"], outcome["status"])
        checks |= roll_up({name: score for name, (score, _) in checks.items()})
        outcome["scores"] = checks
        outcome["evidence"] = evidence_for(outcome["tool_calls"],
                                           documents.chunks[first_chunk:], earlier)
        outcome["trace_url"] = finish_turn(root, outcome["answer"], {
            "status": outcome["status"], "turn_pass": checks["turn_pass"][0],
            "evidence": outcome["evidence"]})
        attach_scores(langfuse.get_current_trace_id(), checks)
    return outcome


def attach_scores(trace_id: str, checks: dict[str, tuple[float, str]]) -> None:
    """One Langfuse score per check, its reason as the comment."""
    for name, (value, reason) in checks.items():
        langfuse.create_score(trace_id=trace_id, name=name, value=value,
                              data_type=DATA_TYPES[name], comment=reason)


# THE WHOLE RUN ------------------------------------------------------------------------

def print_turn(name: str, outcome: dict) -> None:
    scores = outcome["scores"]
    verdict = "PASS" if scores["turn_pass"][0] == 1.0 else f"FAIL ({scores['turn_pass'][1]})"
    tools = ", ".join(call["name"] for call in outcome["tool_calls"]) or "(none)"
    print(f"\n{name}  {verdict}  · status {outcome['status']} · score {scores['turn_score'][0]:.2f}")
    print(f"  webex tools: {tools}")
    print(f"  answer: {one_line(outcome['answer'], 100, count_dropped=False)}")
    print(f"  trace:  {outcome['trace_url']}")


def access_request_count() -> int:
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        return conn.execute("SELECT COUNT(*) FROM access_requests").fetchone()[0]


async def main() -> int:
    require_langfuse()
    if access_request_count() != SEED_ACCESS_REQUESTS:
        raise SystemExit("The database isn't the seed: stop the server, run "
                         "python -m server.reset_db, start it again.")
    cases = json.loads(CASES_FILE.read_text())["cases"]
    tools = await load_tools()
    print(f"Webex scoring: {len(cases)} cases from {CASES_FILE.relative_to(REPO_ROOT)} · "
          f"Langfuse run {RUN_STAMP}")

    records = []
    documents = DocumentRecorder(OpenSearchRetriever())  # what Webex searches, for the judges
    async with real_webex_port(tools, documents) as webex:
        graph = build_graph(tools, OpenSearchRetriever(), webex)
        for case in cases:
            waiting = None  # the approval a paused run waits on, for an approval turn
            earlier = []    # what this case's earlier turns read, for the judges
            for turn in case["turns"]:
                await asyncio.sleep(SECONDS_BETWEEN_TURNS if records else 0)
                outcome = await run_turn(graph, case, turn, webex, documents, waiting, earlier)
                evidence = outcome["evidence"]
                earlier = earlier + evidence["records_read"] + evidence["documents_read"]
                waiting = (webex.results[-1].request_id if webex.results else None)
                name = f"{case['id']} turn {turn['n']}"
                print_turn(name, outcome)
                records.append({"turn": name, **outcome})

    passed = sum(record["scores"]["turn_pass"][0] for record in records)
    print(f"\nturn_pass {passed:.0f}/{len(records)} = {passed / len(records):.0%} · "
          f"scores attached to each trace in Langfuse")
    path = RUNS_DIR / f"score_webex_{RUN_STAMP}.json"
    path.write_text(json.dumps(records, indent=2, default=str))
    print(f"Run record: {path.relative_to(REPO_ROOT)}")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    langfuse.flush()  # send the last traces and scores before exiting
    sys.exit(exit_code)
