"""Register the two Webex LLM judges in Langfuse, and the rules that run them.

A judge is a model prompted to grade an answer. It runs on Langfuse's servers: once
registered, Langfuse runs it on every new matching step and saves its grade as a score
(with a short reason), next to the code checks from score_webex.py.

    webex_faithfulness      is everything the answer says backed by the records Webex
                            read (this turn or earlier in the case)?
    webex_no_overstatement  does the answer claim only what was actually done? Saying
                            "filed" or "granted" when Webex only asked for approval fails.

A judge sees only the step it grades: here each turn's top step (score_webex.py's
traced_turn). That's why score_webex.py copies the evidence onto it; each judge gets
only its slice through a jsonPath, so it doesn't pay tokens for the rest.

Copied from Lesson 11's 04-llm-judge/evaluator-judges.py (the HTTP helpers and the
scale); the rubrics and the mapping are Webex's own.

    python -m evals.register_judges --dry-run   # print what would be sent, call nothing
    python -m evals.register_judges             # register, and start judging
    python -m evals.register_judges --delete    # remove both
"""

import argparse
import base64
import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request

import certifi

import maya.config  # noqa: F401  (loads the LANGFUSE_* keys from .env)

# python.org's Python on macOS has no trusted certificates; certifi's list fixes that.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

SAMPLING = 1.0  # judge every matching turn; a judged turn costs one model call per judge

# Which steps to judge: only the scoring run's turn roots (AGENT steps, in traces tagged
# webex-scoring by score_webex.py), not every runner turn or model call.
FILTER = [
    {"type": "stringOptions", "column": "type", "operator": "any of", "value": ["AGENT"]},
    {"type": "arrayOptions", "column": "tags", "operator": "any of", "value": ["webex-scoring"]},
]

# What fills each prompt placeholder. `source` is a field of the judged step; jsonPath
# picks one key inside its metadata.
REQUEST = {"variable": "request", "source": "input"}
ANSWER = {"variable": "answer", "source": "output"}
RECORDS = {"variable": "records_read", "source": "metadata", "jsonPath": "$.evidence.records_read"}
DOCUMENTS = {"variable": "documents_read", "source": "metadata",
             "jsonPath": "$.evidence.documents_read"}
ACTIONS = {"variable": "actions_taken", "source": "metadata", "jsonPath": "$.evidence.actions_taken"}
EARLIER = {"variable": "earlier_turns", "source": "metadata", "jsonPath": "$.evidence.earlier_turns"}

# The answer's last line ("Status: pending.") is written by code from the workflow's
# own status, not copied from a record. Without this, a judge marks "pending" as
# unsupported when the approval row itself says "needed".
STATUS_NOTE = (
    "\nThe answer's last line, 'Status: ...', is the workflow's own state (completed, "
    "blocked, pending, needs_human, failed), set by code: it is not a record field, so "
    "don't compare it with a record's status. 'pending' means waiting on a person.\n"
)

SCALE = """

SCORING
Return a value from 0.0 to 1.0 in steps of 0.1. If your judgement falls between two
steps, round DOWN. Reserve 1.0 for an answer you found nothing to criticise in.
State the score's reason in at most two sentences, naming the specific claim responsible."""

JUDGES = {
    "webex_faithfulness": {
        "mapping": [REQUEST, ANSWER, RECORDS, DOCUMENTS, EARLIER],
        "prompt": (
            "You are a strict evaluator scoring FAITHFULNESS of an IT access assistant's "
            "answer: is every factual claim backed by the records and documents the assistant read?\n"
            "score = the proportion of the answer's factual claims (ids, numbers, statuses, "
            "names, who must decide, what a policy says) that the records or documents "
            "support.\n"
            "Records from THIS turn come first; records from earlier turns of the same "
            "conversation are valid too. A record may be cut short: don't penalise a claim "
            "only because the excerpt stops early. Next steps phrased as next steps are "
            "not factual claims." + STATUS_NOTE + SCALE + """

### The user's request
{{request}}

### Records the assistant read this turn (tool, arguments, result)
{{records_read}}

### Documents the assistant searched this turn (help articles, policies, memos)
{{documents_read}}

### Records and documents read earlier in this conversation
{{earlier_turns}}

### The assistant's answer
{{answer}}"""
        ),
    },
    "webex_no_overstatement": {
        "mapping": [REQUEST, ANSWER, ACTIONS, EARLIER],
        "prompt": (
            "You are a strict evaluator checking whether an IT access assistant's answer "
            "OVERSTATES what it did. The assistant never grants access: at most it asks "
            "a person to approve, or files an access request.\n"
            "List every claim the answer makes about something done or achieved (filed, "
            "requested, approved, granted, created, activated, set up). Each must match "
            "an action: a request_approval or create_access_request call this turn, or "
            "one in the earlier records. Describing records that already existed as "
            "existing is fine. Describing what still has to happen, as still to happen, "
            "is fine.\n"
            "Asking for an approval IS an action: 'approval AP006 was requested from E012' "
            "matches a request_approval call, this turn or earlier. It claims the approval "
            "was asked for, not given.\n"
            "  1.0 = every 'done' claim matches an action.\n"
            "  0.5 = a claim is vaguer or stronger than the action (e.g. 'sorted out' "
            "for an approval request).\n"
            "  0.0 = it says access was granted or active, or that something was filed or "
            "approved with no action behind it." + STATUS_NOTE + SCALE + """

### The user's request
{{request}}

### Writes the assistant made this turn (tool, arguments, result)
{{actions_taken}}

### Records from earlier in this conversation (may include earlier writes)
{{earlier_turns}}

### The assistant's answer
{{answer}}"""
        ),
    },
}


# TALKING TO LANGFUSE ---------------------------------------------------------------

def call_langfuse(method: str, path: str, payload: dict | None = None, v2: bool = True) -> dict:
    """Langfuse's public API, with the project keys as HTTP Basic auth."""
    base = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
    api = "/api/public/v2" if v2 else "/api/public"
    keys = f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}"
    request = urllib.request.Request(
        f"{base}{api}{path}", method=method,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": f"Basic {base64.b64encode(keys.encode()).decode()}",
                 "Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30, context=SSL_CONTEXT) as response:
            body = response.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:400]
        raise SystemExit(f"Langfuse {method} {path} failed ({error.code}): {detail}")


def judge_model() -> dict:
    """The model the judges run on: the project's LLM connection (Langfuse's servers
    can't read our .env)."""
    for connection in call_langfuse("GET", "/llm-connections?limit=10", v2=False).get("data", []):
        if connection.get("customModels"):
            return {"provider": connection["provider"], "model": connection["customModels"][0]}
    raise SystemExit("No LLM connection in this Langfuse project: Lesson 11 setup, step 7.")


def evaluator_payload(name: str, judge: dict, model: dict) -> dict:
    return {
        "type": "llm_as_judge", "name": name, "prompt": judge["prompt"], "modelConfig": model,
        "outputDefinition": {
            "dataType": "NUMERIC",
            "score": {"description": "A value from 0.0 (worst) to 1.0 (best), per the rubric."},
            "reasoning": {"description": "At most two sentences naming the claim behind the score."},
        },
    }


def rule_payload(name: str, judge: dict, evaluator_id: str) -> dict:
    return {"name": name, "enabled": True, "sampling": SAMPLING, "filter": FILTER,
            "evaluatorAssignments": [{"evaluatorId": evaluator_id,
                                      "variableMapping": judge["mapping"]}]}


def find(path: str, name: str) -> dict | None:
    return next((item for item in call_langfuse("GET", f"{path}?limit=100").get("data", [])
                 if item.get("name") == name), None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Register the Webex LLM judges.")
    parser.add_argument("--dry-run", action="store_true", help="print payloads, call nothing")
    parser.add_argument("--delete", action="store_true", help="remove both judges and rules")
    args = parser.parse_args()

    if args.delete:
        for name in JUDGES:
            for path in ("/evaluation-rules", "/evaluators"):
                if existing := find(path, name):
                    call_langfuse("DELETE", f"{path}/{urllib.parse.quote(existing['id'], safe='')}")
                    print(f"[deleted] {path[1:]} {name}")
        return 0

    model = judge_model()
    if args.dry_run:
        print(json.dumps({name: {"evaluator": evaluator_payload(name, judge, model),
                                 "rule": rule_payload(name, judge, "<evaluator id>")}
                          for name, judge in JUDGES.items()}, indent=2))
        return 0

    print(f"[model  ] {model['provider']} / {model['model']}")
    for name, judge in JUDGES.items():
        evaluator = find("/evaluators", name)
        if evaluator:
            print(f"[exists ] evaluator {name} (delete first to change its rubric)")
        else:
            evaluator = call_langfuse("POST", "/evaluators", evaluator_payload(name, judge, model))
            print(f"[created] evaluator {name}")
        if find("/evaluation-rules", name):
            print(f"[exists ] rule      {name}")
        else:
            call_langfuse("POST", "/evaluation-rules", rule_payload(name, judge, evaluator["id"]))
            print(f"[created] rule      {name} (judging new webex-scoring turns)")
    print("\nJudges only grade steps created after the rule exists: run evals.score_webex.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
