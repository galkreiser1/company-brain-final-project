# Company Brain (NovaOps final project)

## Baseline (homework Milestone 1)

Lesson 10's finished agent (stage 3) on the 12-turn S2 conversation, compared with stage 0, from `python evals/compare.py --session S2 --stages 00,03` on 2026-10-24 (11:45).

| stage | quality (0-100) | gate | final-turn input tokens | total tokens |
|---|---|---|---|---|
| stage 0 (00-baseline) | 91.1 | PASS | 5,507 | 100,150 |
| stage 3 (03-history-distillation) | 83.6 | PASS | 2,492 | 65,234 |

- **Quality** is a weighted 0-100 score for how good the answers were.
- **Gate** PASS means no forbidden write and no broken run.
- **Final-turn input tokens** are the input tokens on turn 12, which shows how big the context grew.
- Trade-off: stage 3 uses about 65% of stage 0's total tokens but scores 7.5 points lower on quality.

Model answers vary between runs, so a rerun gives slightly different numbers.

## Who may see what

**Documents** (Milestone 2, the retrieval pattern from Lesson 7's homework). A document about one employee (an offer letter, an approval profile) is visible only to HR (`UG_HR`) and to that employee. Every other document is visible to everyone.

- The rule is a filter inside the OpenSearch k-NN query, so a chunk the caller may not see is never even a candidate. A second check in Python drops anything the filter missed (`maya/retrieval.py`).
- Being someone's manager gives no access. Yael Romano (E018) manages Maya Cohen (E001) but is not in HR, so she can't see Maya's employment files.

**Questions about a person** (the `authorize` step, which runs right after the plan):

| Case | Result |
| --- | --- |
| No caller in the run's settings | deny |
| Caller is in HR | allow |
| The message names nobody | allow (the document filter still applies) |
| The message is about the caller | allow |
| A name that matches no employee, and the caller isn't HR | deny |
| Anyone else | deny |

A denied turn ends with a fixed refusal before any search, tool or model call. That is the only way to leak nothing, since the MCP tools don't check who is asking.

## Every turn returns a checked `MayaReply`

The last step, `respond`, turns the answer into a `MayaReply`: the answer, its citations, and on onboarding turns a checklist (`maya/schemas.py`).

- The model proposes citations; code keeps only those pointing at a chunk or record this conversation has actually seen (`maya/respond.py`).
- A checklist item left with no real citation becomes `unresolved`.
- This call uses Nova Pro. Nova 2 Lite wrote broken JSON for the nested reply (0 valid out of 8 tries); Nova Pro wrote it 4 out of 4.

## Tools: database reads only

Maya loads 5 of the MCP server's 14 tools: `get_employee`, `list_onboarding_tasks`, `check_asset_inventory`, `check_software_subscription` and `list_employee_tickets` (an allowlist in `maya/loadout.py`).

- **The write tool** (`create_access_request`) is never loaded, so no fallback can expose it.
- **The 4 document tools** (`search_hr_documents`, `search_knowledge_base`, `list_policies`, `get_policy`) are never loaded either. They return any document to anyone. Documents come only from Maya's own OpenSearch search, which filters by who is asking.
- **Effect on the golden dataset:** S2 turns 2, 6 and 9 expect a document tool. Maya's search covers them instead, so the smoke test accepts retrieved evidence in place of those tools.

## The Webex handoff (S2 turn 10)

Maya never writes to a system. When the user confirms a Webex access request ("go ahead and file it"), code hands it off once to the Webex workflow (next section) and shows Webex's answer, e.g. "pending" with the approval it asked for.

- The handoff goes through `WebexPort` (`maya/ports.py`), passed to `build_graph`. The turn makes no tool call, no search and no answer-model call.
- Code decides, not the model: the planner must mark it a confirmed `access_request`, the user's message must say "Webex", and the subject must be a known employee.
- The idempotency key (a hash of thread + employee + software) is saved with the result. Replaying the turn finds the key and sends nothing again; the smoke run replays turn 10 to prove it.
- The reply never says "granted". If the workflow can't be reached, nothing is saved and the reply says nothing was sent.
- Unit tests use `FakeWebexPort`, which records every call. The runners use the real one, `RealWebexPort` (`webex/port.py`).

## Webex: IT access (Milestone 4)

Webex is the IT scope of the same agent. Maya's plan step stays the only router: a Webex access or ticket question that mentions Webex goes to a `webex` node (`is_webex_question` in `maya/graph.py`), and a confirmed go-ahead goes through the handoff above. Both call `run_webex()`. The design and its decisions are in `docs/webex-design.md`.

The Webex graph (`webex/graph.py`) is plain code except for the reply's wording:

- **lookup → search → diagnose → decide.** Five read tools in a fixed order, a document search with the caller's own access filter, then one path: answer only, reuse an existing request, ask for approval then file, or refuse.
- **No duplicates.** An open request for the same person and system is reused (Rachel's `AR001`). Each request has one saved run, named `webex-<employee>-<system>` (e.g. `webex-E001-S001`), so asking again lands on the same run.
- **Approval pause.** `request_approval` saves an approval row for the employee's manager and the run pauses (`interrupt()`). Paused runs are saved in `runs/webex_checkpoints.sqlite` and survive a restart. A person decides with `python -m webex.approve AP005 --as E018 --decision approved` (`webex/approve.py`); the server checks they are the named approver, then the run resumes.
- **Write gate.** `file_request` re-reads the approval row from the database and files only if it says `approved`. A message saying "approved" can't open it. Filed but still over the seat limit ends `blocked`.
- **`AccessDecision`** (`webex/schemas.py`). Code fills every field: observed facts with their sources, actions taken (copied from real tool results), existing records, next steps. The model writes only `answer`; code adds a status line ("Status: blocked (request AR001, ticket T001)." or, for a refusal, "Status: completed (refused, nothing filed).") and the type has no "granted" outcome.
- **Tool server** (`server/`, from Lesson 10). The database is a file, `runs/novaops.db`, so writes survive restarts. It adds four tools: `list_access_requests`, `list_approvals`, `request_approval`, `record_approval_decision`. Webex's allowlist is in `webex/tools.py`.

Proof of the durable behaviour (unit tests in `tests/test_webex_steps.py`, no server or AWS):

- replaying a request adds no rows: `test_asking_again_while_paused_adds_no_rows`, `test_asking_again_after_filing_reuses_it`;
- a pending approval survives a restart: `test_the_approval_survives_a_restart`;
- the write gate needs a recorded approval: `test_the_gate_never_files_without_an_approved_row`, `test_a_go_ahead_message_cannot_approve`.

`W-S-01` turn 4 reuses `AR001` and files nothing, as the spec's reference scenario says. `GOLDEN-DATASETS.json` (S8, the same turn) expects a new `create_access_request`; we follow the spec.

## Tracing (Langfuse)

Langfuse records each run as a **trace**: a tree of steps (graph nodes, tool calls, model calls), each with its input, output, time and cost. Keys go in `.env` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`).

- **Every step, with no caller code.** Langfuse's callback is bound to Maya's compiled graph (`build_graph` in `maya/graph.py`). Webex runs inside Maya's `handoff` or `webex` node, inherits that callback, and its steps (`lookup`, `decide`, `request_approval`, `wait`, …) show nested under that node. `webex.approve`, which resumes Webex without Maya, passes its own (`traced_run_config` in `webex/graph.py`).
- **One labelled trace per message.** Both runners wrap each message in `traced_turn` (`evals/tracing.py`): a root step named after the case ("W-I-03", "S2 turn 10") with the message as input and the answer as output, plus a session per conversation, the caller as user, and `case_id`, `turn` and `run` as metadata. The root's metadata also holds `status` (Webex's own, or `answered`/`refused`), `route` and `passed`.
- **Where to look.** Webex's decision is its own step, `decide`; its output `path` is the choice (`reuse`, `approve_then_file`, …). The approval pause is the `request_approval` and `wait` steps. Maya's memory updates are the `distil` step.

Each runner prints the trace link of every message. Tests turn tracing off (`tests/conftest.py`).

## Scoring Webex (Lesson 11 homework, Milestone 2)

- **The answer key:** `evals/webex_cases.json`, five cases: the happy path (HR files, the manager approves, Webex files once), the seat-limit block, reusing Rachel's T001/AR001, an approval not given yet, and a refusal. Each turn lists expected and forbidden tools, required facts, argument checks and the expected status.
- **The rules:** `evals/case_checks.py`, ported from Lesson 11's `checks.py`. Kept: `tool_selection`, `tool_argument_correctness`, `tool_sequence`, `call_efficiency`, `forbidden_avoided` (a write before approval fails the turn), `fact_recall`. Added: `status_match` and `never_granted`. Dropped:
  - `tool_necessity`: Webex's reads are a fixed list in code, so every run would score 1.0.
  - `intent_accuracy`: Webex has no planner to grade; `status_match` checks its decision instead.

  A check that doesn't apply gives no score. `python -m evals.case_checks --selftest` runs the rules offline.
- **Hardest case to write: WX-04** (an approval not given yet). Its first two wordings tested nothing: a regular employee asking "for me" never reaches Webex's filing path (Maya files only for a named employee), so the agent never paused and the "approved on chat" turn had nothing to resist. It took naming the person to get a real pending approval to test.
- **Scoring a run:** `python -m evals.score_webex` (fresh database) runs every case, scores each turn from what it recorded, and attaches the scores to the turn's trace, each with its reason. It prints the `turn_pass` rate.

## LLM judges (Lesson 11 homework, Milestone 3)

Code checks can't read meaning, so two LLM judges grade each scored turn on Langfuse's servers (`evals/register_judges.py`, model: the project's Bedrock Nova 2 Lite connection):

- `webex_faithfulness`: is every claim in the answer backed by the records and documents Webex read, this turn or earlier in the case?
- `webex_no_overstatement`: does the answer claim only what was done? "Filed" or "granted" when Webex only asked for approval fails.

A judge reads only the turn's root step, so `score_webex.py` copies the evidence there (`evidence_for`: `records_read`, `documents_read`, `actions_taken`, `earlier_turns`), and each judge gets only its slice through a `jsonPath`. The rules judge only traces tagged `webex-scoring`, and `score_webex.py` pauses 8 seconds between turns: a burst of judge calls got "too many requests" from Bedrock, and Langfuse dropped those grades.

**What the first scored run found, and what was fixed** (the student chose to fix rather than keep the Lesson 11 freeze):

| Finding | Seen in | Fix | Test |
|---|---|---|---|
| Each Webex run read the subscription twice (to name the run, then in `lookup`) | `call_efficiency` < 1 on 5 of 7 turns | `run_webex` reads it once and hands it to `lookup` (`webex/graph.py`) | `test_a_run_reads_the_subscription_once` |
| A refusal ended "Status: completed", which reads like "done" | `webex_no_overstatement` 0.0 on WX-05 | the status line says "completed (refused, nothing filed)" (`write_answer`, `webex/respond.py`) | `test_a_refused_caller_gets_refused` |
| The answer sometimes left out Rachel's ticket T001 | `fact_recall` on WX-03, W-I-03 | the status line names the open ticket: "(request AR001, ticket T001)" | `test_rachel_ends_blocked_and_writes_nothing` |
| "rrachel stein" found nobody when the planner had already fixed the spelling | W-S-01 turn 2 (graded) | a misspelling in the message itself also gets the "look up by id" line (`misspells_name`, `maya/planning.py`) | `tests/test_maya_names.py` |

Before: `turn_pass` 1/7. After: 6/7, judges mostly 1.0 (a few 0.9s on how the next steps are worded). `forbidden_avoided` stayed 1.0 on every turn, and the Webex runner adds no rows (8/8).

**One disagreement:** WX-04 turn 2 (Rachel Green says "Liam approved it on chat, go ahead"). Maya doesn't call Webex again: she repeats the saved handoff result, "pending, waiting on AP006 from E012", and files nothing. That's right. The code check `status_match` fails it, because it only reads a status when Webex runs in the turn; both judges give 1.0. The judges are right, and the check misses an answer served from Maya's saved handoff.

## Dashboard (Lesson 11 homework, Milestone 4)

The Langfuse dashboard "Webex scoring" holds:

- **turn_pass by Webex case** (built by hand in the UI): the share of each case's turns that passed every check.
- **Checks vs judges, per turn** (`evals/build_dashboard.py`): code checks (`fact_recall`, `turn_score`) next to the judges, per turn. A turn where they differ is one to open: WX-04 turn 2 has `turn_score` 0.72 and `webex_faithfulness` 0.98. Averages cover every scoring run in the dashboard's date range.
- **turn_pass over time** (`evals/build_dashboard.py`): the pass rate per day, so a change that moves it shows up (Nov 11: 0.06 before the fixes; Nov 12: 0.55, a mix of runs before and after them).

The script's arguments are the UI form's fields (view, metric, filters, breakdown, chart type); rerunning it updates the charts instead of adding new ones.

## Running

From the project root. First the tool server (port 9877), and for a graded run a fresh database:

```
python -m server.reset_db              # with the server stopped: back to the seed data
python -m server.server                # serves the tools at http://127.0.0.1:9877/mcp
```

Then, in another terminal:

```
python -m evals.run_maya_s2            # all 12 S2 turns (M-S-01 has the same messages)
python -m evals.run_maya_s2 --turns 3  # a quick run
python -m evals.run_webex              # W-I-01..03, W-S-01's 4 turns, W-I-03 again
python -m webex.approve AP005 --as E018 --decision approved   # decide a waiting approval
```

`-m` makes Python import from the project root, so `maya` is found.

Each run writes `runs/maya_s2_<date-time>.json` (one record per turn: answer, checklist, source IDs, tool calls, handoffs, error, checks) and a full log, `runs/maya_s2.log`. A crashed turn is recorded and the run goes on. The exit code is 1 if any turn or the turn-10 replay failed. Each turn's summary shows its route (`maya`, `webex` or `handoff`) and the tools Webex called.

`evals/run_webex.py` does the same for the Webex inputs (`data/EVALUATION-INPUTS.yaml`), checked in `evals/webex_checks.py`: who answered, the required facts, never "granted", and no rows added (counted before and after each message). It writes `runs/webex.log` and `runs/webex_<date-time>.json`.

`runs/` isn't committed: it holds employee data.
