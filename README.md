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
- **`AccessDecision`** (`webex/schemas.py`). Code fills every field: observed facts with their sources, actions taken (copied from real tool results), existing records, next steps. The model writes only `answer`; code adds a status line ("Status: blocked (request AR001).") and the type has no "granted" outcome.
- **Tool server** (`server/`, from Lesson 10). The database is a file, `runs/novaops.db`, so writes survive restarts. It adds four tools: `list_access_requests`, `list_approvals`, `request_approval`, `record_approval_decision`. Webex's allowlist is in `webex/tools.py`.

Proof of the durable behaviour (unit tests in `tests/test_webex_steps.py`, no server or AWS):

- replaying a request adds no rows: `test_asking_again_while_paused_adds_no_rows`, `test_asking_again_after_filing_reuses_it`;
- a pending approval survives a restart: `test_the_approval_survives_a_restart`;
- the write gate needs a recorded approval: `test_the_gate_never_files_without_an_approved_row`, `test_a_go_ahead_message_cannot_approve`.

`W-S-01` turn 4 reuses `AR001` and files nothing, as the spec's reference scenario says. `GOLDEN-DATASETS.json` (S8, the same turn) expects a new `create_access_request`; we follow the spec.

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
