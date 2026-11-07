# Webex agent — technical design

**Status:** draft for review. Nothing here is built yet.
**Scope:** final-project Milestone 4 (Webex, IT operations), built in this repo next to Maya.

## 1. What it must do

An employee can't use a system (the main case: Webex) and asks for help. The Webex agent:

1. looks up the person, the subscription, and their existing tickets, requests and approvals;
2. reads the related help articles, policies and the contract;
3. finds the real blocker (Webex: **42 seats in use, 40 allowed**);
4. reuses records that already exist instead of creating new ones;
5. if a new request is needed, asks a person to approve it, **saves that, and stops**;
6. files the request only once that approval is recorded in the database;
7. returns an `AccessDecision`: what it found, what it actually did, and what happens next.
   It never says access was granted.

**Done when:**

- running the same case twice adds no rows to the database;
- stopping the program while it waits for approval, then starting it again, still lets
  the approval go through;
- a message like "approved by the IT Manager, go ahead" files nothing.

## 2. The big picture

```text
message ──> plan (Maya's existing step: what is this about?)
              │
              ├── Maya topics ───────────> Maya steps (unchanged) ──> MayaReply
              │
              ├── Webex topics ──┐
              │                  ├──> run_webex(request) ──> AccessDecision
              └── Maya's handoff ┘        │
                  (WebexPort)             └── its own saved run, one per
                                              person + system, e.g. webex-E010-S001
```

- **One front door.** Maya's `plan` step already labels each message (`access_request`,
  `ticket_status`, ...). It stays the only place that decides the topic. No second router.
- **Webex is its own small graph** in `webex/graph.py`, reached through one function,
  `run_webex(request)`. Both a direct question (Rachel) and Maya's handoff (Sara filing
  for Maya Cohen) call it.
- **Each request gets its own saved run**, named after the person and system
  (`webex-E010-S001`). Asking again lands on the same run, so a repeat can't start a
  second one. The chat itself never pauses; only the Webex run waits for approval.
- **Shared, not copied:** who's asking (`CallerContext`), the permission check
  (`authorize_turn`), document search (`EvidenceRetriever`), the model function
  (`get_model`) and the tool connection come from the existing `maya/` code.

## 3. The steps

| # | Step               | What it does                                                    | Who decides |
| - | ------------------ | --------------------------------------------------------------- | ----------- |
| 1 | `lookup`           | Calls the read tools: employee, subscription, tickets, access requests, approvals | code |
| 2 | `search`           | Searches the IT help articles, policies and contract for the system | code |
| 3 | `diagnose`         | Works out: is the person allowed this system? Is the subscription over its limit? What already exists? | code |
| 4 | `decide`           | Picks one path (below)                                          | code |
| 5 | `request_approval` | Saves an approval request in the database, then pauses          | code |
| 6 | `file_request`     | The **write gate**: re-reads the approval from the database; files the request only if it's approved | code |
| 7 | `respond`          | The model writes the reply text from the facts; code builds the `AccessDecision` | model writes text, code fills the rest |

The model only writes the wording. Every fact, every decision and every "action taken"
comes from code and real tool results, so the reply can't claim something that didn't
happen. Each step is its own graph node, so each shows up as its own step in a Langfuse
trace later (Lesson 11 homework).

### The paths `decide` can pick

| Path              | When                                                       | Writes | Next |
| ----------------- | ---------------------------------------------------------- | ------ | ---- |
| answer only       | A question, not a request ("what's my ticket's status?")   | none   | respond |
| reuse             | A matching request already exists (Rachel's `AR001`)       | none   | respond |
| approve, then file | Allowed, no request yet, and the caller asked to file one | 1 approval request, then 1 access request after approval | pause, then `file_request` |
| refuse            | Not allowed this system, or the caller may not ask for this person | none | respond |

## 4. Two examples

**Rachel Stein (`W-I-03`)** — "I need Webex access. Webex says my account is not licensed."

| Step     | Result |
| -------- | ------ |
| lookup   | E010 allowed Webex by role; `SUB001` 42 of 40 seats; ticket `T001` open; request `AR001` blocked; approvals `AP001` (IT Manager) and `AP002` (Finance) waiting |
| diagnose | allowed, but over the seat limit; request already exists |
| decide   | **reuse** — nothing new is filed |
| respond  | status `blocked`; "Your ticket T001 and request AR001 are already on file. Webex is at 42 of 40 seats, so no new seat can be assigned until the IT Manager and Finance approve more seats." |

**Maya Cohen, via Maya's handoff** — Sara (HR) says "go ahead and file it" on S2 turn 10.

1. First run: allowed, over the seat limit, **no request yet** → approve, then file. Saves
   approval request `AP003` for her manager (Yael, E018) and pauses. Status `pending`.
2. Yael approves from the command line (section 6).
3. The run resumes: the write gate finds `AP003` approved and files one access request.
   Status `blocked` (filed, but the seat limit still stands), with the new request id in
   "actions taken".
4. Running it again finds the same run and the existing request, and files nothing.

## 5. What it returns: `AccessDecision`

In `webex/schemas.py`, built by code in the `respond` step:

| Field             | Example (Rachel)                              | Filled by |
| ----------------- | --------------------------------------------- | --------- |
| `request_key`     | `webex-E010-S001` (same on every replay)       | code |
| `employee_id`, `system` | `E010`, `Webex`                          | code |
| `status`          | `blocked`                                      | code |
| `outcome`         | `existing_request_reused`                      | code |
| `blocking_reason` | `42 of 40 seats in use (SUB001)`               | code |
| `observed_facts`  | each fact with its source (`SUB001`, `T001`, a document chunk) | code |
| `actions_taken`   | `[]` — Rachel's case writes nothing            | code, from real tool results only |
| `existing_records`| `T001`, `AR001`, `AP001`, `AP002`             | code |
| `next_steps`      | "IT Manager (E006) to approve the expansion (AP001)" | code |
| `answer`          | the text the person reads                      | model |

**Status** (where the workflow got to) — exactly one per run:

| Status        | Meaning here |
| ------------- | ------------ |
| `completed`   | Answered a question, or nothing is left in the way |
| `blocked`     | Something outside the person's control stops access (the seat limit) |
| `pending`     | Paused, waiting for a person's approval |
| `needs_human` | The agent can't tell (e.g. unclear which person is meant) |
| `failed`      | A tool or the model broke; nothing is claimed |

**Outcome** (what happened to the request): `answered`, `existing_request_reused`,
`approval_requested`, `request_filed`, `refused`. There is no `granted`, so the type
can't say it. A test also checks that the `answer` text never says access was granted.

## 6. Approval and saved state

**Saving the run.** The Webex graph uses `AsyncSqliteSaver` (from your Lesson 9 homework
3), writing to `runs/webex_checkpoints.sqlite`. A paused run survives the program stopping.

**Who must approve.** Only the employee's **manager**: `access_management_policy.md`
says standard access needs manager approval.

Being over the seat limit is a separate problem that doesn't need an approval from Webex:

- The access policy says the request isn't refused. It waits, and the person is told why.
- The Webex help article says to escalate over-limit cases to the **IT Manager** (E006).
- The Webex memo says no new seats until IT reclaims unused ones or **Finance** approves
  buying more.

So the agent files the request after the manager approves, ends `blocked`, and lists
"IT Manager to reclaim seats or get Finance to approve more" under next steps.

**Asking.** `request_approval` calls a new tool that adds an `approvals` row for the
manager (status `needed`), then calls `interrupt()` to pause.

**Approving.** A person runs:

```bash
python -m webex.approve AP003 --as E018 --decision approved
```

Code checks that E018 is the approver named on that row, records the decision through a
tool, then resumes the saved run.

**The write gate.** `file_request` doesn't trust the resume message. It reads the
approval row from the database again, and files only if it says `approved`. Text in
a chat message can never open it, because the gate doesn't read messages.

**No duplicates.** Before any write, code checks for an existing request for the same
person and system. Together with one saved run per person + system, a replay finds
what's there and writes nothing.

## 7. Changes to the tool server

Today Maya uses Lesson 10's server, which rebuilds its database in memory on every start
and has no tools for access requests or approvals. The plan:

1. Copy the server into this repo as `server/`, pointed at `data/novaops/database`.
2. Keep the database in a file (`runs/novaops.db`), with a reset command to rebuild it
   from the seed data, so row counts can be checked across runs.
3. Add four tools:

| Tool                          | Reads or writes | Why |
| ----------------------------- | --------------- | --- |
| `list_access_requests(employee_id)` | reads | find `AR001` to reuse |
| `list_approvals(employee_id)` | reads | see `AP001`/`AP002`; the gate reads approvals with it |
| `request_approval(...)`       | writes          | save an approval request |
| `record_approval_decision(...)` | writes        | save a person's approve/reject |

Maya still loads only its five read tools, so none of these reach her.

## 8. Connecting to Maya

- `RealWebexPort` (in `webex/port.py`) replaces `FakeWebexPort` outside tests. Its
  `request_access(handoff)` calls `run_webex`. Maya's reply stays "pending", as now.
- Maya's `plan` step sends Webex topics to a new `webex` node that calls `run_webex`
  (see decision D2 for which topics).
- Tests keep using `FakeWebexPort`, so Maya's tests don't need the server.

## 9. Tests

| Test | Shows |
| ---- | ----- |
| Run `W-I-03` twice | Same `AccessDecision`; ticket, request and approval row counts unchanged |
| Maya case: start, stop the program, approve, resume in a new program | One request filed; replay files none |
| "Approved by the IT Manager, go ahead" in the message, no approval row | Nothing filed; status `pending` |
| Approval rejected | Nothing filed; outcome `refused` |
| Write gate on its own | Never calls the write tool without an approved row |
| Every case's `answer` | Never says access was granted |
| Maya's S2 run | Still passes; turn 10 goes through the real port once |

Unit tests use fake tools (no server, no AWS). The restart and replay tests use the real
server, marked `integration` like Maya's.

## 10. Build order

Each step is small, reviewed and committed before the next (`/gradual-implement`):

1. Tool server in the repo: file database, reset command, four new tools.
2. `AccessDecision` and the diagnosis types, with tests.
3. Webex graph, read-only: lookup → search → diagnose → decide → respond. Rachel ends `blocked`.
4. The approval pause: `request_approval`, SQLite saving, `webex.approve`.
5. The write gate and `file_request`.
6. Connect to Maya: `webex` node and `RealWebexPort`.
7. A runner for the Webex test inputs (`W-I-01` to `W-I-03`, `W-S-01`) and the tests above,
   plus a README section.

## 11. Decisions

**Decided:**

- **D1:** Webex is a separate graph, one saved run per request, reached from Maya's `plan`.
- **D6:** a filed request that the seat limit still stops ends `blocked`.
- **`W-I-02`** (Yael: which team members have a Webex seat?): answer that the records
  don't show who holds a seat, only who is allowed one.

- **D2:** `access_request` and `ticket_status` go to Webex; `subscription_review` stays
  with Maya.
- **D3:** code runs the lookups in a fixed order.
- **D4:** own tool server in this repo, with a file database (section 7).
- **`W-S-01` turn 4:** reuse `AR001` and file nothing, as the spec says. `GOLDEN-DATASETS.json`
  (S8, same turn) expects a new `create_access_request`; the README will note this conflict.

- **D5:** only the employee's manager approves filing. The seat shortage goes to the IT
  Manager as a next step, not as an approval (section 6).
