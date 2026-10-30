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

Maya loads 5 of the MCP server's 10 tools: `get_employee`, `list_onboarding_tasks`, `check_asset_inventory`, `check_software_subscription` and `list_employee_tickets` (an allowlist in `maya/loadout.py`).

- **The write tool** (`create_access_request`) is never loaded, so no fallback can expose it.
- **The 4 document tools** (`search_hr_documents`, `search_knowledge_base`, `list_policies`, `get_policy`) are never loaded either. They return any document to anyone. Documents come only from Maya's own OpenSearch search, which filters by who is asking.
- **Effect on the golden dataset:** S2 turns 2, 6 and 9 expect a document tool. Maya's search covers them instead, so the smoke test accepts retrieved evidence in place of those tools.

## Running

From the project root (needs Lesson 10's MCP server running on 127.0.0.1:9877):

```
python -m scripts.smoke_s2
```

`-m` makes Python import from the project root, so `maya` is found.
