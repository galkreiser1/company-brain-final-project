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

## Running

From the project root (needs Lesson 10's MCP server running on 127.0.0.1:9877):

```
python -m scripts.smoke_s2
```

`-m` makes Python import from the project root, so `maya` is found.
