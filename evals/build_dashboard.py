"""Add the scripted charts to the "Webex scoring" Langfuse dashboard (Milestone 4).

The first chart, "turn_pass by Webex case", was built by hand in the Langfuse UI. These
use the same fields through the API; each argument is a field of the UI's widget form:

    UI form field            API argument
    View                     view           ("scores-numeric", "scores-boolean", ...)
    Metric (Value, Avg)      metrics        [{"measure": "value", "agg": "avg"}]
    Filters                  filters        [{"column": "name", "operator": "any of", ...}]
    Breakdown Dimension      dimensions     [{"field": "tags"}]
    Chart Type               chart_type     ("PIVOT_TABLE", "LINE_TIME_SERIES", ...)
    Name, Description        name, description

The dashboards API is marked unstable, so requirements.txt pins langfuse below 5.
Running it again adds nothing: a chart whose name is already on the dashboard is updated
to match CHARTS instead.

    .venv/bin/python -m evals.build_dashboard
"""

import maya.config  # noqa: F401  (loads the LANGFUSE_* keys from .env)
from langfuse import get_client

DASHBOARD = "Webex scoring"

# Only the scoring run's traces (score_webex.py tags them), not the lab's or the runners'.
SCORING_RUNS = {"column": "tags", "operator": "any of", "type": "arrayOptions",
                "value": ["webex-scoring"]}


def score_names(*names: str) -> dict:
    return {"column": "name", "operator": "any of", "type": "stringOptions", "value": list(names)}


CHARTS = [
    {
        # Code checks and judges side by side for each turn: a turn where they disagree
        # is worth opening (WX-04 turn 2: turn_score 0.75 from status_match, judges 1.0).
        # Per turn (the trace name), not per case: a case's turns would average away
        # the one turn that disagrees.
        "name": "Checks vs judges, per turn",
        "description": "Average of each score per turn: code checks (fact_recall, turn_score) "
                       "next to the LLM judges (webex_faithfulness, webex_no_overstatement).",
        "view": "scores-numeric",
        "dimensions": [{"field": "traceName"}, {"field": "name"}],
        "metrics": [{"measure": "value", "agg": "avg"}],
        "filters": [SCORING_RUNS, score_names("fact_recall", "turn_score", "webex_faithfulness",
                                              "webex_no_overstatement")],
        "chart_type": "PIVOT_TABLE",
    },
    {
        # One point per day: did a change move the pass rate? The fixes in "Langfuse 6"
        # should show as a jump.
        "name": "turn_pass over time",
        "description": "Share of scored turns that passed every check, per day.",
        "view": "scores-boolean",
        "dimensions": [],
        "metrics": [{"measure": "value", "agg": "avg"}],
        "filters": [SCORING_RUNS, score_names("turn_pass")],
        # Bars, not a line: a line drops to 0 on days with no run, which reads as "all
        # failed".
        "chart_type": "BAR_TIME_SERIES",
    },
]


def main() -> None:
    dashboards = get_client().api.unstable.dashboards
    widgets = get_client().api.unstable.dashboard_widgets
    dashboard = next((d for d in dashboards.list().data if d.name == DASHBOARD), None)
    if dashboard is None:
        raise SystemExit(f'No "{DASHBOARD}" dashboard: create it in the Langfuse UI first.')

    placed = dashboards.get(dashboard.id).definition.widgets
    on_board = {widgets.get(p.widget_id).name: p.widget_id for p in placed if p.type == "widget"}
    for row, chart in enumerate(CHARTS):
        if chart["name"] in on_board:
            widgets.update(on_board[chart["name"]], **chart)
            print(f"[updated] {chart['name']}")
            continue
        widget = widgets.create(**chart)
        # Right of the hand-built charts, one under the other (the grid is 12 wide).
        dashboards.add_placement(dashboard.id, request={
            "type": "widget", "widgetId": widget.id, "x": 6, "y": row * 6, "width": 6, "height": 6})
        print(f"[created] {chart['name']}")


if __name__ == "__main__":
    main()
