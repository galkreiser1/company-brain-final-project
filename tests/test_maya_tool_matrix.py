"""The matrix test: can every S2 turn reach the tools it needs? (No model calls.)

    .venv/bin/python -m pytest -v tests/test_maya_tool_matrix.py

One case per (turn, accepted topic) in session S2 of the golden dataset. The planner
may pick any accepted topic, so each one's loadout must hold the turn's tools. A
failing case is a bug in LOADOUTS (or REQUIRED_COLLECTIONS), not in the model: fix
the table, not the test.

Then, once: the write tool is unreachable on every path.
"""

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from maya.config import REPO_ROOT
from maya.graph import rearm
from maya.loadout import (
    DOCUMENT_TOOLS, LOADOUTS, MAYA_READ_TOOLS, WRITE_TOOLS, maya_tools, select_tools,
)
from maya.planning import REQUIRED_COLLECTIONS, TurnPlan, fallback_plan


GOLDEN_DATASETS = REPO_ROOT / "data" / "GOLDEN-DATASETS.json"  # same as smoke_s2.py
SESSION_ID = "S2-onboarding-maya"
WRITE_TOOL = "create_access_request"

# data/ isn't committed, so without it there's nothing to test against.
if not GOLDEN_DATASETS.exists():
    pytest.skip(f"{GOLDEN_DATASETS} not found", allow_module_level=True)


def matrix_cases() -> list:
    """One pytest case per (turn, accepted topic). Recall turns (topic null) are skipped."""

    sessions = json.loads(GOLDEN_DATASETS.read_text())["sessions"]
    session = next(s for s in sessions if s["id"] == SESSION_ID)
    cases = []
    for number, turn in enumerate(session["turns"], start=1):
        if not turn.get("expected_intent"):
            continue
        for topic in turn["expected_intent"].split("|"):
            cases.append(pytest.param(topic, turn.get("expected_tools", []),
                                      id=f"turn{number:02d}-{topic}"))
    return cases


def plan_for(topic: str, action_confirmed: bool = False) -> TurnPlan:
    return TurnPlan(current_intent=topic, needs_documents=True, needs_records=True,
                    action_confirmed=action_confirmed)


# THE MATRIX ------------------------------------------------------------------

@pytest.mark.parametrize("topic, expected_tools", matrix_cases())
def test_topic_reaches_the_turns_tools(topic, expected_tools):
    loadout = select_tools(plan_for(topic))
    for tool in expected_tools:
        if tool in MAYA_READ_TOOLS:
            assert tool in loadout, f"{tool} is not in the {topic} loadout"
        elif tool in DOCUMENT_TOOLS:
            # Maya doesn't load document tools: its own search must cover this topic.
            assert REQUIRED_COLLECTIONS.get(topic), f"{topic} has no collections to search"
        else:
            pytest.fail(f"unknown tool {tool!r}")


# THE WRITE TOOL IS UNREACHABLE -----------------------------------------------

def fake_tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(lambda: "{}", name=name, description=name)


# The ten tools the Lesson 10 MCP server offers.
MCP_TOOL_NAMES = sorted(set(MAYA_READ_TOOLS) | DOCUMENT_TOOLS | WRITE_TOOLS)


def test_the_ten_mcp_tool_names_are_known():
    assert len(MCP_TOOL_NAMES) == 10 and WRITE_TOOL in MCP_TOOL_NAMES


def test_write_tool_is_never_loaded():
    loaded = [tool.name for tool in maya_tools([fake_tool(n) for n in MCP_TOOL_NAMES])]
    assert loaded == list(MAYA_READ_TOOLS)


@pytest.mark.parametrize("action_confirmed", [False, True])
@pytest.mark.parametrize("topic", sorted(LOADOUTS))
def test_write_tool_is_in_no_topic_loadout(topic, action_confirmed):
    assert WRITE_TOOL not in select_tools(plan_for(topic, action_confirmed))


def test_write_tool_is_not_in_the_rearm_loadout():
    loaded = [tool.name for tool in maya_tools([fake_tool(n) for n in MCP_TOOL_NAMES])]
    state = {"messages": [AIMessage("an answer without a lookup", id="a1")]}
    assert WRITE_TOOL not in rearm(state, all_tool_names=loaded)["loadout"]


def test_write_tool_is_not_in_the_fallback_loadout():
    assert WRITE_TOOL not in select_tools(fallback_plan())
