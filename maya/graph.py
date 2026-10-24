"""The Maya agent's graph: its state, its steps in run order, and the wiring.

Each turn runs these steps:
  1. plan   - the planner reads the conversation, and writes this turn's plan.
  2. select - plain code picks the turn's loadout (the tools the model may see).
  3. model  - the answer call: the plan + the last two turns, with only the loadout's
              tools attached.
  4. route  - after the model replies, decide what happens next (step 5, or end).
  5. tools  - (when the model asked for tools) run them, then back to step 3.

Read in this order: AgentState, then the steps, then build_graph at the bottom.
"""

from functools import partial
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from maya.loadout import select_tools
from maya.model import get_model
from maya.planning import TurnPlan, answer_messages, planner_messages


# STATE -----------------------------------------------------------------------

class AgentState(TypedDict):
    """Everything saved for one conversation.

    `messages` appends: its reducer (add_messages, the function that merges a step's
    update into the old value) adds each new message to the end of the list. The other
    fields are replaced by each update.
    """

    messages: Annotated[list, add_messages]  # the whole conversation
    plan: dict                              # this turn's plan (a TurnPlan, as a dict)
    loadout: list[str]                      # tools the model may see this turn


# THE STEPS -------------------------------------------------------------------
# Each step gets the state, plus whatever build_graph() passes in, and returns only
# the state fields it changes.

async def plan_turn(state: AgentState, planner) -> dict:
    """Step 1: ask the planner for this turn's plan, and save it in the state.

    The planner gets the whole conversation as text (planner_messages in
    planning.py). If its answer can't be read as a TurnPlan, use a fallback plan:
    topic "other", tools needed.
    """

    result = await planner.ainvoke(planner_messages(state))
    plan = result["parsed"]
    if not plan:
        plan = TurnPlan(current_intent="other", requires_tools=True)
    return {"plan": plan.model_dump()}


def select(state: AgentState, all_tool_names: list[str]) -> dict:
    """Step 2: turn the plan into this turn's loadout. No model call.

    select_tools() returns a list of tool names ([] means none), or None for topic
    "other", which means every tool. Note: "every tool" includes the write tool.
    """

    chosen = select_tools(TurnPlan(**state["plan"]))
    if chosen is None:
        loadout = all_tool_names
    else:
        loadout = chosen
    return {"loadout": loadout}


def model_with_tools(base_model, tools_by_name: dict, names: list[str]):
    """Return the model with exactly these tools attached (no names: no tools)."""

    if not names:
        return base_model
    chosen_tools = [tools_by_name[name] for name in sorted(names)]
    return base_model.bind_tools(chosen_tools)


def call_model(state: AgentState, base_model, tools_by_name: dict) -> dict:
    """Step 3: ask the model for an answer, with only the loadout's tools.

    It gets the plan plus the last two turns (answer_messages in planning.py).
    Returns the model's reply, which may be an answer or a request to call tools.
    """

    model = model_with_tools(base_model, tools_by_name, state["loadout"])
    plan = TurnPlan(**state["plan"])  # the state stores the plan as a plain dict
    response = model.invoke(answer_messages(state, plan))
    return {"messages": [response]}


def route(state: AgentState) -> str:
    """Step 4: after the model replies, choose the next step.

    - The model asked for tools  -> "tools" (step 5)
    - It answered                -> END (the turn is over)
    """

    if state["messages"][-1].tool_calls:
        return "tools"
    return END


# WIRING ----------------------------------------------------------------------

def build_graph(tools: list):
    """Create the models, then connect the steps into a graph.

    `partial(f, x=...)` makes a copy of f with x already filled in, so LangGraph can
    call each step with just the state.
    """

    base_model = get_model()  # tools are attached per turn, in call_model
    tools_by_name = {tool.name: tool for tool in tools}
    all_tool_names = sorted(tools_by_name)  # all ten, including the write tool
    # The planner must reply with a TurnPlan. include_raw=True returns a dict whose
    # "parsed" is None when the reply can't be read as one, instead of raising an error.
    planner = get_model().with_structured_output(TurnPlan, include_raw=True)

    graph = StateGraph(AgentState)
    graph.add_node("plan", partial(plan_turn, planner=planner))
    graph.add_node("select", partial(select, all_tool_names=all_tool_names))
    graph.add_node("model", partial(
        call_model, base_model=base_model, tools_by_name=tools_by_name,
    ))
    # Step 5: LangGraph's ToolNode runs the tools the model asked for.
    graph.add_node("tools", ToolNode(tools))

    # START -> plan -> select -> model, then model -> tools / END, and tools goes back
    # to model.
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "select")
    graph.add_edge("select", "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    # The checkpointer saves the state per conversation (thread_id) between turns, so
    # each turn starts with the messages of all earlier turns.
    return graph.compile(checkpointer=InMemorySaver())
