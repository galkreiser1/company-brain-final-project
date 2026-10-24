"""The Maya agent's graph: its state, its steps in run order, and the wiring.

Each turn runs these steps:
  1. plan  - the planner reads the conversation, and writes this turn's plan.
  2. model - the answer call: the plan + the conversation, with every tool attached.
  3. route - after the model replies, decide what happens next (step 4, or end).
  4. tools - (when the model asked for tools) run them, then back to step 2.

Read in this order: AgentState, then the steps, then build_graph at the bottom.
"""

from functools import partial
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

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


def call_model(state: AgentState, model) -> dict:
    """Step 2: ask the model for an answer, with every tool attached.

    It gets the plan (in its system prompt) plus the conversation (answer_messages).
    Returns the model's reply, which may be an answer or a request to call tools.
    """

    plan = TurnPlan(**state["plan"])  # the state stores the plan as a plain dict
    response = model.invoke(answer_messages(state, plan))
    return {"messages": [response]}


def route(state: AgentState) -> str:
    """Step 3: after the model replies, choose the next step.

    - The model asked for tools  -> "tools" (step 4)
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

    # All ten tools are attached once, so every model call sees every tool's schema.
    model = get_model().bind_tools(tools)
    # The planner must reply with a TurnPlan. include_raw=True returns a dict whose
    # "parsed" is None when the reply can't be read as one, instead of raising an error.
    planner = get_model().with_structured_output(TurnPlan, include_raw=True)

    graph = StateGraph(AgentState)
    graph.add_node("plan", partial(plan_turn, planner=planner))
    graph.add_node("model", partial(call_model, model=model))
    # Step 4: LangGraph's ToolNode runs the tools the model asked for.
    graph.add_node("tools", ToolNode(tools))

    # START -> plan -> model, then model -> tools / END, and tools goes back to model.
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    # The checkpointer saves the state per conversation (thread_id) between turns, so
    # each turn starts with the messages of all earlier turns.
    return graph.compile(checkpointer=InMemorySaver())
