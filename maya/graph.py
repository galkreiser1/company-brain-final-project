"""The Maya agent's graph: its state, its steps in run order, and the wiring.

Each turn runs these steps:
  1. model - the answer call, with every tool attached.
  2. route - after the model replies, decide what happens next (step 3, or end).
  3. tools - (when the model asked for tools) run them, then back to step 1.

Read in this order: AgentState, then the steps, then build_graph at the bottom.
"""

from functools import partial
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from maya.model import get_model
from maya.planning import answer_messages


# STATE -----------------------------------------------------------------------

class AgentState(TypedDict):
    """Everything saved for one conversation.

    `messages` appends: its reducer (add_messages, the function that merges a step's
    update into the old value) adds each new message to the end of the list.
    """

    messages: Annotated[list, add_messages]  # the whole conversation


# THE STEPS -------------------------------------------------------------------
# Each step gets the state, plus whatever build_graph() passes in, and returns only
# the state fields it changes.

def call_model(state: AgentState, model) -> dict:
    """Step 1: ask the model for an answer, with every tool attached.

    Returns the model's reply, which may be an answer or a request to call tools.
    """

    response = model.invoke(answer_messages(state))
    return {"messages": [response]}


def route(state: AgentState) -> str:
    """Step 2: after the model replies, choose the next step.

    - The model asked for tools  -> "tools" (step 3)
    - It answered                -> END (the turn is over)
    """

    if state["messages"][-1].tool_calls:
        return "tools"
    return END


# WIRING ----------------------------------------------------------------------

def build_graph(tools: list):
    """Create the model, then connect the steps into a graph.

    `partial(f, x=...)` makes a copy of f with x already filled in, so LangGraph can
    call each step with just the state.
    """

    # All ten tools are attached once, so every model call sees every tool's schema.
    model = get_model().bind_tools(tools)

    graph = StateGraph(AgentState)
    graph.add_node("model", partial(call_model, model=model))
    # Step 3: LangGraph's ToolNode runs the tools the model asked for.
    graph.add_node("tools", ToolNode(tools))

    # START -> model, then model -> tools / END, and tools goes back to model.
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    # The checkpointer saves the state per conversation (thread_id) between turns, so
    # each turn starts with the messages of all earlier turns.
    return graph.compile(checkpointer=InMemorySaver())
