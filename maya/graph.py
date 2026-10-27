"""The Maya agent's graph: its state, its steps in run order, and the wiring.

Each turn runs these steps:
  1. check  - is it time to distil? (enough new messages since the last time)
  2. distil - (only when step 1 says yes) turn the new messages into updated memory.
  3. plan     - the planner reads the memory + newer messages, and writes the plan.
  4. retrieve - plain code searches the documents with the plan's search_query, as
                the caller (who is set by code, never by a model).
  5. select   - plain code picks the turn's loadout (the tools the model may see).
  6. model    - the answer call: the plan + the last two turns, with only the
                loadout's tools attached.
  7. route    - after the model replies, decide what happens next (8a, 8b, or end).
  8a. tools   - (when the model asked for tools) run them, then back to step 6.
  8b. rearm   - (when a needed lookup was missed) delete that answer, and retry
                step 6 once with every read tool.

Read in this order: AgentState, then the steps, then build_graph at the bottom.
"""

from functools import partial
from typing import Annotated, TypedDict

from langchain_core.messages import RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from maya.loadout import REQUIRED_COLLECTIONS, WRITE_TOOLS, select_tools
from maya.memory import (
    ConversationMemory,
    distiller_messages,
    memory_is_safe,
    needs_distillation,
    planner_messages,
)
from maya.model import get_model
from maya.planning import TurnPlan, answer_messages, message_text, recent_turns
from maya.retrieval import EvidenceRetriever
from maya.schemas import CallerContext, Evidence


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
    rearmed: bool                           # has this turn's fallback been used?
    memory: dict                            # the notes (a ConversationMemory, as a dict)
    distilled_upto: int                     # messages before this index are in the memory
    evidence: list[dict]                    # this turn's document chunks (Evidence, as dicts)


# THE STEPS -------------------------------------------------------------------
# Each step gets the state, plus whatever build_graph() passes in, and returns only
# the state fields it changes.

def distil_or_plan(state: AgentState) -> str:
    """Step 1: at the start of a turn, decide whether to distil first.

    Returns "distil" if enough new messages have piled up (see needs_distillation),
    otherwise "plan". Most turns skip distillation, so it costs nothing on them.
    """

    if needs_distillation(state):
        return "distil"
    return "plan"


async def distil(state: AgentState, distiller) -> dict:
    """Step 2 (only when step 1 says so): turn the new messages into updated memory.

    - Sends the old memory plus every message since it to the distiller.
    - Keeps the new memory only if memory_is_safe() accepts it. Otherwise nothing
      changes: the old memory and every message stay as they were.
    - Never deletes a message. It only moves `distilled_upto`, so the planner stops
      reading the messages that are now in the memory.
    """

    result = await distiller.ainvoke(distiller_messages(state))
    new_memory = result["parsed"]
    old_memory = state.get("memory") or {}

    if new_memory is None or not memory_is_safe(old_memory, new_memory):
        return {}

    return {
        "memory": new_memory.model_dump(),
        "distilled_upto": len(state["messages"]),  # everything so far is now in memory
    }


async def plan_turn(state: AgentState, planner) -> dict:
    """Step 3: ask the planner for this turn's plan, and save it in the state.

    The planner gets the memory plus only the newer messages (planner_messages in
    memory.py). If its answer can't be read as a TurnPlan, use a fallback plan:
    topic "other", tools needed.
    """

    result = await planner.ainvoke(planner_messages(state))
    plan = result["parsed"]
    if not plan:
        plan = TurnPlan(current_intent="other", requires_tools=True)
    return {"plan": plan.model_dump()}


def caller_from_config(config: RunnableConfig) -> CallerContext | None:
    """Who is asking, from the run's config. None if caller_id or user_group is missing.

    The caller is set by the code that starts the turn (config["configurable"]), never
    read from a message or a model's output, so no prompt can change it.
    """

    settings = config.get("configurable", {})
    caller_id = settings.get("caller_id")
    user_group = settings.get("user_group")
    if not caller_id or not user_group:
        return None
    return CallerContext(caller_id=caller_id, user_group=user_group)


def missing_collections(intent: str, evidence: list[Evidence]) -> list[str]:
    """The collections this topic needs (REQUIRED_COLLECTIONS) that the evidence lacks."""

    found = {chunk.collection for chunk in evidence}
    return [name for name in REQUIRED_COLLECTIONS.get(intent, []) if name not in found]


def merge_evidence(first: list[Evidence], extra: list[Evidence]) -> list[Evidence]:
    """The first search's chunks, then the extra search's new ones (no duplicates)."""

    seen = {chunk.chunk_id for chunk in first}
    return first + [chunk for chunk in extra if chunk.chunk_id not in seen]


def retrieve(state: AgentState, config: RunnableConfig, retriever: EvidenceRetriever) -> dict:
    """Step 4: search the documents for this turn. No model call of its own.

    - Unknown caller: no evidence ("fail closed": no search beats a search as nobody).
    - The plan needs no lookup: no search.
    - Otherwise: one search with the plan's search_query (or the user's message, if
      the planner left it empty).
    - If the results lack a collection the topic needs, one more search, limited to
      the missing collections. Never a third, even if that one finds nothing.
    Always sets `evidence`, so last turn's chunks never carry over.

    LangGraph passes `config` in because the function has a parameter of that name.
    """

    caller = caller_from_config(config)
    plan = TurnPlan(**state["plan"])
    if caller is None or not plan.requires_tools:
        return {"evidence": []}

    query = plan.search_query or message_text(state["messages"][-1])
    evidence = retriever.search(query, caller)

    missing = missing_collections(plan.current_intent, evidence)
    if missing:
        extra = retriever.search(query, caller, collections=missing)
        evidence = merge_evidence(evidence, extra)
    return {"evidence": [chunk.model_dump() for chunk in evidence]}


def select(state: AgentState, all_tool_names: list[str]) -> dict:
    """Step 5: turn the plan into this turn's loadout. No model call.

    select_tools() returns a list of tool names ([] means none), or None for topic
    "other", which means every tool. Note: "every tool" includes the write tool.
    Also resets `rearmed` for the new turn.
    """

    chosen = select_tools(TurnPlan(**state["plan"]))
    if chosen is None:
        loadout = all_tool_names
    else:
        loadout = chosen
    return {"loadout": loadout, "rearmed": False}


def model_with_tools(base_model, tools_by_name: dict, names: list[str]):
    """Return the model with exactly these tools attached (no names: no tools)."""

    if not names:
        return base_model
    chosen_tools = [tools_by_name[name] for name in sorted(names)]
    return base_model.bind_tools(chosen_tools)


def call_model(state: AgentState, base_model, tools_by_name: dict) -> dict:
    """Step 6: ask the model for an answer, with only the loadout's tools.

    It gets the plan plus the last two turns (answer_messages in planning.py).
    Returns the model's reply, which may be an answer or a request to call tools.
    """

    model = model_with_tools(base_model, tools_by_name, state["loadout"])
    plan = TurnPlan(**state["plan"])  # the state stores the plan as a plain dict
    response = model.invoke(answer_messages(state, plan))
    return {"messages": [response]}


def used_tool_this_turn(messages: list) -> bool:
    """True if any tool ran in the current turn (since the latest user message)."""

    current_turn = recent_turns(messages, keep=1)
    return any(isinstance(message, ToolMessage) for message in current_turn)


def lookup_was_missed(state: AgentState) -> bool:
    """True if the plan needed a lookup, none happened, and we haven't retried yet."""

    plan = TurnPlan(**state["plan"])
    return (
        plan.requires_tools
        and not state["rearmed"]
        and not used_tool_this_turn(state["messages"])
    )


def route(state: AgentState) -> str:
    """Step 7: after the model replies, choose the next step.

    - The model asked for tools          -> "tools" (step 8a)
    - It answered, but missed a lookup   -> "rearm" (step 8b, retry once)
    - It answered                        -> END (the turn is over)
    """

    if state["messages"][-1].tool_calls:
        return "tools"
    if lookup_was_missed(state):
        return "rearm"
    return END


def rearm(state: AgentState, all_tool_names: list[str]) -> dict:
    """Step 8b (only when a lookup was missed): retry once with every read tool.

    - Loadout: every read tool. The write tool stays only if it was already allowed
      this turn, so a retry can never be what turns the write on.
    - Deletes the answer the model just wrote without a lookup (the user never saw it).
    - Sets `rearmed`, so route() won't send this turn here a second time.
    """

    allowed = [
        name for name in all_tool_names
        if name not in WRITE_TOOLS or name in state["loadout"]
    ]
    unseen_answer = state["messages"][-1]
    return {
        "loadout": allowed,
        "rearmed": True,
        "messages": [RemoveMessage(id=unseen_answer.id)],  # delete by id, don't append
    }


# WIRING ----------------------------------------------------------------------

def build_graph(tools: list, retriever: EvidenceRetriever):
    """Create the models, then connect the steps into a graph.

    `retriever` searches the documents (an OpenSearchRetriever, or a fake in tests).

    `partial(f, x=...)` makes a copy of f with x already filled in, so LangGraph can
    call each step with just the state.
    """

    base_model = get_model()  # tools are attached per turn, in call_model
    tools_by_name = {tool.name: tool for tool in tools}
    all_tool_names = sorted(tools_by_name)  # all ten, including the write tool
    # The planner must reply with a TurnPlan. include_raw=True returns a dict whose
    # "parsed" is None when the reply can't be read as one, instead of raising an error.
    planner = get_model().with_structured_output(TurnPlan, include_raw=True)
    distiller = get_model().with_structured_output(ConversationMemory, include_raw=True)

    graph = StateGraph(AgentState)
    graph.add_node("distil", partial(distil, distiller=distiller))
    graph.add_node("plan", partial(plan_turn, planner=planner))
    graph.add_node("retrieve", partial(retrieve, retriever=retriever))
    graph.add_node("select", partial(select, all_tool_names=all_tool_names))
    graph.add_node("model", partial(
        call_model, base_model=base_model, tools_by_name=tools_by_name,
    ))
    graph.add_node("rearm", partial(rearm, all_tool_names=all_tool_names))
    # Step 8a: LangGraph's ToolNode runs the tools the model asked for.
    graph.add_node("tools", ToolNode(tools))

    # START -> (distil) -> plan -> retrieve -> select -> model, then
    # model -> tools / rearm / END,
    # and tools and rearm both go back to model.
    graph.add_conditional_edges(
        START, distil_or_plan, {"distil": "distil", "plan": "plan"},
    )
    graph.add_edge("distil", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "select")
    graph.add_edge("select", "model")
    graph.add_conditional_edges(
        "model", route, {"tools": "tools", "rearm": "rearm", END: END},
    )
    graph.add_edge("tools", "model")
    graph.add_edge("rearm", "model")

    # The checkpointer saves the state per conversation (thread_id) between turns, so
    # each turn starts with the messages of all earlier turns.
    return graph.compile(checkpointer=InMemorySaver())
